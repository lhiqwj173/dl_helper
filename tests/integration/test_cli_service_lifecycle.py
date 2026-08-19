"""任务 OSR-002：CLI 服务生命周期接入 —— start/finalize/通知/审计/required 策略。"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from dl_helper.training.config import default_schema


def _base_cfg(tmp_path, run_id, notify_type="none", policy="record"):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["run"]["output_root"] = str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    if notify_type == "wecom":
        schema["notifications"] = {
            "type": "wecom",
            "corp_id_secret_key": "WECOM_CORP_ID",
            "corp_secret_key": "WECOM_CORP_SECRET",
            "agent_id_secret_key": "WECOM_AGENT_ID",
            "to_user": "user1",
            "connect_timeout_seconds": 1,
            "read_timeout_seconds": 1,
            "max_attempts": 2,
            "failure_policy": policy,
        }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    return str(path)


class _FakeWecom:
    def __init__(self, fail_content=()):
        self.sent = []
        self._fail = tuple(fail_content)

    def send_text(self, content, redactor=None):
        if any(m in content for m in self._fail):
            raise ConnectionError("net down")
        self.sent.append(content)


class _FakeStore:
    def publish_run_bundle(self, local_dir, run_id):
        pass

    def publish_sweep_bundle(self, local_dir, sweep_id):
        pass

    def fetch_latest_checkpoint(self, run_id, checkpoints_dir):
        # D-003：省略 --resume 时内部 auto 会向远端查询最新 checkpoint；测试无远端返回 None
        return None


def test_cli_train_invokes_services(tmp_path, monkeypatch):
    """启用服务时 CLI 调用 start_run/finalize_run 并写审计。"""
    from dl_helper.training.services import (
        LifecycleServices,
        ServiceAudit,
        ServiceDeliveryError,
    )

    calls = {"start": [], "finalize": []}

    class _Recording(LifecycleServices):
        def start_run(self, run_id, platform="local"):
            calls["start"].append(run_id)
            super().start_run(run_id, platform=platform)

        def finalize_run(self, run_id, status, **fields):
            calls["finalize"].append((run_id, status))
            super().finalize_run(run_id, status, **fields)

    import dl_helper.training.cli as cli

    def fake_build(config, platform, layout):
        if config.notifications.type != "wecom":
            return None
        wecom = _FakeWecom()
        audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
        return _Recording(layout=layout, secret_resolver=_Resolver(),
                          stores=[_FakeStore()], async_sync=None,
                          wecom_client=wecom, audit=audit, failure_policy="record")

    monkeypatch.setattr(cli, "_build_services", fake_build)
    cfg = _base_cfg(tmp_path, "svc-run", notify_type="wecom", policy="record")
    code = cli.main(["train", "--config", cfg, "--experiment", "experiments.toy_multiclass:build_experiment"])
    assert code == 0
    assert calls["start"] == ["svc-run"]
    assert ("svc-run", "succeeded") in calls["finalize"]
    # 审计写入
    audit_path = os.path.join(str(tmp_path), "runs", "svc-run", "services", "service-audit.jsonl")
    assert os.path.exists(audit_path)
    lines = [json.loads(l) for l in open(audit_path, encoding="utf-8")]
    assert any(r["event"] == "RUN_STARTED" for r in lines)


class _Resolver:
    def redact(self, t):
        return t


def test_cli_required_notification_blocks_success(tmp_path, monkeypatch):
    """required 策略下终态通知失败阻止成功（CLI 异常非零）。"""
    import dl_helper.training.cli as cli
    from dl_helper.training.services import ServiceAudit, ServiceDeliveryError

    def fake_build(config, platform, layout):
        wecom = _FakeWecom(fail_content=("[训练成功]",))
        audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
        return type(
            "L",
            (),
            {
                "start_run": lambda self, run_id, platform="local": None,
                "restore_latest_checkpoint": lambda self, run_id: None,
                "finalize_run": lambda self, run_id, status, **f: (
                    (_ for _ in ()).throw(ServiceDeliveryError("wecom required 失败"))
                    if status == "succeeded" else None
                ),
            },
        )()

    monkeypatch.setattr(cli, "_build_services", fake_build)
    cfg = _base_cfg(tmp_path, "svc-required", notify_type="wecom", policy="required")
    with pytest.raises(Exception):
        cli.main(["train", "--config", cfg, "--experiment", "experiments.toy_multiclass:build_experiment"])


def test_cli_fetches_remote_checkpoint_before_required_resume(tmp_path, monkeypatch):
    """跨 Session required 恢复在 worker 启动前先调用远端恢复。"""
    from types import SimpleNamespace

    import dl_helper.training.cli as cli
    import dl_helper.training.backends.torch_backend as torch_backend

    calls: list[str] = []

    class Services:
        _resolver = None

        def restore_latest_checkpoint(self, run_id):
            calls.append(f"restore:{run_id}")
            return "ck-remote"

    def fake_worker(experiment_ref, config, layout, local_rank, world_size, resume,
                    services=None, execution_policy=None):
        calls.append(f"worker:{resume}")
        return SimpleNamespace(status="succeeded")

    monkeypatch.setattr(cli, "_build_services", lambda config, platform, layout: Services())
    monkeypatch.setattr(torch_backend, "run_worker", fake_worker)
    cfg = _base_cfg(tmp_path, "remote-resume")

    code = cli.main([
        "train", "--config", cfg,
        "--experiment", "experiments.toy_multiclass:build_experiment",
        "--resume", "required",
    ])

    assert code == 0
    assert calls == ["restore:remote-resume", "worker:required"]


def test_cli_multiprocess_preempt_publishes_latest_before_finalize(tmp_path, monkeypatch):
    """多进程 75 由父进程先发布 latest checkpoint，再执行暂停终结。"""
    import dl_helper.training.cli as cli
    import dl_helper.training.launcher as launcher
    from dl_helper.training.artifacts import write_json
    from dl_helper.training.platform import Platform

    calls: list[str] = []

    class Services:
        _resolver = None

        def start_run(self, run_id):
            calls.append(f"start:{run_id}")

        def submit_checkpoint(self, run_id, checkpoint_id):
            calls.append(f"checkpoint:{run_id}:{checkpoint_id}")

        def finalize_run(self, run_id, status, **kwargs):
            calls.append(f"finalize:{run_id}:{status}")

    def fake_launch(experiment_ref, config, run_dir, num_procs, resume, **kwargs):
        checkpoint_id = "epoch-000001-step-00000010"
        os.makedirs(os.path.join(run_dir, "checkpoints", checkpoint_id))
        write_json(os.path.join(run_dir, "checkpoints", "latest.json"), {
            "schema_version": 1,
            "checkpoint_id": checkpoint_id,
            "path": checkpoint_id,
        })
        return 75

    monkeypatch.setattr(cli, "_build_services", lambda config, platform, layout: Services())
    monkeypatch.setattr(launcher, "launch_torch", fake_launch)
    monkeypatch.setattr(
        Platform, "resolve_torch_resources",
        lambda self, config, nominal_batch_size: type("Resources", (), {"num_processes": 2})(),
    )
    cfg_path = _base_cfg(tmp_path, "mp-preempt")
    schema = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    schema["distributed"]["num_processes"] = 2
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(schema, f, allow_unicode=True)

    code = cli.main([
        "train", "--config", cfg_path,
        "--experiment", "experiments.toy_multiclass:build_experiment",
        "--resume", "none",
    ])

    assert code == 75
    assert calls == [
        "start:mp-preempt",
        "checkpoint:mp-preempt:epoch-000001-step-00000010",
        "finalize:mp-preempt:preempted",
    ]
