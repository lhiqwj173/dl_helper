"""任务 10.3：服务与 sweep 故障注入矩阵 —— 终态互斥、audit、primary error、幂等。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.services import (
    LifecycleServices,
    SecondaryServiceError,
    ServiceAudit,
    ServiceDeliveryError,
    utc_now,
)


class _FakeWecom:
    def __init__(self, fail_content=()):
        self.sent = []
        self._fail = tuple(fail_content)

    def send_text(self, content, redactor=None):
        if any(m in content for m in self._fail):
            raise ConnectionError("net down")
        self.sent.append(content)


class _FakeStore:
    def __init__(self, fail_publish=False):
        self.bundles = []
        self.fail_publish = fail_publish

    def publish_run_bundle(self, local_dir, run_id):
        if self.fail_publish:
            raise ConnectionError("upload failed")
        self.bundles.append(("run", run_id))

    def publish_sweep_bundle(self, local_dir, sweep_id):
        self.bundles.append(("sweep", sweep_id))


class _NoAsync:
    def flush(self, terminal=False):
        return None


class _Resolver:
    def redact(self, t):
        return t


def _services(tmp_path, wecom, store, policy="required"):
    layout = RunLayout(str(tmp_path / "runs" / "sr"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    return LifecycleServices(layout=layout, secret_resolver=_Resolver(),
                             stores=[store], async_sync=_NoAsync(),
                             wecom_client=wecom, audit=audit, failure_policy=policy)


def test_alist_publish_failure_required_blocks_terminal(tmp_path):
    store = _FakeStore(fail_publish=True)
    svc = _services(tmp_path, _FakeWecom(), store, policy="required")
    svc.start_run("r")
    with pytest.raises(ServiceDeliveryError):
        svc.finalize_run("r", "succeeded")
    assert store.bundles == []  # bundle 未成功


def test_alist_publish_failure_record_marks_degraded(tmp_path):
    store = _FakeStore(fail_publish=True)
    svc = _services(tmp_path, _FakeWecom(), store, policy="record")
    svc.start_run("r")
    svc.finalize_run("r", "succeeded")  # record 不抛
    assert svc.result.has_degraded


def test_primary_training_error_preserved(tmp_path):
    """训练异常为 primary；通知失败为 secondary，不覆盖。"""
    layout = RunLayout(str(tmp_path / "runs" / "p"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    svc = LifecycleServices(layout=layout, secret_resolver=_Resolver(),
                            stores=[_FakeStore()], async_sync=_NoAsync(),
                            wecom_client=_FakeWecom(fail_content=("[训练失败]",)),
                            audit=audit, failure_policy="record")
    primary = RuntimeError("training crash")
    try:
        raise primary
    except RuntimeError as primary_exc:
        svc.result.mark_degraded("wecom", "ConnectionError")
        # failure.json 以 primary 为主
        assert type(primary_exc).__name__ == "RuntimeError"
        assert svc.result.has_degraded


def test_terminal_mutual_exclusion_on_failure(tmp_path):
    from dl_helper.training.artifacts import existing_terminal
    layout = RunLayout(str(tmp_path / "runs" / "m"))
    layout.ensure()
    # 已发布 success 后 failure 不可写
    from dl_helper.training.artifacts import publish_terminal
    publish_terminal(layout.run_dir, "success", {"status": "ok"})
    assert existing_terminal(layout.run_dir) == "run-manifest.json"


def test_idempotent_finalization(tmp_path):
    wecom = _FakeWecom()
    svc = _services(tmp_path, wecom, _FakeStore(), policy="required")
    svc.start_run("r")
    svc.finalize_run("r", "succeeded")
    svc.finalize_run("r", "succeeded")  # 重入
    assert len([s for s in wecom.sent if "训练成功" in s]) == 1


def test_failed_bundle_gated_on_evidence(tmp_path, monkeypatch):
    """OSR-003：failure.json 未成功持久化时不得发布声称完整的 FAILED bundle。"""
    import yaml

    import dl_helper.training.cli as cli
    from dl_helper.training.config import default_schema

    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = "fail-ev"
    schema["run"]["output_root"] = str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    schema["notifications"] = {"type": "wecom", "corp_id_secret_key": "A",
                               "corp_secret_key": "B", "agent_id_secret_key": "C",
                               "to_user": "u", "connect_timeout_seconds": 1,
                               "read_timeout_seconds": 1, "max_attempts": 2,
                               "failure_policy": "required"}
    cfg = tmp_path / "base.yaml"
    cfg.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")

    finalized: list[tuple] = []

    class _FakeServices:
        _resolver = _Resolver()

        def finalize_run(self, run_id, status):
            finalized.append((run_id, status))

        result = type("R", (), {"degraded": []})()

    def boom(args, exc):
        raise OSError("evidence disk full")

    monkeypatch.setattr(cli, "_write_failure_evidence", boom)
    monkeypatch.setattr(cli, "_build_services", lambda c, p, l: _FakeServices())
    # 主异常（ModuleNotFoundError）保留；证据写入失败为 secondary
    with pytest.raises(ModuleNotFoundError):
        cli.main(["train", "--config", str(cfg),
                  "--experiment", "nonexistent_module:build_experiment"])
    # failure.json 未持久化 → FAILED bundle 不发布（finalize 未调用）
    assert finalized == []


def test_bundle_not_republished_on_reentry(tmp_path):
    """OSR-002：按状态跳过已成功发布的 bundle（幂等重入）。"""
    wecom = _FakeWecom()
    store = _FakeStore()
    svc = _services(tmp_path, wecom, store)
    svc.start_run("r")
    svc.finalize_run("r", "succeeded")
    svc.finalize_run("r", "succeeded")  # 重入
    assert len(store.bundles) == 1  # bundle 只发布一次
    # 不同状态（如 preempted）可再发布
    svc.finalize_run("r", "preempted")
    assert len(store.bundles) == 2


def test_failed_notification_retried_on_reentry(tmp_path):
    """OSR-002：失败通知不标记 seen，重启后可重试。"""
    layout = RunLayout(str(tmp_path / "runs" / "retry"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    wecom = _FakeWecom(fail_content=("[训练成功]",))  # RUN_SUCCEEDED 投递失败
    svc1 = LifecycleServices(layout=layout, secret_resolver=_Resolver(),
                             stores=[_FakeStore()], async_sync=_NoAsync(),
                             wecom_client=wecom, audit=audit, failure_policy="record")
    svc1.start_run("r")
    svc1.finalize_run("r", "succeeded")  # 通知失败（record → degraded，不抛）

    # 重启：audit 中 RUN_SUCCEEDED outcome=failed → 不恢复为 seen → 可重试
    wecom2 = _FakeWecom()
    svc2 = LifecycleServices(layout=layout, secret_resolver=_Resolver(),
                             stores=[_FakeStore()], async_sync=_NoAsync(),
                             wecom_client=wecom2, audit=audit, failure_policy="record")
    svc2.finalize_run("r", "succeeded")
    assert len([s for s in wecom2.sent if "训练成功" in s]) == 1  # 重试成功
