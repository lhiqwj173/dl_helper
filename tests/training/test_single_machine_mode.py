"""AList 开关仅控制远程存储，通知和本地恢复保持独立。"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import requests
import yaml

from dl_helper.training.cli import build_parser, main
from dl_helper.training.config import ConfigError, NoRemoteConfig, default_schema, parse_config
from dl_helper.training.doctor import _check_kaggle_requirements, validate_training_start
from dl_helper.training.platform import Platform, SecretResolver, execution_policy_for


@pytest.fixture(autouse=True)
def single_thread_torch():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _schema(tmp_path, backend="torch"):
    schema = default_schema()
    schema["run"].update(id="standalone", output_root=str(tmp_path), source_revision="test-v1")
    schema["training"]["max_epochs"] = 1
    schema["distributed"]["num_processes"] = 1
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["report"]["prediction_splits"] = ["val"]
    schema["remote"] = {
        "type": "alist", "host": "https://alist.example.invalid", "base_path": "/runs",
        "user_secret_key": "ALIST_USER", "password_secret_key": "ALIST_PWD",
        "connect_timeout_seconds": 1, "read_timeout_seconds": 1,
        "max_attempts": 1, "async_upload": True, "failure_policy": "required",
    }
    if backend == "sklearn":
        schema["backend"] = {
            "type": "sklearn", "torch": None,
            "sklearn": {"fit_mode": "incremental", "evaluation_batch_size": 4096,
                        "n_jobs": None, "random_state": "run_seed", "sample_weight_parameter": None},
        }
        schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 10, "min_delta": 0.0}
    return schema


def _argv(tmp_path, schema, backend="torch"):
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    experiment = "sklearn_incremental" if backend == "sklearn" else "toy_multiclass"
    return ["train", "--config", str(path), "--experiment", f"experiments.{experiment}:build_experiment"]


@pytest.mark.parametrize("flag, expected", [(None, True), ("--use-alist", True), ("--no-use-alist", False)])
def test_switch_defaults_to_enabled(flag, expected):
    argv = ["train", "--config", "base.yaml", "--experiment", "project:build_experiment"]
    if flag:
        argv.append(flag)
    assert build_parser().parse_args(argv).use_alist is expected


@pytest.mark.parametrize("backend", ["torch", "sklearn"])
@pytest.mark.parametrize("notification_policy", [None, "required", "record"])
def test_single_machine_trains_without_alist(tmp_path, monkeypatch, backend, notification_policy):
    from dl_helper.training.remote import AListArtifactStore, AsyncArtifactSync
    from dl_helper.training.services import LifecycleServices
    from dl_helper.training.notifications import WecomClient

    def forbidden(*args, **kwargs):
        raise AssertionError("禁用 AList 后不应访问真实网络、AList、异步上传或远程恢复流程")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    keys, messages = [], []

    def resolve_secret(self, key):
        assert notification_policy is not None
        assert key.startswith("WECOM_")
        keys.append(key)
        return "1" if key == "WECOM_AGENT_ID" else "test-secret"

    monkeypatch.setattr(SecretResolver, "resolve", resolve_secret)
    monkeypatch.setattr(WecomClient, "send_text",
                        lambda self, content, **kwargs: messages.append(content) or {"errcode": 0})
    monkeypatch.setattr(AListArtifactStore, "__init__", forbidden)
    monkeypatch.setattr(AsyncArtifactSync, "__init__", forbidden)
    monkeypatch.setattr(LifecycleServices, "restore_latest_checkpoint", forbidden)
    resolve_resources = Platform.resolve_torch_resources

    def test_resources(self, config, nominal_batch_size):
        return replace(resolve_resources(self, config, nominal_batch_size),
                       num_workers=0, persistent_workers=False, prefetch_factor=None)

    monkeypatch.setattr(Platform, "resolve_torch_resources", test_resources)
    # 若执行自动本地恢复，损坏的 latest 必然失败；训练成功证明跳过本地恢复。
    checkpoints = tmp_path / "runs" / "standalone" / "checkpoints"
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "latest.json"
    latest.write_text("损坏的旧检查点", encoding="utf-8")
    schema = _schema(tmp_path, backend)
    if notification_policy is not None:
        schema["notifications"] = _notifications(notification_policy)
    assert main([*_argv(tmp_path, schema, backend), "--no-use-alist", "--resume", "none"]) == 0
    run_dir = checkpoints.parent
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["remote"] == {"type": "none"}
    assert resolved["notifications"] == schema["notifications"]
    policy = json.loads((run_dir / "execution-policy.json").read_text(encoding="utf-8"))
    assert policy["resume"] == "none"
    assert policy["use_alist"] is False
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["epoch"] == 1
    assert (run_dir / "run-manifest.json").is_file()
    assert (run_dir / "report" / "index.html").is_file()
    if notification_policy is not None:
        assert keys
        assert len(messages) >= 2
        audit = [json.loads(line) for line in (run_dir / "services" / "service-audit.jsonl")
                 .read_text(encoding="utf-8").splitlines()]
        successful_events = {item["event"] for item in audit
                             if item["service"] == "wecom" and item["outcome"] == "success"}
        assert {"RUN_STARTED", "RUN_SUCCEEDED"} <= successful_events
    else:
        assert keys == messages == []


@pytest.mark.parametrize("requested, expected", [(None, "auto"), ("none", "none"), ("required", "required")])
def test_no_alist_preserves_resume_policy(tmp_path, monkeypatch, requested, expected):
    import dl_helper.training.doctor as doctor

    checked = []

    def preflight(config, platform, experiment_ref, **kwargs):
        assert config.remote.type == "none"
        checked.append(kwargs["resume"])

    monkeypatch.setattr(doctor, "validate_training_start", preflight)
    argv = [*_argv(tmp_path, _schema(tmp_path)), "--no-use-alist", "--preflight-only"]
    if requested is not None:
        argv.extend(["--resume", requested])
    assert main(argv) == 0
    assert checked == [expected]
    assert not (tmp_path / "runs").exists()


def test_kaggle_single_machine_skips_remote_secrets_and_keeps_budget(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("未启用服务不应读取 Secret")

    monkeypatch.setattr(SecretResolver, "resolve", forbidden)
    config = replace(parse_config(_schema(tmp_path)), remote=NoRemoteConfig(type="none"))
    platform = Platform("kaggle")
    policy = execution_policy_for(platform)
    assert _check_kaggle_requirements(config, platform, policy, use_alist=False) == []
    assert _check_kaggle_requirements(config, platform, None, use_alist=False)
    errors = _check_kaggle_requirements(config, platform, policy)
    assert any("remote.type=alist" in item for item in errors)
    assert not any("企业微信" in item or "notifications" in item for item in errors)


def _notifications(policy="required"):
    return {
        "type": "wecom", "corp_id_secret_key": "WECOM_CORP_ID",
        "corp_secret_key": "WECOM_CORP_SECRET", "agent_id_secret_key": "WECOM_AGENT_ID",
        "to_user": "@all", "connect_timeout_seconds": 1, "read_timeout_seconds": 1,
        "max_attempts": 1, "failure_policy": policy,
    }


@pytest.mark.parametrize("use_alist", [True, False])
@pytest.mark.parametrize("notification_policy", ["required", "record"])
def test_notification_secret_validation_is_independent(tmp_path, monkeypatch, use_alist, notification_policy):
    schema = _schema(tmp_path)
    if not use_alist:
        schema["remote"] = {"type": "none"}
    schema["notifications"] = _notifications(notification_policy)
    keys = []
    monkeypatch.setattr(SecretResolver, "resolve", lambda self, key: keys.append(key) or "test")
    platform = Platform("kaggle")
    assert _check_kaggle_requirements(parse_config(schema), platform, execution_policy_for(platform),
                                      use_alist=use_alist) == []
    expected = ["ALIST_USER", "ALIST_PWD"] if use_alist else []
    assert keys == expected + ["WECOM_CORP_ID", "WECOM_CORP_SECRET", "WECOM_AGENT_ID"]


def test_single_machine_preflight_rejects_inconsistent_internal_state(tmp_path):
    config = parse_config(_schema(tmp_path))
    with pytest.raises(ConfigError, match="必须禁用 remote"):
        validate_training_start(config, Platform("local"), "project:build_experiment",
                                use_alist=False, resume="none")


@pytest.mark.parametrize("resume", ["auto", "none", "required"])
def test_no_alist_doctor_accepts_local_resume(tmp_path, monkeypatch, resume):
    import dl_helper.training.doctor as doctor

    checked = []

    def check(config, platform, experiment_ref, **kwargs):
        checked.append(kwargs["resume"])
        return []

    monkeypatch.setattr(doctor, "run_doctor", check)
    config = replace(parse_config(_schema(tmp_path)), remote=NoRemoteConfig(type="none"))
    validate_training_start(config, Platform("local"), "project:build_experiment",
                            use_alist=False, resume=resume)
    assert checked == [resume]


def test_no_alist_missing_notification_secrets_are_reported(tmp_path, monkeypatch):
    from dl_helper.training.platform import SecretError

    schema = _schema(tmp_path)
    schema["remote"] = {"type": "none"}
    schema["notifications"] = _notifications()
    keys = []

    def missing(self, key):
        keys.append(key)
        raise SecretError(f"缺少凭证: {key}")

    monkeypatch.setattr(SecretResolver, "resolve", missing)
    platform = Platform("kaggle")
    errors = _check_kaggle_requirements(parse_config(schema), platform, execution_policy_for(platform),
                                       use_alist=False)
    assert keys == ["WECOM_CORP_ID", "WECOM_CORP_SECRET", "WECOM_AGENT_ID"]
    assert len(errors) == 3
    assert all(any(key in error for error in errors) for key in keys)


@pytest.mark.parametrize("resume", [None, "required"])
def test_no_alist_cli_resumes_local_checkpoint(tmp_path, monkeypatch, resume):
    import dl_helper.training.platform as platform_module
    import dl_helper.training.backends.torch_backend as backend
    from dl_helper.training.platform import ExecutionPolicy
    from dl_helper.training.remote import AListArtifactStore

    original_worker = backend.run_worker
    calls = []

    class AdvancingClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return self.calls * 100.0

    def worker(*args, **kwargs):
        if not calls:
            kwargs["budget_monotonic"] = AdvancingClock()
        result = original_worker(*args, **kwargs)
        calls.append(result)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("本地恢复不应构造 AList 客户端")

    monkeypatch.setattr(backend, "run_worker", worker)
    monkeypatch.setattr(AListArtifactStore, "__init__", forbidden)
    monkeypatch.setattr(platform_module, "execution_policy_for", lambda platform: ExecutionPolicy(
        platform="local", max_minutes=10., shutdown_grace_minutes=2.))
    resources = Platform.resolve_torch_resources
    monkeypatch.setattr(Platform, "resolve_torch_resources", lambda self, config, batch: replace(
        resources(self, config, batch), num_workers=0, persistent_workers=False, prefetch_factor=None))
    schema = _schema(tmp_path)
    schema["training"]["max_epochs"] = 2
    schema["checkpoint"]["every_epochs"] = 1
    argv = _argv(tmp_path, schema)
    argv[argv.index("--experiment") + 1] = "experiments.toy_multiclass_resumable:build_experiment"
    argv.append("--no-use-alist")
    assert main(argv) == 75
    checkpoint = tmp_path / "runs" / "standalone" / "checkpoints" / "latest.json"
    assert checkpoint.is_file()
    second = argv if resume is None else [*argv, "--resume", resume]
    assert main(second) == 0
    assert [result.status for result in calls] == ["preempted", "succeeded"]
    assert calls[1].epoch == 2
    assert calls[1].global_step > calls[0].global_step > 0
