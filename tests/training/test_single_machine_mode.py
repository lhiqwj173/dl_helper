"""单机开关：默认兼容、无 AList 网络与凭证依赖、两个后端均从头训练。"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import requests
import yaml

from dl_helper.training.cli import CliError, build_parser, main
from dl_helper.training.config import ConfigError, NoRemoteConfig, default_schema, parse_config
from dl_helper.training.doctor import _check_kaggle_requirements, validate_training_start
from dl_helper.training.platform import Platform, SecretResolver, execution_policy_for


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
def test_single_machine_trains_without_remote_or_resume(tmp_path, monkeypatch, backend):
    from dl_helper.training.remote import AListArtifactStore, AsyncArtifactSync
    from dl_helper.training.services import LifecycleServices

    def forbidden(*args, **kwargs):
        raise AssertionError("单机模式不应访问网络、Secret、AList、异步上传或恢复流程")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(SecretResolver, "resolve", forbidden)
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
    assert main([*_argv(tmp_path, schema, backend), "--no-use-alist"]) == 0
    run_dir = checkpoints.parent
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["remote"] == {"type": "none"}
    policy = json.loads((run_dir / "execution-policy.json").read_text(encoding="utf-8"))
    assert policy["resume"] == "none"
    assert policy["use_alist"] is False
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["epoch"] == 1
    assert (run_dir / "run-manifest.json").is_file()
    assert (run_dir / "report" / "index.html").is_file()


def test_required_resume_rejected_before_preflight(tmp_path, monkeypatch):
    import dl_helper.training.doctor as doctor

    def forbidden(*args, **kwargs):
        raise AssertionError("冲突参数应在预检和训练前失败")

    monkeypatch.setattr(doctor, "validate_training_start", forbidden)
    with pytest.raises(CliError, match="不支持恢复训练"):
        main([*_argv(tmp_path, _schema(tmp_path)), "--no-use-alist", "--resume", "required"])
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
    assert any("notifications.type=wecom" in item for item in errors)


def test_single_machine_preserves_enabled_notification_secret_validation(tmp_path, monkeypatch):
    schema = _schema(tmp_path)
    schema["remote"] = {"type": "none"}
    schema["notifications"] = {
        "type": "wecom", "corp_id_secret_key": "WECOM_CORP_ID",
        "corp_secret_key": "WECOM_CORP_SECRET", "agent_id_secret_key": "WECOM_AGENT_ID",
        "to_user": "@all", "connect_timeout_seconds": 1, "read_timeout_seconds": 1,
        "max_attempts": 1, "failure_policy": "required",
    }
    keys = []
    monkeypatch.setattr(SecretResolver, "resolve", lambda self, key: keys.append(key) or "test")
    platform = Platform("kaggle")
    assert _check_kaggle_requirements(parse_config(schema), platform, execution_policy_for(platform),
                                      use_alist=False) == []
    assert keys == ["WECOM_CORP_ID", "WECOM_CORP_SECRET", "WECOM_AGENT_ID"]


def test_single_machine_preflight_rejects_inconsistent_internal_state(tmp_path):
    config = parse_config(_schema(tmp_path))
    with pytest.raises(ConfigError, match="必须禁用 remote"):
        validate_training_start(config, Platform("local"), "project:build_experiment",
                                use_alist=False, resume="none")
    config = replace(config, remote=NoRemoteConfig(type="none"))
    with pytest.raises(ConfigError, match="不支持恢复"):
        validate_training_start(config, Platform("local"), "project:build_experiment", use_alist=False)
