"""补充 platform / doctor 分支覆盖（OSR-009 覆盖率门禁）。

用 dataclasses.replace 绕过 parse_config 校验，触发 doctor 防御分支；
platform 边界：Kaggle 路径合同、backend 缺失、auto 资源解析、revision。
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from dl_helper.training.config import default_schema, parse_config


def _torch_cfg(output_root=None, **run_patch):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["run"]["id"] = "edges"
    if output_root is not None:
        schema["run"]["output_root"] = output_root
    cfg = parse_config(schema)
    if run_patch:
        cfg = replace(cfg, run=replace(cfg.run, **run_patch))
    return cfg


def _skl_cfg(output_root=None):
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096,
                                     "n_jobs": None, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 5, "min_delta": 0.0}
    schema["run"]["id"] = "edges-skl"
    if output_root is not None:
        schema["run"]["output_root"] = output_root
    schema["checkpoint"]["every_epochs"] = 1
    return parse_config(schema)


# ---------- platform ----------
def test_platform_kaggle_output_root_outside_working():
    from dl_helper.training.platform import Platform, PlatformError
    cfg = _torch_cfg(output_root="C:/not-kaggle-working")
    with pytest.raises(PlatformError):
        Platform("kaggle").resolve_output_root(cfg)


def test_platform_torch_backend_none_raises():
    from dl_helper.training.platform import Platform, PlatformError
    cfg = replace(_torch_cfg(), backend=replace(_torch_cfg().backend, torch=None))
    with pytest.raises(PlatformError):
        Platform("local").resolve_torch_resources(cfg, None)


def test_platform_sklearn_backend_none_raises():
    from dl_helper.training.platform import Platform, PlatformError
    cfg = replace(_torch_cfg(), backend=replace(_torch_cfg().backend, sklearn=None))
    with pytest.raises(PlatformError):
        Platform("local").resolve_sklearn_resources(cfg)


def test_platform_num_procs_auto_cpu():
    from dl_helper.training.platform import Platform
    schema = default_schema()
    schema["distributed"]["num_processes"] = "auto"
    cfg = parse_config(schema)
    res = Platform("local").resolve_torch_resources(cfg, 64)
    assert res.num_processes == 1
    assert res.effective_batch_size == 64  # nominal 提供 → 数值


def test_platform_sklearn_njobs_auto():
    from dl_helper.training.platform import Platform
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096,
                                     "n_jobs": "auto", "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = None
    cfg = parse_config(schema)
    res = Platform("local").resolve_sklearn_resources(cfg)
    assert res.n_jobs >= 1


def test_platform_valid_revision_returns():
    from dl_helper.training.platform import resolve_source_revision
    rev = "a" * 40
    cfg = _torch_cfg(source_revision=rev)
    assert resolve_source_revision(cfg) == rev


def test_platform_kaggle_valid_output_root_returns():
    from dl_helper.training.platform import Platform
    cfg = _torch_cfg(output_root="/kaggle/working/dl-helper-runs")
    out = Platform("kaggle").resolve_output_root(cfg)
    assert out == "/kaggle/working/dl-helper-runs"


def _kaggle_cfg(experiment):
    schema = default_schema()
    schema["run"]["output_root"] = "/kaggle/working/dl-helper-runs"
    schema["run"]["id"] = "kg-inputs"
    schema["experiment"] = experiment
    return parse_config(schema)


def test_platform_kaggle_validate_inputs_empty():
    from dl_helper.training.platform import Platform
    Platform("kaggle").validate_kaggle_inputs(_kaggle_cfg({}))  # 无 path key → 不抛


def test_platform_kaggle_validate_inputs_non_path_key():
    from dl_helper.training.platform import Platform
    Platform("kaggle").validate_kaggle_inputs(_kaggle_cfg({"lr": 0.01}))  # 循环回边


def test_platform_kaggle_validate_inputs_missing_input_dir():
    from dl_helper.training.platform import Platform, PlatformError
    cfg = _kaggle_cfg({"data_path": "/kaggle/input/nonexistent-dataset/x"})
    with pytest.raises(PlatformError):
        Platform("kaggle").validate_kaggle_inputs(cfg)


# ---------- doctor ----------
def test_doctor_alist_missing_secret_keys(tmp_path):
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    schema = default_schema()
    schema["remote"] = {"type": "alist", "host": "https://h", "base_path": "/x",
                        "user_secret_key": "U", "password_secret_key": "P",
                        "connect_timeout_seconds": 1, "read_timeout_seconds": 1,
                        "max_attempts": 2, "async_upload": False, "failure_policy": "required"}
    schema["run"]["output_root"] = str(tmp_path)
    schema["run"]["id"] = "doc-alist"
    schema["training"]["max_epochs"] = 1
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    cfg = parse_config(schema)
    bad = replace(cfg, remote=replace(cfg.remote, user_secret_key="", password_secret_key=""))
    errors = run_doctor(bad, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert any("user/password" in e for e in errors)


def test_doctor_wecom_missing_key(tmp_path):
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    schema = default_schema()
    schema["notifications"] = {"type": "wecom", "corp_id_secret_key": "A",
                               "corp_secret_key": "B", "agent_id_secret_key": "C",
                               "to_user": "u", "connect_timeout_seconds": 1,
                               "read_timeout_seconds": 1, "max_attempts": 2,
                               "failure_policy": "record"}
    schema["run"]["output_root"] = str(tmp_path)
    schema["run"]["id"] = "doc-wecom"
    schema["training"]["max_epochs"] = 1
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    cfg = parse_config(schema)
    bad = replace(cfg, notifications=replace(cfg.notifications, agent_id_secret_key=""))
    errors = run_doctor(bad, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert any("agent_id_secret_key" in e for e in errors)


def test_doctor_disk_low(monkeypatch, tmp_path):
    import dl_helper.training.doctor as doc
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform

    monkeypatch.setattr(doc, "free_disk_bytes", lambda path: 1)
    cfg = _torch_cfg(output_root=str(tmp_path))
    errors = run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert any("空间不足" in e for e in errors)


def test_doctor_validate_inputs_raises(tmp_path):
    from dl_helper.training.doctor import run_doctor

    class _RaisingPlatform:
        is_kaggle = False
        kind = "local"

        def resolve_output_root(self, config):
            return str(tmp_path)

        def validate_kaggle_inputs(self, config):
            raise RuntimeError("kaggle inputs 非法")

    cfg = _torch_cfg(output_root=str(tmp_path))
    errors = run_doctor(cfg, _RaisingPlatform(), "experiments.toy_multiclass:build_experiment")
    assert any("kaggle inputs 非法" in e for e in errors)


def test_doctor_emit_contract_with_errors(tmp_path, capsys):
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    cfg = _torch_cfg(output_root=str(tmp_path), source_revision="short")
    errors = run_doctor(cfg, Platform("local"), "nonexistent_module:build", emit_contract=True)
    assert errors  # 有错误 → contract valid=False，跳过 extract
    out = capsys.readouterr().out
    assert "config_fingerprint" in out


def test_doctor_sklearn_contract_extract(tmp_path, capsys):
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    cfg = _skl_cfg(output_root=str(tmp_path))
    errors = run_doctor(cfg, Platform("local"), "experiments.sklearn_batch:build_experiment",
                        emit_contract=True)
    assert errors == []
    out = capsys.readouterr().out
    assert "metric_definitions" in out


def test_doctor_run_id_none_path(tmp_path):
    """run.id None → _check_paths 跳过字符集检查。"""
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["run"]["id"] = None
    schema["run"]["output_root"] = str(tmp_path)
    cfg = parse_config(schema)
    errors = run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert errors == []


def _wecom_cfg(tmp_path, to_user="u", keys=("A", "B", "C")):
    schema = default_schema()
    schema["notifications"] = {"type": "wecom", "corp_id_secret_key": keys[0],
                               "corp_secret_key": keys[1], "agent_id_secret_key": keys[2],
                               "to_user": to_user, "connect_timeout_seconds": 1,
                               "read_timeout_seconds": 1, "max_attempts": 2,
                               "failure_policy": "record"}
    schema["run"]["output_root"] = str(tmp_path)
    schema["run"]["id"] = "doc-wecom2"
    schema["training"]["max_epochs"] = 1
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def test_doctor_wecom_all_present(tmp_path):
    """wecom 全键存在 + to_user 存在 → 无服务错误。"""
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    cfg = _wecom_cfg(tmp_path)
    errors = run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert not any("wecom" in e for e in errors)


def test_doctor_wecom_empty_to_user(tmp_path):
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    cfg = replace(_wecom_cfg(tmp_path), notifications=replace(_wecom_cfg(tmp_path).notifications,
                                                             to_user=""))
    errors = run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert any("to_user" in e for e in errors)


def test_doctor_torch_no_experiment_ref(tmp_path):
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    cfg = _torch_cfg(output_root=str(tmp_path))
    errors = run_doctor(cfg, Platform("local"), "")
    assert any("experiment 引用" in e for e in errors)


def test_launcher_single_success_returns_0(tmp_path):
    from dl_helper.training.backends.base import BackendResult
    from dl_helper.training.launcher import launch_torch
    from dl_helper.training.artifacts import RunLayout
    cfg = _torch_cfg(output_root=str(tmp_path))
    layout = RunLayout(str(tmp_path / "runs" / "lp-ok2"))
    layout.ensure()

    def fake_worker(ref, config, layout, rank, world, resume, publish_terminal=True,
                    budget_monotonic=None, execution_policy=None):
        return BackendResult(status="succeeded", epoch=1, global_step=3)

    code = launch_torch("ref", cfg, layout.run_dir, 1, "none", worker_fn=fake_worker)
    assert code == 0
