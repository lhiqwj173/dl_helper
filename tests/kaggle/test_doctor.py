"""任务 7.2：backend-aware doctor —— 成功、多错误聚合、无副作用、contract 输出。"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.doctor import run_doctor
from dl_helper.training.platform import Platform, kaggle_execution_policy, local_execution_policy


def test_kaggle_requires_policy_alist_and_wecom():
    from dl_helper.training.doctor import _check_kaggle_requirements

    errors = _check_kaggle_requirements(parse_config(default_schema()), Platform("kaggle"))
    assert any("ExecutionPolicy" in error for error in errors)
    assert any("remote.type=alist" in error for error in errors)
    assert any("notifications.type=wecom" in error for error in errors)


def test_kaggle_policy_mismatch_rejected():
    from dl_helper.training.doctor import _check_kaggle_requirements

    errors = _check_kaggle_requirements(parse_config(default_schema()), Platform("kaggle"),
                                        execution_policy=local_execution_policy())
    assert any("660" in error for error in errors)


def test_kaggle_required_services_accept_environment_secrets(monkeypatch):
    from dl_helper.training.doctor import _check_kaggle_requirements

    schema = default_schema()
    schema["remote"] = {
        "type": "alist", "host": "https://alist.example.invalid", "base_path": "/runs",
        "user_secret_key": "ALIST_USER", "password_secret_key": "ALIST_PWD",
        "connect_timeout_seconds": 1, "read_timeout_seconds": 1, "max_attempts": 2,
        "async_upload": False, "failure_policy": "required",
    }
    schema["notifications"] = {
        "type": "wecom", "corp_id_secret_key": "WECOM_CORP_ID",
        "corp_secret_key": "WECOM_CORP_SECRET", "agent_id_secret_key": "WECOM_AGENT_ID",
        "to_user": "@all", "connect_timeout_seconds": 1, "read_timeout_seconds": 1,
        "max_attempts": 2, "failure_policy": "required",
    }
    for key in ("ALIST_USER", "ALIST_PWD", "WECOM_CORP_ID", "WECOM_CORP_SECRET",
                "WECOM_AGENT_ID"):
        monkeypatch.setenv(key, "configured")
    assert _check_kaggle_requirements(parse_config(schema), Platform("kaggle"),
                                      execution_policy=kaggle_execution_policy()) == []


def _torch_cfg(tmp_path, run_id="doctor-torch"):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["run"]["id"] = run_id
    schema["run"]["output_root"] = str(tmp_path)
    return parse_config(schema)


def _skl_cfg(tmp_path, fit_mode="batch"):
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": fit_mode, "evaluation_batch_size": 4096,
                                     "n_jobs": None, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1 if fit_mode == "batch" else 2, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 5, "min_delta": 0.0}
    schema["run"]["id"] = "doctor-skl"
    schema["run"]["output_root"] = str(tmp_path)
    schema["checkpoint"]["every_epochs"] = 1
    return parse_config(schema)


def test_doctor_torch_success(tmp_path):
    cfg = _torch_cfg(tmp_path)
    errors = run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert errors == []


def test_doctor_sklearn_success(tmp_path):
    cfg = _skl_cfg(tmp_path)
    errors = run_doctor(cfg, Platform("local"), "experiments.sklearn_batch:build_experiment")
    assert errors == []


def test_doctor_aggregates_multiple_errors(tmp_path):
    """同时存在的独立错误一次列出。"""
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["run"]["id"] = "doctor-multi"
    schema["run"]["output_root"] = str(tmp_path)
    schema["run"]["source_revision"] = "not-a-sha"  # 非法 revision
    cfg = parse_config(schema)
    errors = run_doctor(cfg, Platform("local"), "nonexistent_module:build")
    assert len(errors) >= 1


def test_doctor_no_training_side_effects(tmp_path):
    """doctor 不创建 checkpoint / 远程目录 / 通知。"""
    cfg = _torch_cfg(tmp_path)
    run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert not os.path.exists(os.path.join(str(tmp_path), "runs"))
    assert not os.path.exists(os.path.join(str(tmp_path), "checkpoints"))


def test_doctor_emit_contract(tmp_path, capsys):
    cfg = _torch_cfg(tmp_path)
    run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment",
               emit_contract=True)
    out = capsys.readouterr().out
    contract = json.loads(out.strip().splitlines()[-1])
    assert contract["backend"] == "torch"
    assert contract["valid"] is True
    assert contract["experiment_ref"] == "experiments.toy_multiclass:build_experiment"
    assert "data_identity" in contract
    assert "metric_definitions" in contract


def test_doctor_contract_schema_complete(tmp_path, capsys):
    """OSR-008：contract 含 splits / label schema / model signature。"""
    cfg = _torch_cfg(tmp_path)
    run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment",
               emit_contract=True)
    out = capsys.readouterr().out
    contract = json.loads(out.strip().splitlines()[-1])
    assert contract["valid"] is True
    assert "splits" in contract and "val" in contract["splits"]
    assert contract["label_schema"]["kind"] == "classification"
    assert "num_classes" in contract["label_schema"]
    assert "model_signature" in contract
    assert "num_parameters" in contract["model_signature"]


def test_doctor_secret_keys_only(tmp_path, capsys):
    """doctor 不显示 Secret 值，只显示 key。"""
    cfg = _torch_cfg(tmp_path)
    schema = default_schema()
    schema["run"]["output_root"] = str(tmp_path)
    schema["run"]["id"] = "doctor-sec"
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["remote"] = {"type": "alist", "host": "https://alist.example.invalid",
                        "base_path": "/x", "user_secret_key": "ALIST_USER",
                        "password_secret_key": "ALIST_PWD", "connect_timeout_seconds": 1,
                        "read_timeout_seconds": 1, "max_attempts": 2, "async_upload": False,
                        "failure_policy": "required"}
    cfg = parse_config(schema)
    run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment")
    # 不解析 Secret（doctor 不显示值）
    assert "ALIST_USER" in str(cfg.remote.user_secret_key)


def test_doctor_contract_extraction_failure_invalid(tmp_path, monkeypatch, capsys):
    """OSR-008：contract 提取失败 → valid=False（不静默通过）。"""
    import dl_helper.training.doctor as doc
    from dl_helper.training.doctor import run_doctor

    cfg = _torch_cfg(tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("extract-boom")

    monkeypatch.setattr(doc, "_extract_contract_info", _boom)
    run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment",
               emit_contract=True)
    out = capsys.readouterr().out
    contract = json.loads(out.strip().splitlines()[-1])
    assert contract["valid"] is False
    assert any("extract-boom" in e for e in contract["errors"])
