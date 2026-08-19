"""任务 1.2：严格 base 配置 schema v1 测试。"""
from __future__ import annotations

import copy

import pytest
import yaml

from dl_helper.training.config import (
    Config,
    ConfigError,
    config_canonical_json,
    config_fingerprint,
    default_schema,
    parse_config,
    parse_config_text,
    tuning_fingerprint,
    yaml_load_strict,
)


def _schema(**patch):
    schema = default_schema()
    merged = _deep_merge(schema, patch)
    return merged


def _deep_merge(base, patch):
    result = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _to_yaml(data) -> str:
    return yaml.safe_dump(data, allow_unicode=True)


def test_valid_snapshot_parses():
    cfg = parse_config(default_schema())
    assert cfg.schema_version == 1
    assert cfg.backend.type == "torch"
    assert cfg.backend.sklearn is None
    assert cfg.run.seed == 42


def test_valid_sklearn_snapshot():
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn",
        "torch": None,
        "sklearn": {
            "fit_mode": "batch",
            "evaluation_batch_size": 4096,
            "n_jobs": "auto",
            "random_state": "run_seed",
            "sample_weight_parameter": None,
        },
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    cfg = parse_config(schema)
    assert cfg.backend.sklearn.fit_mode == "batch"
    assert cfg.backend.torch is None


def test_duplicate_key_rejected():
    text = _to_yaml(default_schema()).replace("seed: 42", "seed: 42\nseed: 43")
    with pytest.raises(ConfigError):
        parse_config_text(text)


def test_unknown_field_rejected():
    schema = _schema(training={"max_epochs": 5, "log_every_steps": 1, "bogus": 1})
    with pytest.raises(ConfigError):
        parse_config(schema)


def test_string_bool_rejected():
    schema = _schema(backend={"type": "torch",
                              "torch": _schema()["backend"]["torch"]})
    schema["backend"]["torch"]["compile"] = "false"
    with pytest.raises(ConfigError):
        parse_config(schema)


def test_nan_inf_rejected():
    schema = _schema(training={"max_epochs": float("nan"), "log_every_steps": 1})
    with pytest.raises(ConfigError):
        parse_config(schema)
    schema = _schema(selection={"metric": "val/loss", "mode": "min", "patience": 1, "min_delta": float("inf")})
    with pytest.raises(ConfigError):
        parse_config(schema)


def test_unselected_backend_non_null():
    schema = default_schema()
    schema["backend"]["sklearn"] = {
        "fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": 1,
        "random_state": "run_seed", "sample_weight_parameter": None,
    }
    with pytest.raises(ConfigError):
        parse_config(schema)


def test_invalid_enum_rejected():
    schema = _schema(backend={"type": "torch", "torch": default_schema()["backend"]["torch"]})
    schema["backend"]["torch"]["mixed_precision"] = "fp32"
    with pytest.raises(ConfigError):
        parse_config(schema)
    schema2 = _schema(backend={"type": "cuda"})
    with pytest.raises(ConfigError):
        parse_config(schema2)


def test_sklearn_batch_constraints():
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": 1,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 2, "log_every_steps": 1}  # batch 要求 max_epochs=1
    with pytest.raises(ConfigError):
        parse_config(schema)
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["checkpoint"] = {"every_epochs": None, "every_optimizer_steps": 1, "keep_last": 1}
    with pytest.raises(ConfigError):
        parse_config(schema)
    schema["checkpoint"] = {"every_epochs": 1, "every_optimizer_steps": None, "keep_last": 1}
    # D-003：budget 由平台执行策略决定，batch 配置本身可解析
    cfg = parse_config(schema)
    assert cfg.backend.sklearn.fit_mode == "batch"


def test_runtime_and_resume_removed_from_user_config():
    # D-002/D-003：根级 runtime 与 checkpoint.resume 必须按未知字段立即失败
    schema = _schema(runtime={"max_minutes": 10, "shutdown_grace_minutes": 5})
    with pytest.raises(ConfigError):
        parse_config(schema)
    schema = _schema(checkpoint={"every_epochs": 1, "every_optimizer_steps": None,
                                 "keep_last": 2, "resume": "auto"})
    with pytest.raises(ConfigError):
        parse_config(schema)


def test_sklearn_distributed_must_be_one():
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "incremental", "evaluation_batch_size": 4096, "n_jobs": 1,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }
    schema["distributed"] = {"num_processes": 2}
    with pytest.raises(ConfigError):
        parse_config(schema)


def test_run_id_charset():
    schema = _schema(run={"id": "bad id!"})
    with pytest.raises(ConfigError):
        parse_config(schema)
    schema = _schema(run={"id": "good-id.v1_2"})
    cfg = parse_config(schema)
    assert cfg.run.id == "good-id.v1_2"


def test_merge_key_rejected():
    schema = default_schema()
    text = _to_yaml(schema) + "\nanchor: &a {x: 1}\n"
    with pytest.raises(ConfigError):
        parse_config_text(text)


def test_env_interpolation_rejected():
    text = _to_yaml(default_schema()).replace("seed: 42", "seed: ${SEED}")
    with pytest.raises(ConfigError):
        parse_config_text(text)


def test_canonical_json_and_fingerprint():
    cfg = parse_config(default_schema())
    js = config_canonical_json(cfg)
    assert '"schema_version"' in js
    fp = config_fingerprint(cfg)
    assert len(fp) == 64
    tfp = tuning_fingerprint(cfg)
    assert len(tfp) == 64
    # 相同配置产生相同指纹
    assert config_fingerprint(parse_config(default_schema())) == fp
    assert tuning_fingerprint(parse_config(default_schema())) == tfp


def test_resume_fingerprint_allows_expected_keys():
    cfg = parse_config(default_schema())
    fp = config_fingerprint(cfg, resume=True)
    schema = copy.deepcopy(default_schema())
    schema["training"]["max_epochs"] = 50  # 允许恢复时变化
    fp2 = config_fingerprint(parse_config(schema), resume=True)
    assert fp == fp2
    schema = copy.deepcopy(default_schema())
    schema["training"]["log_every_steps"] = 99  # 不允许
    fp3 = config_fingerprint(parse_config(schema), resume=True)
    assert fp != fp3
