"""补充 config.py 校验分支覆盖。"""
from __future__ import annotations

import pytest

from dl_helper.training.config import ConfigError, default_schema, parse_config


def _s(**patch):
    schema = default_schema()
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(schema.get(k), dict):
            schema[k] = {**schema[k], **v}
        else:
            schema[k] = v
    return schema


def test_torch_branch_requirements():
    # torch 分支缺失
    s = _s(backend={"type": "torch", "torch": None, "sklearn": None})
    with pytest.raises(ConfigError):
        parse_config(s)
    # torch 分支 + sklearn 非 null
    s = _s(backend={"type": "torch", "torch": default_schema()["backend"]["torch"],
                    "sklearn": {}})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_selection_metric_empty():
    s = _s(selection={"metric": "", "mode": "min", "patience": 1, "min_delta": 0.0})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_checkpoint_invalid():
    s = _s(checkpoint={"every_epochs": -1, "every_optimizer_steps": None,
                       "keep_last": 2})
    with pytest.raises(ConfigError):
        parse_config(s)
    s = _s(checkpoint={"every_epochs": None, "every_optimizer_steps": 0,
                       "keep_last": 2})
    with pytest.raises(ConfigError):
        parse_config(s)
    s = _s(checkpoint={"every_epochs": None, "every_optimizer_steps": None,
                       "keep_last": 0})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_runtime_rejected_as_unknown_field():
    # D-003：runtime 已移出用户 schema，任何 runtime 键按未知字段拒绝
    s = _s(runtime={"max_minutes": -1, "shutdown_grace_minutes": 10})
    with pytest.raises(ConfigError):
        parse_config(s)
    s = _s(runtime={"max_minutes": None, "shutdown_grace_minutes": -1})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_report_invalid():
    s = _s(report={"enabled": "yes", "curve_sample_limit": 100, "prediction_sample_limit": 10,
                   "prediction_splits": ["val"]})
    with pytest.raises(ConfigError):
        parse_config(s)
    s = _s(report={"enabled": True, "curve_sample_limit": 0, "prediction_sample_limit": 10,
                   "prediction_splits": ["val"]})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_remote_invalid():
    s = _s(remote={"type": "alist", "host": "https://x.example", "base_path": "",
                   "user_secret_key": "u", "password_secret_key": "p",
                   "connect_timeout_seconds": 1, "read_timeout_seconds": 1,
                   "max_attempts": 2, "async_upload": False, "failure_policy": "required"})
    with pytest.raises(ConfigError):
        parse_config(s)
    # 非法枚举
    s = _s(remote={"type": "ftp"})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_notifications_invalid():
    s = _s(notifications={"type": "wecom", "corp_id_secret_key": "k1",
                          "corp_secret_key": "k2", "agent_id_secret_key": "k3",
                          "to_user": "", "connect_timeout_seconds": 1,
                          "read_timeout_seconds": 1, "max_attempts": 2, "failure_policy": "record"})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_run_invalid():
    s = _s(run={"name": "", "id": None, "output_root": None, "source_revision": None,
                "seed": 1, "tags": {}})
    with pytest.raises(ConfigError):
        parse_config(s)
    s = _s(run={"name": "x", "id": None, "output_root": None, "source_revision": None,
                "seed": 1, "tags": {"k": 1}})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_schema_version_wrong():
    s = _s()
    s["schema_version"] = 2
    with pytest.raises(ConfigError):
        parse_config(s)


def test_experiment_non_mapping():
    s = _s()
    s["experiment"] = [1, 2]
    with pytest.raises(ConfigError):
        parse_config(s)


def test_sklearn_incremental_every_steps_rejected():
    s = _s(backend={"type": "sklearn", "torch": None,
                    "sklearn": {"fit_mode": "incremental", "evaluation_batch_size": 4096,
                                "n_jobs": 1, "random_state": "run_seed",
                                "sample_weight_parameter": None}},
           distributed={"num_processes": 1},
           training={"max_epochs": 2, "log_every_steps": 1},
           selection=None,
           checkpoint={"every_epochs": 1, "every_optimizer_steps": 1,
                       "keep_last": 2})
    with pytest.raises(ConfigError):
        parse_config(s)


def test_selection_enum():
    s = _s(selection={"metric": "val/loss", "mode": "sideways", "patience": 1, "min_delta": 0.0})
    with pytest.raises(ConfigError):
        parse_config(s)
