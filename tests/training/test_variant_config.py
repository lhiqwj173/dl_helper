"""任务 1.3：严格 variant resolver 与 tuning fingerprint 测试。"""
from __future__ import annotations

import copy
import os

import pytest
import yaml

from dl_helper.training.config import (
    ConfigError,
    default_schema,
    parse_config,
    resolve_variant_text,
    tuning_fingerprint,
)


def _base_yaml(**patch) -> str:
    data = copy.deepcopy(default_schema())
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k] = _merge(data[k], v)
        else:
            data[k] = v
    return yaml.safe_dump(data, allow_unicode=True)


def _merge(base, patch):
    result = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def _variant_yaml(**patch) -> str:
    return yaml.safe_dump(patch, allow_unicode=True)


def test_mapping_recursive_merge():
    base = _base_yaml(training={"max_epochs": 20, "log_every_steps": 20})
    variant = _variant_yaml(training={"max_epochs": 5})
    cfg = resolve_variant_text(base, variant)
    assert cfg.training.max_epochs == 5
    assert cfg.training.log_every_steps == 20  # 未覆盖字段保留


def test_scalar_list_null_replace():
    base = _base_yaml(report={"prediction_splits": ["val", "test"]},
                      selection={"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0})
    variant = _variant_yaml(report={"prediction_splits": ["val"]})
    cfg = resolve_variant_text(base, variant)
    assert cfg.report.prediction_splits == ("val",)


def test_variant_missing_schema_version_ok():
    base = _base_yaml()
    variant = _variant_yaml(training={"max_epochs": 3})
    cfg = resolve_variant_text(base, variant)
    assert cfg.schema_version == 1
    assert cfg.training.max_epochs == 3


def test_forbidden_variant_fields():
    base = _base_yaml()
    cases = [
        {"schema_version": 2},
        {"distributed": {"num_processes": 2}},
        {"run": {"id": "other-id"}},
        {"run": {"seed": 123}},
        {"run": {"output_root": "/tmp/x"}},
        {"run": {"source_revision": "abc"}},
        {"backend": {"type": "sklearn"}},
        {"checkpoint": {"resume": "auto"}},
        {"remote": {"type": "alist", "host": "https://x.example"}},
    ]
    for patch in cases:
        with pytest.raises(ConfigError):
            resolve_variant_text(base, _variant_yaml(**patch))


def test_forbidden_unknown_top_field():
    base = _base_yaml()
    with pytest.raises(ConfigError):
        resolve_variant_text(base, _variant_yaml(bogus={"x": 1}))


def test_cross_field_conflict_after_merge():
    # 合并后 sklearn batch 与 max_epochs=2 冲突
    base = _base_yaml(backend={
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": 1,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }, distributed={"num_processes": 1}, training={"max_epochs": 1, "log_every_steps": 1})
    variant = _variant_yaml(training={"max_epochs": 2})
    with pytest.raises(ConfigError):
        resolve_variant_text(base, variant)


def test_tuning_fingerprint_same_for_infra_only_difference():
    base = _base_yaml()
    cfg_base = resolve_variant_text(base, _variant_yaml(report={"curve_sample_limit": 100000}))
    cfg_infra = resolve_variant_text(base, _variant_yaml(report={"curve_sample_limit": 50000}))
    # 报告采样上限不同 → tuning 相同
    assert tuning_fingerprint(cfg_base) == tuning_fingerprint(cfg_infra)
    # 训练参数不同 → tuning 不同
    cfg_train = resolve_variant_text(base, _variant_yaml(training={"max_epochs": 3}))
    assert tuning_fingerprint(cfg_base) != tuning_fingerprint(cfg_train)


def test_variant_merge_reruns_full_schema():
    base = _base_yaml()
    variant = _variant_yaml(training={"max_epochs": 0})  # 非法
    with pytest.raises(ConfigError):
        resolve_variant_text(base, variant)


def test_path_escape_rejected(tmp_path):
    from dl_helper.training.config import _resolve_variant_path

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "good.yaml").write_text("x: 1", encoding="utf-8")
    # 相对路径合法
    good = _resolve_variant_path(str(base_dir), "good.yaml", "variant")
    assert good == str((base_dir / "good.yaml").resolve())
    # 绝对路径拒绝
    with pytest.raises(ConfigError):
        _resolve_variant_path(str(base_dir), str(tmp_path / "outside.yaml"), "variant")
    # URL 拒绝
    with pytest.raises(ConfigError):
        _resolve_variant_path(str(base_dir), "https://example.com/a.yaml", "variant")
    # .. 逃逸拒绝
    with pytest.raises(ConfigError):
        _resolve_variant_path(str(base_dir), "../outside.yaml", "variant")
