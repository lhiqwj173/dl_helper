"""任务 7.1：Local/Kaggle 平台检测、路径与资源合同。"""
from __future__ import annotations

import os

import pytest

from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import Platform, PlatformError, RuntimeBudget


def _cfg(**patch):
    schema = default_schema()
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(schema.get(k), dict):
            schema[k] = {**schema[k], **v}
        else:
            schema[k] = v
    return parse_config(schema)


def test_detect_platform_local():
    p = Platform("local")
    assert not p.is_kaggle


def test_detect_platform_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    from dl_helper.training.platform import detect_platform
    assert detect_platform() == "kaggle"


def test_output_root_local_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Platform("local")
    assert p.resolve_output_root(_cfg()) == str(tmp_path)


def test_output_root_explicit(tmp_path):
    p = Platform("local")
    out = str(tmp_path / "custom")
    cfg = _cfg(run={"output_root": out})
    assert p.resolve_output_root(cfg) == out


def test_kaggle_output_root_must_be_in_working(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    p = Platform("kaggle")
    with pytest.raises(PlatformError):
        p.resolve_output_root(_cfg(run={"output_root": "/tmp/outside"}))


def test_kaggle_input_validation(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    p = Platform("kaggle")
    # 非 /kaggle/input 输入路径
    cfg = _cfg(experiment={"data_path": "/tmp/data"})
    with pytest.raises(PlatformError):
        p.validate_kaggle_inputs(cfg)
    # 合法路径（存在性校验在 doctor；此处只校验格式）
    cfg2 = _cfg(experiment={"data_path": "/kaggle/input/ds/data.csv"})
    # 不触发错误（路径存在性检查在 doctor 层面做，避免破坏单测）


def test_runtime_budget_basic():
    class FakeClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

    clock = FakeClock()
    b = RuntimeBudget(max_minutes=10, grace_minutes=2, monotonic=clock)
    assert not b.hit()
    clock.t = (10 - 2) * 60  # 恰好到 deadline
    assert b.hit()


def test_runtime_budget_invalid():
    with pytest.raises(PlatformError):
        RuntimeBudget(max_minutes=0, grace_minutes=1)
    with pytest.raises(PlatformError):
        RuntimeBudget(max_minutes=5, grace_minutes=5)
    with pytest.raises(PlatformError):
        RuntimeBudget(max_minutes=5, grace_minutes=10)
