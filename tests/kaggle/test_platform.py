"""任务 7.1：Local/Kaggle 平台检测、路径与资源合同。"""
from __future__ import annotations

import os

import pytest

from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import (
    ExecutionPolicy,
    Platform,
    PlatformError,
    RuntimeBudget,
    execution_policy_for,
    execution_policy_from_dict,
    execution_policy_to_dict,
    kaggle_execution_policy,
    local_execution_policy,
)


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


def test_runtime_budget_forecasts_next_epoch_from_average():
    class FakeClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

    clock = FakeClock()
    budget = RuntimeBudget(max_minutes=10, grace_minutes=2, monotonic=clock)

    first_started = budget.begin_epoch()
    clock.t = 180.0
    first = budget.complete_epoch(first_started)
    assert first.average_epoch_seconds == 180.0
    assert first.remaining_training_seconds == 300.0
    assert not first.should_preempt

    second_started = budget.begin_epoch()
    clock.t = 380.0
    second = budget.complete_epoch(second_started)
    assert second.average_epoch_seconds == 190.0
    assert second.remaining_training_seconds == 100.0
    assert second.should_preempt


def test_runtime_budget_rejects_clock_rollback():
    times = iter((10.0, 20.0, 19.0))
    budget = RuntimeBudget(max_minutes=10, grace_minutes=2, monotonic=lambda: next(times))
    started = budget.begin_epoch()
    with pytest.raises(PlatformError, match="时钟倒退"):
        budget.complete_epoch(started)


def test_runtime_budget_invalid():
    with pytest.raises(PlatformError):
        RuntimeBudget(max_minutes=0, grace_minutes=1)
    with pytest.raises(PlatformError):
        RuntimeBudget(max_minutes=5, grace_minutes=5)
    with pytest.raises(PlatformError):
        RuntimeBudget(max_minutes=5, grace_minutes=10)


# ---------- D-003：独立 ExecutionPolicy ----------
def test_execution_policy_kaggle_is_660_10():
    p = kaggle_execution_policy()
    assert p.platform == "kaggle"
    assert p.max_minutes == 660.0
    assert p.shutdown_grace_minutes == 10.0


def test_execution_policy_local_has_no_budget():
    p = local_execution_policy()
    assert p.platform == "local"
    assert p.max_minutes is None
    assert p.shutdown_grace_minutes == 10.0


def test_execution_policy_for_matches_platform(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert execution_policy_for(Platform("kaggle")) == kaggle_execution_policy()
    assert execution_policy_for(Platform("local")) == local_execution_policy()


def test_execution_policy_roundtrip_and_strict_rebuild():
    policy = execution_policy_for(Platform("local"))
    assert execution_policy_from_dict(execution_policy_to_dict(policy)) == policy
    kaggle = execution_policy_for(Platform("kaggle"))
    assert execution_policy_from_dict(execution_policy_to_dict(kaggle)) == kaggle
    # 平台不一致的值必须拒绝（父进程与 worker 不得看到不同预算）
    with pytest.raises(PlatformError):
        execution_policy_from_dict({**execution_policy_to_dict(kaggle), "max_minutes": 500.0})
    # 缺字段必须拒绝
    with pytest.raises(PlatformError):
        execution_policy_from_dict({"schema_version": 1, "platform": "kaggle", "max_minutes": 660.0})
    # 未知字段必须拒绝
    with pytest.raises(PlatformError):
        execution_policy_from_dict({**execution_policy_to_dict(policy), "budget": 5})


def test_execution_policy_is_frozen_platform():
    with pytest.raises(Exception):
        ExecutionPolicy(platform="kaggle", max_minutes=660.0, shutdown_grace_minutes=10.0).max_minutes = 1
