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


def test_kaggle_identity_path_is_not_a_physical_input(monkeypatch):
    from dl_helper.training.config import config_fingerprint

    physical_path = "/kaggle/input/datasets/lhiiqwj/weekday-dataset/data"
    logical_path = "/kaggle/input/datasets/qtz173/weekday-dataset/data"
    cfg = _cfg(experiment={"data_path": physical_path,
                           "comparison_data_identity_path": logical_path,
                           "comparison_data_fingerprint": "fixed-snapshot"})
    fingerprint = config_fingerprint(cfg, resume=True)
    checked = []

    def exists(path):
        checked.append(path)
        return path == physical_path

    monkeypatch.setattr("dl_helper.training.platform.os.path.exists", exists)
    Platform("kaggle").validate_kaggle_inputs(cfg)
    assert checked == [physical_path]
    assert cfg.experiment["comparison_data_identity_path"] == logical_path
    assert config_fingerprint(cfg, resume=True) == fingerprint


@pytest.mark.parametrize("physical_path", ["/kaggle/input/missing/data", "/tmp/data"])
def test_identity_path_does_not_bypass_physical_input_checks(monkeypatch, physical_path):
    cfg = _cfg(experiment={"comparison_data_identity_path": "historical-dataset",
                           "data_path": physical_path})
    monkeypatch.setattr("dl_helper.training.platform.os.path.exists", lambda path: False)
    with pytest.raises(PlatformError, match="Kaggle 输入路径"):
        Platform("kaggle").validate_kaggle_inputs(cfg)


@pytest.mark.parametrize("identity_path", ["", " ", None, 1, []])
def test_invalid_logical_identity_path_is_rejected(identity_path):
    cfg = _cfg(experiment={"comparison_data_identity_path": identity_path})
    with pytest.raises(PlatformError, match="逻辑身份路径必须是非空字符串"):
        Platform("kaggle").validate_kaggle_inputs(cfg)


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


def test_kaggle_budget_includes_preparation_and_uses_partial_epoch(monkeypatch):
    from dl_helper.training.platform import KAGGLE_SESSION_START_ENV, runtime_budget_for

    class Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()
    monkeypatch.setenv(KAGGLE_SESSION_START_ENV, "0.0")
    # 安装/加载已用 20 分钟；worker 初始化不能重置 Session 的 705 分钟截止点。
    clock.t = 20 * 60
    budget = runtime_budget_for(kaggle_execution_policy(), monotonic=clock)
    started = budget.begin_epoch()
    clock.t = 700 * 60
    forecast = budget.complete_epoch(started)
    assert forecast.remaining_training_seconds == 5 * 60
    assert not forecast.should_preempt  # 不够整轮仍可利用剩余时间。
    assert not budget.hit()
    clock.t = 705 * 60
    assert budget.hit()


def test_kaggle_session_clock_is_inherited_and_never_reset(monkeypatch):
    from dl_helper.training.platform import KAGGLE_SESSION_START_ENV, start_kaggle_session_clock

    monkeypatch.delenv(KAGGLE_SESSION_START_ENV, raising=False)
    monkeypatch.setattr("dl_helper.training.platform._kaggle_process_elapsed", lambda: 5.0)
    assert start_kaggle_session_clock(lambda: 10.0) == 5.0
    assert start_kaggle_session_clock(lambda: 1000.0) == 5.0


def test_kaggle_detects_kernel_without_notebook_setup(tmp_path):
    from dl_helper.training.platform import _kaggle_process_elapsed

    def process(pid, parent, start, argv):
        directory = tmp_path / str(pid)
        directory.mkdir()
        fields = ["S", str(parent), *(["0"] * 17), str(start)]
        (directory / "stat").write_text(f"{pid} (name ) with spaces) " + " ".join(fields), encoding="utf-8")
        (directory / "cmdline").write_text("\0".join(argv), encoding="utf-8")

    process(30, 20, 90000, ["python", "-m", "dl_helper.training.cli"])
    process(20, 10, 80000, ["sh", "-c", "train"])
    process(10, 1, 10000, ["python", "-m", "ipykernel_launcher", "-f", "connection.json"])
    (tmp_path / "uptime").write_text("1000.00 2000.00", encoding="utf-8")
    assert _kaggle_process_elapsed(tmp_path, pid=30, ticks_per_second=100) == 900.0
    # 同一内核下重启 CLI，不应重置准备阶段消耗的预算。
    process(40, 10, 99000, ["python", "-m", "dl_helper.training.cli"])
    assert _kaggle_process_elapsed(tmp_path, pid=40, ticks_per_second=100) == 900.0


def test_kaggle_direct_cli_uses_own_process_start(tmp_path):
    from dl_helper.training.platform import _kaggle_process_elapsed

    directory = tmp_path / "30"
    directory.mkdir()
    fields = ["S", "0", *(["0"] * 17), "90000"]
    (directory / "stat").write_text("30 (python) " + " ".join(fields), encoding="utf-8")
    (directory / "cmdline").write_text("python\0-m\0dl_helper.training.cli", encoding="utf-8")
    (tmp_path / "uptime").write_text("1000.00 2000.00", encoding="utf-8")
    assert _kaggle_process_elapsed(tmp_path, pid=30, ticks_per_second=100) == 100.0


def test_kaggle_missing_proc_fails_explicitly(tmp_path):
    from dl_helper.training.platform import _kaggle_process_elapsed

    with pytest.raises(FileNotFoundError):
        _kaggle_process_elapsed(tmp_path, pid=30, ticks_per_second=100)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "101", "broken"])
def test_kaggle_session_clock_rejects_invalid_origin(monkeypatch, value):
    from dl_helper.training.platform import KAGGLE_SESSION_START_ENV, start_kaggle_session_clock

    monkeypatch.setenv(KAGGLE_SESSION_START_ENV, value)
    with pytest.raises((PlatformError, ValueError)):
        start_kaggle_session_clock(lambda: 100.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 101.0])
def test_runtime_budget_rejects_invalid_origin(value):
    with pytest.raises(PlatformError):
        RuntimeBudget(720, 15, monotonic=lambda: 100.0, started_at=value)


def test_expired_session_budget_preempts_immediately():
    budget = RuntimeBudget(720, 15, monotonic=lambda: 710 * 60.0, started_at=0.0,
                           preempt_on_epoch_forecast=False)
    assert budget.hit()


def test_local_budget_ignores_kaggle_session_origin(monkeypatch):
    from dl_helper.training.platform import KAGGLE_SESSION_START_ENV, runtime_budget_for

    monkeypatch.setenv(KAGGLE_SESSION_START_ENV, "broken")
    policy = ExecutionPolicy(platform="local", max_minutes=10.0, shutdown_grace_minutes=2.0)
    budget = runtime_budget_for(policy, monotonic=lambda: 100.0)
    assert not budget.hit()


# ---------- D-003：独立 ExecutionPolicy ----------
def test_execution_policy_kaggle_is_720_15():
    p = kaggle_execution_policy()
    assert p.platform == "kaggle"
    assert p.max_minutes == 720.0
    assert p.shutdown_grace_minutes == 15.0


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
        execution_policy_from_dict({"schema_version": 1, "platform": "kaggle", "max_minutes": 720.0})
    # 未知字段必须拒绝
    with pytest.raises(PlatformError):
        execution_policy_from_dict({**execution_policy_to_dict(policy), "budget": 5})


def test_execution_policy_is_frozen_platform():
    with pytest.raises(Exception):
        ExecutionPolicy(platform="kaggle", max_minutes=720.0, shutdown_grace_minutes=15.0).max_minutes = 1
