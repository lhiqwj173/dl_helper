"""任务 3.1：backend-neutral 状态机与 EngineState 序列化。"""
from __future__ import annotations

import pytest

from dl_helper.training.engine import (
    EngineState,
    EngineStateError,
    STAGE_CHECKPOINTING,
    STAGE_CREATED,
    STAGE_FAILED,
    STAGE_FINALIZING,
    STAGE_FITTING,
    STAGE_PREEMPTED,
    STAGE_PREFLIGHTED,
    STAGE_SERVICES_FINALIZED,
    STAGE_SERVICES_STARTED,
    STAGE_SUCCEEDED,
    STAGE_TESTING,
    StageMachine,
    TERMINAL_STAGES,
)


def test_valid_path_to_success():
    m = StageMachine()
    assert m.stage == STAGE_CREATED
    m.transition(STAGE_PREFLIGHTED)
    m.transition(STAGE_SERVICES_STARTED)
    m.transition("PREPARED")
    m.transition(STAGE_FITTING)
    m.transition(STAGE_TESTING)
    m.transition(STAGE_FINALIZING)
    m.transition(STAGE_SERVICES_FINALIZED)
    m.transition(STAGE_SUCCEEDED)
    assert m.stage == STAGE_SUCCEEDED


def test_invalid_transition():
    m = StageMachine()
    with pytest.raises(EngineStateError):
        m.transition(STAGE_FITTING)  # CREATED 不能直接到 FITTING
    m.transition(STAGE_PREFLIGHTED)
    with pytest.raises(EngineStateError):
        m.transition(STAGE_SUCCEEDED)  # 跳过中间态
    m.transition(STAGE_FAILED)


def test_terminal_is_final():
    m = StageMachine()
    m.transition(STAGE_FAILED)
    with pytest.raises(EngineStateError):
        m.transition(STAGE_FITTING)


def test_runtime_budget_path():
    m = StageMachine()
    m.transition(STAGE_PREFLIGHTED)
    m.transition(STAGE_SERVICES_STARTED)
    m.transition("PREPARED")
    m.transition(STAGE_FITTING)
    m.transition(STAGE_CHECKPOINTING)
    m.transition(STAGE_FINALIZING)
    m.transition(STAGE_SERVICES_FINALIZED)
    m.transition(STAGE_PREEMPTED)
    assert m.stage == STAGE_PREEMPTED


def test_any_non_terminal_to_failed():
    m = StageMachine()
    m.transition(STAGE_PREFLIGHTED)
    m.transition(STAGE_SERVICES_STARTED)
    m.transition("PREPARED")
    m.transition(STAGE_FITTING)
    m.transition(STAGE_CHECKPOINTING)
    m.transition(STAGE_FINALIZING)
    m.transition(STAGE_FAILED)
    assert m.stage == STAGE_FAILED


def test_engine_state_roundtrip():
    s = EngineState("torch", "run-1", "fp")
    s.transition(STAGE_PREFLIGHTED)
    s.transition(STAGE_SERVICES_STARTED)
    s.transition("PREPARED")
    s.transition(STAGE_FITTING)
    s.advance_epoch()
    s.advance_batch()
    s.increment_global_step()
    s.selection_update(0.9)
    saved = s.state_dict()
    s2 = EngineState("torch", "run-1", "fp")
    s2.load_state_dict(saved)
    assert s2.stage == STAGE_FITTING
    assert s2.epoch == 1
    assert s2.global_step == 1
    assert s2.best_value == 0.9


def test_engine_state_backend_mismatch():
    s = EngineState("torch", "run-1", "fp")
    s2 = EngineState("sklearn", "run-1", "fp")
    with pytest.raises(EngineStateError):
        s2.load_state_dict(s.state_dict())


def test_engine_state_fingerprint_mismatch():
    s = EngineState("torch", "run-1", "fp")
    s2 = EngineState("torch", "run-1", "different-fp")
    with pytest.raises(EngineStateError):
        s2.load_state_dict(s.state_dict())


def test_engine_state_schema_version():
    s = EngineState("torch", "run-1", "fp")
    saved = dict(s.state_dict())
    saved["schema_version"] = 2
    s2 = EngineState("torch", "run-1", "fp")
    with pytest.raises(EngineStateError):
        s2.load_state_dict(saved)


def test_engine_state_advance():
    s = EngineState("torch", "r", "fp")
    s.advance_batch()
    s.advance_batch()
    s.advance_epoch()
    assert s.epoch == 1
    assert s.batch_in_epoch == 0


def test_terminal_stages_are_three():
    assert set(TERMINAL_STAGES) == {STAGE_SUCCEEDED, STAGE_FAILED, STAGE_PREEMPTED}
