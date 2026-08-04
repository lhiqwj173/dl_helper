"""补充 engine 状态机 / EngineState 漂移分支覆盖（OSR-009 覆盖率门禁）。"""
from __future__ import annotations

import pytest


def test_stage_machine_unknown_stage():
    from dl_helper.training.engine import EngineStateError, StageMachine
    with pytest.raises(EngineStateError):
        StageMachine("bogus-stage")


def test_stage_machine_terminal_transition_rejected():
    from dl_helper.training.engine import (
        EngineStateError,
        STAGE_FITTING,
        STAGE_SUCCEEDED,
        StageMachine,
    )
    m = StageMachine(STAGE_SUCCEEDED)
    with pytest.raises(EngineStateError):
        m.transition(STAGE_FITTING)


def test_engine_state_load_state_dict_drifts():
    from dl_helper.training.engine import EngineState, EngineStateError
    s = EngineState("torch", "r1", "fp")

    bad_run = s.state_dict()
    bad_run["run_id"] = "r2"
    with pytest.raises(EngineStateError):
        s.load_state_dict(bad_run)

    bad_backend = s.state_dict()
    bad_backend["backend"] = "sklearn"
    with pytest.raises(EngineStateError):
        s.load_state_dict(bad_backend)

    bad_schema = s.state_dict()
    bad_schema["schema_version"] = 99
    with pytest.raises(EngineStateError):
        s.load_state_dict(bad_schema)

    bad_fp = s.state_dict()
    bad_fp["config_fingerprint"] = "other"
    with pytest.raises(EngineStateError):
        s.load_state_dict(bad_fp)
