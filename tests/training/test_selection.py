"""任务 3.1：selection/early-stop 与方向/定义错误。"""
from __future__ import annotations

import pytest

from dl_helper.training.contracts import MetricDefinition
from dl_helper.training.engine import (
    EngineState,
    EngineStateError,
    resolve_definition,
    validate_selection,
)
from dl_helper.training.config import Config, SelectionConfig


def _def(name, direction):
    return MetricDefinition(
        name=name, direction=direction, formula_id="f", formula_version=1,
        averaging="macro", sample_weight_policy="supported", zero_division="zero",
        exact=True, evaluation_scope="full", parameters={}, implementation="builtin_verified",
    )


def _selection(metric, mode, patience=5, min_delta=0.0):
    return SelectionConfig(metric=metric, mode=mode, patience=patience, min_delta=min_delta)


class TestSelectionTracker:
    def test_min_mode_improvement(self):
        s = EngineState("torch", "r", "fp", metric_name="val/loss", mode="min", patience=3)
        assert s.selection_update(1.0) is True
        assert s.best_value == 1.0
        assert s.selection_update(1.5) is False  # 变差
        assert s.no_improve == 1
        assert s.selection_update(0.8) is True  # 改善
        assert s.best_value == 0.8
        assert s.no_improve == 0

    def test_min_delta_required(self):
        s = EngineState("torch", "r", "fp", metric_name="val/loss", mode="min", patience=2, min_delta=0.1)
        assert s.selection_update(1.0) is True
        assert s.selection_update(0.95) is False  # 0.05 < 0.1 min_delta
        assert s.best_value == 1.0
        assert s.selection_update(0.8) is True  # 0.2 > 0.1

    def test_max_mode(self):
        s = EngineState("torch", "r", "fp", metric_name="val/f1", mode="max", patience=2)
        assert s.selection_update(0.5) is True
        assert s.selection_update(0.6) is True
        assert s.selection_update(0.55) is False

    def test_early_stop(self):
        s = EngineState("torch", "r", "fp", metric_name="val/loss", mode="min", patience=2)
        s.selection_update(1.0)
        assert s.should_early_stop() is False
        s.selection_update(1.1)
        assert s.should_early_stop() is False
        s.selection_update(1.2)
        assert s.should_early_stop() is True  # 连续 2 次未改善

    def test_non_finite_fails(self):
        s = EngineState("torch", "r", "fp", metric_name="val/loss", mode="min", patience=2)
        with pytest.raises(EngineStateError):
            s.selection_update(float("nan"))
        with pytest.raises(EngineStateError):
            s.selection_update(float("inf"))

    def test_no_early_stop_without_patience(self):
        s = EngineState("torch", "r", "fp", metric_name="val/loss", mode="min", patience=None)
        s.selection_update(1.0)
        s.selection_update(2.0)
        assert s.should_early_stop() is False


class TestSelectionValidation:
    def _defs(self):
        return {
            "loss": _def("loss", "min"),
            "f1_macro": _def("f1_macro", "max"),
            "sampled_f1": MetricDefinition(
                name="sampled_f1", direction="max", formula_id="s", formula_version=1,
                averaging="macro", sample_weight_policy="supported", zero_division="zero",
                exact=False, evaluation_scope="sampled", parameters={}, implementation="custom",
            ),
        }

    def test_has_val_requires_selection(self):
        with pytest.raises(EngineStateError):
            validate_selection(None, self._defs(), has_val=True)
        # 无 val 且 selection null → 通过
        validate_selection(None, self._defs(), has_val=False)

    def test_no_val_forbids_selection(self):
        with pytest.raises(EngineStateError):
            validate_selection(_selection("val/loss", "min"), self._defs(), has_val=False)

    def test_metric_must_exist(self):
        with pytest.raises(EngineStateError):
            validate_selection(_selection("val/nonexistent", "min"), self._defs(), has_val=True)

    def test_mode_must_match_direction(self):
        with pytest.raises(EngineStateError):
            validate_selection(_selection("val/loss", "max"), self._defs(), has_val=True)
        validate_selection(_selection("val/loss", "min"), self._defs(), has_val=True)

    def test_sampled_or_non_exact_rejected(self):
        with pytest.raises(EngineStateError):
            validate_selection(_selection("val/sampled_f1", "max"), self._defs(), has_val=True)

    def test_non_val_metric_rejected(self):
        with pytest.raises(EngineStateError):
            validate_selection(_selection("test/loss", "min"), self._defs(), has_val=True)

    def test_resolve_definition(self):
        defs = self._defs()
        assert resolve_definition("val/loss", defs) is defs["loss"]
        assert resolve_definition("loss", defs) is defs["loss"]
        assert resolve_definition("test/loss", defs) is None
        assert resolve_definition("val/missing", defs) is None
