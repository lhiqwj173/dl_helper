"""补充 metrics 校验辅助分支覆盖（OSR-009 覆盖率门禁）。"""
from __future__ import annotations

import numpy as np
import pytest

from dl_helper.training.metrics import (
    MetricStateError,
    _as_float64_weight,
    _class_index_map,
    _require_1d,
    _to_class_indices,
)


def test_as_float64_weight_negative_cases():
    with pytest.raises(MetricStateError):
        _as_float64_weight(np.zeros((2, 2)), 4)  # 非一维
    with pytest.raises(MetricStateError):
        _as_float64_weight(np.array([1.0, 2.0]), 3)  # 长度 != sample_count
    with pytest.raises(MetricStateError):
        _as_float64_weight(np.array([1.0, np.inf]), 2)  # 非有限
    with pytest.raises(MetricStateError):
        _as_float64_weight(np.array([1.0, -1.0]), 2)  # 含负值
    with pytest.raises(MetricStateError):
        _as_float64_weight(np.array([0.0, 0.0]), 2)  # 权重和 <= 0


def test_as_float64_weight_none_returns_ones():
    w = _as_float64_weight(None, 3)
    assert w.shape == (3,) and w.dtype == np.float64 and w.sum() == 3.0


def test_class_index_map_duplicate_raises():
    with pytest.raises(MetricStateError):
        _class_index_map(np.array([1, 1]))


def test_class_index_map_np_scalar_keys():
    mapping = _class_index_map(np.array([0, 1, 2]))
    assert mapping == {0: 0, 1: 1, 2: 2}


def test_to_class_indices_negative_cases():
    mapping = {0: 0, 1: 1}
    with pytest.raises(MetricStateError):
        _to_class_indices(np.array([[0], [1]]), mapping, "x")  # 非一维
    with pytest.raises(MetricStateError):
        _to_class_indices(np.array([0.5, 1.5]), mapping, "x")  # 非整数 dtype
    with pytest.raises(MetricStateError):
        _to_class_indices(np.array([0, 2]), mapping, "x")  # 未知类别


def test_require_1d_negative_cases():
    with pytest.raises(MetricStateError):
        _require_1d(np.zeros((2, 2)), "x")
    with pytest.raises(MetricStateError):
        _require_1d(np.zeros(3), "x", expected_len=5)
