"""任务 2.1：MetricDefinition 公式元数据与 MetricState 归约合同。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_helper.training.contracts import MetricDefinition, PredictedBatch
from dl_helper.training.metrics import (
    LossState,
    MetricStateError,
    MulticlassState,
    MultilabelState,
    RegressionState,
    StageMetricState,
    combine_reduction_states,
)


def _def(**kw):
    base = dict(
        name="m", direction="max", formula_id="f", formula_version=1,
        averaging="macro", sample_weight_policy="supported", zero_division="zero",
        exact=True, evaluation_scope="full", parameters={}, implementation="builtin_verified",
    )
    base.update(kw)
    return MetricDefinition(**base)


def test_metric_definition_validation_negative():
    with pytest.raises(ValueError):
        _def(name="")
    with pytest.raises(ValueError):
        _def(formula_version=0)
    with pytest.raises(ValueError):
        _def(direction="sideways")
    with pytest.raises(ValueError):
        _def(averaging="nonexistent")
    with pytest.raises(ValueError):
        _def(sample_weight_policy="sometimes")
    with pytest.raises(ValueError):
        _def(zero_division="sometimes")
    with pytest.raises(ValueError):
        _def(evaluation_scope="partial")
    with pytest.raises(ValueError):
        _def(implementation="magic")
    with pytest.raises(ValueError):
        _def(parameters={"x": float("nan")})
    with pytest.raises(ValueError):
        _def(evaluation_scope="sampled", exact=True)
    with pytest.raises(ValueError):
        _def(name="a", formula_id="")  # 空 formula_id


def test_loss_state_roundtrip_and_reduction():
    s = LossState("loss")
    s.update(2.0, 2.0)
    s.update(4.0, 4.0)
    assert s.compute() == {"loss/loss": 1.0}
    s2 = LossState("loss")
    s2.load_state_dict(s.state_dict())
    assert s2.compute() == {"loss/loss": 1.0}
    reduced = combine_reduction_states([s.reduction_state(), LossState("loss").reduction_state()])
    s3 = LossState("loss")
    s3.load_reduced_state(reduced)
    assert s3.compute() == {"loss/loss": 1.0}


def test_loss_state_rejects_nonpositive_denominator():
    s = LossState("loss")
    with pytest.raises(MetricStateError):
        s.update(1.0, 0.0)
    with pytest.raises(MetricStateError):
        s.update(1.0, float("nan"))
    with pytest.raises(MetricStateError):
        s.update(float("inf"), 1.0)


def _mc_batch(rng, n=50):
    y = rng.integers(0, 3, n)
    yhat = rng.integers(0, 3, n)
    w = (rng.random(n) + 0.1).astype(np.float64)
    return PredictedBatch(targets=y, predictions=yhat, sample_count=n, sample_weight=w)


def test_multiclass_state_reset_and_roundtrip():
    rng = np.random.default_rng(1)
    s = MulticlassState("metric", [0, 1, 2])
    for _ in range(3):
        s.update(_mc_batch(rng))
    c1 = s.compute()
    saved = s.state_dict()
    s2 = MulticlassState("metric", [0, 1, 2])
    s2.load_state_dict(saved)
    c2 = s2.compute()
    for k in c1:
        assert c1[k] == pytest.approx(c2[k])
    s.reset()
    with pytest.raises(MetricStateError):
        s.compute()


def test_multiclass_state_negative():
    rng = np.random.default_rng(2)
    with pytest.raises(MetricStateError):
        MulticlassState("m", [0, 0, 1])  # 重复类别
    s = MulticlassState("metric", [0, 1, 2])
    with pytest.raises(MetricStateError):
        s.update(PredictedBatch(targets=np.array([5]), predictions=np.array([0]),
                                sample_count=1))  # 未知类别
    with pytest.raises(ValueError):
        PredictedBatch(targets=np.array([0, 0]), predictions=np.array([0]),
                       sample_count=2)  # 样本维不一致（合同层拒绝）


def test_multilabel_state_roundtrip():
    rng = np.random.default_rng(3)
    s = MultilabelState("metric", 3, threshold=0.5)
    for _ in range(3):
        Y = rng.integers(0, 2, (40, 3))
        S = rng.random((40, 3))
        w = (rng.random(40) + 0.1).astype(np.float64)
        s.update(PredictedBatch(targets=Y, predictions=(S >= 0.5).astype(int),
                                scores=S, sample_count=40, sample_weight=w))
    c1 = s.compute()
    saved = s.state_dict()
    s2 = MultilabelState("metric", 3, threshold=0.5)
    s2.load_state_dict(saved)
    c2 = s2.compute()
    for k in c1:
        assert c1[k] == pytest.approx(c2[k])


def test_multilabel_state_negative():
    s = MultilabelState("metric", 2)
    with pytest.raises(MetricStateError):
        s.update(PredictedBatch(targets=np.array([[0, 1]]), predictions=np.array([[1, 0]]),
                                sample_count=1))  # 缺 scores
    with pytest.raises(MetricStateError):
        s.update(PredictedBatch(targets=np.array([[0, 1]]), predictions=np.array([[1, 0]]),
                                scores=np.array([[0.5]]), sample_count=1))  # shape 错误
    with pytest.raises(MetricStateError):
        s.update(PredictedBatch(targets=np.array([[2, 1]]), predictions=np.array([[1, 0]]),
                                scores=np.array([[0.5, 0.5]]), sample_count=1))  # target 非 0/1


def test_regression_state_roundtrip():
    rng = np.random.default_rng(4)
    s = RegressionState("metric", 2)
    for _ in range(3):
        t = rng.random((40, 2)) * 10
        p = t + rng.random((40, 2))
        w = (rng.random(40) + 0.1).astype(np.float64)
        s.update(PredictedBatch(targets=t, predictions=p, sample_count=40, sample_weight=w))
    c1 = s.compute()
    saved = s.state_dict()
    s2 = RegressionState("metric", 2)
    s2.load_state_dict(saved)
    c2 = s2.compute()
    for k in c1:
        assert c1[k] == pytest.approx(c2[k])


def test_regression_negative_m2_detected():
    s = RegressionState("metric", 1)
    # 伪造损坏状态
    s._m2[0] = -1e-5  # 超出舍入界
    with pytest.raises(MetricStateError):
        s._validate_m2()
    s._m2[0] = -1e-16  # 在舍入界内
    s._validate_m2()
    assert s._m2[0] == 0.0


def test_combine_reduction_consistency_checks():
    rng = np.random.default_rng(5)
    s = MulticlassState("metric", [0, 1, 2])
    s.update(_mc_batch(rng))
    states = [s.reduction_state()]
    ok = combine_reduction_states(states)
    assert "confusion" in ok
    # key 不一致
    bad = dict(states[0])
    bad.pop("confusion")
    with pytest.raises(MetricStateError):
        combine_reduction_states([states[0], bad])
    # shape 不一致
    s2 = MulticlassState("metric", [0, 1])
    s2.update(PredictedBatch(targets=np.array([0, 1]), predictions=np.array([0, 1]),
                             sample_count=2))
    with pytest.raises(MetricStateError):
        combine_reduction_states([states[0], s2.reduction_state()])


def test_stage_metric_state_two_rank_reduction():
    rng = np.random.default_rng(6)
    s1 = StageMetricState({"x": _def(name="x")}, LossState("loss"),
                          [MulticlassState("metric", [0, 1, 2])])
    s2 = StageMetricState({"x": _def(name="x")}, LossState("loss"),
                          [MulticlassState("metric", [0, 1, 2])])
    for _ in range(2):
        s1.update_predicted(_mc_batch(rng))
        s2.update_predicted(_mc_batch(rng))
    reduced = combine_reduction_states([s1.reduction_state(), s2.reduction_state()])
    s_all = StageMetricState({"x": _def(name="x")}, LossState("loss"),
                             [MulticlassState("metric", [0, 1, 2])])
    s_all.load_reduced_state(reduced)
    # 与单进程全部数据对比
    rng2 = np.random.default_rng(6)
    s_ref = StageMetricState({"x": _def(name="x")}, LossState("loss"),
                             [MulticlassState("metric", [0, 1, 2])])
    for _ in range(4):
        s_ref.update_predicted(_mc_batch(rng2))
    c_all = s_all.compute()
    c_ref = s_ref.compute()
    for k in c_ref:
        assert c_all[k] == pytest.approx(c_ref[k], abs=1e-9)
