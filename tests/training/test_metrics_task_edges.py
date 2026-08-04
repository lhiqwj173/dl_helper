"""补充 metrics/task 边界与错误分支。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_helper.training.contracts import EstimatorBatch, PredictedBatch
from dl_helper.training.metrics import (
    MetricStateError,
    MulticlassState,
    MultilabelState,
    RegressionState,
    combine_reduction_states,
)
from dl_helper.training.task import (
    MulticlassClassificationTask,
    MultilabelClassificationTask,
    RegressionTask,
    SklearnMulticlassTask,
    SklearnMultilabelTask,
    SklearnRegressionTask,
)


def test_combine_reduction_op_mismatch():
    s1 = MulticlassState("m", [0, 1, 2])
    s1.update(PredictedBatch(targets=np.array([0, 1, 2]), predictions=np.array([0, 1, 2]),
                             sample_count=3))
    r1 = s1.reduction_state()
    r2 = dict(r1)
    # 篡改第二个状态的 op
    first_key = next(iter(r2))
    v, op = r2[first_key]
    r2[first_key] = (v, "max" if op == "sum" else "sum")
    with pytest.raises(MetricStateError):
        combine_reduction_states([r1, r2])


def test_multilabel_threshold_out_of_range():
    with pytest.raises(MetricStateError):
        MultilabelState("m", 2, threshold=1.5)
    with pytest.raises(MetricStateError):
        MultilabelState("m", 2, threshold=-0.1)


def test_multilabel_threshold_wrong_length():
    with pytest.raises(MetricStateError):
        MultilabelState("m", 2, threshold=[0.5, 0.6, 0.7])


def test_regression_weight_sum_zero():
    s = RegressionState("m", 1)
    # 零权重和 → 合同层拒绝
    with pytest.raises(ValueError):
        s.update(PredictedBatch(targets=np.array([1.0]), predictions=np.array([1.0]),
                                sample_count=1, sample_weight=np.array([0.0])))


def test_multiclass_negative_m2_rounding():
    s = RegressionState("m", 1)
    s._m2[0] = -1e-16  # 在舍入界内
    s._validate_m2()
    assert s._m2[0] == 0.0


def test_sklearn_regression_2d_targets():
    task = SklearnRegressionTask(num_targets=2)

    class Est:
        def predict(self, X):
            return np.random.randn(X.shape[0], 2)

    batch = EstimatorBatch(features=np.zeros((8, 4)), targets=np.random.randn(8, 2),
                           sample_count=8)
    pred = task.predict_batch(Est(), batch)
    assert pred.predictions.shape == (8, 2)


def test_sklearn_multilabel_predict_proba_3d():
    task = SklearnMultilabelTask(num_labels=2)

    class Est:
        def predict_proba(self, X):
            n = X.shape[0]
            return np.stack([np.full((n, 2), 0.4), np.full((n, 2), 0.6)], axis=1)

    batch = EstimatorBatch(features=np.zeros((4, 4)), targets=np.array([[0, 1], [1, 0], [1, 1], [0, 0]]),
                           sample_count=4)
    pred = task.predict_batch(Est(), batch)
    assert pred.scores.shape == (4, 2)


def test_multilabel_bce_loss_weights():
    task = MultilabelClassificationTask(num_labels=3)
    model = torch.nn.Linear(8, 3)
    inputs = torch.randn(8, 8)
    targets = torch.randint(0, 2, (8, 3)).float()
    w = torch.ones(8) * 2.0
    prepared = task.prepare_batch((inputs, targets, w), "train")
    outputs = task.forward(model, prepared)
    loss = task.loss(outputs, prepared)
    assert torch.isfinite(loss.numerator)


def test_regression_single_target_batch():
    task = RegressionTask(num_targets=1)
    model = torch.nn.Linear(8, 1)
    inputs = torch.randn(8, 8)
    targets = torch.randn(8)  # 一维
    prepared = task.prepare_batch((inputs, targets), "train")
    outputs = task.forward(model, prepared)
    loss = task.loss(outputs, prepared)
    assert torch.isfinite(loss.numerator)


def test_sklearn_multiclass_prediction_unknown():
    task = SklearnMulticlassTask(classes=[0, 1, 2])

    class Est:
        def predict(self, X):
            return np.array([9, 9])

        def predict_proba(self, X):
            return np.full((2, 3), 0.5)

    batch = EstimatorBatch(features=np.zeros((2, 4)), targets=np.array([0, 1]), sample_count=2)
    pred = task.predict_batch(Est(), batch)
    # 未知预测类别在指标状态更新时失败
    from dl_helper.training.metrics import MetricStateError
    state = task.metric_state("val")
    with pytest.raises(MetricStateError):
        state.update_predicted(pred)
