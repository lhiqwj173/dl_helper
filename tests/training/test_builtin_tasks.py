"""任务 1.5：内置 Torch/sklearn 任务适配测试。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_helper.training.contracts import EstimatorBatch, LossResult, PredictedBatch, PreparedBatch
from dl_helper.training.metrics import MetricStateError
from dl_helper.training.task import (
    MulticlassClassificationTask,
    MultilabelClassificationTask,
    RegressionTask,
    SklearnMulticlassTask,
    SklearnMultilabelTask,
    SklearnRegressionTask,
    default_model_call,
)


class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def _multiclass_batch(n=16, c=3):
    inputs = torch.randn(n, 8)
    targets = torch.randint(0, c, (n,))
    return inputs, targets


def test_multiclass_task_loss_and_predicted():
    task = MulticlassClassificationTask(num_classes=3)
    model = SimpleMLP(8, 3)
    inputs, targets = _multiclass_batch()
    prepared = task.prepare_batch((inputs, targets), "train")
    outputs = task.forward(model, prepared)
    loss = task.loss(outputs, prepared)
    assert isinstance(loss, LossResult)
    assert torch.isfinite(loss.numerator)
    pred = task.to_predicted_batch(outputs, prepared)
    assert isinstance(pred, PredictedBatch)
    assert pred.sample_count == inputs.shape[0]
    state = task.metric_state("train")
    task.update_metrics(state, pred)
    scalars = state.compute()
    assert "train/accuracy" in scalars


def test_multiclass_task_with_weights():
    task = MulticlassClassificationTask(num_classes=3)
    model = SimpleMLP(8, 3)
    inputs, targets = _multiclass_batch()
    w = torch.ones(inputs.shape[0]) * 0.5
    prepared = task.prepare_batch((inputs, targets, w), "train")
    outputs = task.forward(model, prepared)
    pred = task.to_predicted_batch(outputs, prepared)
    assert pred.sample_weight is not None
    assert np.allclose(pred.sample_weight, 0.5)


def test_multiclass_shape_error():
    task = MulticlassClassificationTask(num_classes=3)
    model = SimpleMLP(8, 3)
    inputs, targets = _multiclass_batch()
    prepared = task.prepare_batch((inputs, targets), "train")
    bad = torch.randn(16, 5)  # 错误类别数
    with pytest.raises(ValueError):
        task.loss(bad, prepared)


def test_multilabel_task():
    task = MultilabelClassificationTask(num_labels=2)
    model = SimpleMLP(8, 2)
    inputs = torch.randn(16, 8)
    targets = torch.randint(0, 2, (16, 2)).float()
    prepared = task.prepare_batch((inputs, targets), "train")
    outputs = task.forward(model, prepared)
    loss = task.loss(outputs, prepared)
    assert torch.isfinite(loss.numerator)
    pred = task.to_predicted_batch(outputs, prepared)
    assert pred.scores.shape == (16, 2)
    assert pred.predictions.shape == (16, 2)
    state = task.metric_state("train")
    task.update_metrics(state, pred)
    scalars = state.compute()
    assert "train/subset_accuracy" in scalars
    assert "train/hamming_loss" in scalars


def test_regression_task_single_and_multi_target():
    for num_targets in (1, 3):
        task = RegressionTask(num_targets=num_targets)
        model = SimpleMLP(8, num_targets)
        inputs = torch.randn(16, 8)
        targets = torch.randn(16, num_targets)
        prepared = task.prepare_batch((inputs, targets), "train")
        outputs = task.forward(model, prepared)
        loss = task.loss(outputs, prepared)
        assert torch.isfinite(loss.numerator)
        pred = task.to_predicted_batch(outputs, prepared)
        assert pred.predictions.shape == (16, num_targets)
        state = task.metric_state("train")
        task.update_metrics(state, pred)
        scalars = state.compute()
        assert "train/mae" in scalars
        assert "train/r2" in scalars


class FakeSklearnEstimator:
    """满足 classifier 协议的 fake estimator。"""

    def __init__(self):
        self.classes_ = np.array([0, 1, 2])

    def predict(self, X):
        n = X.shape[0]
        return np.full(n, 1)

    def predict_proba(self, X):
        n = X.shape[0]
        p = np.full((n, 3), 0.1)
        p[:, 1] = 0.8
        return p


def test_sklearn_multiclass_task():
    task = SklearnMulticlassTask(classes=[0, 1, 2])
    est = FakeSklearnEstimator()
    features = np.random.randn(16, 8)
    targets = np.random.randint(0, 3, (16,))
    batch = EstimatorBatch(features=features, targets=targets, sample_count=16)
    pred = task.predict_batch(est, batch)
    assert pred.sample_count == 16
    assert pred.scores.shape == (16, 3)
    state = task.metric_state("val")
    task.update_metrics(state, pred)
    scalars = state.compute()
    assert "val/accuracy" in scalars


class FakeSklearnRegressor:
    def predict(self, X):
        n = X.shape[0]
        return np.random.randn(n)


def test_sklearn_regression_task():
    task = SklearnRegressionTask(num_targets=1)
    est = FakeSklearnRegressor()
    features = np.random.randn(16, 8)
    targets = np.random.randn(16)
    batch = EstimatorBatch(features=features, targets=targets, sample_count=16)
    pred = task.predict_batch(est, batch)
    assert pred.predictions.shape[0] == 16


class FakeSklearnMultilabel:
    def predict_proba(self, X):
        n = X.shape[0]
        p = np.full((n, 2), 0.5)
        return p


def test_sklearn_multilabel_task():
    task = SklearnMultilabelTask(num_labels=2)
    est = FakeSklearnMultilabel()
    features = np.random.randn(16, 8)
    targets = np.random.randint(0, 2, (16, 2))
    batch = EstimatorBatch(features=features, targets=targets, sample_count=16)
    pred = task.predict_batch(est, batch)
    assert pred.scores.shape == (16, 2)
    assert pred.predictions.shape == (16, 2)


def test_metric_state_roundtrip_and_reduction():
    task = MulticlassClassificationTask(num_classes=3)
    model = SimpleMLP(8, 3)
    state = task.metric_state("train")
    for _ in range(2):
        inputs, targets = _multiclass_batch()
        prepared = task.prepare_batch((inputs, targets), "train")
        outputs = task.forward(model, prepared)
        pred = task.to_predicted_batch(outputs, prepared)
        task.update_metrics(state, pred)
    s1 = state.compute()
    # state roundtrip
    saved = state.state_dict()
    state2 = task.metric_state("train")
    state2.load_state_dict(saved)
    s2 = state2.compute()
    for k in s1:
        assert s1[k] == pytest.approx(s2[k])
    # reduction 单 rank 幂等
    from dl_helper.training.metrics import combine_reduction_states

    reduced = combine_reduction_states([state.reduction_state()])
    state3 = task.metric_state("train")
    state3.load_reduced_state(reduced)
    s3 = state3.compute()
    for k in s1:
        assert s1[k] == pytest.approx(s3[k])


def test_empty_split_fails():
    task = MulticlassClassificationTask(num_classes=3)
    state = task.metric_state("train")
    with pytest.raises(MetricStateError):
        state.compute()
