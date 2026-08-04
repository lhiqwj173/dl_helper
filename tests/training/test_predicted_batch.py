"""任务 1.5：PredictedBatch 合同、prediction arrays 与内置任务正负用例。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_helper.training.contracts import PredictedBatch
from dl_helper.training.task import (
    MulticlassClassificationTask,
    MultilabelClassificationTask,
    RegressionTask,
)


class _SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def _run_task(task, model, inputs, targets, extra=None):
    if extra is not None:
        batch = (inputs, targets, extra)
    else:
        batch = (inputs, targets)
    prepared = task.prepare_batch(batch, "val")
    outputs = task.forward(model, prepared)
    return task.to_predicted_batch(outputs, prepared)


def test_multiclass_prediction_arrays():
    task = MulticlassClassificationTask(num_classes=3)
    model = _SimpleMLP(8, 3)
    pred = _run_task(task, model, torch.randn(10, 8), torch.randint(0, 3, (10,)))
    arrays = task.prediction_arrays(pred)
    assert arrays["targets"].shape == (10,)
    assert arrays["predictions"].shape == (10,)
    assert arrays["scores"].shape == (10, 3)
    for v in arrays.values():
        assert v.shape[0] == pred.sample_count


def test_multilabel_prediction_arrays():
    task = MultilabelClassificationTask(num_labels=2)
    model = _SimpleMLP(8, 2)
    pred = _run_task(task, model, torch.randn(10, 8), torch.randint(0, 2, (10, 2)).float())
    arrays = task.prediction_arrays(pred)
    assert arrays["scores"].shape == (10, 2)
    assert arrays["predictions"].shape == (10, 2)


def test_regression_prediction_arrays():
    task = RegressionTask(num_targets=2)
    model = _SimpleMLP(8, 2)
    pred = _run_task(task, model, torch.randn(10, 8), torch.randn(10, 2))
    arrays = task.prediction_arrays(pred)
    assert arrays["targets"].shape == (10, 2)
    assert arrays["predictions"].shape == (10, 2)


def test_predicted_batch_sample_weight_passthrough():
    task = MulticlassClassificationTask(num_classes=3)
    model = _SimpleMLP(8, 3)
    w = torch.full((10,), 2.0)
    pred = _run_task(task, model, torch.randn(10, 8), torch.randint(0, 3, (10,)), extra=w)
    assert pred.sample_weight is not None
    assert np.allclose(pred.sample_weight, 2.0)


def test_predicted_batch_contract_errors():
    tgt = np.array([0, 1], dtype=np.int64)
    pred = np.array([1, 1], dtype=np.int64)
    # 样本维不一致
    with pytest.raises(ValueError):
        PredictedBatch(targets=tgt, predictions=np.array([1], dtype=np.int64), sample_count=2)
    # 非有限 scores
    with pytest.raises(ValueError):
        PredictedBatch(targets=tgt, predictions=pred, sample_count=2, scores=np.array([np.nan, 1.0]))
    # 负权重
    with pytest.raises(ValueError):
        PredictedBatch(targets=tgt, predictions=pred, sample_count=2,
                       sample_weight=np.array([-0.1, 1.0]))
    # 非 float weight dtype
    with pytest.raises(TypeError):
        PredictedBatch(targets=tgt, predictions=pred, sample_count=2,
                       sample_weight=np.array([1, 0], dtype=np.int64))


def test_invalid_prepared_batch_sample_weight():
    task = MulticlassClassificationTask(num_classes=3)
    with pytest.raises(ValueError):
        task.prepare_batch((torch.randn(10, 8), torch.randint(0, 3, (10,)),
                            torch.full((5,), 1.0)), "train")  # 长度不一致
