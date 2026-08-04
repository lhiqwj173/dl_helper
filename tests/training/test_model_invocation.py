"""任务 1.5：任意 PyTorch 模型调用规则与自定义 Task 测试。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_helper.training.contracts import LossResult, PredictedBatch, PreparedBatch
from dl_helper.training.task import MulticlassClassificationTask, RegressionTask, default_model_call


class MappingModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x=None, y=None):
        return self.fc(x) + y


class TupleModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, a, b):
        return self.fc(a) + b


def test_default_call_mapping():
    model = MappingModel(4, 3)
    y = torch.randn(2, 3)
    inputs = {"x": torch.randn(2, 4), "y": y}
    out = default_model_call(model, inputs)
    assert out.shape == (2, 3)
    # 与直接调用一致
    xv = inputs["x"]
    ref = model(x=xv, y=y)
    assert torch.allclose(out, ref)


def test_default_call_tuple():
    model = TupleModel(4, 3)
    a = torch.randn(2, 4)
    b = torch.randn(2, 3)
    out = default_model_call(model, (a, b))
    ref = model(a, b)
    assert torch.allclose(out, ref)


def test_default_call_list_is_single_argument():
    model = torch.nn.Linear(4, 3)
    inputs = [torch.randn(2, 4)]
    # list 作为单参数 → 模型收到 list 而非 tensor，Linear 会失败
    with pytest.raises(TypeError):
        default_model_call(model, inputs)
    # 位置展开需要显式 tuple
    out = default_model_call(model, tuple(inputs))
    assert out.shape == (2, 3)


def test_mapping_multi_input_training():
    model = MappingModel(4, 3)
    task = MulticlassClassificationTask(num_classes=3)

    class MultiInputTask(MulticlassClassificationTask):
        def prepare_batch(self, batch, stage):
            x, y, targets = batch
            return PreparedBatch(inputs={"x": x, "y": y}, targets=targets, sample_count=targets.shape[0])

    mt = MultiInputTask(num_classes=3)
    x = torch.randn(8, 4)
    y = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    prepared = mt.prepare_batch((x, y, targets), "train")
    outputs = mt.forward(model, prepared)
    assert outputs.shape == (8, 3)


class CustomTask(RegressionTask):
    """自定义 Task 覆盖 forward/to_predicted_batch。"""

    def __init__(self):
        super().__init__(num_targets=1)
        self.name = "custom"
        self.report_kind_value = "general"

    def forward(self, model, prepared):
        raw = default_model_call(model, prepared.inputs)
        return torch.abs(raw)  # 结构化输出变换

    def loss(self, outputs, prepared):
        pred = outputs
        tgt = prepared.targets
        per_sample = (pred - tgt) ** 2
        return LossResult(numerator=per_sample.sum(), denominator=float(prepared.sample_count))

    def to_predicted_batch(self, outputs, prepared):
        pred = outputs.detach().cpu().numpy().astype(np.float64)
        tgt = prepared.targets.detach().cpu().numpy().astype(np.float64)
        return PredictedBatch(targets=tgt, predictions=pred, sample_count=prepared.sample_count)


def test_custom_task_override():
    model = torch.nn.Linear(4, 1)
    task = CustomTask()
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 1)
    prepared = task.prepare_batch((inputs, targets), "train")
    outputs = task.forward(model, prepared)
    assert torch.all(outputs >= 0)  # abs 变换生效
    loss = task.loss(outputs, prepared)
    assert torch.isfinite(loss.numerator)
    pred = task.to_predicted_batch(outputs, prepared)
    assert pred.predictions.shape == (8, 1)
    assert task.report_kind() == "general"
