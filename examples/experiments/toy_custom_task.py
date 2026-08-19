"""toy 自定义 Task 实验：覆盖 forward/to_predicted_batch 处理结构化输出。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_helper.training.contracts import (
    DataIdentity,
    LoaderDataModule,
    LossResult,
    PredictedBatch,
    PreparedBatch,
    TorchExperiment,
)
from dl_helper.training.task import RegressionTask, default_model_call


class CustomHead(nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x)


class CustomTask(RegressionTask):
    """自定义 Task：输出经 abs 变换并覆盖 loss/predicted。"""

    def __init__(self):
        super().__init__(num_targets=1)
        self.name = "custom"
        self.report_kind_value = "general"

    def forward(self, model, prepared):
        raw = default_model_call(model, prepared.inputs)
        return torch.abs(raw)  # 结构化输出：非负

    def loss(self, outputs, prepared):
        pred = outputs
        tgt = prepared.targets
        per_sample = (pred - tgt) ** 2
        return LossResult(numerator=per_sample.sum(), denominator=float(prepared.sample_count))

    def to_predicted_batch(self, outputs, prepared):
        pred = outputs.detach().cpu().numpy().astype(np.float64)
        tgt = prepared.targets.detach().cpu().numpy().astype(np.float64)
        return PredictedBatch(targets=tgt, predictions=pred, sample_count=prepared.sample_count)


def _dm(seed=42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(160, 8, generator=g)
    y = torch.randn(160, 1, generator=g).abs()  # 非负目标
    train_ds = TensorDataset(x[:128], y[:128])
    val_ds = TensorDataset(x[128:], y[128:])
    return LoaderDataModule(
        DataIdentity("toy-custom-task", "1.0", "fp-toy-custom-task"),
        DataLoader(train_ds, batch_size=16),
        val_dataloader=DataLoader(val_ds, batch_size=16),
        nominal_train_batch_size=16,
    )


def build_experiment(config: dict) -> TorchExperiment:
    def task_factory():
        return CustomTask()

    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=float(config.get("lr", 0.05)))

    return TorchExperiment(
        name="toy-custom-task", backend="torch",
        model_factory=CustomHead,
        datamodule_factory=lambda: _dm(),
        task_factory=task_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=lambda o: None,
        model_config=dict(config),
    )
