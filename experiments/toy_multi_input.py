"""toy 双输入多分类实验：Mapping/tuple 输入 + 内置 Task 覆盖 prepare_batch。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_helper.training.contracts import DataIdentity, LoaderDataModule, PreparedBatch, TorchExperiment
from dl_helper.training.task import MulticlassClassificationTask


class DualInputMLP(nn.Module):
    """接受两个输入 tensor 的模型。"""

    def __init__(self, d1=4, d2=4, out=3):
        super().__init__()
        self.fc1 = nn.Linear(d1, 8)
        self.fc2 = nn.Linear(d2, 8)
        self.head = nn.Linear(16, out)

    def forward(self, x1, x2):
        h = torch.cat([torch.relu(self.fc1(x1)), torch.relu(self.fc2(x2))], dim=1)
        return self.head(h)


class MultiInputTask(MulticlassClassificationTask):
    """把 (x1, x2, targets) 转为 PreparedBatch（inputs 为 tuple）。"""

    def prepare_batch(self, batch, stage):
        x1, x2, targets = batch
        return PreparedBatch(inputs=(x1, x2), targets=targets, sample_count=targets.shape[0])


def _dm(seed=42):
    g = torch.Generator().manual_seed(seed)
    x1 = torch.randn(128, 4, generator=g)
    x2 = torch.randn(128, 4, generator=g)
    y = torch.randint(0, 3, (128,), generator=g)
    ds = TensorDataset(x1, x2, y)
    return LoaderDataModule(
        DataIdentity("toy-multi-input", "1.0", "fp-toy-multi-input"),
        DataLoader(ds, batch_size=16),
        nominal_train_batch_size=16,
    )


def build_experiment(config: dict) -> TorchExperiment:
    def task_factory():
        return MultiInputTask(num_classes=3)

    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=float(config.get("lr", 0.05)))

    return TorchExperiment(
        name="toy-multi-input", backend="torch",
        model_factory=lambda: DualInputMLP(),
        datamodule_factory=lambda: _dm(),
        task_factory=task_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=lambda o: None,
        model_config=dict(config),
    )
