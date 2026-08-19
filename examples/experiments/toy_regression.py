"""toy 多目标回归实验。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
from dl_helper.training.task import RegressionTask


class Regressor(nn.Module):
    def __init__(self, in_dim=8, num_targets=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, num_targets))

    def forward(self, x):
        return self.net(x)


def _dm(seed=42, num_targets=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(128, 8, generator=g)
    y = torch.randn(128, num_targets, generator=g)
    ds = TensorDataset(x, y)
    return LoaderDataModule(
        DataIdentity("toy-regression", "1.0", "fp-toy-regression"),
        DataLoader(ds, batch_size=16),
        nominal_train_batch_size=16,
    )


def build_experiment(config: dict) -> TorchExperiment:
    num_targets = int(config.get("num_targets", 2))

    def task_factory():
        return RegressionTask(num_targets=num_targets)

    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=float(config.get("lr", 0.05)))

    return TorchExperiment(
        name="toy-regression", backend="torch",
        model_factory=lambda: Regressor(num_targets=num_targets),
        datamodule_factory=lambda: _dm(num_targets=num_targets),
        task_factory=task_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=lambda o: None,
        model_config=dict(config),
    )
