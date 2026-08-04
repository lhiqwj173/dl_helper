"""toy 多标签实验：BCE-with-logits 与显式阈值。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
from dl_helper.training.task import MultilabelClassificationTask


class MultilabelMLP(nn.Module):
    def __init__(self, in_dim=8, num_labels=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, num_labels))

    def forward(self, x):
        return self.net(x)


def _dm(seed=42, num_labels=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(128, 8, generator=g)
    y = torch.randint(0, 2, (128, num_labels), generator=g).float()
    ds = TensorDataset(x, y)
    return LoaderDataModule(
        DataIdentity("toy-multilabel", "1.0", "fp-toy-multilabel"),
        DataLoader(ds, batch_size=16),
        nominal_train_batch_size=16,
    )


def build_experiment(config: dict) -> TorchExperiment:
    num_labels = int(config.get("num_labels", 3))
    threshold = float(config.get("threshold", 0.5))

    def task_factory():
        return MultilabelClassificationTask(num_labels=num_labels, threshold=threshold)

    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=float(config.get("lr", 0.05)))

    return TorchExperiment(
        name="toy-multilabel", backend="torch",
        model_factory=lambda: MultilabelMLP(num_labels=num_labels),
        datamodule_factory=lambda: _dm(num_labels=num_labels),
        task_factory=task_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=lambda o: None,
        model_config=dict(config),
    )
