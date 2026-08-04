"""toy 多分类（可中途恢复 DataModule）：供运行预算测试使用。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dl_helper.training.contracts import DataIdentity, ResumableMapDataModule, TorchExperiment
from dl_helper.training.task import MulticlassClassificationTask


class ToyResumableMLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def _collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys)


def _resumable_dm(seed=42, n_train=128, n_val=32, batch_size=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_train + n_val, 8, generator=g)
    y = torch.randint(0, 3, (n_train + n_val,), generator=g)
    train_ds = TensorDataset(x[:n_train], y[:n_train])
    val_ds = TensorDataset(x[n_train:], y[n_train:])
    return ResumableMapDataModule(
        DataIdentity("toy-mc-resumable", "1.0", "fp-toy-mc-res"),
        lambda: train_ds, _collate, batch_size=batch_size,
        val_dataset_factory=lambda: val_ds, val_batch_size=batch_size,
    )


def build_experiment(config: dict) -> TorchExperiment:
    num_classes = 3

    def task_factory():
        return MulticlassClassificationTask(num_classes=num_classes)

    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=float(config.get("lr", 0.05)))

    return TorchExperiment(
        name="toy-mc-resumable", backend="torch",
        model_factory=lambda: ToyResumableMLP(),
        datamodule_factory=lambda: _resumable_dm(),
        task_factory=task_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=lambda o: None,
        model_config=dict(config),
    )
