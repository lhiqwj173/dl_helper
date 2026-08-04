"""toy 多分类实验：本地固定 seed 造数，无网络。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
from dl_helper.training.task import MulticlassClassificationTask


class ToyMLP(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _toy_dm(seed: int = 42, n_train: int = 128, n_val: int = 32, batch_size: int = 16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_train + n_val, 8, generator=g)
    y = torch.randint(0, 3, (n_train + n_val,), generator=g)
    train_ds = TensorDataset(x[:n_train], y[:n_train])
    val_ds = TensorDataset(x[n_train:], y[n_train:])
    return LoaderDataModule(
        DataIdentity("toy-multiclass", "1.0", f"fp-toy-mc-{seed}"),
        DataLoader(train_ds, batch_size=batch_size),
        val_dataloader=DataLoader(val_ds, batch_size=batch_size),
        nominal_train_batch_size=batch_size,
    )


def build_experiment(config: dict) -> TorchExperiment:
    in_dim = int(config.get("in_dim", 8))
    num_classes = int(config.get("num_classes", 3))
    hidden = int(config.get("hidden", 16))
    seed = int(config.get("seed", 42))
    batch_size = int(config.get("batch_size", 16))

    dm = _toy_dm(seed=seed, batch_size=batch_size)

    def model_factory():
        return ToyMLP(in_dim=in_dim, out_dim=num_classes, hidden=hidden)

    def task_factory():
        return MulticlassClassificationTask(num_classes=num_classes)

    def optimizer_factory(params):
        lr = float(config.get("lr", 0.05))
        return torch.optim.SGD(params, lr=lr)

    def scheduler_factory(optimizer):
        return None

    return TorchExperiment(
        name="toy-multiclass",
        backend="torch",
        model_factory=model_factory,
        datamodule_factory=lambda: dm,
        task_factory=task_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        model_config=dict(config),
    )
