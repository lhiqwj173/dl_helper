"""MNIST Kaggle 示例：只读取显式挂载的数据路径，构造/测试不联网下载。"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
from dl_helper.training.task import MulticlassClassificationTask


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """从固定格式 NPZ 读取 images/labels；缺文件直接失败。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MNIST 数据路径不存在: {path!r}")
    with np.load(path) as data:
        images = data["images"].astype(np.float32) / 255.0
        labels = data["labels"].astype(np.int64)
    return images, labels


def _mnist_dm(config: dict):
    data_path = config["data_path"]
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("mnist 需要显式 data_path")
    images, labels = _load_npz(data_path)
    n = images.shape[0]
    n_train = int(n * 0.8)
    train_ds = TensorDataset(torch.from_numpy(images[:n_train]), torch.from_numpy(labels[:n_train]))
    val_ds = TensorDataset(torch.from_numpy(images[n_train:]), torch.from_numpy(labels[n_train:]))
    return LoaderDataModule(
        DataIdentity("mnist", "1.0", f"fp-mnist-{os.path.getsize(data_path)}"),
        DataLoader(train_ds, batch_size=64),
        val_dataloader=DataLoader(val_ds, batch_size=64),
        nominal_train_batch_size=64,
    )


def build_experiment(config: dict) -> TorchExperiment:
    def task_factory():
        return MulticlassClassificationTask(num_classes=10)

    def optimizer_factory(params):
        return torch.optim.Adam(params, lr=float(config.get("lr", 0.001)))

    return TorchExperiment(
        name="mnist", backend="torch",
        model_factory=MNISTMLP,
        datamodule_factory=lambda: _mnist_dm(config),
        task_factory=task_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=lambda o: None,
        model_config=dict(config),
    )
