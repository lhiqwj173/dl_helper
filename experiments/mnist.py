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
    """读取 images/labels 或 x_train/x_test/y_train/y_test 格式的 MNIST。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MNIST 数据路径不存在: {path!r}")
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if {"images", "labels"}.issubset(keys):
            images = data["images"]
            labels = data["labels"]
        elif {"x_train", "x_test", "y_train", "y_test"}.issubset(keys):
            images = np.concatenate((data["x_train"], data["x_test"]), axis=0)
            labels = np.concatenate((data["y_train"], data["y_test"]), axis=0)
        else:
            raise ValueError(
                "MNIST NPZ 必须包含 images/labels 或完整的 "
                "x_train/x_test/y_train/y_test 键；实际键为 "
                f"{sorted(keys)!r}"
            )
        if images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError(f"MNIST images 必须为 [N, 28, 28]，实际为 {images.shape!r}")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError(
                f"MNIST labels 必须为与 images 等长的一维数组，实际为 {labels.shape!r}"
            )
        images = images.astype(np.float32) / 255.0
        labels = labels.astype(np.int64)
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
