"""sklearn batch 实验：本地固定 ndarray 造数，Pipeline 预处理只在 train fit。"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dl_helper.training.contracts import (
    DataIdentity,
    EstimatorBatch,
    SklearnExperiment,
)
from dl_helper.training.task import SklearnMulticlassTask


class _ToyBatchDM:
    def __init__(self, seed=42, n_train=80, n_val=20, n_features=6, n_classes=3):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n_train + n_val, n_features))
        y = rng.integers(0, n_classes, n_train + n_val)
        self._x = x
        self._y = y
        self._n_train = n_train
        self._identity = DataIdentity("toy-sklearn-batch", "1.0", f"fp-skl-batch-{seed}")

    def setup(self, stage):
        return None

    def identity(self):
        return self._identity

    def full_train_data(self):
        return EstimatorBatch(
            features=self._x[: self._n_train],
            targets=self._y[: self._n_train],
            sample_count=self._n_train,
        )

    def evaluation_batches(self, stage):
        if stage == "train":
            yield EstimatorBatch(features=self._x[: self._n_train],
                                 targets=self._y[: self._n_train], sample_count=self._n_train)
        elif stage == "val":
            yield EstimatorBatch(features=self._x[self._n_train:],
                                 targets=self._y[self._n_train:],
                                 sample_count=self._x.shape[0] - self._n_train)


def build_experiment(config: dict) -> SklearnExperiment:
    n_classes = int(config.get("n_classes", 3))
    kernel = str(config.get("kernel", "linear"))
    dm = _ToyBatchDM()

    def estimator_factory():
        return make_pipeline(StandardScaler(), SVC(kernel=kernel, probability=True, random_state=42))

    def task_factory():
        return SklearnMulticlassTask(classes=[0, 1, 2][:n_classes])

    return SklearnExperiment(
        name="toy-sklearn-batch",
        backend="sklearn",
        estimator_factory=estimator_factory,
        datamodule_factory=lambda: dm,
        task_factory=task_factory,
        model_config=dict(config),
    )
