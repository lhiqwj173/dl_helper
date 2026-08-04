"""sklearn Pipeline 实验：预处理只在 train fit，val/test 复用已 fit Pipeline。"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dl_helper.training.contracts import DataIdentity, EstimatorBatch, SklearnExperiment
from dl_helper.training.task import SklearnMulticlassTask


class _PipelineDM:
    def __init__(self, seed=42, n_train=80, n_val=20):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n_train + n_val, 6))
        y = rng.integers(0, 3, n_train + n_val)
        self._n_train = n_train
        self._x = x
        self._y = y
        self._identity = DataIdentity("toy-sklearn-pipeline", "1.0", "fp-skl-pipeline")

    def setup(self, stage):
        return None

    def identity(self):
        return self._identity

    def full_train_data(self):
        return EstimatorBatch(features=self._x[: self._n_train],
                              targets=self._y[: self._n_train], sample_count=self._n_train)

    def evaluation_batches(self, stage):
        if stage == "train":
            yield EstimatorBatch(features=self._x[: self._n_train],
                                 targets=self._y[: self._n_train], sample_count=self._n_train)
        elif stage == "val":
            yield EstimatorBatch(features=self._x[self._n_train:], targets=self._y[self._n_train:],
                                 sample_count=self._x.shape[0] - self._n_train)


def build_experiment(config: dict) -> SklearnExperiment:
    n_estimators = int(config.get("n_estimators", 20))

    def estimator_factory():
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=2),
        )

    def task_factory():
        return SklearnMulticlassTask(classes=[0, 1, 2])

    return SklearnExperiment(
        name="toy-sklearn-pipeline", backend="sklearn",
        estimator_factory=estimator_factory,
        datamodule_factory=lambda: _PipelineDM(),
        task_factory=task_factory,
        model_config=dict(config),
    )
