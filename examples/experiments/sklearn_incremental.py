"""sklearn incremental 实验：SGDClassifier partial_fit 与可恢复 batch source。"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from dl_helper.training.contracts import DataIdentity, EstimatorBatch, SklearnExperiment
from dl_helper.training.task import SklearnMulticlassTask


class _ToyIncrementalSource:
    def __init__(self, seed=42, n_train=64, batch_size=16, n_features=4, n_classes=3):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n_train, n_features))
        y = rng.integers(0, n_classes, n_train)
        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._n_train = n_train
        self._consumed_batches = 0
        self.classes = np.arange(n_classes)
        self.nominal_batch_size = batch_size
        self.supports_mid_fit_resume = True

    def iter_epoch(self, epoch):
        for start in range(0, self._n_train, self._batch_size):
            end = min(start + self._batch_size, self._n_train)
            yield EstimatorBatch(
                features=self._x[start:end], targets=self._y[start:end],
                sample_count=end - start,
            )

    def state_dict(self):
        return {"consumed_batches": self._consumed_batches}

    def load_state_dict(self, state):
        self._consumed_batches = int(state["consumed_batches"])


class _ToyIncrementalDM:
    def __init__(self, seed=42, n_train=64, n_val=20, batch_size=16):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n_train + n_val, 4))
        y = rng.integers(0, 3, n_train + n_val)
        self._n_train = n_train
        self._x = x
        self._y = y
        self._source = _ToyIncrementalSource(seed=seed, n_train=n_train, batch_size=batch_size)
        self._identity = DataIdentity("toy-sklearn-incr", "1.0", f"fp-skl-incr-{seed}")

    def setup(self, stage):
        return None

    def identity(self):
        return self._identity

    def incremental_train_data(self):
        return self._source

    def evaluation_batches(self, stage):
        if stage == "val":
            yield EstimatorBatch(features=self._x[self._n_train:], targets=self._y[self._n_train:],
                                 sample_count=self._x.shape[0] - self._n_train)


def build_experiment(config: dict) -> SklearnExperiment:
    loss = str(config.get("loss", "log_loss"))
    n_classes = int(config.get("n_classes", 3))

    def estimator_factory():
        return SGDClassifier(loss=loss, random_state=42)

    def task_factory():
        return SklearnMulticlassTask(classes=[0, 1, 2][:n_classes])

    return SklearnExperiment(
        name="toy-sklearn-incremental",
        backend="sklearn",
        estimator_factory=estimator_factory,
        datamodule_factory=lambda: _ToyIncrementalDM(),
        task_factory=task_factory,
        model_config=dict(config),
    )
