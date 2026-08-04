"""任务 3.4：sklearn estimator 预检与参数解析。"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dl_helper.training.backends.sklearn_backend import (
    SklearnBackendError,
    apply_params,
    clone_estimator,
    resolve_fit_kwargs,
    validate_estimator_contract,
)
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.contracts import EstimatorBatch, SklearnExperiment
from dl_helper.training.task import SklearnMulticlassTask, SklearnRegressionTask


def _skl_cfg(**patch):
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": None,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = None
    schema["runtime"] = {"max_minutes": None, "shutdown_grace_minutes": 10}
    schema["checkpoint"] = {"every_epochs": None, "every_optimizer_steps": None,
                            "keep_last": 1, "resume": "none"}
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(schema.get(k), dict):
            schema[k] = {**schema[k], **v}
        else:
            schema[k] = v
    return parse_config(schema)


def _exp(estimator_factory, task_factory):
    return SklearnExperiment(name="e", backend="sklearn", estimator_factory=estimator_factory,
                             datamodule_factory=lambda: None, task_factory=task_factory, model_config={})


def test_pipeline_preflight_passes():
    cfg = _skl_cfg()
    est = clone_estimator(_exp(
        lambda: make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", random_state=None)),
        lambda: SklearnMulticlassTask(classes=[0, 1, 2])))
    task = SklearnMulticlassTask(classes=[0, 1, 2])
    validate_estimator_contract(est, task, cfg)
    apply_params(est, cfg)
    # n_jobs=null 不要求；random_state=run_seed 递归设置
    assert est.get_params()["sgdclassifier__random_state"] == cfg.run.seed


def test_random_forest_n_jobs():
    cfg = _skl_cfg(backend={"type": "sklearn", "torch": None,
                            "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": 4,
                                        "random_state": "run_seed", "sample_weight_parameter": None}})
    est = clone_estimator(_exp(lambda: RandomForestClassifier(n_estimators=5, random_state=None),
                               lambda: SklearnMulticlassTask(classes=[0, 1, 2])))
    apply_params(est, cfg)
    assert est.get_params()["n_jobs"] == 4
    assert est.get_params()["random_state"] == cfg.run.seed


def test_fitted_estimator_rejected():
    clf = SGDClassifier()
    X = np.random.randn(20, 4)
    y = np.random.randint(0, 3, 20)
    clf.fit(X, y)
    exp = _exp(lambda: clf, lambda: SklearnMulticlassTask(classes=[0, 1, 2]))
    with pytest.raises(SklearnBackendError):
        clone_estimator(exp)


def test_missing_top_level_n_jobs_rejected():
    class NoJobs(BaseEstimator, ClassifierMixin):
        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    cfg = _skl_cfg(backend={"type": "sklearn", "torch": None,
                            "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": 2,
                                        "random_state": "require_explicit", "sample_weight_parameter": None}})
    est = clone_estimator(_exp(lambda: NoJobs(), lambda: SklearnMulticlassTask(classes=[0, 1, 2])))
    with pytest.raises(SklearnBackendError):
        apply_params(est, cfg)


def test_random_state_require_explicit():
    class HasRandomState(BaseEstimator, ClassifierMixin):
        def __init__(self, random_state=None):
            self.random_state = random_state

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    cfg = _skl_cfg(backend={"type": "sklearn", "torch": None,
                            "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": None,
                                        "random_state": "require_explicit", "sample_weight_parameter": None}})
    est = clone_estimator(_exp(lambda: HasRandomState(), lambda: SklearnMulticlassTask(classes=[0, 1, 2])))
    with pytest.raises(SklearnBackendError):
        apply_params(est, cfg)


def test_kind_mismatch_rejected():
    cfg = _skl_cfg()
    est = clone_estimator(_exp(lambda: SGDRegressor(), lambda: SklearnRegressionTask(num_targets=1)))
    # regressor 用于回归 → 合法
    task = SklearnRegressionTask(num_targets=1)
    validate_estimator_contract(est, task, cfg)
    # classifier Task 与 regressor estimator → 拒绝
    with pytest.raises(SklearnBackendError):
        validate_estimator_contract(est, SklearnMulticlassTask(classes=[0, 1, 2]), cfg)


def test_required_prediction_missing():
    class NoProba(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    cfg = _skl_cfg()
    est = clone_estimator(_exp(lambda: NoProba(), lambda: SklearnMulticlassTask(classes=[0, 1, 2])))
    task = SklearnMulticlassTask(classes=[0, 1, 2])  # required predict_proba
    with pytest.raises(SklearnBackendError):
        validate_estimator_contract(est, task, cfg)


def test_resolve_fit_kwargs_sample_weight():
    cfg = _skl_cfg(backend={"type": "sklearn", "torch": None,
                            "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": None,
                                        "random_state": "run_seed", "sample_weight_parameter": "sample_weight"}})
    w = np.array([1.0, 2.0])
    batch = EstimatorBatch(features=np.zeros((2, 2)), targets=np.array([0, 1]),
                           sample_count=2, sample_weight=w)
    kwargs = resolve_fit_kwargs(cfg, batch)
    assert kwargs == {"sample_weight": w}
    # 数据无权重但配置了参数 → 失败
    no_w = EstimatorBatch(features=np.zeros((2, 2)), targets=np.array([0, 1]), sample_count=2)
    with pytest.raises(SklearnBackendError):
        resolve_fit_kwargs(cfg, no_w)
    # 有权重但未配置参数 → 失败
    cfg2 = _skl_cfg()
    with pytest.raises(SklearnBackendError):
        resolve_fit_kwargs(cfg2, batch)


def test_pipeline_sample_weight_path():
    cfg = _skl_cfg(backend={"type": "sklearn", "torch": None,
                            "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": None,
                                        "random_state": "run_seed",
                                        "sample_weight_parameter": "sgdclassifier__sample_weight"}})
    w = np.array([1.0, 2.0])
    batch = EstimatorBatch(features=np.zeros((2, 2)), targets=np.array([0, 1]),
                           sample_count=2, sample_weight=w)
    kwargs = resolve_fit_kwargs(cfg, batch)
    assert kwargs == {"sgdclassifier__sample_weight": w}
