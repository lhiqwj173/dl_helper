"""分支覆盖补充：metrics/config/contracts/task 负向与边界路径。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_helper.training.contracts import (
    DataIdentity,
    EstimatorBatch,
    LossResult,
    MetricDefinition,
    PredictedBatch,
    PreparedBatch,
    validate_backend_match,
    validate_metric_definition,
    validate_predicted_batch,
    validate_prepared_batch,
)
from dl_helper.training.config import ConfigError, default_schema, parse_config
from dl_helper.training.metrics import (
    LossState,
    MetricStateError,
    MulticlassState,
    MultilabelState,
    RegressionState,
    StageMetricState,
    combine_reduction_states,
)
from dl_helper.training.task import (
    MulticlassClassificationTask,
    MultilabelClassificationTask,
    RegressionTask,
    SklearnMulticlassTask,
    SklearnMultilabelTask,
    SklearnRegressionTask,
)


class TestConfigBranches:
    def test_invalid_nested_types(self):
        s = default_schema()
        s["training"]["max_epochs"] = "twenty"
        with pytest.raises(ConfigError):
            parse_config(s)
        s = default_schema()
        s["run"]["seed"] = -5
        with pytest.raises(ConfigError):
            parse_config(s)

    def test_selection_negative_delta(self):
        s = default_schema()
        s["selection"] = {"metric": "val/loss", "mode": "min", "patience": 0, "min_delta": -1.0}
        with pytest.raises(ConfigError):
            parse_config(s)

    def test_report_split_validation(self):
        s = default_schema()
        s["report"]["prediction_splits"] = ["val", "bogus"]
        with pytest.raises(ConfigError):
            parse_config(s)
        s = default_schema()
        s["report"]["prediction_splits"] = ["val", "val"]
        with pytest.raises(ConfigError):
            parse_config(s)

    def test_remote_host_required(self):
        s = default_schema()
        s["remote"] = {"type": "alist", "host": "", "base_path": "/x",
                       "user_secret_key": "u", "password_secret_key": "p",
                       "connect_timeout_seconds": 1, "read_timeout_seconds": 1,
                       "max_attempts": 2, "async_upload": False, "failure_policy": "required"}
        with pytest.raises(ConfigError):
            parse_config(s)


class TestMetricBranches:
    def test_loss_empty_compute(self):
        s = LossState("val")
        assert s.compute() == {}  # 未记录 loss → 空

    def test_multiclass_extended_negative(self):
        rng = np.random.default_rng(1)
        s = MulticlassState("val", [0, 1, 2])
        with pytest.raises(MetricStateError):
            s.compute()  # 空
        y = rng.integers(0, 3, 10)
        s.update(PredictedBatch(targets=y, predictions=y, sample_count=10))
        # 全零权重 → 合同层拒绝（weight 和必须为正）
        with pytest.raises(ValueError):
            s.update(PredictedBatch(targets=y, predictions=y, sample_count=10,
                                    sample_weight=np.zeros(10)))

    def test_balanced_accuracy_all_missing(self):
        # 所有真实类别都缺失
        s = MulticlassState("val", [0, 1, 2])
        s.update(PredictedBatch(targets=np.array([0, 0]), predictions=np.array([1, 1]),
                                sample_count=2))
        out = s.compute()
        assert out["val/balanced_accuracy"] == 0.0

    def test_regression_force_finite_constant(self):
        s = RegressionState("val", 1)
        # 常量 target、完美预测 → R2=1
        s.update(PredictedBatch(targets=np.array([5.0, 5.0]), predictions=np.array([5.0, 5.0]),
                                sample_count=2))
        assert s.compute()["val/r2"] == 1.0
        # 常量 target、有误差 → R2=0
        s2 = RegressionState("val", 1)
        s2.update(PredictedBatch(targets=np.array([5.0, 5.0]), predictions=np.array([6.0, 6.0]),
                                 sample_count=2))
        assert s2.compute()["val/r2"] == 0.0

    def test_regression_reduction_moment_merge(self):
        rng = np.random.default_rng(2)
        s1 = RegressionState("val", 2)
        s2 = RegressionState("val", 2)
        for _ in range(3):
            t = rng.random((20, 2)) * 1000
            p = t + rng.random((20, 2))
            s1.update(PredictedBatch(targets=t, predictions=p, sample_count=20))
            t2 = rng.random((20, 2)) * 1000
            p2 = t2 + rng.random((20, 2))
            s2.update(PredictedBatch(targets=t2, predictions=p2, sample_count=20))
        combined = combine_reduction_states([s1.reduction_state(), s2.reduction_state()])
        s_all = RegressionState("val", 2)
        s_all.load_reduced_state(combined)
        assert s_all.compute()

    def test_multilabel_threshold_vector(self):
        s = MultilabelState("val", 2, threshold=[0.3, 0.7])
        s.update(PredictedBatch(targets=np.array([[0, 1], [1, 0]]),
                                predictions=np.array([[0, 1], [1, 0]]),
                                scores=np.array([[0.2, 0.8], [0.8, 0.2]]), sample_count=2))
        assert s.compute()["val/subset_accuracy"] == 1.0


class TestTaskBranches:
    def test_multiclass_targets_float(self):
        task = MulticlassClassificationTask(num_classes=3)
        with pytest.raises(ValueError):
            task.prepare_batch((torch.randn(4, 8), torch.tensor([0.5, 1.5, 2.5, 0.0])), "train")

    def test_regression_target_rank(self):
        task = RegressionTask(num_targets=1)
        with pytest.raises(ValueError):
            task.prepare_batch((torch.randn(4, 8), torch.randn(4, 3, 1)), "train")

    def test_sklearn_multiclass_unknown_class(self):
        task = SklearnMulticlassTask(classes=[0, 1, 2])

        class Est:
            def predict(self, X):
                return np.array([5, 5])

            def predict_proba(self, X):
                return np.full((2, 3), 0.5)

        batch = EstimatorBatch(features=np.zeros((2, 4)), targets=np.array([0, 1]),
                               sample_count=2)
        pred = task.predict_batch(Est(), batch)
        with pytest.raises(MetricStateError):
            state = task.metric_state("val")
            state.update_predicted(pred)

    def test_sklearn_multilabel_decision_function(self):
        task = SklearnMultilabelTask(num_labels=2, required_prediction="decision_function")

        class Est:
            def decision_function(self, X):
                return np.array([[0.9, -0.9], [-0.9, 0.9]])

        batch = EstimatorBatch(features=np.zeros((2, 4)), targets=np.array([[1, 0], [0, 1]]),
                               sample_count=2)
        pred = task.predict_batch(Est(), batch)
        assert pred.predictions.shape == (2, 2)


class TestContractBranches:
    def test_prepared_batch_metadata(self):
        with pytest.raises(ValueError):
            PreparedBatch(inputs=1, targets=2, sample_count=0)

    def test_predicted_batch_2d_weight(self):
        with pytest.raises(ValueError):
            PredictedBatch(targets=np.array([0, 1]), predictions=np.array([1, 1]),
                           sample_count=2, sample_weight=np.array([[1.0], [1.0]]))

    def test_metric_definition_sampled_scope(self):
        # sampled 且 exact=True → 拒绝
        with pytest.raises(ValueError):
            MetricDefinition(name="x", direction="max", formula_id="f", formula_version=1,
                             averaging="macro", sample_weight_policy="supported",
                             zero_division="zero", exact=True, evaluation_scope="sampled",
                             parameters={}, implementation="custom")

    def test_backend_match_sklearn(self):
        from dl_helper.training.contracts import SklearnExperiment
        exp = SklearnExperiment(name="s", backend="sklearn",
                                estimator_factory=lambda: None, datamodule_factory=lambda: None,
                                task_factory=lambda: None, model_config={})
        validate_backend_match(exp, "sklearn")
        with pytest.raises(ValueError):
            validate_backend_match(exp, "torch")
