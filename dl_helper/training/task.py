"""统一 Task：PreparedBatch/LossResult/PredictedBatch 与内置多分类/多标签/回归任务。"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import (
    EstimatorBatch,
    LossResult,
    MetricDefinition,
    PredictedBatch,
    PreparedBatch,
    TorchTask,
    validate_loss_result,
    validate_predicted_batch,
    validate_prepared_batch,
)
from .metrics import (
    LossState,
    MetricStateError,
    MulticlassState,
    MultilabelState,
    RegressionState,
    StageMetricState,
)

# --------------------------------------------------------------------------
# 默认模型调用规则
# --------------------------------------------------------------------------

def default_model_call(model: torch.nn.Module, inputs: Any) -> Any:
    """固定调用规则：Mapping->关键字参数、tuple->位置参数、其他->单参数。"""
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _split_batch(batch: Any, stage: str) -> tuple[Any, torch.Tensor, torch.Tensor | None]:
    """从原始 batch 提取 inputs/targets/可选 sample_weight。"""
    if isinstance(batch, Mapping):
        if "inputs" not in batch or "targets" not in batch:
            raise TypeError(f"batch mapping 必须含 inputs/targets 键")
        inputs = batch["inputs"]
        targets = batch["targets"]
        weight = batch.get("sample_weight")
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            inputs, targets = batch
            weight = None
        elif len(batch) == 3:
            inputs, targets, weight = batch
        else:
            raise TypeError(f"内置 Task 只接受 (inputs, targets[, sample_weight])，得到长度 {len(batch)}")
    else:
        raise TypeError(f"内置 Task 无法解析 batch 类型: {type(batch).__name__}")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("内置 Torch Task 的 targets 必须是 Tensor")
    return inputs, targets, weight


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# --------------------------------------------------------------------------
# 内置 Torch 任务
# --------------------------------------------------------------------------

class MulticlassClassificationTask:
    """logits [N,C] 与 long target [N]。"""

    def __init__(self, num_classes: int, threshold: float | None = None) -> None:
        if num_classes < 2:
            raise ValueError("多分类必须至少两个类别")
        self.num_classes = num_classes
        self.classes = np.arange(num_classes, dtype=np.int64)
        self.threshold = threshold
        self.label_names = [str(value) for value in self.classes.tolist()]
        self.name = "multiclass"
        self.report_kind_value = "multiclass"
        self._metric_definitions = _multiclass_definitions(num_classes)

    @property
    def metric_definitions(self) -> Mapping[str, MetricDefinition]:
        return self._metric_definitions

    def report_kind(self) -> str:
        return self.report_kind_value

    def metric_state(self, stage: str) -> StageMetricState:
        return StageMetricState(
            self._metric_definitions,
            LossState(stage),
            [MulticlassState(stage, self.classes)],
        )

    def prepare_batch(self, batch: Any, stage: str) -> PreparedBatch:
        inputs, targets, weight = _split_batch(batch, stage)
        if targets.dim() != 1:
            raise ValueError(f"多分类 targets 必须为一维，得到 {targets.dim()} 维")
        if targets.dtype not in (torch.int64, torch.int32, torch.uint8, torch.int16, torch.int8):
            raise ValueError(f"多分类 targets 必须是整数类别索引，得到 dtype {targets.dtype}")
        targets = targets.long()
        n = targets.shape[0]
        if n == 0:
            raise ValueError("空 batch 不得用于评价")
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32)
        prepared = PreparedBatch(inputs=inputs, targets=targets, sample_count=n, sample_weight=weight)
        validate_prepared_batch(prepared)
        return prepared

    def forward(self, model: torch.nn.Module, prepared: PreparedBatch) -> Any:
        return default_model_call(model, prepared.inputs)

    def loss(self, outputs: Any, prepared: PreparedBatch) -> LossResult:
        logits = outputs
        if not isinstance(logits, torch.Tensor):
            raise TypeError("多分类模型输出必须是 Tensor logits")
        if logits.dim() != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"多分类 logits 必须为 [N,{self.num_classes}]，得到 {tuple(logits.shape)}"
            )
        if logits.shape[0] != prepared.sample_count:
            raise ValueError("logits 样本维与 sample_count 不一致")
        weights = prepared.sample_weight
        per_sample = F.cross_entropy(logits, prepared.targets, reduction="none")
        if weights is not None:
            per_sample = per_sample * weights
            denom = weights.sum().detach()
        else:
            denom = torch.tensor(float(prepared.sample_count), dtype=logits.dtype).detach()
        result = LossResult(numerator=per_sample.sum(), denominator=denom)
        validate_loss_result(result)
        return result

    def to_predicted_batch(self, outputs: Any, prepared: PreparedBatch) -> PredictedBatch:
        logits = outputs
        if not isinstance(logits, torch.Tensor):
            raise TypeError("多分类模型输出必须是 Tensor logits")
        if logits.dim() != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"多分类 logits 必须为 [N,{self.num_classes}]，得到 {tuple(logits.shape)}"
            )
        predictions = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
        predicted = PredictedBatch(
            targets=_to_numpy(prepared.targets),
            predictions=_to_numpy(predictions).astype(np.int64),
            sample_count=prepared.sample_count,
            scores=_to_numpy(probs).astype(np.float64),
            sample_weight=_to_numpy(prepared.sample_weight) if prepared.sample_weight is not None else None,
        )
        validate_predicted_batch(predicted)
        return predicted

    def update_metrics(self, state: StageMetricState, predicted: PredictedBatch) -> None:
        state.update_predicted(predicted)

    def prediction_arrays(self, predicted: PredictedBatch) -> Mapping[str, np.ndarray]:
        return {
            "targets": np.asarray(predicted.targets),
            "predictions": np.asarray(predicted.predictions),
            "scores": np.asarray(predicted.scores),
        }


class MultilabelClassificationTask:
    """logits/target [N,L]、BCE-with-logits 与显式阈值。"""

    def __init__(self, num_labels: int, threshold: float | Sequence[float] = 0.5,
                 label_names: Sequence[str] | None = None) -> None:
        if num_labels < 1:
            raise ValueError("多标签必须至少一个 label")
        self.num_labels = num_labels
        if isinstance(threshold, (int, float)):
            thr = float(threshold)
        else:
            thr = np.asarray(threshold, dtype=np.float64)
            if thr.shape != (num_labels,):
                raise ValueError("threshold 向量长度必须等于 label 数")
        self.threshold = thr
        self.label_names = list(label_names) if label_names else [str(i) for i in range(num_labels)]
        self.name = "multilabel"
        self.report_kind_value = "multilabel"
        self._metric_definitions = _multilabel_definitions(num_labels)

    @property
    def metric_definitions(self) -> Mapping[str, MetricDefinition]:
        return self._metric_definitions

    def report_kind(self) -> str:
        return self.report_kind_value

    def metric_state(self, stage: str) -> StageMetricState:
        return StageMetricState(
            self._metric_definitions,
            LossState(stage),
            [MultilabelState(stage, self.num_labels, self.threshold, self.label_names)],
        )

    def prepare_batch(self, batch: Any, stage: str) -> PreparedBatch:
        inputs, targets, weight = _split_batch(batch, stage)
        if targets.dim() != 2 or targets.shape[1] != self.num_labels:
            raise ValueError(
                f"多标签 targets 必须为 [N,{self.num_labels}]，得到 {tuple(targets.shape)}"
            )
        n = targets.shape[0]
        if n == 0:
            raise ValueError("空 batch 不得用于评价")
        if targets.dtype != torch.float32:
            targets = targets.float()
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32)
        prepared = PreparedBatch(inputs=inputs, targets=targets, sample_count=n, sample_weight=weight)
        validate_prepared_batch(prepared)
        return prepared

    def forward(self, model: torch.nn.Module, prepared: PreparedBatch) -> Any:
        return default_model_call(model, prepared.inputs)

    def loss(self, outputs: Any, prepared: PreparedBatch) -> LossResult:
        logits = outputs
        if not isinstance(logits, torch.Tensor):
            raise TypeError("多标签模型输出必须是 Tensor logits")
        if logits.dim() != 2 or logits.shape[1] != self.num_labels:
            raise ValueError(
                f"多标签 logits 必须为 [N,{self.num_labels}]，得到 {tuple(logits.shape)}"
            )
        weights = prepared.sample_weight
        per_label = F.binary_cross_entropy_with_logits(logits, prepared.targets, reduction="none")
        per_sample = per_label.mean(dim=1)
        if weights is not None:
            per_sample = per_sample * weights
            denom = weights.sum().detach()
        else:
            denom = torch.tensor(float(prepared.sample_count), dtype=logits.dtype).detach()
        result = LossResult(numerator=per_sample.sum(), denominator=denom)
        validate_loss_result(result)
        return result

    def to_predicted_batch(self, outputs: Any, prepared: PreparedBatch) -> PredictedBatch:
        logits = outputs
        if not isinstance(logits, torch.Tensor):
            raise TypeError("多标签模型输出必须是 Tensor logits")
        if logits.dim() != 2 or logits.shape[1] != self.num_labels:
            raise ValueError(
                f"多标签 logits 必须为 [N,{self.num_labels}]，得到 {tuple(logits.shape)}"
            )
        probs = torch.sigmoid(logits)
        thr = np.asarray(self.threshold, dtype=np.float64)
        hard = (probs.detach().cpu().numpy() >= thr).astype(np.int64)
        predicted = PredictedBatch(
            targets=_to_numpy(prepared.targets),
            predictions=hard,
            sample_count=prepared.sample_count,
            scores=_to_numpy(probs).astype(np.float64),
            sample_weight=_to_numpy(prepared.sample_weight) if prepared.sample_weight is not None else None,
        )
        validate_predicted_batch(predicted)
        return predicted

    def update_metrics(self, state: StageMetricState, predicted: PredictedBatch) -> None:
        state.update_predicted(predicted)

    def prediction_arrays(self, predicted: PredictedBatch) -> Mapping[str, np.ndarray]:
        return {
            "targets": np.asarray(predicted.targets),
            "predictions": np.asarray(predicted.predictions),
            "scores": np.asarray(predicted.scores),
        }


class RegressionTask:
    """prediction/target [N] 或 [N,D]。"""

    def __init__(self, num_targets: int = 1, target_names: Sequence[str] | None = None) -> None:
        if num_targets < 1:
            raise ValueError("回归必须至少一个 target")
        self.num_targets = num_targets
        self.target_names = list(target_names) if target_names else [str(i) for i in range(num_targets)]
        self.name = "regression"
        self.report_kind_value = "regression"
        self._metric_definitions = _regression_definitions(num_targets)

    @property
    def metric_definitions(self) -> Mapping[str, MetricDefinition]:
        return self._metric_definitions

    def report_kind(self) -> str:
        return self.report_kind_value

    def metric_state(self, stage: str) -> StageMetricState:
        return StageMetricState(
            self._metric_definitions,
            LossState(stage),
            [RegressionState(stage, self.num_targets, self.target_names)],
        )

    def prepare_batch(self, batch: Any, stage: str) -> PreparedBatch:
        inputs, targets, weight = _split_batch(batch, stage)
        if targets.dim() not in (1, 2):
            raise ValueError(f"回归 targets 必须为一/二维，得到 {targets.dim()} 维")
        n = targets.shape[0]
        if n == 0:
            raise ValueError("空 batch 不得用于评价")
        if targets.dtype != torch.float32:
            targets = targets.float()
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32)
        prepared = PreparedBatch(inputs=inputs, targets=targets, sample_count=n, sample_weight=weight)
        validate_prepared_batch(prepared)
        return prepared

    def forward(self, model: torch.nn.Module, prepared: PreparedBatch) -> Any:
        return default_model_call(model, prepared.inputs)

    def loss(self, outputs: Any, prepared: PreparedBatch) -> LossResult:
        pred = outputs
        if not isinstance(pred, torch.Tensor):
            raise TypeError("回归模型输出必须是 Tensor")
        if pred.dim() == 1:
            pred = pred[:, None]
        if pred.dim() != 2 or pred.shape[1] != self.num_targets:
            raise ValueError(
                f"回归预测必须为 [N,{self.num_targets}]，得到 {tuple(pred.shape)}"
            )
        tgt = prepared.targets if prepared.targets.dim() == 2 else prepared.targets[:, None]
        if tgt.shape != pred.shape:
            raise ValueError(f"回归 target shape {tuple(tgt.shape)} != prediction shape {tuple(pred.shape)}")
        weights = prepared.sample_weight
        per_target = F.mse_loss(pred, tgt, reduction="none")
        per_sample = per_target.mean(dim=1)
        if weights is not None:
            per_sample = per_sample * weights
            denom = weights.sum().detach()
        else:
            denom = torch.tensor(float(prepared.sample_count), dtype=pred.dtype).detach()
        result = LossResult(numerator=per_sample.sum(), denominator=denom)
        validate_loss_result(result)
        return result

    def to_predicted_batch(self, outputs: Any, prepared: PreparedBatch) -> PredictedBatch:
        pred = outputs
        if not isinstance(pred, torch.Tensor):
            raise TypeError("回归模型输出必须是 Tensor")
        if pred.dim() == 1:
            pred = pred[:, None]
        if pred.dim() != 2 or pred.shape[1] != self.num_targets:
            raise ValueError(
                f"回归预测必须为 [N,{self.num_targets}]，得到 {tuple(pred.shape)}"
            )
        tgt = prepared.targets if prepared.targets.dim() == 2 else prepared.targets[:, None]
        predicted = PredictedBatch(
            targets=_to_numpy(tgt).astype(np.float64),
            predictions=_to_numpy(pred).astype(np.float64),
            sample_count=prepared.sample_count,
            sample_weight=_to_numpy(prepared.sample_weight) if prepared.sample_weight is not None else None,
        )
        validate_predicted_batch(predicted)
        return predicted

    def update_metrics(self, state: StageMetricState, predicted: PredictedBatch) -> None:
        state.update_predicted(predicted)

    def prediction_arrays(self, predicted: PredictedBatch) -> Mapping[str, np.ndarray]:
        return {
            "targets": np.asarray(predicted.targets),
            "predictions": np.asarray(predicted.predictions),
        }


# --------------------------------------------------------------------------
# 内置 sklearn 任务
# --------------------------------------------------------------------------

class SklearnMulticlassTask:
    """sklearn 多分类：声明 classes，predict_proba 作为 scores。"""

    estimator_kind = "classifier"
    required_prediction = "predict_proba"

    def __init__(self, classes: Sequence[Any]) -> None:
        classes_arr = np.asarray(classes)
        if classes_arr.ndim != 1 or classes_arr.shape[0] < 2:
            raise ValueError("sklearn 多分类必须声明至少两个唯一类别")
        if classes_arr.shape[0] != len(set(classes_arr.tolist())):
            raise ValueError("sklearn 多分类类别必须唯一")
        self.classes = classes_arr
        self.name = "multiclass"
        self.report_kind_value = "multiclass"
        self._metric_definitions = _multiclass_definitions(classes_arr.shape[0])

    @property
    def metric_definitions(self) -> Mapping[str, MetricDefinition]:
        return self._metric_definitions

    def report_kind(self) -> str:
        return self.report_kind_value

    def metric_state(self, stage: str) -> StageMetricState:
        return StageMetricState(
            self._metric_definitions,
            LossState(stage),
            [MulticlassState(stage, self.classes)],
        )

    def predict_batch(self, estimator: Any, batch: EstimatorBatch) -> PredictedBatch:
        features = batch.features
        predictions = np.asarray(estimator.predict(features))
        scores = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != self.classes.shape[0]:
            raise ValueError("predict_proba 列数必须等于声明类别数")
        targets = np.asarray(batch.targets)
        predicted = PredictedBatch(
            targets=targets,
            predictions=predictions,
            sample_count=batch.sample_count,
            scores=scores,
            sample_weight=batch.sample_weight,
            sample_ids=batch.sample_ids,
        )
        validate_predicted_batch(predicted)
        return predicted

    def update_metrics(self, state: StageMetricState, predicted: PredictedBatch) -> None:
        state.update_predicted(predicted)

    def prediction_arrays(self, predicted: PredictedBatch) -> Mapping[str, np.ndarray]:
        return {
            "targets": np.asarray(predicted.targets),
            "predictions": np.asarray(predicted.predictions),
            "scores": np.asarray(predicted.scores),
        }


class SklearnMultilabelTask:
    """sklearn 多标签：decision_function/predict_proba 作为 scores。"""

    estimator_kind = "classifier"

    def __init__(self, num_labels: int, required_prediction: str = "predict_proba",
                 threshold: float | Sequence[float] = 0.5,
                 label_names: Sequence[str] | None = None) -> None:
        if num_labels < 1:
            raise ValueError("多标签必须至少一个 label")
        if required_prediction not in ("predict_proba", "decision_function"):
            raise ValueError("多标签 sklearn Task 的 required_prediction 必须是 predict_proba/decision_function")
        self.num_labels = num_labels
        self.required_prediction = required_prediction
        self.classes = None
        if isinstance(threshold, (int, float)):
            thr = float(threshold)
        else:
            thr = np.asarray(threshold, dtype=np.float64)
            if thr.shape != (num_labels,):
                raise ValueError("threshold 向量长度必须等于 label 数")
        self.threshold = thr
        self.label_names = list(label_names) if label_names else [str(i) for i in range(num_labels)]
        self.name = "multilabel"
        self.report_kind_value = "multilabel"
        self._metric_definitions = _multilabel_definitions(num_labels)

    @property
    def metric_definitions(self) -> Mapping[str, MetricDefinition]:
        return self._metric_definitions

    def report_kind(self) -> str:
        return self.report_kind_value

    def metric_state(self, stage: str) -> StageMetricState:
        return StageMetricState(
            self._metric_definitions,
            LossState(stage),
            [MultilabelState(stage, self.num_labels, self.threshold, self.label_names)],
        )

    def predict_batch(self, estimator: Any, batch: EstimatorBatch) -> PredictedBatch:
        features = batch.features
        if self.required_prediction == "predict_proba":
            scores = np.asarray(estimator.predict_proba(features), dtype=np.float64)
            if scores.ndim == 3:  # 每个 label 一个二分类概率矩阵
                scores = scores[:, :, 1]
        elif self.required_prediction == "decision_function":
            scores = np.asarray(estimator.decision_function(features), dtype=np.float64)
            if scores.ndim == 2 and scores.shape[1] == 1:
                scores = scores[:, 0]
        if scores.ndim != 2 or scores.shape[1] != self.num_labels:
            raise ValueError(f"scores 必须为 [N,{self.num_labels}]，得到 {scores.shape}")
        thr = np.asarray(self.threshold, dtype=np.float64)
        hard = (scores >= thr).astype(np.int64)
        targets = np.asarray(batch.targets)
        predicted = PredictedBatch(
            targets=targets,
            predictions=hard,
            sample_count=batch.sample_count,
            scores=scores,
            sample_weight=batch.sample_weight,
            sample_ids=batch.sample_ids,
        )
        validate_predicted_batch(predicted)
        return predicted

    def update_metrics(self, state: StageMetricState, predicted: PredictedBatch) -> None:
        state.update_predicted(predicted)

    def prediction_arrays(self, predicted: PredictedBatch) -> Mapping[str, np.ndarray]:
        return {
            "targets": np.asarray(predicted.targets),
            "predictions": np.asarray(predicted.predictions),
            "scores": np.asarray(predicted.scores),
        }


class SklearnRegressionTask:
    """sklearn 回归。"""

    estimator_kind = "regressor"
    required_prediction = "predict"

    def __init__(self, num_targets: int = 1, target_names: Sequence[str] | None = None) -> None:
        if num_targets < 1:
            raise ValueError("回归必须至少一个 target")
        self.num_targets = num_targets
        self.classes = None
        self.target_names = list(target_names) if target_names else [str(i) for i in range(num_targets)]
        self.name = "regression"
        self.report_kind_value = "regression"
        self._metric_definitions = _regression_definitions(num_targets)

    @property
    def metric_definitions(self) -> Mapping[str, MetricDefinition]:
        return self._metric_definitions

    def report_kind(self) -> str:
        return self.report_kind_value

    def metric_state(self, stage: str) -> StageMetricState:
        return StageMetricState(
            self._metric_definitions,
            LossState(stage),
            [RegressionState(stage, self.num_targets, self.target_names)],
        )

    def predict_batch(self, estimator: Any, batch: EstimatorBatch) -> PredictedBatch:
        pred = np.asarray(estimator.predict(batch.features), dtype=np.float64)
        targets = np.asarray(batch.targets)
        if pred.ndim == 1 and targets.ndim == 2:
            pred = pred[:, None]
        if targets.ndim == 1:
            targets = targets[:, None]
        predicted = PredictedBatch(
            targets=targets,
            predictions=pred,
            sample_count=batch.sample_count,
            sample_weight=batch.sample_weight,
            sample_ids=batch.sample_ids,
        )
        validate_predicted_batch(predicted)
        return predicted

    def update_metrics(self, state: StageMetricState, predicted: PredictedBatch) -> None:
        state.update_predicted(predicted)

    def prediction_arrays(self, predicted: PredictedBatch) -> Mapping[str, np.ndarray]:
        return {
            "targets": np.asarray(predicted.targets),
            "predictions": np.asarray(predicted.predictions),
        }


# --------------------------------------------------------------------------
# MetricDefinition 构建
# --------------------------------------------------------------------------

def _multiclass_definitions(num_classes: int) -> Mapping[str, MetricDefinition]:
    base = dict(
        zero_division="zero",
        exact=True,
        evaluation_scope="full",
        sample_weight_policy="supported",
        implementation="builtin_verified",
        parameters={"num_classes": num_classes},
    )
    return {
        "loss": MetricDefinition(
            name="loss", direction="min", formula_id="weighted_mean_loss", formula_version=1,
            averaging="none", sample_weight_policy="supported", zero_division="not_applicable",
            exact=True, evaluation_scope="full", parameters={}, implementation="builtin_verified",
        ),
        "accuracy": MetricDefinition(name="accuracy", direction="max", formula_id="multiclass_accuracy",
                                     formula_version=1, averaging="micro", **base),
        "balanced_accuracy": MetricDefinition(name="balanced_accuracy", direction="max",
                                              formula_id="multiclass_balanced_accuracy", formula_version=1,
                                              averaging="macro", **base),
        "precision_macro": MetricDefinition(name="precision_macro", direction="max",
                                            formula_id="multiclass_precision_macro", formula_version=1,
                                            averaging="macro", **base),
        "recall_macro": MetricDefinition(name="recall_macro", direction="max",
                                         formula_id="multiclass_recall_macro", formula_version=1,
                                         averaging="macro", **base),
        "f1_macro": MetricDefinition(name="f1_macro", direction="max",
                                     formula_id="multiclass_f1_macro", formula_version=1,
                                     averaging="macro", **base),
        "f1_weighted": MetricDefinition(name="f1_weighted", direction="max",
                                        formula_id="multiclass_f1_weighted", formula_version=1,
                                        averaging="weighted", **base),
    }


def _multilabel_definitions(num_labels: int) -> Mapping[str, MetricDefinition]:
    base = dict(
        zero_division="zero",
        exact=True,
        evaluation_scope="full",
        sample_weight_policy="supported",
        implementation="builtin_verified",
        parameters={"num_labels": num_labels},
    )
    return {
        "loss": MetricDefinition(
            name="loss", direction="min", formula_id="weighted_mean_loss", formula_version=1,
            averaging="none", sample_weight_policy="supported", zero_division="not_applicable",
            exact=True, evaluation_scope="full", parameters={}, implementation="builtin_verified",
        ),
        "precision_macro": MetricDefinition(name="precision_macro", direction="max",
                                            formula_id="multilabel_precision_macro", formula_version=1,
                                            averaging="macro", **base),
        "recall_macro": MetricDefinition(name="recall_macro", direction="max",
                                         formula_id="multilabel_recall_macro", formula_version=1,
                                         averaging="macro", **base),
        "f1_macro": MetricDefinition(name="f1_macro", direction="max",
                                     formula_id="multilabel_f1_macro", formula_version=1,
                                     averaging="macro", **base),
        "f1_weighted": MetricDefinition(name="f1_weighted", direction="max",
                                        formula_id="multilabel_f1_weighted", formula_version=1,
                                        averaging="weighted", **base),
        "precision_micro": MetricDefinition(name="precision_micro", direction="max",
                                            formula_id="multilabel_precision_micro", formula_version=1,
                                            averaging="micro", **base),
        "recall_micro": MetricDefinition(name="recall_micro", direction="max",
                                         formula_id="multilabel_recall_micro", formula_version=1,
                                         averaging="micro", **base),
        "f1_micro": MetricDefinition(name="f1_micro", direction="max",
                                     formula_id="multilabel_f1_micro", formula_version=1,
                                     averaging="micro", **base),
        "subset_accuracy": MetricDefinition(name="subset_accuracy", direction="max",
                                            formula_id="multilabel_subset_accuracy", formula_version=1,
                                            averaging="micro", **base),
        "hamming_loss": MetricDefinition(name="hamming_loss", direction="min",
                                         formula_id="multilabel_hamming_loss", formula_version=1,
                                         averaging="micro", **base),
    }


def _regression_definitions(num_targets: int) -> Mapping[str, MetricDefinition]:
    base = dict(
        zero_division="not_applicable",
        exact=True,
        evaluation_scope="full",
        sample_weight_policy="supported",
        implementation="builtin_verified",
        parameters={"num_targets": num_targets},
    )
    return {
        "loss": MetricDefinition(
            name="loss", direction="min", formula_id="weighted_mean_loss", formula_version=1,
            averaging="none", sample_weight_policy="supported", zero_division="not_applicable",
            exact=True, evaluation_scope="full", parameters={}, implementation="builtin_verified",
        ),
        "mae": MetricDefinition(name="mae", direction="min", formula_id="regression_mae",
                                formula_version=1, averaging="uniform_average", **base),
        "mse": MetricDefinition(name="mse", direction="min", formula_id="regression_mse",
                                formula_version=1, averaging="uniform_average", **base),
        "r2": MetricDefinition(name="r2", direction="max", formula_id="regression_r2_uniform",
                               formula_version=1, averaging="uniform_average", **base),
        "r2_variance_weighted": MetricDefinition(
            name="r2_variance_weighted", direction="max", formula_id="regression_r2_variance_weighted",
            formula_version=1, averaging="variance_weighted", **base),
    }
