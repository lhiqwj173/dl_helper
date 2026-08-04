"""版本化流式指标：固定大小状态、分布式 sum/moment 归约、sklearn 金标语义。"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from .contracts import MetricDefinition, PredictedBatch

# 常量 target 的 R2 舍入界系数（design：8 * float64 epsilon）
_ROUNDING_EPS = 8.0 * np.finfo(np.float64).eps
# 最小 R2 分子平方和
FLOAT64_EPS = np.finfo(np.float64).eps


class MetricStateError(Exception):
    """指标状态非法。"""


# --------------------------------------------------------------------------
# 校验辅助
# --------------------------------------------------------------------------

def _as_float64_weight(weight: np.ndarray | None, sample_count: int) -> np.ndarray:
    if weight is None:
        return np.ones(sample_count, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 1:
        raise MetricStateError(f"sample_weight 必须是一维，得到 shape {w.shape}")
    if w.shape[0] != sample_count:
        raise MetricStateError(f"sample_weight 长度 {w.shape[0]} != sample_count {sample_count}")
    if not np.all(np.isfinite(w)):
        raise MetricStateError("sample_weight 含非有限值")
    if np.any(w < 0):
        raise MetricStateError("sample_weight 含负值")
    if w.sum() <= 0:
        raise MetricStateError("sample_weight 权重和必须为正")
    return w


def _class_index_map(classes: np.ndarray) -> dict[Any, int]:
    mapping: dict[Any, int] = {}
    for i, c in enumerate(classes):
        key: Any = c.item() if isinstance(c, (np.generic,)) else c
        if key in mapping:
            raise MetricStateError(f"类别重复: {key!r}")
        mapping[key] = i
    return mapping


def _to_class_indices(values: np.ndarray, mapping: dict[Any, int], name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise MetricStateError(f"{name} 必须是一维，得到 shape {values.shape}")
    if values.dtype.kind not in "iu":
        raise MetricStateError(f"{name} 必须是整数类别索引，得到 dtype {values.dtype}")
    out = np.empty(values.shape[0], dtype=np.int64)
    for i, v in enumerate(values):
        key = v.item()
        if key not in mapping:
            raise MetricStateError(f"{name} 含未知类别: {key!r}")
        out[i] = mapping[key]
    return out


def _require_1d(values: np.ndarray, name: str, expected_len: int | None = None) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise MetricStateError(f"{name} 必须是一维，得到 shape {values.shape}")
    if expected_len is not None and values.shape[0] != expected_len:
        raise MetricStateError(f"{name} 长度 {values.shape[0]} != {expected_len}")
    return values


# --------------------------------------------------------------------------
# 子状态基类
# --------------------------------------------------------------------------

class _MetricSubState(ABC):
    """固定大小流式状态：update 消费 PredictedBatch，compute 产出标量。"""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def reduction_state(self) -> Mapping[str, tuple[torch.Tensor, Literal["sum", "min", "max", "merge_weighted_moments"]]]: ...

    @abstractmethod
    def load_reduced_state(self, state: Mapping[str, torch.Tensor]) -> None: ...

    @abstractmethod
    def update(self, predicted: PredictedBatch) -> None: ...

    @abstractmethod
    def compute(self) -> Mapping[str, float]: ...

    def extended_compute(self) -> Mapping[str, Any]:
        return dict(self.compute())


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

class LossState:
    """loss 加权和 / 权重和状态（独立于 PredictedBatch 消费）。

    输出键为 f"{stage}/loss"。
    """

    formula_id = "weighted_mean_loss"
    formula_version = 1

    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._numerator = 0.0
        self._denominator = 0.0
        self._sample_count = 0

    def reset(self) -> None:
        self._numerator = 0.0
        self._denominator = 0.0
        self._sample_count = 0

    def update(self, numerator: float, denominator: float) -> None:
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            raise MetricStateError(f"loss numerator/denominator 必须有限: {numerator}/{denominator}")
        if denominator <= 0:
            raise MetricStateError(f"loss denominator 必须为正: {denominator}")
        self._numerator += float(numerator)
        self._denominator += float(denominator)
        self._sample_count += 1

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "stage": self._stage,
            "numerator": self._numerator,
            "denominator": self._denominator,
            "sample_count": self._sample_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._stage = str(state["stage"])
        self._numerator = float(state["numerator"])
        self._denominator = float(state["denominator"])
        self._sample_count = int(state["sample_count"])

    def reduction_state(self) -> Mapping[str, tuple[torch.Tensor, Literal["sum"]]]:
        return {
            "numerator": (torch.tensor(self._numerator, dtype=torch.float64), "sum"),
            "denominator": (torch.tensor(self._denominator, dtype=torch.float64), "sum"),
        }

    def load_reduced_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self._numerator = float(state["numerator"])
        self._denominator = float(state["denominator"])
        # 空 loss（denominator=0）合法：该 stage 未记录 loss
        if self._denominator < 0:
            raise MetricStateError("loss 归约后 denominator 不得为负")
        self._sample_count = 0

    def compute(self) -> Mapping[str, float]:
        if self._denominator <= 0:
            # 该 stage 未记录 loss（如 val/test），不产出 loss 指标
            return {}
        return {f"{self._stage}/loss": self._numerator / self._denominator}


# --------------------------------------------------------------------------
# 多分类：float64 加权混淆矩阵
# --------------------------------------------------------------------------

class MulticlassState(_MetricSubState):
    """加权混淆矩阵 + per-class 标量与向量输出。"""

    formula_ids = {
        "accuracy": "multiclass_accuracy",
        "balanced_accuracy": "multiclass_balanced_accuracy",
        "precision_macro": "multiclass_precision_macro",
        "recall_macro": "multiclass_recall_macro",
        "f1_macro": "multiclass_f1_macro",
        "f1_weighted": "multiclass_f1_weighted",
    }
    formula_version = 1

    def __init__(self, name_prefix: str, classes: Sequence[Any]) -> None:
        classes_arr = np.asarray(classes)
        if classes_arr.ndim != 1 or classes_arr.shape[0] < 2:
            raise MetricStateError("多分类必须声明至少两个唯一类别")
        if classes_arr.shape[0] != len(set(classes_arr.tolist())):
            raise MetricStateError("多分类类别必须唯一")
        self._prefix = name_prefix
        self._classes = classes_arr
        self._mapping = _class_index_map(classes_arr)
        self._c = classes_arr.shape[0]
        self._confusion = np.zeros((self._c, self._c), dtype=np.float64)
        self._counts = np.zeros((self._c, self._c), dtype=np.int64)
        self._sample_count = 0
        self._weight_sum = 0.0

    def reset(self) -> None:
        self._confusion = np.zeros((self._c, self._c), dtype=np.float64)
        self._counts = np.zeros((self._c, self._c), dtype=np.int64)
        self._sample_count = 0
        self._weight_sum = 0.0

    def update(self, predicted: PredictedBatch) -> None:
        n = predicted.sample_count
        if n <= 0:
            raise MetricStateError("空 split 不得更新指标")
        targets = _to_class_indices(predicted.targets, self._mapping, "targets")
        preds = _to_class_indices(predicted.predictions, self._mapping, "predictions")
        if targets.shape[0] != n or preds.shape[0] != n:
            raise MetricStateError("targets/predictions 样本维不一致")
        weights = _as_float64_weight(predicted.sample_weight, n)
        np.add.at(self._confusion, (targets, preds), weights)
        np.add.at(self._counts, (targets, preds), 1)
        self._sample_count += n
        self._weight_sum += float(weights.sum())

    def _row_sums(self) -> np.ndarray:
        return self._confusion.sum(axis=1)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "name": self._prefix,
            "classes": self._classes.tolist(),
            "confusion": self._confusion.copy(),
            "counts": self._counts.copy(),
            "sample_count": self._sample_count,
            "weight_sum": self._weight_sum,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        classes = np.asarray(state["classes"])
        if classes.shape != self._classes.shape or not np.array_equal(classes, self._classes):
            raise MetricStateError("恢复状态类别顺序漂移")
        self._confusion = np.asarray(state["confusion"], dtype=np.float64)
        self._counts = np.asarray(state["counts"], dtype=np.int64)
        if self._confusion.shape != (self._c, self._c) or self._counts.shape != (self._c, self._c):
            raise MetricStateError("恢复状态 shape 漂移")
        self._sample_count = int(state["sample_count"])
        self._weight_sum = float(state["weight_sum"])

    def reduction_state(self) -> Mapping[str, tuple[torch.Tensor, Literal["sum"]]]:
        return {
            "confusion": (torch.from_numpy(self._confusion), "sum"),
            "counts": (torch.from_numpy(self._counts), "sum"),
            "sample_count": (torch.tensor(self._sample_count, dtype=torch.int64), "sum"),
            "weight_sum": (torch.tensor(self._weight_sum, dtype=torch.float64), "sum"),
        }

    def load_reduced_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self._confusion = state["confusion"].cpu().numpy()
        self._counts = state["counts"].cpu().numpy()
        self._sample_count = int(state["sample_count"].item())
        self._weight_sum = float(state["weight_sum"].item())
        if self._confusion.shape != (self._c, self._c):
            raise MetricStateError("归约 confusion shape 漂移")

    def compute(self) -> Mapping[str, float]:
        if self._sample_count <= 0:
            raise MetricStateError(f"split 无样本：{self._prefix}")
        total = self._confusion.sum()
        if total <= 0:
            raise MetricStateError(f"split 权重和必须为正：{self._prefix}")
        accuracy = float(np.trace(self._confusion) / total)
        row_sums = self._confusion.sum(axis=1)
        col_sums = self._confusion.sum(axis=0)
        diag = np.diag(self._confusion)
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(col_sums > 0, diag / np.where(col_sums > 0, col_sums, 1), 0.0)
            recall = np.where(row_sums > 0, diag / np.where(row_sums > 0, row_sums, 1), 0.0)
            denom = precision + recall
            f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
        macro = lambda a: float(a.mean())  # noqa: E731
        # weighted: 按真实 weighted support 加权
        support = row_sums
        support_denom = total
        weighted = lambda a: float(np.sum(support * a) / support_denom) if support_denom > 0 else 0.0  # noqa: E731
        present = row_sums > 0
        if present.any():
            balanced = float(recall[present].mean())
        else:
            balanced = 0.0
        return {
            f"{self._prefix}/accuracy": accuracy,
            f"{self._prefix}/balanced_accuracy": balanced,
            f"{self._prefix}/precision_macro": macro(precision),
            f"{self._prefix}/recall_macro": macro(recall),
            f"{self._prefix}/f1_macro": macro(f1),
            f"{self._prefix}/f1_weighted": weighted(f1),
        }

    def extended_compute(self) -> Mapping[str, Any]:
        scalars = self.compute()
        row_sums = self._confusion.sum(axis=1)
        col_sums = self._confusion.sum(axis=0)
        diag = np.diag(self._confusion)
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(col_sums > 0, diag / np.where(col_sums > 0, col_sums, 1), 0.0)
            recall = np.where(row_sums > 0, diag / np.where(row_sums > 0, row_sums, 1), 0.0)
            denom = precision + recall
            f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
        out = dict(scalars)
        out[f"{self._prefix}/per_class"] = {
            "classes": self._classes.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "weighted_support": row_sums.tolist(),
        }
        out[f"{self._prefix}/confusion_weighted"] = self._confusion.tolist()
        out[f"{self._prefix}/confusion_counts"] = self._counts.tolist()
        out[f"{self._prefix}/sample_count"] = int(self._sample_count)
        out[f"{self._prefix}/weight_sum"] = float(self._weight_sum)
        return out


# --------------------------------------------------------------------------
# 多标签：per-label TP/FP/FN/TN
# --------------------------------------------------------------------------

class MultilabelState(_MetricSubState):
    """per-label 加权统计 + subset accuracy + hamming loss。"""

    formula_ids = {
        "precision_macro": "multilabel_precision_macro",
        "recall_macro": "multilabel_recall_macro",
        "f1_macro": "multilabel_f1_macro",
        "f1_weighted": "multilabel_f1_weighted",
        "precision_micro": "multilabel_precision_micro",
        "recall_micro": "multilabel_recall_micro",
        "f1_micro": "multilabel_f1_micro",
        "subset_accuracy": "multilabel_subset_accuracy",
        "hamming_loss": "multilabel_hamming_loss",
    }
    formula_version = 1

    def __init__(self, name_prefix: str, num_labels: int, threshold: float | Sequence[float] = 0.5,
                 label_names: Sequence[str] | None = None) -> None:
        if num_labels < 1:
            raise MetricStateError("多标签必须至少一个 label")
        if isinstance(threshold, (int, float)):
            thr = np.full(num_labels, float(threshold), dtype=np.float64)
        else:
            thr = np.asarray(threshold, dtype=np.float64)
            if thr.shape != (num_labels,):
                raise MetricStateError("threshold 向量长度必须等于 label 数")
        if not np.all(np.isfinite(thr)) or np.any(thr < 0) or np.any(thr > 1):
            raise MetricStateError("threshold 必须在 [0,1]")
        self._prefix = name_prefix
        self._num_labels = num_labels
        self._threshold = thr
        self._label_names = list(label_names) if label_names else [str(i) for i in range(num_labels)]
        if len(self._label_names) != num_labels:
            raise MetricStateError("label_names 长度必须等于 label 数")
        self.reset()

    def reset(self) -> None:
        self._tp = np.zeros(self._num_labels, dtype=np.float64)
        self._fp = np.zeros(self._num_labels, dtype=np.float64)
        self._fn = np.zeros(self._num_labels, dtype=np.float64)
        self._tn = np.zeros(self._num_labels, dtype=np.float64)
        self._raw_tp = np.zeros(self._num_labels, dtype=np.int64)
        self._raw_fp = np.zeros(self._num_labels, dtype=np.int64)
        self._raw_fn = np.zeros(self._num_labels, dtype=np.int64)
        self._raw_tn = np.zeros(self._num_labels, dtype=np.int64)
        self._subset_exact = 0.0
        self._hamming_sum = 0.0
        self._sample_count = 0
        self._weight_sum = 0.0

    def update(self, predicted: PredictedBatch) -> None:
        n = predicted.sample_count
        if n <= 0:
            raise MetricStateError("空 split 不得更新指标")
        targets = np.asarray(predicted.targets)
        if targets.ndim != 2 or targets.shape[1] != self._num_labels:
            raise MetricStateError(f"多标签 targets 必须为 [N,{self._num_labels}]，得到 {targets.shape}")
        if targets.shape[0] != n:
            raise MetricStateError("targets 样本维不一致")
        if targets.dtype == np.bool_:
            tgt = targets.astype(np.float64)
        elif np.issubdtype(targets.dtype, np.number):
            if np.any((targets != 0) & (targets != 1)):
                raise MetricStateError("多标签 targets 只能是 bool 或 0/1")
            tgt = targets.astype(np.float64)
        else:
            raise MetricStateError(f"多标签 targets dtype 非法: {targets.dtype}")
        scores = predicted.scores
        if scores is None:
            raise MetricStateError("多标签必须提供 scores")
        scores = np.asarray(scores)
        if scores.shape != (n, self._num_labels):
            raise MetricStateError(f"多标签 scores 必须为 [N,{self._num_labels}]，得到 {scores.shape}")
        if not np.all(np.isfinite(scores)):
            raise MetricStateError("scores 含非有限值")
        weights = _as_float64_weight(predicted.sample_weight, n)
        pred = (scores >= self._threshold[None, :]).astype(np.float64)
        # 加权
        w = weights[:, None]
        tp = (pred * tgt) * w
        fp = (pred * (1 - tgt)) * w
        fn = ((1 - pred) * tgt) * w
        tn = ((1 - pred) * (1 - tgt)) * w
        self._tp += tp.sum(axis=0)
        self._fp += fp.sum(axis=0)
        self._fn += fn.sum(axis=0)
        self._tn += tn.sum(axis=0)
        self._raw_tp += ((pred * tgt) > 0).sum(axis=0)
        self._raw_fp += ((pred * (1 - tgt)) > 0).sum(axis=0)
        self._raw_fn += (((1 - pred) * tgt) > 0).sum(axis=0)
        self._raw_tn += (((1 - pred) * (1 - tgt)) > 0).sum(axis=0)
        exact = np.all((pred > 0) == (tgt > 0), axis=1)
        self._subset_exact += float(np.sum(weights * exact))
        mismatches = np.sum((pred > 0) != (tgt > 0), axis=1)
        self._hamming_sum += float(np.sum(weights * mismatches))
        self._sample_count += n
        self._weight_sum += float(weights.sum())

    def _per_label(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.errstate(invalid="ignore", divide="ignore"):
            pred_denom = self._tp + self._fp
            precision = np.where(pred_denom > 0, self._tp / np.where(pred_denom > 0, pred_denom, 1), 0.0)
            rec_denom = self._tp + self._fn
            recall = np.where(rec_denom > 0, self._tp / np.where(rec_denom > 0, rec_denom, 1), 0.0)
            f1_denom = precision + recall
            f1 = np.where(f1_denom > 0, 2 * precision * recall / np.where(f1_denom > 0, f1_denom, 1), 0.0)
        return precision, recall, f1

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "name": self._prefix,
            "num_labels": self._num_labels,
            "threshold": self._threshold.tolist(),
            "label_names": self._label_names,
            "tp": self._tp, "fp": self._fp, "fn": self._fn, "tn": self._tn,
            "raw_tp": self._raw_tp, "raw_fp": self._raw_fp, "raw_fn": self._raw_fn, "raw_tn": self._raw_tn,
            "subset_exact": self._subset_exact,
            "hamming_sum": self._hamming_sum,
            "sample_count": self._sample_count,
            "weight_sum": self._weight_sum,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["num_labels"]) != self._num_labels:
            raise MetricStateError("恢复状态 label 数漂移")
        for key in ("tp", "fp", "fn", "tn"):
            setattr(self, f"_{key}", np.asarray(state[key], dtype=np.float64))
        for key in ("raw_tp", "raw_fp", "raw_fn", "raw_tn"):
            setattr(self, f"_{key}", np.asarray(state[key], dtype=np.int64))
        self._subset_exact = float(state["subset_exact"])
        self._hamming_sum = float(state["hamming_sum"])
        self._sample_count = int(state["sample_count"])
        self._weight_sum = float(state["weight_sum"])

    def reduction_state(self) -> Mapping[str, tuple[torch.Tensor, Literal["sum"]]]:
        tensors: dict[str, tuple[torch.Tensor, Literal["sum"]]] = {}
        for key in ("tp", "fp", "fn", "tn"):
            tensors[key] = (torch.from_numpy(getattr(self, f"_{key}")), "sum")
        for key in ("raw_tp", "raw_fp", "raw_fn", "raw_tn"):
            tensors[key] = (torch.from_numpy(getattr(self, f"_{key}")), "sum")
        tensors["subset_exact"] = (torch.tensor(self._subset_exact, dtype=torch.float64), "sum")
        tensors["hamming_sum"] = (torch.tensor(self._hamming_sum, dtype=torch.float64), "sum")
        tensors["sample_count"] = (torch.tensor(self._sample_count, dtype=torch.int64), "sum")
        tensors["weight_sum"] = (torch.tensor(self._weight_sum, dtype=torch.float64), "sum")
        return tensors

    def load_reduced_state(self, state: Mapping[str, torch.Tensor]) -> None:
        for key in ("tp", "fp", "fn", "tn"):
            setattr(self, f"_{key}", state[key].cpu().numpy())
        for key in ("raw_tp", "raw_fp", "raw_fn", "raw_tn"):
            setattr(self, f"_{key}", state[key].cpu().numpy())
        self._subset_exact = float(state["subset_exact"].item())
        self._hamming_sum = float(state["hamming_sum"].item())
        self._sample_count = int(state["sample_count"].item())
        self._weight_sum = float(state["weight_sum"].item())

    def compute(self) -> Mapping[str, float]:
        if self._sample_count <= 0:
            raise MetricStateError(f"split 无样本：{self._prefix}")
        precision, recall, f1 = self._per_label()
        total_tp = self._tp.sum()
        total_pred = total_tp + self._fp.sum()
        total_pos = total_tp + self._fn.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            micro_precision = float(total_tp / total_pred) if total_pred > 0 else 0.0
            micro_recall = float(total_tp / total_pos) if total_pos > 0 else 0.0
            micro_denom = micro_precision + micro_recall
            micro_f1 = float(2 * micro_precision * micro_recall / micro_denom) if micro_denom > 0 else 0.0
        support = self._tp + self._fn  # 正类 weighted support
        support_denom = support.sum()
        weighted_f1 = float(np.sum(support * f1) / support_denom) if support_denom > 0 else 0.0
        denom_w = self._weight_sum
        subset_acc = self._subset_exact / denom_w if denom_w > 0 else 0.0
        hamming = self._hamming_sum / (self._num_labels * denom_w) if denom_w > 0 else 0.0
        return {
            f"{self._prefix}/precision_macro": float(precision.mean()),
            f"{self._prefix}/recall_macro": float(recall.mean()),
            f"{self._prefix}/f1_macro": float(f1.mean()),
            f"{self._prefix}/f1_weighted": weighted_f1,
            f"{self._prefix}/precision_micro": micro_precision,
            f"{self._prefix}/recall_micro": micro_recall,
            f"{self._prefix}/f1_micro": micro_f1,
            f"{self._prefix}/subset_accuracy": subset_acc,
            f"{self._prefix}/hamming_loss": hamming,
        }

    def extended_compute(self) -> Mapping[str, Any]:
        scalars = self.compute()
        precision, recall, f1 = self._per_label()
        support = self._tp + self._fn
        out = dict(scalars)
        out[f"{self._prefix}/per_label"] = {
            "labels": self._label_names,
            "thresholds": self._threshold.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "weighted_support": support.tolist(),
        }
        out[f"{self._prefix}/sample_count"] = int(self._sample_count)
        out[f"{self._prefix}/weight_sum"] = float(self._weight_sum)
        return out


# --------------------------------------------------------------------------
# 回归：weighted Welford 矩
# --------------------------------------------------------------------------

def _chan_merge(
    n_a: np.ndarray, mean_a: np.ndarray, m2_a: np.ndarray,
    n_b: np.ndarray, mean_b: np.ndarray, m2_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = n_a + n_b
    mean = np.zeros_like(n_a)
    m2 = np.zeros_like(n_a)
    # 全零情况
    both_zero = (n_a == 0) & (n_b == 0)
    a_only = (n_a > 0) & (n_b == 0)
    b_only = (n_a == 0) & (n_b > 0)
    both = (n_a > 0) & (n_b > 0)
    mean[both_zero] = 0.0
    mean[a_only] = mean_a[a_only]
    mean[b_only] = mean_b[b_only]
    m2[both_zero] = 0.0
    m2[a_only] = m2_a[a_only]
    m2[b_only] = m2_b[b_only]
    if both.any():
        na = n_a[both]
        nb = n_b[both]
        delta = mean_b[both] - mean_a[both]
        new_mean = mean_a[both] + delta * (nb / (na + nb))
        new_m2 = m2_a[both] + m2_b[both] + delta * delta * (na * nb / (na + nb))
        mean[both] = new_mean
        m2[both] = new_m2
    return n, mean, m2


class RegressionState(_MetricSubState):
    """per-target MAE/MSE/R2，weighted Welford 合并。"""

    formula_ids = {
        "mae": "regression_mae",
        "mse": "regression_mse",
        "r2_uniform": "regression_r2_uniform",
        "r2_variance_weighted": "regression_r2_variance_weighted",
    }
    formula_version = 1

    def __init__(self, name_prefix: str, num_targets: int, target_names: Sequence[str] | None = None) -> None:
        if num_targets < 1:
            raise MetricStateError("回归必须至少一个 target")
        self._prefix = name_prefix
        self._num_targets = num_targets
        self._target_names = list(target_names) if target_names else [str(i) for i in range(num_targets)]
        if len(self._target_names) != num_targets:
            raise MetricStateError("target_names 长度必须等于 target 数")
        self.reset()

    def reset(self) -> None:
        d = self._num_targets
        self._weight_sum = np.zeros(d, dtype=np.float64)
        self._sum_abs = np.zeros(d, dtype=np.float64)
        self._sum_sq = np.zeros(d, dtype=np.float64)
        self._mean = np.zeros(d, dtype=np.float64)
        self._m2 = np.zeros(d, dtype=np.float64)
        self._sample_count = 0
        self._total_weight = 0.0

    def update(self, predicted: PredictedBatch) -> None:
        n = predicted.sample_count
        if n <= 0:
            raise MetricStateError("空 split 不得更新指标")
        targets = np.asarray(predicted.targets, dtype=np.float64)
        preds = np.asarray(predicted.predictions, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets[:, None]
        if preds.ndim == 1:
            preds = preds[:, None]
        if targets.ndim != 2 or targets.shape[1] != self._num_targets:
            raise MetricStateError(f"回归 targets 必须为 [N,{self._num_targets}]，得到 {targets.shape}")
        if preds.shape != targets.shape:
            raise MetricStateError(f"回归 predictions shape {preds.shape} != targets shape {targets.shape}")
        if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(preds)):
            raise MetricStateError("回归 targets/predictions 含非有限值")
        weights = _as_float64_weight(predicted.sample_weight, n)
        w = weights[:, None]
        abs_err = np.abs(preds - targets)
        sq_err = (preds - targets) ** 2
        self._sum_abs += (w * abs_err).sum(axis=0)
        self._sum_sq += (w * sq_err).sum(axis=0)
        # weighted Welford per target
        self._total_weight += float(weights.sum())
        self._sample_count += n
        for d in range(self._num_targets):
            for i in range(n):
                x = targets[i, d]
                wi = weights[i]
                if wi <= 0:
                    continue
                self._weight_sum[d], self._mean[d], self._m2[d] = _welford_update(
                    self._weight_sum[d], self._mean[d], self._m2[d], x, wi
                )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "name": self._prefix,
            "num_targets": self._num_targets,
            "target_names": self._target_names,
            "weight_sum": self._weight_sum,
            "sum_abs": self._sum_abs,
            "sum_sq": self._sum_sq,
            "mean": self._mean,
            "m2": self._m2,
            "sample_count": self._sample_count,
            "total_weight": self._total_weight,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["num_targets"]) != self._num_targets:
            raise MetricStateError("恢复状态 target 数漂移")
        for key in ("weight_sum", "sum_abs", "sum_sq", "mean", "m2"):
            setattr(self, f"_{key}", np.asarray(state[key], dtype=np.float64))
        self._sample_count = int(state["sample_count"])
        self._total_weight = float(state["total_weight"])
        self._validate_m2()

    def _validate_m2(self) -> None:
        scale = np.maximum.reduce([self._weight_sum, np.abs(self._mean), np.ones_like(self._mean)])
        bound = _ROUNDING_EPS * scale
        neg = self._m2 < 0
        if np.any(neg):
            if np.all(np.abs(self._m2[neg]) <= bound[neg]):
                self._m2[neg] = 0.0
            else:
                raise MetricStateError("回归 M2 出现负值（状态损坏）")

    def reduction_state(self) -> Mapping[str, tuple[torch.Tensor, str]]:
        moments = np.stack([self._weight_sum, self._mean, self._m2], axis=1)
        return {
            "sum_abs": (torch.from_numpy(self._sum_abs), "sum"),
            "sum_sq": (torch.from_numpy(self._sum_sq), "sum"),
            "sample_count": (torch.tensor(self._sample_count, dtype=torch.int64), "sum"),
            "total_weight": (torch.tensor(self._total_weight, dtype=torch.float64), "sum"),
            "moments": (torch.from_numpy(moments), "merge_weighted_moments"),
        }

    def load_reduced_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self._sum_abs = state["sum_abs"].cpu().numpy()
        self._sum_sq = state["sum_sq"].cpu().numpy()
        self._sample_count = int(state["sample_count"].item())
        self._total_weight = float(state["total_weight"].item())
        moments = state["moments"].cpu().numpy()
        if moments.ndim != 2 or moments.shape[1] != 3:
            raise MetricStateError("moments 状态必须为 [D,3] (weight,mean,M2)")
        self._weight_sum = moments[:, 0]
        self._mean = moments[:, 1]
        self._m2 = moments[:, 2]
        self._validate_m2()

    def compute(self) -> Mapping[str, float]:
        if self._sample_count <= 0:
            raise MetricStateError(f"split 无样本：{self._prefix}")
        ws = self._weight_sum
        if np.any(ws <= 0):
            raise MetricStateError("回归 target 权重和必须为正")
        mae = self._sum_abs / ws
        mse = self._sum_sq / ws
        r2 = _r2_force_finite(self._sum_sq, self._m2)
        total_weight = float(ws.sum())
        uniform_mae = float(np.sum(self._sum_abs) / total_weight)
        uniform_mse = float(np.sum(self._sum_sq) / total_weight)
        uniform_r2 = float(r2.mean())
        denom = self._m2.sum()
        if denom > 0:
            vw_r2 = float(np.sum(self._m2 * r2) / denom)
        else:
            vw_r2 = uniform_r2
        out: dict[str, float] = {
            f"{self._prefix}/mae": uniform_mae,
            f"{self._prefix}/mse": uniform_mse,
            f"{self._prefix}/r2": uniform_r2,
            f"{self._prefix}/r2_variance_weighted": vw_r2,
        }
        for d, name in enumerate(self._target_names):
            out[f"{self._prefix}/mae_target_{name}"] = float(mae[d])
            out[f"{self._prefix}/mse_target_{name}"] = float(mse[d])
            out[f"{self._prefix}/r2_target_{name}"] = float(r2[d])
        return out

    def extended_compute(self) -> Mapping[str, Any]:
        scalars = self.compute()
        ws = self._weight_sum
        out = dict(scalars)
        out[f"{self._prefix}/per_target"] = {
            "targets": self._target_names,
            "mae": (self._sum_abs / ws).tolist(),
            "mse": (self._sum_sq / ws).tolist(),
            "r2": _r2_force_finite(self._sum_sq, self._m2).tolist(),
            "weight_sum": ws.tolist(),
            "target_variance": (self._m2 / np.where(ws > 0, ws, 1)).tolist(),
        }
        out[f"{self._prefix}/sample_count"] = int(self._sample_count)
        out[f"{self._prefix}/weight_sum"] = float(self._total_weight)
        return out


def _welford_update(n: float, mean: float, m2: float, x: float, w: float) -> tuple[float, float, float]:
    if n == 0:
        return w, x, 0.0
    new_n = n + w
    delta = x - mean
    new_mean = mean + delta * (w / new_n)
    new_m2 = m2 + w * delta * (x - new_mean)
    return new_n, new_mean, new_m2


def _r2_force_finite(sse: np.ndarray, m2_y: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(m2_y > 0, 1.0 - sse / np.where(m2_y > 0, m2_y, 1), 0.0)
    # M2_y==0：force_finite 语义
    constant = m2_y == 0
    r2[constant & (sse == 0)] = 1.0
    r2[constant & (sse != 0)] = 0.0
    if not np.all(np.isfinite(r2)):
        raise MetricStateError("R2 出现非有限值")
    return r2


# --------------------------------------------------------------------------
# Stage 聚合状态
# --------------------------------------------------------------------------

class StageMetricState:
    """一个 stage 的完整指标状态：loss + 任务指标子状态。"""

    def __init__(
        self,
        definitions: Mapping[str, MetricDefinition],
        loss_state: LossState,
        task_states: Sequence[_MetricSubState],
    ) -> None:
        self._definitions = dict(definitions)
        self._loss = loss_state
        self._states = list(task_states)

    @property
    def definitions(self) -> Mapping[str, MetricDefinition]:
        return self._definitions

    def update_loss(self, numerator: float, denominator: float) -> None:
        self._loss.update(numerator, denominator)

    def update_predicted(self, predicted: PredictedBatch) -> None:
        for st in self._states:
            st.update(predicted)

    def reset(self) -> None:
        self._loss.reset()
        for st in self._states:
            st.reset()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "loss": self._loss.state_dict(),
            "task_states": [st.state_dict() for st in self._states],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._loss.load_state_dict(state["loss"])
        task_states = state["task_states"]
        if len(task_states) != len(self._states):
            raise MetricStateError("恢复状态子状态数量漂移")
        for st, st_state in zip(self._states, task_states):
            st.load_state_dict(st_state)

    def reduction_state(self) -> Mapping[str, tuple[torch.Tensor, str]]:
        out: dict[str, tuple[torch.Tensor, str]] = {}
        for key, (t, op) in self._loss.reduction_state().items():
            out[f"loss/{key}"] = (t, op)
        for idx, st in enumerate(self._states):
            for key, (t, op) in st.reduction_state().items():
                out[f"state{idx}/{key}"] = (t, op)
        return out

    def load_reduced_state(self, state: Mapping[str, torch.Tensor]) -> None:
        loss_state = {}
        for key in ("numerator", "denominator"):
            loss_state[key] = state[f"loss/{key}"]
        self._loss.load_reduced_state(loss_state)
        for idx, st in enumerate(self._states):
            prefix = f"state{idx}/"
            sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            st.load_reduced_state(sub)

    def compute(self) -> Mapping[str, float]:
        out: dict[str, float] = {}
        out.update(self._loss.compute())
        for st in self._states:
            out.update(st.compute())
        return out

    def extended_compute(self) -> Mapping[str, Any]:
        out: dict[str, Any] = {}
        out.update(self._loss.compute())
        for st in self._states:
            out.update(st.extended_compute())
        return out


# --------------------------------------------------------------------------
# 归约合并
# --------------------------------------------------------------------------

def combine_reduction_states(
    per_rank: Sequence[Mapping[str, tuple[torch.Tensor, str]]],
) -> Mapping[str, torch.Tensor]:
    """按固定 rank 顺序合并各 rank 的 reduction state。

    校验键、shape、dtype 与操作完全一致；sum/min/max 求和，weighted moments 顺序 Chan 合并。
    """
    if not per_rank:
        raise MetricStateError("无 rank 状态可归约")
    reference = per_rank[0]
    keys = set(reference.keys())
    ops: dict[str, str] = {}
    for key, (t, op) in reference.items():
        ops[key] = op
    for rank_idx, rank_state in enumerate(per_rank[1:]):
        if set(rank_state.keys()) != keys:
            raise MetricStateError(f"rank {rank_idx + 1} 状态键不一致")
        for key, (t, op) in rank_state.items():
            ref_t, ref_op = reference[key]
            if op != ref_op:
                raise MetricStateError(f"rank {rank_idx + 1} 状态 op 不一致: {key}")
            if t.shape != ref_t.shape:
                raise MetricStateError(f"rank {rank_idx + 1} 状态 shape 不一致: {key}")
            if t.dtype != ref_t.dtype:
                raise MetricStateError(f"rank {rank_idx + 1} 状态 dtype 不一致: {key}")

    result: dict[str, torch.Tensor] = {}
    for key in keys:
        op = ops[key]
        tensors = [rank_state[key][0] for rank_state in per_rank]
        if op == "sum":
            result[key] = torch.stack(tensors, dim=0).sum(dim=0)
        elif op == "merge_weighted_moments":
            merged = tensors[0]
            for t in tensors[1:]:
                merged = _merge_moment_tensors(merged, t)
            result[key] = merged
        elif op in ("min", "max"):
            result[key] = torch.stack(tensors, dim=0).min(dim=0).values if op == "min" else torch.stack(tensors, dim=0).max(dim=0).values
        else:
            raise MetricStateError(f"未知归约操作: {op!r}")
    return result


def _merge_moment_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dim() != 2 or a.shape[1] != 3:
        raise MetricStateError("weighted moments 状态必须为 [D,3]")
    if a.shape != b.shape:
        raise MetricStateError("weighted moments 状态 shape 不一致")
    n_a, mean_a, m2_a = a[:, 0], a[:, 1], a[:, 2]
    n_b, mean_b, m2_b = b[:, 0], b[:, 1], b[:, 2]
    n, mean, m2 = _chan_merge(
        n_a.double(), mean_a.double(), m2_a.double(),
        n_b.double(), mean_b.double(), m2_b.double(),
    )
    out = torch.stack([torch.as_tensor(n), torch.as_tensor(mean), torch.as_tensor(m2)], dim=1)
    return out.to(a.dtype)
