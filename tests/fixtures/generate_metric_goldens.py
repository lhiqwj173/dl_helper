"""生成 metric_goldens_v1.json —— 使用 scikit-learn 1.6.1 计算期望值。

只允许在实现指标之前或公式升级时用本脚本重新生成 fixture；
运行测试时不得用待测实现现场生成期望。修改公式语义必须增加 formula_version
并生成 metric_goldens_v{n}.json，保持旧版本 fixture 不变。
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "metric_goldens_v1.json")


def _arr(a):
    return [float(x) for x in np.asarray(a, dtype=np.float64).ravel()]


def _multiclass_cases(rng):
    cases = []
    # 无权重 / 整数权重 / 非整数权重 / 缺失类别 / 极端不平衡
    y = rng.integers(0, 3, 200)
    yhat = rng.integers(0, 3, 200)
    w_int = rng.integers(1, 5, 200).astype(np.float64)
    w_float = (rng.random(200) + 0.01).astype(np.float64)
    y_missing = np.where(y == 2, 1, y)
    y_ext = np.concatenate([np.zeros(190, dtype=int), np.ones(10, dtype=int), np.full(200, 2, dtype=int)])
    yhat_ext = np.zeros(400, dtype=int)
    w_ext = (rng.random(400) + 0.01).astype(np.float64)
    classes = [0, 1, 2]
    for name, yy, yyhat, ww in (
        ("unweighted", y, yhat, None),
        ("int_weights", y, yhat, w_int),
        ("float_weights", y, yhat, w_float),
        ("missing_class", y_missing, yhat, w_float),
        ("extreme_imbalance", y_ext, yhat_ext, w_ext),
    ):
        p, r, f, sup = precision_recall_fscore_support(
            yy, yyhat, labels=classes, average=None, zero_division=0, sample_weight=ww)
        pm = precision_recall_fscore_support(yy, yyhat, labels=classes, average="macro",
                                             zero_division=0, sample_weight=ww)[2]
        fw = precision_recall_fscore_support(yy, yyhat, labels=classes, average="weighted",
                                             zero_division=0, sample_weight=ww)[2]
        cases.append({
            "name": f"multiclass/{name}",
            "formula_id": {
                "accuracy": "multiclass_accuracy",
                "balanced_accuracy": "multiclass_balanced_accuracy",
                "precision_macro": "multiclass_precision_macro",
                "recall_macro": "multiclass_recall_macro",
                "f1_macro": "multiclass_f1_macro",
                "f1_weighted": "multiclass_f1_weighted",
            },
            "formula_version": 1,
            "classes": classes,
            "targets": _arr(yy),
            "predictions": _arr(yyhat),
            "sample_weight": (None if ww is None else _arr(ww)),
            "expected": {
                "accuracy": float(accuracy_score(yy, yyhat, sample_weight=ww)),
                "balanced_accuracy": float(balanced_accuracy_score(yy, yyhat, sample_weight=ww)),
                "precision_macro": float(precision_recall_fscore_support(
                    yy, yyhat, labels=classes, average="macro", zero_division=0, sample_weight=ww)[0]),
                "recall_macro": float(precision_recall_fscore_support(
                    yy, yyhat, labels=classes, average="macro", zero_division=0, sample_weight=ww)[1]),
                "f1_macro": float(pm),
                "f1_weighted": float(fw),
                "per_class_precision": _arr(p),
                "per_class_recall": _arr(r),
                "per_class_f1": _arr(f),
            },
        })
    return cases


def _multilabel_cases(rng):
    cases = []
    Y = rng.integers(0, 2, (100, 3))
    S = rng.random((100, 3))
    w = (rng.random(100) + 0.01).astype(np.float64)
    Yp = (S >= 0.5).astype(int)
    for name, yy, ss, ww in (
        ("weighted", Y, S, w),
        ("unweighted", Y, S, None),
        ("all_negative", np.zeros((100, 3), dtype=int), S, w),
    ):
        yp = (ss >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(yy, yp, average=None, zero_division=0, sample_weight=ww)
        cases.append({
            "name": f"multilabel/{name}",
            "formula_id": {
                "precision_macro": "multilabel_precision_macro",
                "recall_macro": "multilabel_recall_macro",
                "f1_macro": "multilabel_f1_macro",
                "f1_weighted": "multilabel_f1_weighted",
                "precision_micro": "multilabel_precision_micro",
                "recall_micro": "multilabel_recall_micro",
                "f1_micro": "multilabel_f1_micro",
                "subset_accuracy": "multilabel_subset_accuracy",
                "hamming_loss": "multilabel_hamming_loss",
            },
            "formula_version": 1,
            "threshold": 0.5,
            "num_labels": 3,
            "targets": [[int(x) for x in row] for row in yy],
            "scores": _arr(ss),
            "sample_weight": (None if ww is None else _arr(ww)),
            "expected": {
                "precision_macro": float(precision_recall_fscore_support(
                    yy, yp, average="macro", zero_division=0, sample_weight=ww)[0]),
                "recall_macro": float(precision_recall_fscore_support(
                    yy, yp, average="macro", zero_division=0, sample_weight=ww)[1]),
                "f1_macro": float(precision_recall_fscore_support(
                    yy, yp, average="macro", zero_division=0, sample_weight=ww)[2]),
                "f1_weighted": float(precision_recall_fscore_support(
                    yy, yp, average="weighted", zero_division=0, sample_weight=ww)[2]),
                "precision_micro": float(precision_recall_fscore_support(
                    yy, yp, average="micro", zero_division=0, sample_weight=ww)[0]),
                "recall_micro": float(precision_recall_fscore_support(
                    yy, yp, average="micro", zero_division=0, sample_weight=ww)[1]),
                "f1_micro": float(precision_recall_fscore_support(
                    yy, yp, average="micro", zero_division=0, sample_weight=ww)[2]),
                "subset_accuracy": float(accuracy_score(yy, yp, sample_weight=ww)),
                "hamming_loss": float(hamming_loss(yy, yp, sample_weight=ww)),
                "per_label_precision": _arr(p),
                "per_label_recall": _arr(r),
                "per_label_f1": _arr(f),
            },
        })
    return cases


def _regression_cases(rng):
    cases = []
    t = rng.random((100, 3)) * 10
    p = t + rng.random((100, 3))
    w = (rng.random(100) + 0.01).astype(np.float64)
    t_const = np.full((100, 1), 5.0)
    p_perfect = t_const.copy()
    p_miss = t_const + 0.5
    for name, tt, pp, ww in (
        ("weighted", t, p, w),
        ("unweighted", t, p, None),
        ("constant_perfect", t_const, p_perfect, w),
        ("constant_miss", t_const, p_miss, w),
    ):
        cases.append({
            "name": f"regression/{name}",
            "formula_id": {
                "mae": "regression_mae",
                "mse": "regression_mse",
                "r2": "regression_r2_uniform",
                "r2_variance_weighted": "regression_r2_variance_weighted",
            },
            "formula_version": 1,
            "targets": _arr(tt),
            "predictions": _arr(pp),
            "num_targets": tt.shape[1],
            "sample_weight": (None if ww is None else _arr(ww)),
            "expected": {
                "mae": float(mean_absolute_error(tt, pp, sample_weight=ww)),
                "mse": float(mean_squared_error(tt, pp, sample_weight=ww)),
                "r2": float(r2_score(tt, pp, sample_weight=ww, multioutput="uniform_average", force_finite=True)),
                "r2_variance_weighted": float(r2_score(tt, pp, sample_weight=ww, multioutput="variance_weighted", force_finite=True)),
            },
        })
    return cases


def main() -> None:
    rng = np.random.default_rng(20260802)
    fixture = {
        "schema_version": 1,
        "scikit_learn_version": "1.6.1",
        "generated_by": "generate_metric_goldens.py",
        "cases": _multiclass_cases(rng) + _multilabel_cases(rng) + _regression_cases(rng),
    }
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(fixture['cases'])} cases to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
