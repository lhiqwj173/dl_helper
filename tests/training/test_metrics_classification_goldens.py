"""任务 2.2：多分类/多标签指标与 sklearn 1.6.1 金标对比。"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
)

from dl_helper.training.contracts import PredictedBatch
from dl_helper.training.metrics import MulticlassState, MultilabelState

GOLDEN_TOL = 1e-6


def _pb(targets, predictions, scores=None, weight=None):
    return PredictedBatch(
        targets=np.asarray(targets), predictions=np.asarray(predictions),
        sample_count=len(np.asarray(targets)), scores=scores,
        sample_weight=(None if weight is None else np.asarray(weight, dtype=np.float64)),
    )


def check_multiclass(y, yhat, w, classes):
    acc = accuracy_score(y, yhat, sample_weight=w)
    ba = balanced_accuracy_score(y, yhat, sample_weight=w)
    p, r, f, _ = precision_recall_fscore_support(
        y, yhat, labels=classes, average=None, zero_division=0, sample_weight=w)
    pm = precision_recall_fscore_support(y, yhat, labels=classes, average="macro",
                                         zero_division=0, sample_weight=w)[0]
    rm = precision_recall_fscore_support(y, yhat, labels=classes, average="macro",
                                         zero_division=0, sample_weight=w)[1]
    fm = precision_recall_fscore_support(y, yhat, labels=classes, average="macro",
                                         zero_division=0, sample_weight=w)[2]
    fw = precision_recall_fscore_support(y, yhat, labels=classes, average="weighted",
                                         zero_division=0, sample_weight=w)[2]
    st = MulticlassState("metric", classes)
    st.update(_pb(y, yhat, weight=w))
    out = st.compute()
    assert abs(out["metric/accuracy"] - acc) <= GOLDEN_TOL, (out["metric/accuracy"], acc)
    assert abs(out["metric/balanced_accuracy"] - ba) <= GOLDEN_TOL, (out["metric/balanced_accuracy"], ba)
    assert abs(out["metric/precision_macro"] - pm) <= GOLDEN_TOL, (out["metric/precision_macro"], pm)
    assert abs(out["metric/recall_macro"] - rm) <= GOLDEN_TOL, (out["metric/recall_macro"], rm)
    assert abs(out["metric/f1_macro"] - fm) <= GOLDEN_TOL, (out["metric/f1_macro"], fm)
    assert abs(out["metric/f1_weighted"] - fw) <= GOLDEN_TOL, (out["metric/f1_weighted"], fw)
    ext = st.extended_compute()
    pc = ext["metric/per_class"]
    assert np.allclose(pc["precision"], p, atol=GOLDEN_TOL)
    assert np.allclose(pc["recall"], r, atol=GOLDEN_TOL)
    assert np.allclose(pc["f1"], f, atol=GOLDEN_TOL)


class TestMulticlassGoldens:
    @pytest.fixture(autouse=True)
    def _rng(self):
        self.rng = np.random.default_rng(0)

    def test_unweighted(self):
        y = self.rng.integers(0, 3, 200)
        yhat = self.rng.integers(0, 3, 200)
        check_multiclass(y, yhat, None, [0, 1, 2])

    def test_integer_weights(self):
        y = self.rng.integers(0, 3, 200)
        yhat = self.rng.integers(0, 3, 200)
        w = self.rng.integers(1, 5, 200).astype(np.float64)
        check_multiclass(y, yhat, w, [0, 1, 2])

    def test_float_weights(self):
        y = self.rng.integers(0, 3, 200)
        yhat = self.rng.integers(0, 3, 200)
        w = self.rng.random(200) + 0.01
        check_multiclass(y, yhat, w, [0, 1, 2])

    def test_missing_true_class(self):
        y = self.rng.integers(0, 3, 200)
        y = np.where(y == 2, 1, y)  # class 2 从 truth 消失
        yhat = self.rng.integers(0, 3, 200)
        check_multiclass(y, yhat, None, [0, 1, 2])

    def test_never_predicted_class(self):
        y = self.rng.integers(0, 3, 200)
        yhat = self.rng.integers(0, 3, 200)
        yhat = np.where(yhat == 0, 1, yhat)  # class 0 从未预测
        w = self.rng.random(200) + 0.01
        check_multiclass(y, yhat, w, [0, 1, 2])

    def test_extreme_imbalance(self):
        y = np.concatenate([np.zeros(190, dtype=int), np.ones(10, dtype=int), np.full(200, 2, dtype=int)])
        yhat = np.zeros(400, dtype=int)
        w = self.rng.random(400) + 0.01
        check_multiclass(y, yhat, w, [0, 1, 2])


def check_multilabel(Y, S, thr, w):
    Yp = (S >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(Y, Yp, average=None, zero_division=0, sample_weight=w)
    pm = precision_recall_fscore_support(Y, Yp, average="macro", zero_division=0, sample_weight=w)[0]
    rm = precision_recall_fscore_support(Y, Yp, average="macro", zero_division=0, sample_weight=w)[1]
    fm = precision_recall_fscore_support(Y, Yp, average="macro", zero_division=0, sample_weight=w)[2]
    fw = precision_recall_fscore_support(Y, Yp, average="weighted", zero_division=0, sample_weight=w)[2]
    pmi = precision_recall_fscore_support(Y, Yp, average="micro", zero_division=0, sample_weight=w)[0]
    rmi = precision_recall_fscore_support(Y, Yp, average="micro", zero_division=0, sample_weight=w)[1]
    fmi = precision_recall_fscore_support(Y, Yp, average="micro", zero_division=0, sample_weight=w)[2]
    sub = accuracy_score(Y, Yp, sample_weight=w)
    ham = hamming_loss(Y, Yp, sample_weight=w)
    st = MultilabelState("metric", Y.shape[1], threshold=thr)
    st.update(_pb(Y, Yp, scores=S, weight=w))
    out = st.compute()
    assert abs(out["metric/precision_macro"] - pm) <= GOLDEN_TOL, (out["metric/precision_macro"], pm)
    assert abs(out["metric/recall_macro"] - rm) <= GOLDEN_TOL, (out["metric/recall_macro"], rm)
    assert abs(out["metric/f1_macro"] - fm) <= GOLDEN_TOL, (out["metric/f1_macro"], fm)
    assert abs(out["metric/f1_weighted"] - fw) <= GOLDEN_TOL, (out["metric/f1_weighted"], fw)
    assert abs(out["metric/precision_micro"] - pmi) <= GOLDEN_TOL, (out["metric/precision_micro"], pmi)
    assert abs(out["metric/recall_micro"] - rmi) <= GOLDEN_TOL, (out["metric/recall_micro"], rmi)
    assert abs(out["metric/f1_micro"] - fmi) <= GOLDEN_TOL, (out["metric/f1_micro"], fmi)
    assert abs(out["metric/subset_accuracy"] - sub) <= GOLDEN_TOL, (out["metric/subset_accuracy"], sub)
    assert abs(out["metric/hamming_loss"] - ham) <= GOLDEN_TOL, (out["metric/hamming_loss"], ham)
    ext = st.extended_compute()
    pc = ext["metric/per_label"]
    assert np.allclose(pc["precision"], p, atol=GOLDEN_TOL)


class TestMultilabelGoldens:
    @pytest.fixture(autouse=True)
    def _rng(self):
        self.rng = np.random.default_rng(1)

    def test_weighted(self):
        Y = self.rng.integers(0, 2, (100, 3))
        S = self.rng.random((100, 3))
        w = self.rng.random(100) + 0.01
        check_multilabel(Y, S, 0.5, w)

    def test_all_negative_labels(self):
        Y = np.zeros((100, 3), dtype=int)
        S = self.rng.random((100, 3))
        w = self.rng.random(100) + 0.01
        check_multilabel(Y, S, 0.5, w)

    def test_custom_threshold(self):
        Y = self.rng.integers(0, 2, (80, 2))
        S = self.rng.random((80, 2))
        w = self.rng.random(80) + 0.01
        check_multilabel(Y, S, 0.3, w)

    def test_unweighted(self):
        Y = self.rng.integers(0, 2, (80, 2))
        S = self.rng.random((80, 2))
        check_multilabel(Y, S, 0.5, None)
