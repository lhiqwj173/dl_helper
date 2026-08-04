"""任务 2.4：固定 seed 随机分块 / 状态恢复 / 不可变 golden 门禁。"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dl_helper.training.contracts import PredictedBatch
from dl_helper.training.metrics import (
    MetricStateError,
    MulticlassState,
    MultilabelState,
    RegressionState,
)

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "metric_goldens_v1.json")
TOL = 1e-6

_MC_METRICS = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
_ML_METRICS = ("precision_macro", "recall_macro", "f1_macro", "f1_weighted",
               "precision_micro", "recall_micro", "f1_micro", "subset_accuracy", "hamming_loss")
_REG_METRICS = ("mae", "mse", "r2", "r2_variance_weighted")


def _load_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _mc_state_from_data(case):
    classes = np.asarray(case["classes"])
    targets = np.asarray(case["targets"], dtype=np.int64)
    predictions = np.asarray(case["predictions"], dtype=np.int64)
    w = None if case["sample_weight"] is None else np.asarray(case["sample_weight"], dtype=np.float64)
    return MulticlassState("metric", classes), targets, predictions, w


def _ml_state_from_data(case):
    targets = np.asarray(case["targets"], dtype=np.int64)
    n, l = targets.shape
    scores = np.asarray(case["scores"], dtype=np.float64).reshape(n, l)
    w = None if case["sample_weight"] is None else np.asarray(case["sample_weight"], dtype=np.float64)
    return MultilabelState("metric", l, threshold=case["threshold"]), targets, scores, w


def _reg_state_from_data(case):
    d = case["num_targets"]
    targets = np.asarray(case["targets"], dtype=np.float64).reshape(-1, d)
    predictions = np.asarray(case["predictions"], dtype=np.float64).reshape(-1, d)
    w = None if case["sample_weight"] is None else np.asarray(case["sample_weight"], dtype=np.float64)
    return RegressionState("metric", d), targets, predictions, w


def _pb(targets, predictions, scores=None, weight=None):
    return PredictedBatch(
        targets=np.asarray(targets), predictions=np.asarray(predictions),
        sample_count=np.asarray(targets).shape[0], scores=scores,
        sample_weight=(None if weight is None else np.asarray(weight, dtype=np.float64)),
    )


def _chunk_indices(rng, n, min_size=1):
    """随机分块（100 组以内固定 seed）。"""
    if n <= min_size:
        return [np.arange(n)]
    cut = np.sort(rng.integers(0, n + 1, size=min(n, 64)))
    cut = np.unique(np.concatenate([[0], cut, [n]]))
    out = []
    for i in range(len(cut) - 1):
        if cut[i + 1] - cut[i] >= min_size:
            out.append(np.arange(cut[i], cut[i + 1]))
    return out


class TestGoldenFixtures:
    @pytest.fixture(autouse=True)
    def _fixture(self):
        self.fixture = _load_fixture()

    def test_fixture_schema_and_version(self):
        assert self.fixture["schema_version"] == 1
        assert self.fixture["scikit_learn_version"] == "1.6.1"

    def test_multiclass_golden_full_batch(self):
        for case in self.fixture["cases"]:
            if not case["name"].startswith("multiclass/"):
                continue
            state, targets, predictions, w = _mc_state_from_data(case)
            state.update(_pb(targets, predictions, weight=w))
            out = state.compute()
            for metric in _MC_METRICS:
                assert abs(out[f"metric/{metric}"] - case["expected"][metric]) <= TOL, (
                    f"{case['name']} {metric}: {out[f'metric/{metric}']} vs {case['expected'][metric]}"
                )
            ext = state.extended_compute()
            pc = ext["metric/per_class"]
            assert np.allclose(pc["precision"], case["expected"]["per_class_precision"], atol=TOL)

    def test_multilabel_golden_full_batch(self):
        for case in self.fixture["cases"]:
            if not case["name"].startswith("multilabel/"):
                continue
            state, targets, scores, w = _ml_state_from_data(case)
            hard = (scores >= case["threshold"]).astype(int)
            state.update(_pb(targets, hard, scores=scores, weight=w))
            out = state.compute()
            for metric in _ML_METRICS:
                assert abs(out[f"metric/{metric}"] - case["expected"][metric]) <= TOL, (
                    f"{case['name']} {metric}: {out[f'metric/{metric}']} vs {case['expected'][metric]}"
                )

    def test_regression_golden_full_batch(self):
        for case in self.fixture["cases"]:
            if not case["name"].startswith("regression/"):
                continue
            state, targets, predictions, w = _reg_state_from_data(case)
            state.update(_pb(targets, predictions, weight=w))
            out = state.compute()
            for metric in _REG_METRICS:
                assert abs(out[f"metric/{metric}"] - case["expected"][metric]) <= TOL, (
                    f"{case['name']} {metric}: {out[f'metric/{metric}']} vs {case['expected'][metric]}"
                )


class TestChunking:
    """随机分块与单批全量结果一致，状态保存/恢复不改变结果。"""

    @pytest.fixture(autouse=True)
    def _cases(self):
        self.fixture = _load_fixture()

    @pytest.mark.parametrize("seed", list(range(100)))
    def test_multiclass_chunking(self, seed):
        case = next(c for c in self.fixture["cases"] if c["name"] == "multiclass/float_weights")
        state, targets, predictions, w = _mc_state_from_data(case)
        rng = np.random.default_rng(seed)
        state.reset()
        for idx in _chunk_indices(rng, targets.shape[0]):
            ww = None if w is None else w[idx]
            state.update(_pb(targets[idx], predictions[idx], weight=ww))
        # 与单批比较
        ref = MulticlassState("metric", case["classes"])
        ref.update(_pb(targets, predictions, weight=w))
        out = state.compute()
        ref_out = ref.compute()
        for metric in _MC_METRICS:
            assert abs(out[f"metric/{metric}"] - ref_out[f"metric/{metric}"]) <= TOL

    @pytest.mark.parametrize("seed", list(range(100)))
    def test_regression_chunking(self, seed):
        case = next(c for c in self.fixture["cases"] if c["name"] == "regression/weighted")
        state, targets, predictions, w = _reg_state_from_data(case)
        rng = np.random.default_rng(seed)
        state.reset()
        for idx in _chunk_indices(rng, targets.shape[0]):
            ww = None if w is None else w[idx]
            state.update(_pb(targets[idx], predictions[idx], weight=ww))
        ref = RegressionState("metric", case["num_targets"])
        ref.update(_pb(targets, predictions, weight=w))
        out = state.compute()
        ref_out = ref.compute()
        for metric in _REG_METRICS:
            assert abs(out[f"metric/{metric}"] - ref_out[f"metric/{metric}"]) <= TOL

    @pytest.mark.parametrize("seed", list(range(100)))
    def test_multilabel_chunking(self, seed):
        case = next(c for c in self.fixture["cases"] if c["name"] == "multilabel/weighted")
        state, targets, scores, w = _ml_state_from_data(case)
        hard = (scores >= case["threshold"]).astype(int)
        rng = np.random.default_rng(seed)
        state.reset()
        for idx in _chunk_indices(rng, targets.shape[0]):
            ww = None if w is None else w[idx]
            state.update(_pb(targets[idx], hard[idx], scores=scores[idx], weight=ww))
        ref = MultilabelState("metric", case["num_labels"], threshold=case["threshold"])
        ref.update(_pb(targets, hard, scores=scores, weight=w))
        out = state.compute()
        ref_out = ref.compute()
        for metric in _ML_METRICS:
            assert abs(out[f"metric/{metric}"] - ref_out[f"metric/{metric}"]) <= TOL

    def test_state_save_load_between_chunks(self):
        case = next(c for c in self.fixture["cases"] if c["name"] == "regression/weighted")
        state, targets, predictions, w = _reg_state_from_data(case)
        rng = np.random.default_rng(7)
        for idx in _chunk_indices(rng, targets.shape[0], min_size=1):
            ww = None if w is None else w[idx]
            state.update(_pb(targets[idx], predictions[idx], weight=ww))
            saved = state.state_dict()
            state2 = RegressionState("metric", case["num_targets"])
            state2.load_state_dict(saved)
            state = state2
        out = state.compute()
        ref = RegressionState("metric", case["num_targets"])
        ref.update(_pb(targets, predictions, weight=w))
        ref_out = ref.compute()
        for metric in _REG_METRICS:
            assert abs(out[f"metric/{metric}"] - ref_out[f"metric/{metric}"]) <= TOL

    def test_empty_split_fails(self):
        state = MulticlassState("metric", [0, 1, 2])
        with pytest.raises(MetricStateError):
            state.compute()
