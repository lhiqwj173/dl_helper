"""任务 2.3：回归指标与 sklearn 1.6.1 金标、大偏置 R2 稳定性。"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dl_helper.training.contracts import PredictedBatch
from dl_helper.training.metrics import RegressionState

GOLDEN_TOL = 1e-6
BIG_OFFSET_TOL = 1e-10


def _pb(targets, predictions, weight=None):
    return PredictedBatch(
        targets=np.asarray(targets, dtype=np.float64),
        predictions=np.asarray(predictions, dtype=np.float64),
        sample_count=np.asarray(targets).shape[0],
        sample_weight=(None if weight is None else np.asarray(weight, dtype=np.float64)),
    )


def check_regression(t, p, w, multioutput, tol=GOLDEN_TOL):
    tt = np.asarray(t)
    if tt.ndim == 1:
        tt = tt[:, None]
    pp = np.asarray(p)
    if pp.ndim == 1:
        pp = pp[:, None]
    st = RegressionState("metric", tt.shape[1])
    st.update(_pb(tt, pp, weight=w))
    out = st.compute()
    mae = mean_absolute_error(tt, pp, sample_weight=w)
    mse = mean_squared_error(tt, pp, sample_weight=w)
    r2 = r2_score(tt, pp, sample_weight=w, multioutput=multioutput, force_finite=True)
    assert abs(out["metric/mae"] - mae) <= tol, (out["metric/mae"], mae)
    assert abs(out["metric/mse"] - mse) <= tol, (out["metric/mse"], mse)
    if multioutput == "uniform_average":
        assert abs(out["metric/r2"] - r2) <= tol, (out["metric/r2"], r2)
    else:
        assert abs(out["metric/r2_variance_weighted"] - r2) <= tol, (out["metric/r2_variance_weighted"], r2)
    # per-target
    ext = st.extended_compute()
    pt = ext["metric/per_target"]
    r2_per = r2_score(tt, pp, sample_weight=w, multioutput="raw_values", force_finite=True)
    assert np.allclose(pt["r2"], r2_per, atol=tol)


class TestRegressionGoldens:
    @pytest.fixture(autouse=True)
    def _rng(self):
        self.rng = np.random.default_rng(2)

    def test_unweighted_single_output(self):
        t = self.rng.random(100) * 10
        p = t + self.rng.random(100)
        check_regression(t, p, None, "uniform_average")
        check_regression(t, p, None, "variance_weighted")

    def test_weighted(self):
        t = self.rng.random((100, 2)) * 10
        p = t + self.rng.random((100, 2))
        w = self.rng.random(100) + 0.01
        check_regression(t, p, w, "uniform_average")
        check_regression(t, p, w, "variance_weighted")

    def test_constant_target(self):
        t = np.full((100, 1), 5.0)
        w = self.rng.random(100) + 0.01
        check_regression(t, t.copy(), w, "uniform_average")  # perfect → R2=1
        p = t + 0.5
        check_regression(t, p, w, "uniform_average")  # constant miss → R2=0 (force_finite)

    def test_near_constant_target(self):
        t = np.full((100, 1), 5.0) + self.rng.normal(0, 1e-9, (100, 1))
        p = t + self.rng.normal(0, 1e-9, (100, 1))
        w = self.rng.random(100) + 0.01
        check_regression(t, p, w, "uniform_average")  # 不 epsilon 改判

    def test_multi_output_different_variance(self):
        n = 200
        t = np.column_stack([
            self.rng.normal(0, 1, n),
            self.rng.normal(0, 100, n),
            self.rng.normal(0, 0.001, n),
        ])
        p = t + np.column_stack([self.rng.normal(0, 0.1, n),
                                 self.rng.normal(0, 5, n),
                                 self.rng.normal(0, 0.0005, n)])
        w = self.rng.random(n) + 0.01
        check_regression(t, p, w, "uniform_average")
        check_regression(t, p, w, "variance_weighted")


def test_large_offset_r2_stability():
    """大偏置小波动：Welford M2 与直接两遍计算误差不超过 1e-10。"""
    rng = np.random.default_rng(3)
    n = 500
    offset = 1e6
    t = np.full(n, offset) + rng.normal(0, 1, n)
    p = t + rng.normal(0, 0.5, n)
    w = (rng.random(n) + 0.1).astype(np.float64)
    tt = t[:, None]
    pp = p[:, None]
    st = RegressionState("metric", 1)
    st.update(_pb(tt, pp, weight=w))
    out = st.compute()
    # 直接两遍计算
    ymean = np.average(tt[:, 0], weights=w)
    m2 = np.sum(w * (tt[:, 0] - ymean) ** 2)
    sse = np.sum(w * (pp[:, 0] - tt[:, 0]) ** 2)
    r2_ref = 1.0 - sse / m2
    assert abs(out["metric/r2"] - r2_ref) <= BIG_OFFSET_TOL, (out["metric/r2"], r2_ref)
    # 与 sklearn
    r2_sk = r2_score(tt, pp, sample_weight=w, force_finite=True)
    assert abs(out["metric/r2"] - r2_sk) <= BIG_OFFSET_TOL, (out["metric/r2"], r2_sk)


def test_large_offset_mse_stability():
    rng = np.random.default_rng(4)
    n = 300
    t = np.full(n, 1e9) + rng.normal(0, 1, n)
    p = t + rng.normal(0, 0.1, n)
    st = RegressionState("metric", 1)
    st.update(_pb(t[:, None], p[:, None], weight=None))
    out = st.compute()
    mse_ref = np.mean((p - t) ** 2)
    assert abs(out["metric/mse"] - mse_ref) <= 1e-6


def test_variance_weighted_all_zero_variance_falls_back_to_uniform():
    rng = np.random.default_rng(5)
    # 两输出，其中一个常量 target 方差为零 → variance_weighted 退化为 uniform
    t = np.column_stack([np.full(100, 5.0), rng.normal(0, 1, 100)])
    p = np.column_stack([np.full(100, 5.0), t[:, 1] + rng.normal(0, 0.1, 100)])
    w = rng.random(100) + 0.01
    st = RegressionState("metric", 2)
    st.update(_pb(t, p, weight=w))
    out = st.compute()
    r2_u = r2_score(t, p, sample_weight=w, multioutput="uniform_average", force_finite=True)
    r2_vw = r2_score(t, p, sample_weight=w, multioutput="variance_weighted", force_finite=True)
    assert abs(out["metric/r2"] - r2_u) <= GOLDEN_TOL
    assert abs(out["metric/r2_variance_weighted"] - r2_vw) <= GOLDEN_TOL
