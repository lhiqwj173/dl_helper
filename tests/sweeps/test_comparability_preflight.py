"""任务 6.2：零优化步可比性预检。"""
from __future__ import annotations

import pytest

from dl_helper.training.sweep import SweepError, SweepManifest, TrialSpec, _compare_contracts


def _manifest(metric="val/f1_macro", mode="max"):
    return SweepManifest(
        schema_version=1, sweep_id="s", experiment="e", base_config="/tmp/b",
        comparison_metric=metric, mode=mode,
        trials=[TrialSpec(name="a", variant="", resolved_config=""),
                TrialSpec(name="b", variant="", resolved_config="")],
    )


def _contract(backend="torch", data_fp="fp1", task="multiclass", formula_ver=1,
              direction="max", metric=None, splits=None, label_schema=None):
    return {
        "backend": backend,
        "experiment_ref": "experiments.toy_multiclass:build_experiment",
        "data_identity": {"name": "toy", "version": "1.0", "fingerprint": data_fp},
        "task_name": task,
        "splits": splits or {"train": {"fingerprint": "a"}, "val": {"fingerprint": "b"}},
        "label_schema": label_schema or {"kind": "multiclass", "num_classes": 3},
        "metric_definitions": metric or {
            "f1_macro": {"formula_id": "f", "formula_version": formula_ver,
                         "direction": direction, "exact": True, "evaluation_scope": "full",
                         "averaging": "macro", "sample_weight_policy": "supported"},
        },
    }


def test_identical_contracts_pass():
    m = _manifest()
    _compare_contracts(m, [_contract(), _contract()])


def test_threshold_drift_fails():
    """OSR-008：多标签阈值漂移（0.2 vs 0.8）→ 零优化步拒绝。"""
    m = _manifest()
    c1 = _contract(label_schema={"kind": "multilabel", "num_labels": 3, "threshold": 0.2})
    c2 = _contract(label_schema={"kind": "multilabel", "num_labels": 3, "threshold": 0.8})
    with pytest.raises(SweepError):
        _compare_contracts(m, [c1, c2])


def test_split_fingerprint_drift_fails():
    """OSR-008：split fingerprint 漂移 → 零优化步拒绝。"""
    m = _manifest()
    c1 = _contract(splits={"train": {"fingerprint": "x"}, "val": {"fingerprint": "y"}})
    c2 = _contract(splits={"train": {"fingerprint": "x"}, "val": {"fingerprint": "z"}})
    with pytest.raises(SweepError):
        _compare_contracts(m, [c1, c2])


def test_backend_drift_fails():
    m = _manifest()
    with pytest.raises(SweepError):
        _compare_contracts(m, [_contract(), _contract(backend="sklearn")])


def test_data_drift_fails():
    m = _manifest()
    with pytest.raises(SweepError):
        _compare_contracts(m, [_contract(data_fp="fp1"), _contract(data_fp="fp2")])


def test_task_drift_fails():
    m = _manifest()
    with pytest.raises(SweepError):
        _compare_contracts(m, [_contract(task="multiclass"), _contract(task="regression")])


def test_formula_version_drift_fails():
    m = _manifest()
    with pytest.raises(SweepError):
        _compare_contracts(m, [_contract(formula_ver=1), _contract(formula_ver=2)])


def test_comparison_metric_not_produced_fails():
    m = _manifest(metric="val/nonexistent")
    with pytest.raises(SweepError):
        _compare_contracts(m, [_contract(), _contract()])


def test_comparison_metric_not_full_exact_fails():
    m = _manifest()
    sampled = {"f1_macro": {"formula_id": "f", "formula_version": 1, "direction": "max",
                            "exact": False, "evaluation_scope": "sampled",
                            "averaging": "macro", "sample_weight_policy": "supported"}}
    with pytest.raises(SweepError):
        _compare_contracts(m, [_contract(metric=sampled), _contract(metric=sampled)])


def test_direction_mismatch_fails():
    m = _manifest(mode="min")
    with pytest.raises(SweepError):
        _compare_contracts(m, [_contract(direction="max"), _contract(direction="max")])


def test_test_metric_not_comparable():
    m = _manifest(metric="test/accuracy")
    from dl_helper.training.sweep import parse_sweep_manifest, SweepError as SE

    # comparison_metric 必须 val/ 前缀，解析阶段已拒绝
    import yaml, os, tempfile
    data = {
        "schema_version": 1,
        "sweep": {"id": "s", "experiment": "e", "base_config": "./b.yaml",
                  "comparison_metric": "test/accuracy", "mode": "max",
                  "trials": [{"name": "a", "variant": "./a.yaml"},
                             {"name": "b", "variant": "./b.yaml"}]},
    }
    tmp = tempfile.mkdtemp()
    for f, content in [("b.yaml", "x: 1"), ("a.yaml", "x: 2")]:
        open(os.path.join(tmp, f), "w", encoding="utf-8").write(content)
    open(os.path.join(tmp, "sweep.yaml"), "w", encoding="utf-8").write(
        yaml.safe_dump(data, allow_unicode=True))
    with pytest.raises(SE):
        parse_sweep_manifest(os.path.join(tmp, "sweep.yaml"))
