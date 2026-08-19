"""任务 6.1：严格 sweep manifest parser。"""
from __future__ import annotations

import yaml
import pytest

from dl_helper.training.sweep import SweepError, parse_sweep_manifest, resolve_trial_configs


def _manifest(tmp_path, **patch):
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    base = sweep_dir / "base.yaml"
    base.write_text(_base_yaml(), encoding="utf-8")
    v1 = sweep_dir / "v1.yaml"
    v1.write_text(yaml.safe_dump({"training": {"max_epochs": 2}}, allow_unicode=True), encoding="utf-8")
    v2 = sweep_dir / "v2.yaml"
    v2.write_text(yaml.safe_dump({"training": {"max_epochs": 3}}, allow_unicode=True), encoding="utf-8")
    data = {
        "schema_version": 1,
        "sweep": {
            "id": "sweep-1",
            "experiment": "experiments.toy_multiclass:build_experiment",
            "base_config": "./base.yaml",
            "comparison_metric": "val/loss",
            "mode": "min",
            "trials": [
                {"name": "a", "variant": "./v1.yaml"},
                {"name": "b", "variant": "./v2.yaml"},
            ],
        },
    }
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(data["sweep"].get(k), dict):
            data["sweep"][k] = {**data["sweep"][k], **v}
        else:
            data["sweep"][k] = v
    path = sweep_dir / "sweep.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return str(path)


def _base_yaml():
    schema = {
        "schema_version": 1,
        "run": {"name": "s", "id": None, "output_root": None, "source_revision": None,
                "seed": 42, "tags": {}},
        "experiment": {},
        "training": {"max_epochs": 20, "log_every_steps": 20},
        "backend": {"type": "torch", "torch": {
            "gradient_accumulation_steps": 1, "mixed_precision": "no", "compile": False,
            "clip_grad_norm": 1.0, "deterministic": "strict", "matmul_precision": "high",
            "find_unused_parameters": False}, "sklearn": None},
        "distributed": {"num_processes": 1},
        "selection": {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0},
        "checkpoint": {"every_epochs": None, "every_optimizer_steps": None, "keep_last": 1},

        "report": {"enabled": True, "curve_sample_limit": 100000, "prediction_sample_limit": 10000,
                   "prediction_splits": ["val"]},
        "remote": {"type": "none"},
        "notifications": {"type": "none"},
    }
    return yaml.safe_dump(schema, allow_unicode=True)


def test_valid_manifest_parses(tmp_path):
    m = parse_sweep_manifest(_manifest(tmp_path))
    assert m.sweep_id == "sweep-1"
    assert len(m.trials) == 2
    assert m.derived_run_id("a") == "sweep-1--a"
    assert m.comparison_metric == "val/loss"
    assert m.mode == "min"


def test_less_than_two_trials_rejected(tmp_path):
    with pytest.raises(SweepError):
        parse_sweep_manifest(_manifest(tmp_path, trials=[{"name": "a", "variant": "./v1.yaml"}]))


def test_duplicate_trial_names_rejected(tmp_path):
    with pytest.raises(SweepError):
        parse_sweep_manifest(_manifest(tmp_path, trials=[
            {"name": "a", "variant": "./v1.yaml"},
            {"name": "a", "variant": "./v2.yaml"},
        ]))


def test_bad_sweep_id_rejected(tmp_path):
    with pytest.raises(SweepError):
        parse_sweep_manifest(_manifest(tmp_path, id="bad id!"))


def test_comparison_metric_must_be_val(tmp_path):
    with pytest.raises(SweepError):
        parse_sweep_manifest(_manifest(tmp_path, comparison_metric="test/acc"))


def test_absolute_variant_rejected(tmp_path):
    with pytest.raises(SweepError):
        parse_sweep_manifest(_manifest(tmp_path, trials=[
            {"name": "a", "variant": "./v1.yaml"},
            {"name": "b", "variant": "/etc/passwd"},
        ]))


def test_url_variant_rejected(tmp_path):
    with pytest.raises(SweepError):
        parse_sweep_manifest(_manifest(tmp_path, trials=[
            {"name": "a", "variant": "./v1.yaml"},
            {"name": "b", "variant": "https://example.com/x.yaml"},
        ]))


def test_path_escape_rejected(tmp_path):
    with pytest.raises(SweepError):
        parse_sweep_manifest(_manifest(tmp_path, trials=[
            {"name": "a", "variant": "./v1.yaml"},
            {"name": "b", "variant": "../escape.yaml"},
        ]))


def test_tuning_fingerprint_unique(tmp_path):
    m = parse_sweep_manifest(_manifest(tmp_path))
    resolved = resolve_trial_configs(m)
    assert len(resolved) == 2
    fps = {tuning_fp(c) for _, c in resolved}
    assert len(fps) == 2  # max_epochs 不同 → tuning 不同


def tuning_fp(config):
    from dl_helper.training.config import tuning_fingerprint
    return tuning_fingerprint(config)


def test_identical_tuning_rejected(tmp_path):
    manifest_path = _manifest(tmp_path)
    sweep_dir = tmp_path / "sweep"
    (sweep_dir / "v1.yaml").write_text(yaml.safe_dump({"run": {"tags": {"a": "1"}}}, allow_unicode=True), encoding="utf-8")
    (sweep_dir / "v2.yaml").write_text(yaml.safe_dump({"run": {"tags": {"a": "2"}}}, allow_unicode=True), encoding="utf-8")
    m = parse_sweep_manifest(manifest_path)
    with pytest.raises(SweepError):
        resolve_trial_configs(m)
