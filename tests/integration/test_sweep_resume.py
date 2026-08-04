"""任务 6.4：sweep 暂停与严格恢复 —— 漂移/FAILED 负向与已成功跳过。"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from dl_helper.training.sweep import (
    SweepError,
    _SweepLayout,
    _trial_completed,
    _validate_resume,
    parse_sweep_manifest,
    sweep_checksum,
)


def _base_yaml(output_root):
    return {
        "schema_version": 1,
        "run": {"name": "t", "id": None, "output_root": output_root,
                "source_revision": None, "seed": 42, "tags": {}},
        "experiment": {},
        "training": {"max_epochs": 1, "log_every_steps": 1},
        "backend": {"type": "torch", "torch": {
            "gradient_accumulation_steps": 1, "mixed_precision": "no", "compile": False,
            "clip_grad_norm": 1.0, "deterministic": "off", "matmul_precision": "high",
            "find_unused_parameters": False}, "sklearn": None},
        "distributed": {"num_processes": 1},
        "selection": {"metric": "val/loss", "mode": "min", "patience": 20, "min_delta": 0.0},
        "checkpoint": {"every_epochs": None, "every_optimizer_steps": None,
                       "keep_last": 1, "resume": "none"},
        "runtime": {"max_minutes": None, "shutdown_grace_minutes": 10},
        "report": {"enabled": True, "curve_sample_limit": 100000,
                   "prediction_sample_limit": 10000, "prediction_splits": ["val"]},
        "remote": {"type": "none"},
        "notifications": {"type": "none"},
    }


def _write_sweep(tmp_path, output_root):
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "base.yaml").write_text(yaml.safe_dump(_base_yaml(output_root), allow_unicode=True),
                                         encoding="utf-8")
    for name, lr in [("lr-1e-2", 0.01), ("lr-5e-2", 0.05)]:
        (sweep_dir / f"{name}.yaml").write_text(
            yaml.safe_dump({"experiment": {"lr": lr}}, allow_unicode=True), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "sweep": {
            "id": "toy-lr-sweep",
            "experiment": "experiments.toy_multiclass:build_experiment",
            "base_config": "./base.yaml",
            "comparison_metric": "val/loss",
            "mode": "min",
            "trials": [{"name": "lr-1e-2", "variant": "./lr-1e-2.yaml"},
                       {"name": "lr-5e-2", "variant": "./lr-5e-2.yaml"}],
        },
    }
    path = sweep_dir / "sweep.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    return str(path)


def _sweep_dir(output_root):
    return os.path.join(output_root, "sweeps", "toy-lr-sweep")


def test_resume_without_pause_manifest_fails(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    os.makedirs(_sweep_dir(output_root), exist_ok=True)
    m = parse_sweep_manifest(manifest)
    with pytest.raises(SweepError):
        _validate_resume(m, _sweep_dir(output_root))


def test_resume_failed_sweep_rejected(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    sd = _sweep_dir(output_root)
    os.makedirs(sd, exist_ok=True)
    m = parse_sweep_manifest(manifest)
    json.dump({"sweep_id": m.sweep_id, "current_run_id": "x", "checksum": sweep_checksum(m)},
              open(os.path.join(sd, "pause-manifest.json"), "w", encoding="utf-8"))
    json.dump({"error": "x"}, open(os.path.join(sd, "failure.json"), "w", encoding="utf-8"))
    with pytest.raises(SweepError):
        _validate_resume(m, sd)


def test_resume_checksum_drift_rejected(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    sd = _sweep_dir(output_root)
    os.makedirs(sd, exist_ok=True)
    m = parse_sweep_manifest(manifest)
    json.dump({"sweep_id": m.sweep_id, "current_run_id": "x",
               "checksum": "0" * 64},
              open(os.path.join(sd, "pause-manifest.json"), "w", encoding="utf-8"))
    with pytest.raises(SweepError):
        _validate_resume(m, sd)


def test_resume_sweep_id_drift_rejected(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    sd = _sweep_dir(output_root)
    os.makedirs(sd, exist_ok=True)
    m = parse_sweep_manifest(manifest)
    json.dump({"sweep_id": "OTHER", "current_run_id": "x", "checksum": sweep_checksum(m)},
              open(os.path.join(sd, "pause-manifest.json"), "w", encoding="utf-8"))
    with pytest.raises(SweepError):
        _validate_resume(m, sd)


def test_trial_completed_detection(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    sd = _sweep_dir(output_root)
    os.makedirs(sd, exist_ok=True)
    layout = _SweepLayout(sd)
    from dl_helper.training.artifacts import append_jsonl, write_json
    append_jsonl(layout.trials_jsonl, {"trial": "lr-1e-2", "run_id": "toy-lr-sweep--lr-1e-2",
                                       "status": "succeeded"})
    # 需 run 终态（run-manifest.json）
    run_dir = os.path.join(output_root, "runs", "toy-lr-sweep--lr-1e-2")
    os.makedirs(run_dir, exist_ok=True)
    write_json(os.path.join(run_dir, "run-manifest.json"), {"status": "succeeded", "run_id": "toy-lr-sweep--lr-1e-2"})
    assert _trial_completed(layout, "toy-lr-sweep--lr-1e-2")
    assert not _trial_completed(layout, "toy-lr-sweep--lr-5e-2")
