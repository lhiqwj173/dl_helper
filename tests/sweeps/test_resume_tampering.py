"""任务 OSR-008：resume 篡改 —— 已完成 run/contract/pause checksum 漂移必须失败。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import write_json
from dl_helper.training.sweep import (
    SweepError,
    _SweepLayout,
    _contracts_checksum,
    _run_terminal_checksum,
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
    import yaml
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "base.yaml").write_text(yaml.safe_dump(_base_yaml(output_root), allow_unicode=True),
                                         encoding="utf-8")
    for name, lr in [("lr-1e-2", 0.01), ("lr-5e-2", 0.05)]:
        (sweep_dir / f"{name}.yaml").write_text(
            yaml.safe_dump({"experiment": {"lr": lr}}, allow_unicode=True), encoding="utf-8")
    man = {
        "schema_version": 1,
        "sweep": {"id": "tamper-sweep", "experiment": "experiments.toy_multiclass:build_experiment",
                  "base_config": "./base.yaml", "comparison_metric": "val/loss", "mode": "min",
                  "trials": [{"name": "lr-1e-2", "variant": "./lr-1e-2.yaml"},
                             {"name": "lr-5e-2", "variant": "./lr-5e-2.yaml"}]},
    }
    path = sweep_dir / "sweep.yaml"
    path.write_text(yaml.safe_dump(man, allow_unicode=True), encoding="utf-8")
    return str(path)


def _sweep_dir(output_root):
    return os.path.join(output_root, "sweeps", "tamper-sweep")


def _make_pause(tmp_path, output_root, m):
    sd = _sweep_dir(output_root)
    os.makedirs(sd, exist_ok=True)
    os.makedirs(os.path.join(sd, "contracts"), exist_ok=True)
    write_json(os.path.join(sd, "contracts", "lr-1e-2.json"), {"valid": True})
    write_json(os.path.join(sd, "contracts", "lr-5e-2.json"), {"valid": True})
    # 已完成 run 终态
    run_dir = os.path.join(output_root, "runs", "tamper-sweep--lr-1e-2")
    os.makedirs(run_dir, exist_ok=True)
    write_json(os.path.join(run_dir, "run-manifest.json"), {"status": "succeeded"})
    # 当前被暂停 run 的暂停终态（含 resume_checkpoint）与 checkpoint 目录/manifest
    cur_run = "tamper-sweep--lr-5e-2"
    cur_run_dir = os.path.join(output_root, "runs", cur_run)
    os.makedirs(cur_run_dir, exist_ok=True)
    write_json(os.path.join(cur_run_dir, "pause-manifest.json"),
               {"status": "preempted", "resume_checkpoint": "epoch-000000-step-00000004"})
    ckpt_dir = os.path.join(cur_run_dir, "checkpoints", "epoch-000000-step-00000004")
    os.makedirs(ckpt_dir, exist_ok=True)
    from dl_helper.training.artifacts import sha256_file
    write_json(os.path.join(ckpt_dir, "checkpoint-manifest.json"),
               {"schema_version": 1, "complete": True, "files": {}})
    ckpt_checksum = sha256_file(os.path.join(ckpt_dir, "checkpoint-manifest.json"))
    pause = {
        "schema_version": 1,
        "sweep_id": m.sweep_id,
        "current_run_id": cur_run,
        "current_checkpoint": "epoch-000000-step-00000004",
        "current_checkpoint_checksum": ckpt_checksum,
        "completed": [{"trial": "lr-1e-2", "run_id": "tamper-sweep--lr-1e-2", "status": "succeeded"}],
        "remaining": ["lr-5e-2"],
        "checksum": sweep_checksum(m),
        "contract_checksum": _contracts_checksum(m, output_root),
        "run_checksums": {"tamper-sweep--lr-1e-2": _run_terminal_checksum(output_root, "tamper-sweep--lr-1e-2")},
    }
    write_json(os.path.join(sd, "pause-manifest.json"), pause)
    return sd


def test_untampered_resume_passes(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    sd = _make_pause(tmp_path, output_root, m)
    _validate_resume(m, sd)  # 不抛


def test_completed_run_tampered_fails(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    sd = _make_pause(tmp_path, output_root, m)
    # 篡改已完成 run 终态
    run_dir = os.path.join(output_root, "runs", "tamper-sweep--lr-1e-2")
    write_json(os.path.join(run_dir, "run-manifest.json"), {"status": "succeeded", "tampered": True})
    with pytest.raises(SweepError):
        _validate_resume(m, sd)


def test_contract_tampered_fails(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    sd = _make_pause(tmp_path, output_root, m)
    write_json(os.path.join(sd, "contracts", "lr-1e-2.json"), {"valid": False})
    with pytest.raises(SweepError):
        _validate_resume(m, sd)


def test_pause_checksum_tampered_fails(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    sd = _make_pause(tmp_path, output_root, m)
    pause = json.load(open(os.path.join(sd, "pause-manifest.json"), encoding="utf-8"))
    pause["checksum"] = "0" * 64
    write_json(os.path.join(sd, "pause-manifest.json"), pause)
    with pytest.raises(SweepError):
        _validate_resume(m, sd)


def test_pause_binds_current_checkpoint(tmp_path):
    """OSR-008：pause 严格绑定当前被暂停 run 的 resume checkpoint。"""
    from dl_helper.training.sweep import _write_pause_manifest

    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    sweep_dir = _sweep_dir(output_root)
    os.makedirs(sweep_dir, exist_ok=True)
    current_run = "tamper-sweep--lr-5e-2"
    run_dir = os.path.join(output_root, "runs", current_run)
    os.makedirs(run_dir, exist_ok=True)
    write_json(os.path.join(run_dir, "pause-manifest.json"),
               {"status": "preempted", "resume_checkpoint": "epoch-000000-step-00000004"})
    os.makedirs(os.path.join(run_dir, "checkpoints", "epoch-000000-step-00000004"), exist_ok=True)
    write_json(os.path.join(run_dir, "checkpoints", "epoch-000000-step-00000004",
                            "checkpoint-manifest.json"), {"complete": True, "files": {}})
    # 已完成 run 的终态（run_checksums 非 null 所需）
    done_run_dir = os.path.join(output_root, "runs", "tamper-sweep--lr-1e-2")
    os.makedirs(done_run_dir, exist_ok=True)
    write_json(os.path.join(done_run_dir, "run-manifest.json"), {"status": "succeeded"})
    completed = [{"trial": "lr-1e-2", "run_id": "tamper-sweep--lr-1e-2", "status": "succeeded"}]
    _write_pause_manifest(m, sweep_dir, current_run, completed, output_root)
    pause = json.load(open(os.path.join(sweep_dir, "pause-manifest.json"), encoding="utf-8"))
    assert pause["current_run_id"] == current_run
    assert pause["current_checkpoint"] == "epoch-000000-step-00000004"
    assert pause["current_checkpoint_checksum"]
    assert pause["run_checksums"]["tamper-sweep--lr-1e-2"]
    assert pause["checksum"] == sweep_checksum(m)
    assert "contract_checksum" in pause


def test_pause_missing_current_checkpoint_fails(tmp_path):
    """OSR-008：pause 缺 current_checkpoint 立即失败。"""
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    sd = _make_pause(tmp_path, output_root, m)
    pause = json.load(open(os.path.join(sd, "pause-manifest.json"), encoding="utf-8"))
    del pause["current_checkpoint"]
    write_json(os.path.join(sd, "pause-manifest.json"), pause)
    with pytest.raises(SweepError):
        _validate_resume(m, sd)


def test_pause_missing_completed_run_checksum_fails(tmp_path):
    """OSR-008：已完成 run 缺 run_checksums 条目立即失败。"""
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    sd = _make_pause(tmp_path, output_root, m)
    pause = json.load(open(os.path.join(sd, "pause-manifest.json"), encoding="utf-8"))
    pause["run_checksums"] = {}
    write_json(os.path.join(sd, "pause-manifest.json"), pause)
    with pytest.raises(SweepError):
        _validate_resume(m, sd)
