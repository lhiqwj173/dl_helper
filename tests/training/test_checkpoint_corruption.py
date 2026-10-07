"""任务 4.3/4.4：检查点损坏、篡改与可信来源校验。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import read_json, sha256_file, write_json
from dl_helper.training.checkpoint import (
    CHECKPOINT_MANIFEST,
    CheckpointError,
    validate_manifest_complete,
    validate_sklearn_checkpoint_source,
    write_manifest,
)


def _make_checkpoint_dir(tmp_path, run_id="run-1", fp="fp", data_fp="data", model_sig=None):
    ckpt_dir = tmp_path / "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    (ckpt_dir / "estimator.joblib").write_bytes(b"fake-joblib-bytes")
    write_json(os.path.join(str(ckpt_dir), "engine-state.json"), {"epoch": 2})
    files = {
        "estimator.joblib": {"size": os.path.getsize(str(ckpt_dir / "estimator.joblib")),
                             "sha256": sha256_file(str(ckpt_dir / "estimator.joblib"))},
        "engine-state.json": {"size": os.path.getsize(str(ckpt_dir / "engine-state.json")),
                              "sha256": sha256_file(str(ckpt_dir / "engine-state.json"))},
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "checkpoint_id": "epoch-000002-step-00000010",
        "created_utc": "2026-01-01T00:00:00Z",
        "epoch": 2,
        "batch_in_epoch": 0,
        "global_step": 10,
        "config_fingerprint": fp,
        "backend": "sklearn",
        "data_fingerprint": data_fp,
        "model_signature": model_sig or {"class": "x"},
        "runtime_versions": {"python": "3.10.16", "sklearn": "1.6.1",
                             "numpy": "2.2.4", "scipy": "1.15.2", "joblib": "1.4.2"},
        "files": files,
        "complete": True,
    }
    write_manifest(str(ckpt_dir), manifest)
    return str(ckpt_dir), manifest


def test_validate_manifest_complete_ok(tmp_path):
    ckpt_dir, _ = _make_checkpoint_dir(tmp_path)
    manifest = read_json(os.path.join(ckpt_dir, CHECKPOINT_MANIFEST))
    validate_manifest_complete(manifest, ckpt_dir)  # 不抛


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_nested_inventory_is_portable_across_operating_systems(tmp_path, separator):
    root = tmp_path / "checkpoint"
    member = root / "accelerator-state" / "model.bin"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"model")
    manifest = {"complete": True, "files": {
        f"accelerator-state{separator}model.bin": {"size": 5, "sha256": sha256_file(str(member))}}}
    validate_manifest_complete(manifest, str(root))


def test_inventory_rejects_duplicate_normalized_paths(tmp_path):
    manifest = {"complete": True, "files": {"state/model.bin": {}, "state\\model.bin": {}}}
    with pytest.raises(CheckpointError, match="规范化后重复"):
        validate_manifest_complete(manifest, str(tmp_path))


def test_missing_file_fails(tmp_path):
    ckpt_dir, manifest = _make_checkpoint_dir(tmp_path)
    os.remove(os.path.join(ckpt_dir, "engine-state.json"))
    with pytest.raises(CheckpointError):
        validate_manifest_complete(manifest, ckpt_dir)


def test_checksum_mismatch_fails(tmp_path):
    ckpt_dir, manifest = _make_checkpoint_dir(tmp_path)
    with open(os.path.join(ckpt_dir, "estimator.joblib"), "wb") as f:
        f.write(b"tampered-bytes")
    with pytest.raises(CheckpointError):
        validate_manifest_complete(manifest, ckpt_dir)


def test_incomplete_manifest_fails(tmp_path):
    ckpt_dir, manifest = _make_checkpoint_dir(tmp_path)
    manifest["complete"] = False
    write_manifest(ckpt_dir, manifest)
    with pytest.raises(CheckpointError):
        validate_manifest_complete(manifest, ckpt_dir)


def test_sklearn_source_wrong_run_id(tmp_path):
    ckpt_dir, _ = _make_checkpoint_dir(tmp_path, run_id="run-1")
    with pytest.raises(CheckpointError):
        validate_sklearn_checkpoint_source(
            str(tmp_path), "ckpt", "run-OTHER", "fp", "data", {"class": "x"})


def test_sklearn_source_fingerprint_mismatch(tmp_path):
    ckpt_dir, _ = _make_checkpoint_dir(tmp_path)
    with pytest.raises(CheckpointError):
        validate_sklearn_checkpoint_source(
            str(tmp_path), "ckpt", "run-1", "different-fp", "data", {"class": "x"})


def test_sklearn_source_model_signature_mismatch(tmp_path):
    ckpt_dir, _ = _make_checkpoint_dir(tmp_path)
    with pytest.raises(CheckpointError):
        validate_sklearn_checkpoint_source(
            str(tmp_path), "ckpt", "run-1", "fp", "data", {"class": "different"})


def test_sklearn_source_symlink_rejected(tmp_path):
    ckpt_dir, _ = _make_checkpoint_dir(tmp_path)
    target = tmp_path / "target.joblib"
    target.write_bytes(b"data")
    os.remove(os.path.join(ckpt_dir, "estimator.joblib"))
    os.symlink(str(target), os.path.join(ckpt_dir, "estimator.joblib"))
    with pytest.raises(CheckpointError):
        validate_sklearn_checkpoint_source(
            str(tmp_path), "ckpt", "run-1", "fp", "data", {"class": "x"})


def test_sklearn_source_path_escape(tmp_path):
    ckpt_dir, _ = _make_checkpoint_dir(tmp_path)
    with pytest.raises(CheckpointError):
        validate_sklearn_checkpoint_source(
            str(tmp_path), "../escape", "run-1", "fp", "data", {"class": "x"})


def test_runtime_version_drift_fails(tmp_path):
    ckpt_dir, manifest = _make_checkpoint_dir(tmp_path)
    manifest["runtime_versions"]["sklearn"] = "0.24.2"  # 版本漂移
    write_manifest(ckpt_dir, manifest)
    with pytest.raises(CheckpointError):
        validate_sklearn_checkpoint_source(
            str(tmp_path), "ckpt", "run-1", "fp", "data", {"class": "x"})


def test_torch_runtime_version_drift_allows_resume(monkeypatch, capsys):
    from dl_helper.training import checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module, "runtime_versions",
        lambda backend: {"python": "9.9.9", "torch": "9.9.9",
                         "accelerate": "9.9.9", "numpy": "9.9.9"})
    checkpoint_module.verify_runtime_versions(
        "torch", {"python": "3.12.13", "torch": "2.10.0+cu128",
                  "accelerate": "1.6.0", "numpy": "2.0.2"})
    output = capsys.readouterr().out
    assert "runtime 版本漂移" in output
    assert "python recorded=3.12.13 current=9.9.9" in output
