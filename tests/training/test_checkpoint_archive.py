"""单机 ZIP 的发布、安全恢复及 Dataset 选择合同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import yaml

from dl_helper.training.checkpoint_archive import (
    _inspect_archive, apply_archive_retention, export_checkpoint_archive,
    restore_checkpoint_source, select_checkpoint_source,
)
from dl_helper.training.config import config_fingerprint, config_to_dict, default_schema, parse_config


@pytest.fixture
def config():
    schema = default_schema()
    schema["run"]["id"] = "archive-test"
    schema["run"]["source_revision"] = "source-v1"
    return parse_config(schema)


def make_checkpoint(root, config, *, epoch=0, step=2, batch=2, suffix=""):
    identifier = f"epoch-{epoch:06d}-step-{step:08d}{suffix}"
    directory = root / "checkpoints" / identifier
    directory.mkdir(parents=True)
    payload = b"checkpoint-data" * 5000
    (directory / "weights.bin").write_bytes(payload)
    engine = json.dumps({"best_value": None, "no_improve": 0}).encode("utf-8")
    (directory / "engine-state.json").write_bytes(engine)
    files = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
             for name, raw in (("weights.bin", payload), ("engine-state.json", engine))}
    manifest = {"schema_version": 1, "complete": True, "backend": "torch",
                "run_id": config.run.id, "checkpoint_id": identifier,
                "epoch": epoch, "global_step": step, "batch_in_epoch": batch,
                "created_utc": "2026-10-07T00:00:00Z", "files": files,
                "config_fingerprint": config_fingerprint(config, resume=True),
                "data_fingerprint": "data-v1", "runtime_versions": {}, "model_signature": {}}
    (directory / "checkpoint-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(config_to_dict(config)), encoding="utf-8")
    (root / "checkpoints" / "latest.json").write_text(json.dumps(
        {"schema_version": 1, "checkpoint_id": identifier, "path": identifier}), encoding="utf-8")
    return directory


def test_export_is_compressed_complete_and_idempotent(tmp_path, config):
    root = tmp_path / "run"
    cp = make_checkpoint(root, config)
    output = tmp_path / "downloads"
    archive = export_checkpoint_archive(root, cp, output, config)
    outer, inner = _inspect_archive(archive)
    assert outer["schema_version"] == 2
    assert outer["phase"] == "mid-epoch"
    assert outer["position"]["global_step"] == 2
    assert "training_returncode" not in outer
    with ZipFile(archive) as stream:
        assert stream.getinfo(f"run/checkpoints/{cp.name}/weights.bin").compress_type == ZIP_DEFLATED
    assert archive.stat().st_size < 10000
    assert (output / "last-checkpoint.zip").read_bytes() == archive.read_bytes()
    before = archive.read_bytes()
    assert export_checkpoint_archive(root, cp, output, config) == archive
    assert archive.read_bytes() == before
    assert inner["checkpoint_id"] == cp.name


def test_restore_to_existing_layout_preserves_dataset_and_migrates_failure(tmp_path, config):
    source = tmp_path / "dataset"
    cp = make_checkpoint(source, config)
    archive = export_checkpoint_archive(source, cp, tmp_path / "input", config)
    before = archive.read_bytes()
    destination = tmp_path / "working"
    destination.mkdir()
    cfg = destination / "config.resolved.yaml"
    cfg.write_text(yaml.safe_dump(config_to_dict(config)), encoding="utf-8")
    (destination / "failure.json").write_text('{"message":"old failure"}', encoding="utf-8")
    candidate = select_checkpoint_source([archive], destination, config)
    restore_checkpoint_source(candidate, destination, cfg, config)
    assert (destination / "checkpoints" / cp.name / "weights.bin").read_bytes() == (cp / "weights.bin").read_bytes()
    assert not (destination / "failure.json").exists()
    history = json.loads((destination / "recovery" / "terminal-history.json").read_text(encoding="utf-8"))
    assert history["entries"][0]["name"] == "failure.json"
    assert archive.read_bytes() == before


def test_latest_external_wins_and_alias_deduplicates(tmp_path, config):
    local = tmp_path / "local"
    make_checkpoint(local, config, step=2)
    remote = tmp_path / "remote"
    cp = make_checkpoint(remote, config, step=5)
    archives = tmp_path / "input"
    export_checkpoint_archive(remote, cp, archives, config)
    candidate = select_checkpoint_source([archives], local, config)
    assert candidate.manifest["global_step"] == 5


def test_early_stop_beats_periodic_same_step(tmp_path, config):
    local = tmp_path / "local"
    make_checkpoint(local, config, step=2)
    remote = tmp_path / "remote"
    cp = make_checkpoint(remote, config, step=2, suffix="-early-stop")
    archive = export_checkpoint_archive(remote, cp, tmp_path / "input", config)
    assert select_checkpoint_source([archive], local, config).manifest["checkpoint_id"].endswith("-early-stop")


def test_corrupt_latest_does_not_fall_back(tmp_path, config):
    local = tmp_path / "local"
    make_checkpoint(local, config)
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"broken archive")
    with pytest.raises(Exception):
        select_checkpoint_source([bad], local, config)


def test_same_identity_different_content_fails(tmp_path, config):
    local = tmp_path / "local"
    make_checkpoint(local, config)
    source = tmp_path / "source"
    cp = make_checkpoint(source, config)
    manifest = json.loads((cp / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprint"] = "conflict"
    (cp / "checkpoint-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = export_checkpoint_archive(source, cp, tmp_path / "input", config)
    with pytest.raises(ValueError, match="冲突"):
        select_checkpoint_source([archive], local, config)


def test_v1_and_expanded_dataset_compatibility(tmp_path, config):
    source = tmp_path / "source"
    cp = make_checkpoint(source, config)
    v2 = export_checkpoint_archive(source, cp, tmp_path / "input", config)
    expanded = tmp_path / "expanded"
    with ZipFile(v2) as stream:
        stream.extractall(expanded)
    manifest_path = expanded / "resume-manifest.json"
    outer = json.loads(manifest_path.read_text(encoding="utf-8"))
    outer["schema_version"] = 1
    outer["training_returncode"] = 75
    for key in ("backend", "position", "phase", "data_fingerprint"):
        outer.pop(key)
    manifest_path.write_text(json.dumps(outer), encoding="utf-8")
    assert select_checkpoint_source([expanded], tmp_path / "working", config).manifest["global_step"] == 2
    restored = tmp_path / "working"
    restored.mkdir()
    cfg = restored / "config.resolved.yaml"
    cfg.write_text(yaml.safe_dump(config_to_dict(config)), encoding="utf-8")
    candidate = select_checkpoint_source([expanded], restored, config)
    restore_checkpoint_source(candidate, restored, cfg, config)
    assert (restored / "checkpoints" / cp.name).is_dir()


def test_optional_dataset_absent_or_empty(tmp_path, config):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert select_checkpoint_source([empty, tmp_path / "not-mounted"], tmp_path / "working", config) is None


def test_archive_retention_follows_raw_directories(tmp_path, config):
    root = tmp_path / "run"
    output = tmp_path / "archives"
    first = make_checkpoint(root, config, step=2)
    first_zip = export_checkpoint_archive(root, first, output, config)
    second = make_checkpoint(root, config, step=3)
    second_zip = export_checkpoint_archive(root, second, output, config)
    apply_archive_retention(output, root / "checkpoints", None, config.run.id)
    assert first_zip.exists()
    import shutil
    shutil.rmtree(first)
    apply_archive_retention(output, root / "checkpoints", 1, config.run.id)
    assert not first_zip.exists()
    assert second_zip.exists() and (output / "last-checkpoint.zip").exists()


def test_zip_traversal_is_rejected(tmp_path, config):
    bad = tmp_path / "bad.zip"
    with ZipFile(bad, "w") as stream:
        stream.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="路径非法"):
        select_checkpoint_source([bad], tmp_path / "working", config)
    assert not (tmp_path / "escape").exists()


def test_allowed_budget_change_reuses_immutable_zip(tmp_path, config):
    from dataclasses import replace
    root = tmp_path / "run"
    checkpoint = make_checkpoint(root, config)
    output = tmp_path / "archives"
    archive = export_checkpoint_archive(root, checkpoint, output, config)
    before = archive.read_bytes()
    changed = replace(config, training=replace(config.training, max_epochs=config.training.max_epochs + 1))
    export_checkpoint_archive(root, checkpoint, output, changed)
    assert archive.read_bytes() == before


@pytest.mark.parametrize("directory", ["checkpoints", "models"])
def test_export_cannot_mutate_checkpoint_or_include_itself(tmp_path, config, directory):
    root = tmp_path / "run"
    checkpoint = make_checkpoint(root, config)
    with pytest.raises(ValueError, match="不得导出"):
        export_checkpoint_archive(root, checkpoint, root / directory / "archives", config)


def test_archive_callback_failure_keeps_previous_latest_and_allows_retry(tmp_path):
    from dl_helper.training.checkpoint import read_latest, write_torch_checkpoint
    from dl_helper.training.engine import EngineState

    class Accelerator:
        is_main_process = True

        def wait_for_everyone(self):
            return None

        def save_state(self, directory):
            Path(directory, "model.bin").write_bytes(b"model state")

    state = EngineState(backend="torch", run_id="callback", config_fingerprint="fp")
    root = tmp_path / "checkpoints"

    def save(step, callback=None):
        return write_torch_checkpoint(Accelerator(), str(root), "callback", state, {}, {}, "fp", "data", {},
                                      0, step, step, archive_callback=callback)

    first = save(1)

    def broken(_):
        raise OSError("模拟ZIP失败")

    with pytest.raises(OSError, match="模拟ZIP失败"):
        save(2, broken)
    assert read_latest(str(root))["checkpoint_id"] == first
    assert (root / first).is_dir()
    second = save(2)
    assert read_latest(str(root))["checkpoint_id"] == second


def test_old_full_working_directory_dataset(tmp_path, config):
    dataset = tmp_path / "dataset"
    source = dataset / "dl-helper-runs" / "runs" / config.run.id
    cp = make_checkpoint(source, config)
    auxiliary = source / "models" / "diagnostic" / "model.safetensors"
    auxiliary.parent.mkdir(parents=True)
    auxiliary.write_bytes(b"auxiliary-weights")
    (source / "failure.json").write_text('{"message":"old crash"}', encoding="utf-8")
    destination = tmp_path / "working"
    destination.mkdir()
    cfg_path = destination / "config.resolved.yaml"
    cfg_path.write_text(yaml.safe_dump(config_to_dict(config)), encoding="utf-8")
    candidate = select_checkpoint_source([dataset], destination, config)
    assert candidate.kind == "run"
    restore_checkpoint_source(candidate, destination, cfg_path, config)
    assert (destination / "checkpoints" / cp.name).is_dir()
    assert not (destination / "failure.json").exists()
    assert (source / "failure.json").is_file()
    assert (destination / "models" / "diagnostic" / "model.safetensors").read_bytes() == auxiliary.read_bytes()
    exported = export_checkpoint_archive(destination, destination / "checkpoints" / cp.name,
                                          tmp_path / "new-downloads", config)
    with ZipFile(exported) as archive:
        assert archive.read("run/models/diagnostic/model.safetensors") == auxiliary.read_bytes()
