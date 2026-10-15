"""单机成果归档、后处理重打包、远端一致性及失败阻断。"""
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest

from dl_helper.training import bundles
from dl_helper.training.artifacts import ArtifactError, RunLayout, sha256_file, write_json
from dl_helper.training.cli import main
from dl_helper.training.remote import LocalArtifactStore, _make_zip
from dl_helper.training.services import LifecycleServices, ServiceAudit


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "runs" / "unit-run"
    root.mkdir(parents=True)
    (root / "模型.bin").write_bytes(b"weights")
    (root / "checkpoint-archives").mkdir()
    (root / "checkpoint-archives/last-checkpoint.zip").write_bytes(b"resume")
    write_json(str(root / "run-manifest.json"), {
        "run_id": "unit-run", "status": "succeeded", "artifacts": {
            "模型.bin": {"size": 7, "sha256": sha256_file(str(root / "模型.bin"))},
            "checkpoints/removed.bin": {"size": 100, "sha256": "excluded"},
        },
    })
    return root


def test_bundle_preserves_original_and_has_exact_inventory(run_dir):
    original = (run_dir / "run-manifest.json").read_bytes()
    output = bundles.export_run_bundle(run_dir)
    assert output == bundles.default_bundle_path(run_dir)
    assert output == run_dir.parent / "unit-run.run-bundle.zip"
    assert (run_dir / "run-manifest.json").read_bytes() == original
    with ZipFile(output) as archive:
        inventory = json.loads(archive.read("bundle-manifest.json"))
        assert set(inventory["files"]) == set(archive.namelist()) - {"bundle-manifest.json"}
        assert archive.read("run-manifest.json") == original
        assert "checkpoint-archives/last-checkpoint.zip" not in archive.namelist()
        for name, expected in inventory["files"].items():
            assert expected == {"size": len(archive.read(name)),
                                "sha256": hashlib.sha256(archive.read(name)).hexdigest()}
    assert output.with_suffix(".zip.sha256").read_text(encoding="utf-8") == (
        f"{sha256_file(str(output))}  {output.name}\n")
    assert (run_dir / "checkpoint-archives/last-checkpoint.zip").is_file()


def test_postprocessing_repackages_without_rewriting_training_manifest(run_dir):
    output = bundles.export_run_bundle(run_dir)
    original = (run_dir / "run-manifest.json").read_bytes()
    before = output.read_bytes()
    (run_dir / "exports").mkdir()
    (run_dir / "exports/model.torchscript").write_bytes(b"postprocessing")
    assert bundles.export_run_bundle(run_dir) == output
    assert output.read_bytes() != before
    assert (run_dir / "run-manifest.json").read_bytes() == original
    with ZipFile(output) as archive:
        assert archive.read("exports/model.torchscript") == b"postprocessing"
    # 本地和 AList 使用完全相同的 ZIP 生成器。
    assert _make_zip(str(run_dir), bundles.EXCLUDED_DIRECTORIES) == output.read_bytes()
    os.utime(run_dir / "模型.bin", (123456789, 123456789))
    first = output.read_bytes()
    bundles.export_run_bundle(run_dir)
    assert output.read_bytes() == first


@pytest.mark.parametrize("terminal", ["pause-manifest.json", "failure.json"])
def test_unsuccessful_runs_do_not_replace_old_bundle(run_dir, tmp_path, terminal):
    (run_dir / "run-manifest.json").unlink()
    write_json(str(run_dir / terminal), {"status": "preempted"})
    target = tmp_path / "result.zip"
    target.write_bytes(b"old")
    with pytest.raises(ArtifactError, match="成功运行"):
        bundles.export_run_bundle(run_dir, target)
    assert target.read_bytes() == b"old"


@pytest.mark.parametrize("mutation", ["missing", "changed", "temporary", "recursive_output"])
def test_invalid_artifacts_fail_before_publishing(run_dir, tmp_path, mutation):
    target = tmp_path / "result.zip"
    target.write_bytes(b"old")
    if mutation == "missing":
        (run_dir / "模型.bin").unlink()
    elif mutation == "changed":
        (run_dir / "模型.bin").write_bytes(b"changed")
    elif mutation == "temporary":
        (run_dir / "unfinished.tmp").write_bytes(b"partial")
    else:
        target = run_dir / "result.zip"
        target.write_bytes(b"old")
    with pytest.raises(ArtifactError):
        bundles.export_run_bundle(run_dir, target)
    assert target.read_bytes() == b"old"


def test_source_mutation_during_zip_keeps_previous_bundle(run_dir, tmp_path, monkeypatch):
    target = tmp_path / "result.zip"
    target.write_bytes(b"old")
    copy = bundles.shutil.copyfileobj

    def mutate(source, destination, length):
        destination.write(b"unexpected bytes")
        copy(source, destination, length)

    monkeypatch.setattr(bundles.shutil, "copyfileobj", mutate)
    with pytest.raises(ArtifactError, match="回读校验失败"):
        bundles.export_run_bundle(run_dir, target)
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".result.zip.*.tmp"))


def test_windows_manifest_paths_are_validated(run_dir):
    (run_dir / "models").mkdir()
    path = run_dir / "models/weight.bin"
    path.write_bytes(b"abc")
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["models\\weight.bin"] = {"size": 3, "sha256": sha256_file(str(path))}
    write_json(str(run_dir / "run-manifest.json"), manifest)
    bundles.export_run_bundle(run_dir)
    path.write_bytes(b"bad")
    with pytest.raises(ArtifactError, match="models/weight.bin"):
        bundles.export_run_bundle(run_dir)


def test_bundle_cli_does_not_need_remote_or_secrets(run_dir, tmp_path):
    output = tmp_path / "result.zip"
    assert main(["bundle", "--run", str(run_dir), "--out", str(output)]) == 0
    assert output.is_file()


def test_single_machine_lifecycle_exports_and_propagates_errors(run_dir, monkeypatch):
    layout = RunLayout(str(run_dir))
    layout.ensure()
    services = LifecycleServices(
        layout=layout, secret_resolver=None, stores=[LocalArtifactStore()], async_sync=None,
        wecom_client=None, audit=ServiceAudit(layout.service_audit_jsonl, redactor=lambda value: value),
        failure_policy="record",
    )
    services.finalize_run("unit-run", "succeeded")
    assert bundles.default_bundle_path(run_dir).is_file()
    first = bundles.default_bundle_path(run_dir).read_bytes()
    services.finalize_run("unit-run", "succeeded")
    assert bundles.default_bundle_path(run_dir).read_bytes() == first

    def disk_failure(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(bundles, "write_directory_bundle", disk_failure)
    with pytest.raises(OSError, match="disk full"):
        services.finalize_run("unit-run", "succeeded")
