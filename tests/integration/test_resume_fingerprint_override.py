"""显式临时指纹跳过须贯通真实恢复，且不能跳过完整性或模型签名。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends import torch_backend
from dl_helper.training.checkpoint import CheckpointError
from dl_helper.training.checkpoint_archive import select_checkpoint_source
from dl_helper.training.cli import main, CliError
from dl_helper.training.config import config_to_dict, config_fingerprint, load_config_file
from dl_helper.training.platform import Platform
from test_torch_checkpoint_resume import _cfg, _AdvancingClock, _BUDGET


@pytest.fixture(autouse=True)
def single_threaded_loaders(monkeypatch):
    resolve = Platform.resolve_torch_resources

    def local_loaders(self, config, nominal_batch):
        return replace(resolve(self, config, nominal_batch), num_workers=0,
                       persistent_workers=False, prefetch_factor=None)

    monkeypatch.setattr(Platform, "resolve_torch_resources", local_loaders)


@pytest.fixture
def archive_case(tmp_path, request):
    config = _cfg("override-resume", 2)
    config = replace(config, run=replace(config.run, source_revision="stable-source",
                                        output_root=str(tmp_path / "output")))
    reference = RunLayout(str(tmp_path / "reference"))
    reference.ensure()
    torch_backend.run_worker("experiments.toy_multiclass_resumable:build_experiment", config, reference)
    partial = RunLayout(str(tmp_path / "partial"))
    partial.ensure()
    if getattr(request, "param", True):
        partial.write_text("dataset-files.json", json.dumps({
            "schema_version": 1, "files": ["1/a.png", "2/b.png"],
            "train_files": ["1/a.png"], "validation_files": ["2/b.png"]}) + "\n")
    result = torch_backend.run_worker("experiments.toy_multiclass_resumable:build_experiment", config,
                                      partial, budget_monotonic=_AdvancingClock(), execution_policy=_BUDGET)
    assert result.status == "preempted"
    archive = Path(partial.path("checkpoint-archives", "last-checkpoint.zip"))
    moved_config = replace(config, experiment={**config.experiment, "data_dir": "/relocated/dataset"})
    config_path = tmp_path / "moved.yaml"
    config_path.write_text(yaml.safe_dump(config_to_dict(moved_config)), encoding="utf-8")
    argv = ["train", "--config", str(config_path), "--experiment",
            "experiments.toy_multiclass_resumable:build_experiment", "--no-use-alist",
            "--resume", "required", "--checkpoint-input", str(archive)]
    return config, moved_config, reference, archive, argv


def test_explicit_override_resumes_and_exports_strictly_valid_checkpoint(archive_case, monkeypatch):
    old, current, reference, archive, argv = archive_case
    original_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="不兼容"):
        main(argv)
    monkeypatch.setattr(torch_backend, "data_fingerprint", lambda datamodule: "b" * 64)
    assert main([*argv, "--resume-allow-fingerprint-mismatch"]) == 0
    resumed = Path(current.run.output_root) / "runs" / current.run.id
    for name in ("best", "last"):
        expected = load_file(reference.path("models", name, "model.safetensors"))
        actual = load_file(str(resumed / "models" / name / "model.safetensors"))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    final_archive = resumed / "checkpoint-archives" / "last-checkpoint.zip"
    # 新归档可重新通过默认严格选择，不能把跳过开关变成永久依赖。
    saved_config = load_config_file(str(resumed / "config.resolved.yaml"))
    candidate = select_checkpoint_source([final_archive], resumed / "unused", saved_config)
    assert candidate.manifest["config_fingerprint"] == config_fingerprint(saved_config, resume=True)
    assert candidate.manifest["data_fingerprint"] == "b" * 64
    policy = json.loads((resumed / "execution-policy.json").read_text(encoding="utf-8"))
    assert policy["resume_allow_fingerprint_mismatch"] is True
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == original_sha


def test_override_still_rejects_incompatible_model(archive_case, monkeypatch):
    old, current, reference, archive, argv = archive_case
    original_signature = torch_backend.model_signature

    def changed_signature(model):
        return {**original_signature(model), "class": "incompatible.Model"}

    monkeypatch.setattr(torch_backend, "model_signature", changed_signature)
    with pytest.raises(CheckpointError, match="模型签名"):
        main([*argv, "--resume-allow-fingerprint-mismatch"])


def test_override_still_rejects_corrupt_archive(archive_case, tmp_path):
    from zipfile import ZipFile

    old, current, reference, archive, argv = archive_case
    expanded = tmp_path / "expanded"
    with ZipFile(archive) as source:
        source.extractall(expanded)
    config_path = expanded / "run" / "config.resolved.yaml"
    config_path.write_text("corrupted", encoding="utf-8")
    argv[-1] = str(expanded)
    with pytest.raises(ValueError, match="校验"):
        main([*argv, "--resume-allow-fingerprint-mismatch"])


def test_override_requires_explicit_required_resume(archive_case):
    old, current, reference, archive, argv = archive_case
    argv[argv.index("--resume") + 1] = "none"
    with pytest.raises(CliError, match="required"):
        main([*argv, "--resume-allow-fingerprint-mismatch"])
    changed_revision = replace(current, run=replace(current.run, source_revision="other-source"))
    with pytest.raises(ValueError, match="不兼容"):
        select_checkpoint_source([archive], Path(current.run.output_root) / "absent", changed_revision,
                                 allow_fingerprint_mismatch=True)


def test_dataset_files_roundtrip_and_changed_names_fail(archive_case, tmp_path):
    from zipfile import ZipFile

    old, current, reference, archive, argv = archive_case
    manifest = {"schema_version": 1, "files": ["1/c.png", "2/b.png"],
                "train_files": ["1/c.png"], "validation_files": ["2/b.png"]}
    names_path = tmp_path / "dataset-files.json"
    names_path.write_text(json.dumps(manifest), encoding="utf-8")
    command = [*argv, "--resume-allow-fingerprint-mismatch", "--dataset-file-list", str(names_path)]
    with pytest.raises(ValueError, match="数据文件清单与检查点不同"):
        main(command)
    manifest["files"] = ["1/a.png", "2/b.png"]
    manifest["train_files"] = ["1/a.png"]
    names_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert main(command) == 0
    exported = Path(current.run.output_root) / "runs" / current.run.id / "checkpoint-archives/last-checkpoint.zip"
    with ZipFile(exported) as saved:
        assert json.loads(saved.read("run/dataset-files.json").decode("utf-8")) == manifest


@pytest.mark.parametrize("archive_case", [False], indirect=True)
def test_legacy_checkpoint_without_file_list_adds_it_on_next_save(archive_case, tmp_path):
    from zipfile import ZipFile

    old, current, reference, archive, argv = archive_case
    with ZipFile(archive) as original:
        assert "run/dataset-files.json" not in original.namelist()
    manifest = {"schema_version": 1, "files": ["1/a.png", "2/b.png"],
                "train_files": ["1/a.png"], "validation_files": ["2/b.png"]}
    names = tmp_path / "dataset-files.json"
    names.write_text(json.dumps(manifest), encoding="utf-8")
    assert main([*argv, "--resume-allow-fingerprint-mismatch", "--dataset-file-list", str(names)]) == 0
    exported = Path(current.run.output_root) / "runs" / current.run.id / "checkpoint-archives/last-checkpoint.zip"
    with ZipFile(exported) as saved:
        assert json.loads(saved.read("run/dataset-files.json").decode("utf-8")) == manifest
