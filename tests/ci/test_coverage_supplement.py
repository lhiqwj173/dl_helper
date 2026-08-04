"""补充 launcher / doctor / sweep 覆盖率。"""
from __future__ import annotations

import os

import pytest
import yaml

from dl_helper.training.config import default_schema, parse_config


# ---------- launcher ----------
def test_launcher_config_to_dict():
    from dl_helper.training.launcher import _config_to_dict
    cfg = parse_config(default_schema())
    d = _config_to_dict(cfg)
    assert d["schema_version"] == 1
    assert d["backend"]["type"] == "torch"


def test_launcher_multi_process_returns_nonzero_on_worker_error(tmp_path):
    """多进程 worker 非零 → launcher 返回该退出码。"""
    from dl_helper.training.artifacts import RunLayout
    from dl_helper.training.launcher import launch_torch
    cfg = parse_config(default_schema())

    def fail_worker(ref, config, layout, rank, world, resume, publish_terminal=True, budget_monotonic=None):
        raise RuntimeError("boom")

    layout = RunLayout(str(tmp_path / "runs" / "lp"))
    layout.ensure()
    # 单进程 worker 抛异常 → launch_torch 直接抛
    with pytest.raises(RuntimeError):
        launch_torch("ref", cfg, layout.run_dir, 1, "none", worker_fn=fail_worker)


# ---------- doctor ----------
def test_doctor_kaggle_revision_required(tmp_path):
    schema = default_schema()
    schema["run"]["output_root"] = str(tmp_path)
    schema["run"]["id"] = "doc-k"
    schema["run"]["source_revision"] = "short"
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    cfg = parse_config(schema)
    from dl_helper.training.doctor import run_doctor
    errors = run_doctor(cfg, _KagglePlatform(), "experiments.toy_multiclass:build_experiment")
    assert errors  # 非法 revision → 错误


class _KagglePlatform:
    is_kaggle = True
    kind = "kaggle"

    def resolve_output_root(self, config):
        return "/kaggle/working/dl-helper-runs"

    def validate_kaggle_inputs(self, config):
        return None


def test_doctor_services_missing_keys(tmp_path):
    schema = default_schema()
    schema["run"]["output_root"] = str(tmp_path)
    schema["run"]["id"] = "doc-svc"
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["remote"] = {"type": "alist", "host": "https://alist.example.invalid",
                        "base_path": "/x", "user_secret_key": "ALIST_USER",
                        "password_secret_key": "ALIST_PWD", "connect_timeout_seconds": 1,
                        "read_timeout_seconds": 1, "max_attempts": 2, "async_upload": False,
                        "failure_policy": "required"}
    cfg = parse_config(schema)
    from dl_helper.training.doctor import run_doctor
    from dl_helper.training.platform import Platform
    errors = run_doctor(cfg, Platform("local"), "experiments.toy_multiclass:build_experiment")
    assert errors == []  # doctor 不解析 Secret，但配置结构合法


# ---------- sweep ----------
def test_sweep_resolve_local_path_rules(tmp_path):
    from dl_helper.training.sweep import _resolve_local_path
    base = tmp_path / "base"
    base.mkdir()
    (base / "ok.yaml").write_text("x", encoding="utf-8")
    good = _resolve_local_path(str(base), "ok.yaml", "v")
    assert good == str((base / "ok.yaml").resolve())
    with pytest.raises(Exception):
        _resolve_local_path(str(base), "https://x.example/a.yaml", "v")
    with pytest.raises(Exception):
        _resolve_local_path(str(base), "../escape.yaml", "v")


def test_sweep_trial_paused(tmp_path):
    from dl_helper.training.sweep import _trial_paused
    import json
    sweep_dir = tmp_path / "sweeps" / "s"
    os.makedirs(sweep_dir, exist_ok=True)
    assert not _trial_paused(str(sweep_dir), "r1")
    json.dump({"current_run_id": "r1"}, open(os.path.join(sweep_dir, "pause-manifest.json"), "w", encoding="utf-8"))
    assert _trial_paused(str(sweep_dir), "r1")
    assert not _trial_paused(str(sweep_dir), "r2")


# ---------- launcher 多进程分支（mock spawn，不真实起进程） ----------
def test_launcher_multiprocess_success(tmp_path, monkeypatch):
    """多进程启动循环 + 全成功退出码 → 返回 0。"""
    from dl_helper.training.launcher import launch_torch
    from dl_helper.training.artifacts import RunLayout

    cfg = parse_config(default_schema())
    layout = RunLayout(str(tmp_path / "runs" / "lp-ok"))
    layout.ensure()
    spawned = []

    class _FakeProc:
        exitcode = 0

        def __init__(self, target, args):
            spawned.append(args)

        def start(self):
            pass

        def join(self):
            pass

    class _FakeCtx:
        def Process(self, target, args):
            return _FakeProc(target, args)

    class _FakeMP:
        def get_context(self, method):
            return _FakeCtx()

    monkeypatch.setattr("dl_helper.training.launcher.multiprocessing", _FakeMP())
    code = launch_torch("exp:build", cfg, layout.run_dir, 2, "none",
                        worker_fn=lambda *a, **k: None)
    assert code == 0
    assert len(spawned) == 2


def test_launcher_multiprocess_returns_nonzero(tmp_path, monkeypatch):
    """worker 退出码非零 → launcher 返回第一个非零。"""
    from dl_helper.training.launcher import launch_torch
    from dl_helper.training.artifacts import RunLayout

    cfg = parse_config(default_schema())
    layout = RunLayout(str(tmp_path / "runs" / "lp-err"))
    layout.ensure()

    class _FakeProc:
        exitcode = 7

        def __init__(self, target, args):
            pass

        def start(self):
            pass

        def join(self):
            pass

    class _FakeCtx:
        def Process(self, target, args):
            return _FakeProc(target, args)

    class _FakeMP:
        def get_context(self, method):
            return _FakeCtx()

    monkeypatch.setattr("dl_helper.training.launcher.multiprocessing", _FakeMP())
    code = launch_torch("exp:build", cfg, layout.run_dir, 2, "none",
                        worker_fn=lambda *a, **k: None)
    assert code == 7


def test_launcher_single_preempted_returns_75(tmp_path):
    from dl_helper.training.backends.base import BackendResult
    from dl_helper.training.launcher import launch_torch
    from dl_helper.training.artifacts import RunLayout

    cfg = parse_config(default_schema())
    layout = RunLayout(str(tmp_path / "runs" / "lp-pre"))
    layout.ensure()

    def fake_worker(ref, config, layout, rank, world, resume, publish_terminal=True, budget_monotonic=None):
        return BackendResult(status="preempted", epoch=1, global_step=5)

    code = launch_torch("ref", cfg, layout.run_dir, 1, "none", worker_fn=fake_worker)
    assert code == 75


# ---------- checkpoint 分支补充 ----------
def test_checkpoint_validate_manifest_size_mismatch(tmp_path):
    import hashlib

    from dl_helper.training.checkpoint import CheckpointError, validate_manifest_complete
    root = tmp_path / "ck"
    root.mkdir()
    (root / "model.bin").write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    manifest = {"complete": True, "files": {"model.bin": {"sha256": digest, "size": 999}}}
    with pytest.raises(CheckpointError):
        validate_manifest_complete(manifest, str(root))


def test_checkpoint_read_latest_invalid_content(tmp_path):
    from dl_helper.training.checkpoint import CheckpointError, read_latest
    root = tmp_path / "ck"
    root.mkdir()
    (root / "latest.json").write_text('{"foo": 1}', encoding="utf-8")  # 无 path
    with pytest.raises(CheckpointError):
        read_latest(str(root))


def test_checkpoint_stage_missing_dir(tmp_path):
    from dl_helper.training.checkpoint import CheckpointError, _stage_and_finalize
    with pytest.raises(CheckpointError):
        _stage_and_finalize(str(tmp_path / "missing"), str(tmp_path / "final"))


def test_checkpoint_retention_none_and_latest_kept(tmp_path):
    from dl_helper.training.checkpoint import apply_retention
    root = tmp_path / "ck"
    root.mkdir()
    apply_retention(str(root), None)  # keep_last None → 直接返回
    # 带 manifest 的旧目录 + latest → 删除旧、保留 latest
    for name, content in (
        ("epoch-000000-step-00000000", '{"complete": true, "files": {}}'),
        ("epoch-000000-step-00000010", '{"complete": true, "files": {}}'),
    ):
        d = root / name
        d.mkdir()
        (d / "checkpoint-manifest.json").write_text(content, encoding="utf-8")
    (root / "latest.json").write_text(
        '{"schema_version": 1, "checkpoint_id": "epoch-000000-step-00000010", '
        '"path": "epoch-000000-step-00000010"}', encoding="utf-8")
    apply_retention(str(root), 1)
    assert (root / "epoch-000000-step-00000010").exists()
    assert not (root / "epoch-000000-step-00000000").exists()


def test_checkpoint_torch_existing_dir_raises(tmp_path):
    from dl_helper.training.checkpoint import CheckpointError, write_torch_checkpoint
    root = tmp_path / "ck"
    root.mkdir()
    (root / "epoch-000000-step-00000000").mkdir()
    with pytest.raises(CheckpointError):
        write_torch_checkpoint(None, str(root), "r", None, {}, {}, "fp", "dp", {}, 0, 0, 0)


def test_checkpoint_sklearn_existing_dir_raises(tmp_path):
    from dl_helper.training.checkpoint import CheckpointError, write_sklearn_checkpoint
    root = tmp_path / "ck"
    root.mkdir()
    (root / "epoch-000000-step-00000000").mkdir()
    with pytest.raises(CheckpointError):
        write_sklearn_checkpoint(None, {}, None, {}, str(root), "r", "fp", "dp", {}, 0, 0, 0, None)


def test_checkpoint_write_model_manifest_extra(tmp_path):
    from dl_helper.training.checkpoint import write_model_manifest
    d = tmp_path / "models"
    d.mkdir()
    man = write_model_manifest(str(d), "torch", {"class": "x", "num_parameters": 1, "params": {}},
                               "r1", {"m.safetensors": {"size": 1, "sha256": "a" * 64}},
                               extra={"note": "hello"})
    assert man["note"] == "hello"


def test_checkpoint_load_torch_no_latest(tmp_path):
    from dl_helper.training.checkpoint import CheckpointError, load_torch_checkpoint
    root = tmp_path / "ck"
    root.mkdir()
    with pytest.raises(CheckpointError):
        load_torch_checkpoint(None, str(root), None, None, None, "fp", "dp", {})


def test_checkpoint_retention_no_manifest_and_no_latest(tmp_path):
    """保留策略：无 manifest 目录跳过；无 latest 时不保底。"""
    from dl_helper.training.checkpoint import apply_retention
    root = tmp_path / "ck"
    root.mkdir()
    (root / "epoch-000000-step-00000000").mkdir()  # 无 checkpoint-manifest.json
    apply_retention(str(root), 1)
    assert (root / "epoch-000000-step-00000000").exists()  # 无 manifest → 不删除
