"""任务 OSR-005：已完成 run 不可变 —— 重跑拒绝、终态内容不被改写。"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import TorchBackendError, run_worker
from dl_helper.training.config import default_schema, parse_config


def _cfg(run_id, max_epochs=1):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 20, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def test_completed_run_rerun_rejected(tmp_path):
    run_dir = str(tmp_path / "runs" / "immutable")
    cfg = _cfg("immutable")
    layout = RunLayout(run_dir)
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    manifest_path = layout.path("run-manifest.json")
    original = json.load(open(manifest_path, encoding="utf-8"))
    # 二次运行被拒绝
    with pytest.raises(TorchBackendError):
        run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    # 终态内容未改写
    after = json.load(open(manifest_path, encoding="utf-8"))
    assert after == original


def test_completed_run_checksums_stable(tmp_path):
    run_dir = str(tmp_path / "runs" / "immutable2")
    cfg = _cfg("immutable2")
    layout = RunLayout(run_dir)
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    manifest = json.load(open(layout.path("run-manifest.json"), encoding="utf-8"))
    # 全部 artifacts 与 checksum 一致
    for rel, meta in manifest["artifacts"].items():
        full = os.path.join(layout.run_dir, rel)
        assert os.path.exists(full), f"artifact 缺失: {rel}"
        assert os.path.getsize(full) == meta["size"]
        with open(full, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == meta["sha256"]


def test_cli_rejects_completed_run_before_writing(tmp_path):
    """OSR-005：CLI 在任何写入前拒绝已完成 run，config.resolved.yaml 不被改写。"""
    import yaml

    from dl_helper.training.cli import CliError, main

    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = "cli-immutable"
    schema["run"]["output_root"] = str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    cfg = tmp_path / "base.yaml"
    cfg.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")

    assert main(["train", "--config", str(cfg),
                 "--experiment", "experiments.toy_multiclass:build_experiment"]) == 0
    resolved = os.path.join(str(tmp_path), "runs", "cli-immutable", "config.resolved.yaml")
    original = hashlib.sha256(open(resolved, "rb").read()).hexdigest()

    with pytest.raises(CliError):
        main(["train", "--config", str(cfg),
              "--experiment", "experiments.toy_multiclass:build_experiment"])
    # config.resolved.yaml 未被改写（预写入拒绝）
    assert hashlib.sha256(open(resolved, "rb").read()).hexdigest() == original
    # 终态内容仍为首次运行产物
    manifest = json.load(open(os.path.join(str(tmp_path), "runs", "cli-immutable",
                                           "run-manifest.json"), encoding="utf-8"))
    assert manifest["status"] == "succeeded"
