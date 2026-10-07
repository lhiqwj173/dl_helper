"""任务 4.3/OSR-004：Torch 连续/恢复 step、权重、指标一致。"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import ExecutionPolicy, Platform

_BUDGET = ExecutionPolicy(platform="local", max_minutes=10.0, shutdown_grace_minutes=2.0)


def _cfg(run_id, max_epochs):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 30, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = 1
    schema["checkpoint"]["keep_last"] = 2
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


@pytest.fixture(autouse=True)
def cpu_test_loaders(monkeypatch):
    resolve = Platform.resolve_torch_resources

    def single_threaded(self, config, nominal_batch):
        return replace(resolve(self, config, nominal_batch), num_workers=0,
                       persistent_workers=False, prefetch_factor=None)

    monkeypatch.setattr(Platform, "resolve_torch_resources", single_threaded)


class _AdvancingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def test_torch_resume_reaches_same_final_position(tmp_path):
    """第一段 PREEMPTED 生成检查点，第二段 resume 到最终位置。"""
    run_dir = str(tmp_path / "runs" / "resume-pos")
    cfg1 = _cfg("resume-pos", max_epochs=2)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    r1 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg1, layout1, 0, 1, "auto",
                    budget_monotonic=_AdvancingClock(), execution_policy=_BUDGET)
    assert r1.status == "preempted"

    cfg2 = _cfg("resume-pos", max_epochs=4)
    layout2 = RunLayout(run_dir)
    layout2.ensure()
    r2 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg2, layout2, 0, 1, "auto")
    assert r2.status == "succeeded"
    assert r2.epoch == 4
    summary = json.load(open(layout2.summary_json, encoding="utf-8"))
    assert summary["epoch"] == 4


def test_torch_no_resume_starts_fresh(tmp_path):
    """resume=auto 但无 checkpoint → 从零开始。"""
    run_dir = str(tmp_path / "runs" / "fresh")
    cfg = _cfg("fresh", max_epochs=1)
    layout = RunLayout(run_dir)
    layout.ensure()
    r = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, layout, 0, 1, "auto")
    assert r.epoch == 1
    assert os.path.exists(layout.path("checkpoints", "latest.json"))


def test_zip_resume_matches_continuous_training(tmp_path):
    """真实 ZIP 迁移到新 run 目录，比较训练参数、优化器状态和逐轮指标。"""
    from pathlib import Path
    import torch
    import yaml
    from safetensors.torch import load_file
    from dl_helper.training.config import config_to_dict
    from dl_helper.training.checkpoint_archive import select_checkpoint_source, restore_checkpoint_source

    cfg = _cfg("zip-numeric", max_epochs=2)
    reference = RunLayout(str(tmp_path / "continuous"))
    reference.ensure()
    run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, reference, resume="auto")

    interrupted = RunLayout(str(tmp_path / "interrupted"))
    interrupted.ensure()
    partial = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, interrupted, resume="auto",
                         budget_monotonic=_AdvancingClock(), execution_policy=_BUDGET)
    assert partial.status == "preempted"
    archive = Path(interrupted.path("checkpoint-archives", "last-checkpoint.zip"))
    assert archive.is_file()

    resumed = RunLayout(str(tmp_path / "resumed"))
    resumed.ensure()
    cfg_path = Path(resumed.path("config.resolved.yaml"))
    cfg_path.write_text(yaml.safe_dump(config_to_dict(cfg)), encoding="utf-8")
    candidate = select_checkpoint_source([archive], Path(resumed.run_dir), cfg)
    restore_checkpoint_source(candidate, Path(resumed.run_dir), cfg_path, cfg)
    result = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, resumed, resume="required")
    assert result.global_step == 16 and result.epoch == 2
    assert Path(resumed.path("checkpoint-archives", "last-checkpoint.zip")).is_file()

    for name in ("models/last/model.safetensors", "models/best/model.safetensors"):
        expected = load_file(reference.path(*name.split("/")))
        actual = load_file(resumed.path(*name.split("/")))
        assert set(expected) == set(actual)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def checkpoint_state(layout):
        latest = json.loads(Path(layout.path("checkpoints", "latest.json")).read_text(encoding="utf-8"))
        return Path(layout.path("checkpoints", latest["path"]))

    expected_dir, actual_dir = checkpoint_state(reference), checkpoint_state(resumed)
    expected_optimizer = torch.load(expected_dir / "accelerator-state" / "optimizer.bin", weights_only=False)
    actual_optimizer = torch.load(actual_dir / "accelerator-state" / "optimizer.bin", weights_only=False)
    assert expected_optimizer == actual_optimizer
    expected_engine = json.loads((expected_dir / "engine-state.json").read_text(encoding="utf-8"))
    actual_engine = json.loads((actual_dir / "engine-state.json").read_text(encoding="utf-8"))
    assert expected_engine == actual_engine
    for path in ("datamodule-state.json", "metric-states.json"):
        assert json.loads((expected_dir / path).read_text(encoding="utf-8")) == json.loads((actual_dir / path).read_text(encoding="utf-8"))
    def records(layout):
        rows = [json.loads(line) for line in Path(layout.path("metrics", "metrics.jsonl")).read_text(
            encoding="utf-8").splitlines()]
        for row in rows:
            row.pop("computed_utc")
        return rows

    assert records(reference) == records(resumed)
