"""早停终点跨会话恢复应直接收尾，不重复拟合、验证或检查点提交。"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends import torch_backend
from dl_helper.training.checkpoint import read_latest
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.contracts import ResumableMapDataModule
from dl_helper.training.platform import ExecutionPolicy
from experiments import toy_multiclass_resumable


@pytest.fixture(autouse=True)
def single_process_loader(monkeypatch):
    """小型状态恢复测试不创建 Windows DataLoader 子进程。"""
    original = ResumableMapDataModule.configure_resources

    def configure(datamodule, **requested):
        return original(datamodule, num_workers=0, pin_memory=requested["pin_memory"],
                        persistent_workers=False, prefetch_factor=None)

    monkeypatch.setattr(ResumableMapDataModule, "configure_resources", configure)


def _config(patience):
    schema = default_schema()
    schema["run"]["id"] = "early-stop-resume"
    schema["training"]["max_epochs"] = 5
    schema["selection"] = {
        "metric": "val/loss", "mode": "min", "patience": patience,
        "min_delta": 100.0,
    }
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    schema["report"]["prediction_splits"] = ["val", "test"]
    return parse_config(schema)


def _checkpoint_files(layout):
    root = Path(layout.path("checkpoints"))
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("patience", [0, 1])
def test_cross_session_early_stop_resume_only_finalizes(tmp_path, monkeypatch, patience):
    original_builder = toy_multiclass_resumable.build_experiment

    def build_with_test(config):
        experiment = original_builder(config)
        original_factory = experiment.datamodule_factory

        def datamodule_factory():
            datamodule = original_factory()
            # 本测试复用验证数据来检查 test 收尾，不评估模型泛化。
            datamodule._test_dataset = datamodule._val_dataset
            return datamodule

        return replace(experiment, datamodule_factory=datamodule_factory)

    monkeypatch.setattr(toy_multiclass_resumable, "build_experiment", build_with_test)
    config = _config(patience)
    reference = "experiments.toy_multiclass_resumable:build_experiment"
    original = RunLayout(str(tmp_path / "original"))
    original.ensure()
    first = torch_backend.run_worker(reference, config, original, 0, 1, "none")
    assert first.status == "succeeded"
    assert first.epoch == patience
    assert first.global_step == (patience + 1) * 8
    checkpoints = _checkpoint_files(original)

    # 模拟 AList 恢复：新 Session 只有检查点，没有成功终态或根目录历史。
    restored = RunLayout(str(tmp_path / "restored"))
    restored.ensure()
    shutil.copytree(original.path("checkpoints"), restored.path("checkpoints"), dirs_exist_ok=True)
    latest = read_latest(restored.path("checkpoints"))
    checkpoint = Path(restored.path("checkpoints", latest["path"]))
    saved_engine = json.loads((checkpoint / "engine-state.json").read_text(encoding="utf-8"))
    assert saved_engine["batch_in_epoch"] == 8
    assert saved_engine["no_improve"] == patience

    def reject_repeated_transition(*args, **kwargs):
        raise AssertionError("早停恢复不得重复验证选择或保存检查点")

    monkeypatch.setattr(torch_backend, "_apply_selection", reject_repeated_transition)
    monkeypatch.setattr(torch_backend, "_save_torch_checkpoint", reject_repeated_transition)
    evaluated_stages = []
    original_evaluate = torch_backend._evaluate

    def evaluate(*args, **kwargs):
        evaluated_stages.append(args[4])
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(torch_backend, "_evaluate", evaluate)
    resumed = torch_backend.run_worker(reference, config, restored, 0, 1, "required")
    assert resumed.status == "succeeded"
    assert (resumed.epoch, resumed.global_step, resumed.batch_in_epoch) == (
        first.epoch, first.global_step, first.batch_in_epoch)
    assert evaluated_stages == ["test"]
    assert _checkpoint_files(restored) == checkpoints
    for filename in ("metrics/metrics.jsonl", "metrics/summary.json"):
        original_records = Path(original.path(filename)).read_text(encoding="utf-8")
        restored_records = Path(restored.path(filename)).read_text(encoding="utf-8")
        if filename.endswith("jsonl"):
            before = [json.loads(line) for line in original_records.splitlines()]
            after = [json.loads(line) for line in restored_records.splitlines()]
            assert [(r["stage"], r["epoch"], r["global_step"], r["metrics"]) for r in after] == [
                (r["stage"], r["epoch"], r["global_step"], r["metrics"]) for r in before]
        else:
            before, after = json.loads(original_records), json.loads(restored_records)
            assert after["selection"] == before["selection"]
            assert after["stage_metrics"] == before["stage_metrics"]
    for kind in ("best", "last"):
        before = load_file(original.path("models", kind, "model.safetensors"))
        after = load_file(restored.path("models", kind, "model.safetensors"))
        assert all(torch.equal(before[key], after[key]) for key in before)
    assert Path(restored.report_index).is_file()
    assert Path(restored.path("run-manifest.json")).is_file()


def test_patience_zero_mid_epoch_resume_still_runs_first_validation(tmp_path):
    """patience=0 的初始中途检查点没有 best_value，不能误判为早停终点。"""
    layout = RunLayout(str(tmp_path / "mid-epoch"))
    layout.ensure()
    config = _config(0)
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return calls * 100.0

    reference = "experiments.toy_multiclass_resumable:build_experiment"
    paused = torch_backend.run_worker(
        reference, config, layout, 0, 1, "none", budget_monotonic=clock,
        execution_policy=ExecutionPolicy(
            platform="local", max_minutes=10.0, shutdown_grace_minutes=2.0),
    )
    assert paused.status == "preempted"
    assert 0 < paused.global_step < 8
    resumed = torch_backend.run_worker(reference, config, layout, 0, 1, "required")
    assert resumed.status == "succeeded"
    assert resumed.global_step == 8
    summary = json.loads(Path(layout.summary_json).read_text(encoding="utf-8"))
    assert summary["selection"]["best_epoch"] == 0
    assert summary["selection"]["best_value"] is not None
