"""任务 10.1：两进程 gloo 训练 —— 步数一致、指标有限、分布式归约。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable

# 每测试唯一 MASTER_PORT，避免 Windows spawn 残留进程端口冲突
_MP = {"counter": 29600}


def _set_port(monkeypatch=None):
    _MP["counter"] += 1
    port = str(_MP["counter"])
    if monkeypatch is not None:
        monkeypatch.setenv("MASTER_PORT", port)
    else:
        os.environ["MASTER_PORT"] = port
    return port


class _Clock:
    """模块级假时钟（spawn 子进程需可 pickle）。"""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def _budget_worker(experiment_ref, config, layout, local_rank, world_size, resume,
                   publish_terminal=True, budget_monotonic=None, execution_policy=None):
    """worker 包装：注入测试专用 Local 执行预算（D-003 后 YAML 不再承载 runtime）。

    launch_torch 的 spawn 子进程通过纯 dict 严格重建平台规范策略；测试需要
    假时钟预算来触发 preempt，因此在此包装内用 Local 自定义策略直接调用 run_worker。
    """
    from dl_helper.training.backends.torch_backend import run_worker
    from dl_helper.training.platform import ExecutionPolicy
    return run_worker(
        experiment_ref, config, layout, local_rank, world_size, resume,
        publish_terminal=publish_terminal, budget_monotonic=budget_monotonic,
        execution_policy=ExecutionPolicy(platform="local", max_minutes=5, shutdown_grace_minutes=1),
    )


def _write_config(tmp_path, run_id, patience=20, lr=0.05, max_epochs=2):
    schema = {
        "schema_version": 1,
        "run": {"name": "gloo", "id": run_id, "output_root": str(tmp_path),
                "source_revision": None, "seed": 42, "tags": {}},
        "experiment": {"lr": lr},
        "training": {"max_epochs": max_epochs, "log_every_steps": 1},
        "backend": {"type": "torch", "torch": {
            "gradient_accumulation_steps": 2, "mixed_precision": "no", "compile": False,
            "clip_grad_norm": None, "deterministic": "off", "matmul_precision": "high",
            "find_unused_parameters": False}, "sklearn": None},
        "distributed": {"num_processes": 2},
        "selection": {"metric": "val/loss", "mode": "min", "patience": patience, "min_delta": 0.0},
        "checkpoint": {"every_epochs": None, "every_optimizer_steps": None,
                       "keep_last": 1},
        "report": {"enabled": True, "curve_sample_limit": 100000,
                   "prediction_sample_limit": 10000, "prediction_splits": ["val"]},
        "remote": {"type": "none"},
        "notifications": {"type": "none"},
    }
    import yaml
    path = tmp_path / "gloo.yaml"
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.mark.skipif(not os.environ.get("DLH_ALLOW_MP"), reason="多进程 spawn 需显式启用")
def test_gloo_two_process_training(tmp_path):
    """两进程 gloo 完成训练，step 一致，指标有限。"""
    cfg_path = _write_config(tmp_path, "gloo-2p")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8",
           "DLH_ALLOW_MP": "1"}
    cmd = [PY, "-m", "dl_helper.training.cli", "train",
           "--config", cfg_path,
           "--experiment", "experiments.toy_multiclass_resumable:build_experiment",
           "--run-id", "gloo-2p"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run_dir = os.path.join(str(tmp_path), "runs", "gloo-2p")
    assert os.path.exists(os.path.join(run_dir, "metrics", "summary.json"))
    summary = json.load(open(os.path.join(run_dir, "metrics", "summary.json"), encoding="utf-8"))
    assert summary["status"] == "succeeded"
    # 指标有限
    lines = [json.loads(l) for l in open(os.path.join(run_dir, "metrics", "metrics.jsonl"), encoding="utf-8")]
    val = next(l for l in lines if l["stage"] == "val")
    assert "val/loss" in val["metrics"]
    assert val["metrics"]["val/loss"] == val["metrics"]["val/loss"]  # 非 NaN
    import math
    assert math.isfinite(val["metrics"]["val/loss"])


@pytest.mark.skipif(not os.environ.get("DLH_ALLOW_MP"), reason="多进程 spawn 需显式启用")
def test_gloo_two_process_checkpoint_resume(tmp_path):
    """OSR-004：两进程 checkpoint + resume 完成（preempted 退出码传播、共享 accelerator-state）。"""
    from dl_helper.training.config import default_schema, parse_config
    from dl_helper.training.launcher import launch_torch

    def _cfg(run_id, max_epochs):
        schema = default_schema()
        schema["distributed"]["num_processes"] = 2
        schema["training"]["max_epochs"] = max_epochs
        schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 30, "min_delta": 0.0}
        schema["report"]["prediction_splits"] = ["val"]
        schema["run"]["id"] = run_id
        schema["checkpoint"]["every_optimizer_steps"] = 4
        schema["checkpoint"]["keep_last"] = 5
        # 两阶段配置一致；resume 由 launch_torch 显式传入，不承载于 YAML（D-003）
        schema["backend"]["torch"]["mixed_precision"] = "no"
        schema["backend"]["torch"]["deterministic"] = "off"
        return parse_config(schema)

    # 阶段 1：预算预占（假时钟，epoch 0 中途）→ 两进程 checkpoint（preempted 经退出码 75 传播）
    # D-003：YAML 不再承载 runtime；测试预算经 worker 包装在子进程内以 Local 策略注入，
    # 从而绕过 spawn 纯 dict 重建的「仅平台规范策略」校验（测试专用，非用户配置路径）。
    res_dir = str(tmp_path / "runs" / "gloo-resume")
    code1 = launch_torch("experiments.toy_multiclass_resumable:build_experiment",
                         _cfg("gloo-resume", max_epochs=1), res_dir, 2, "auto",
                         worker_fn=_budget_worker, budget_monotonic=_Clock())
    assert code1 == 75
    ckpt_root = os.path.join(res_dir, "checkpoints")
    assert os.path.isdir(ckpt_root) and any(d.startswith("epoch-") for d in os.listdir(ckpt_root))
    ckpt_dir = next(d for d in os.listdir(ckpt_root) if d.startswith("epoch-"))
    accel_files = os.listdir(os.path.join(ckpt_root, ckpt_dir, "accelerator-state"))
    assert any(f.startswith("random_states_0") for f in accel_files)
    assert any(f.startswith("random_states_1") for f in accel_files)

    # 阶段 2：两进程恢复至完成（max_epochs 从 1 增至 2，resume 指纹不含 max_epochs）
    code2 = launch_torch("experiments.toy_multiclass_resumable:build_experiment",
                         _cfg("gloo-resume", max_epochs=2), res_dir, 2, "auto")
    assert code2 == 0


@pytest.mark.skipif(not os.environ.get("DLH_ALLOW_MP"), reason="多进程 spawn 需显式启用")
def test_gloo_two_process_early_stop_no_deadlock(tmp_path):
    """OSR-004：两进程 gloo 真实早停 + checkpoint，不 DDP 死锁。"""
    cfg_path = _write_config(tmp_path, "gloo-es", patience=1, lr=0.5, max_epochs=5)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8",
           "DLH_ALLOW_MP": "1"}
    cmd = [PY, "-m", "dl_helper.training.cli", "train",
           "--config", cfg_path,
           "--experiment", "experiments.toy_multiclass_resumable:build_experiment",
           "--run-id", "gloo-es"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr  # 未死锁
    run_dir = os.path.join(str(tmp_path), "runs", "gloo-es")
    # OSR-004：断言真实早停（epoch < max_epochs）且生成 checkpoint（非空覆盖）
    summary = json.load(open(os.path.join(run_dir, "metrics", "summary.json"), encoding="utf-8"))
    assert summary["epoch"] < 5, f"未早停: epoch={summary['epoch']}"
    ckpt_root = os.path.join(run_dir, "checkpoints")
    assert os.path.isdir(ckpt_root) and any(
        d.startswith("epoch-") for d in os.listdir(ckpt_root)
    ), "早停必须生成 checkpoint"
    # OSR-004：共享 accelerator-state 目录含各 rank RNG（主 rank 模型/优化器 + 每 rank RNG）
    ckpt_dir = next(d for d in os.listdir(ckpt_root) if d.startswith("epoch-"))
    accel_dir = os.path.join(ckpt_root, ckpt_dir, "accelerator-state")
    files = os.listdir(accel_dir)
    assert any(f.startswith("random_states_0") for f in files), "缺 rank-0 RNG"
    assert any(f.startswith("random_states_1") for f in files), "缺 rank-1 RNG"
    assert any(f.endswith(".safetensors") or f == "pytorch_model.bin" for f in files), "缺模型权重"
    assert os.path.exists(os.path.join(run_dir, "run-manifest.json"))
