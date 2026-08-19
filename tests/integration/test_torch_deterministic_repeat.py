"""任务 OSR-006：strict 确定性 + 固定 seed → 两次子进程运行结果一致。"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cfg(tmp_path, run_id):
    schema = {
        "schema_version": 1,
        "run": {"name": "det", "id": run_id, "output_root": str(tmp_path),
                "source_revision": None, "seed": 42, "tags": {}},
        "experiment": {"lr": 0.05},
        "training": {"max_epochs": 2, "log_every_steps": 20},
        "backend": {"type": "torch", "torch": {
            "gradient_accumulation_steps": 1, "mixed_precision": "no", "compile": False,
            "clip_grad_norm": 1.0, "deterministic": "strict", "matmul_precision": "high",
            "find_unused_parameters": False}, "sklearn": None},
        "distributed": {"num_processes": 1},
        "selection": {"metric": "val/loss", "mode": "min", "patience": 20, "min_delta": 0.0},
        "checkpoint": {"every_epochs": None, "every_optimizer_steps": None,
                       "keep_last": 1},
        "report": {"enabled": True, "curve_sample_limit": 100000,
                   "prediction_sample_limit": 10000, "prediction_splits": ["val"]},
        "remote": {"type": "none"},
        "notifications": {"type": "none"},
    }
    path = tmp_path / "det.yaml"
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    return str(path)


def _run(tmp_path, run_id):
    cfg = _cfg(tmp_path, run_id)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "dl_helper.training.cli", "train",
           "--config", cfg, "--experiment", "experiments.toy_multiclass:build_experiment"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          env=env, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    model_path = os.path.join(str(tmp_path), "runs", run_id, "models", "last", "model.safetensors")
    with open(model_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_deterministic_repeat_identical(tmp_path):
    h1 = _run(tmp_path, "det-rep-1")
    h2 = _run(tmp_path, "det-rep-2")
    assert h1 == h2, "strict 确定性下两次运行模型权重应一致"
