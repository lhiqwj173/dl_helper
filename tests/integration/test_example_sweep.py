"""任务 8.4：同一实验多 variant sweep 示例端到端。"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
import yaml

from dl_helper.training.sweep import run_sweep

SWEEP = os.path.join("configs", "sweeps", "toy-learning-rate")


@pytest.mark.slow
def test_example_sweep_completes(tmp_path):
    """sweep 顺序完成、稳定 best、聚合报告与 manifest。"""
    out_root = str(tmp_path / "out")
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    # 复制 base + variants，改写 output_root
    base_data = yaml.safe_load(open(os.path.join(SWEEP, "base.yaml"), encoding="utf-8"))
    base_data["run"]["output_root"] = out_root
    yaml.safe_dump(base_data, open(os.path.join(sweep_dir, "base.yaml"), "w", encoding="utf-8"),
                   allow_unicode=True)
    for v in ("lr-1e-2.yaml", "lr-3e-3.yaml"):
        shutil.copy(os.path.join(SWEEP, "variants", v), os.path.join(sweep_dir, v))
    manifest = {
        "schema_version": 1,
        "sweep": {
            "id": "toy-lr-sweep-test",
            "experiment": "experiments.toy_multiclass:build_experiment",
            "base_config": "./base.yaml",
            "comparison_metric": "val/loss",
            "mode": "min",
            "trials": [
                {"name": "lr-1e-2", "variant": "./lr-1e-2.yaml"},
                {"name": "lr-3e-3", "variant": "./lr-3e-3.yaml"},
            ],
        },
    }
    manifest_path = os.path.join(sweep_dir, "sweep.yaml")
    yaml.safe_dump(manifest, open(manifest_path, "w", encoding="utf-8"), allow_unicode=True)

    code = run_sweep(manifest_path)
    assert code == 0
    sd = os.path.join(out_root, "sweeps", "toy-lr-sweep-test")
    assert os.path.exists(os.path.join(sd, "sweep-manifest.json"))
    assert os.path.exists(os.path.join(sd, "best-trial.json"))
    m = json.load(open(os.path.join(sd, "sweep-manifest.json"), encoding="utf-8"))
    assert len(m["ranking"]) == 2
    assert m["best_trial"] in ("lr-1e-2", "lr-3e-3")
