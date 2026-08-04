"""任务 OSR-006：运行时控制 —— seed/deterministic/matmul/compile 应用于实际训练。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import (
    TorchBackendError,
    _apply_runtime_controls,
    run_worker,
)
from dl_helper.training.config import default_schema, parse_config


def _cfg(run_id, max_epochs=1, deterministic="off", seed=42, compile_=False, matmul="high"):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 20, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["run"]["seed"] = seed
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = deterministic
    schema["backend"]["torch"]["compile"] = compile_
    schema["backend"]["torch"]["matmul_precision"] = matmul
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def test_seed_controls_model_init():
    """相同 seed → 相同初始模型；不同 seed → 不同。"""
    import torch

    cfg1 = _cfg("seed-a", seed=42)
    cfg2 = _cfg("seed-b", seed=43)
    _apply_runtime_controls(cfg1)
    m1 = torch.nn.Linear(4, 3)
    w1 = m1.weight.clone()
    _apply_runtime_controls(cfg2)
    m2 = torch.nn.Linear(4, 3)
    w2 = m2.weight.clone()
    _apply_runtime_controls(cfg1)
    m3 = torch.nn.Linear(4, 3)
    w3 = m3.weight.clone()
    assert torch.equal(w1, w3)  # 同 seed 同初始化
    assert not torch.equal(w1, w2)  # 不同 seed 不同初始化


def test_deterministic_strict_applies():
    import torch

    cfg = _cfg("det-strict", deterministic="strict")
    _apply_runtime_controls(cfg)
    assert torch.are_deterministic_algorithms_enabled()


def test_deterministic_off_does_not_enable():
    import torch

    cfg = _cfg("det-off", deterministic="off")
    _apply_runtime_controls(cfg)
    assert not torch.are_deterministic_algorithms_enabled()


def test_compile_run(tmp_path):
    cfg = _cfg("compile-run", max_epochs=1, compile_=True)
    layout = RunLayout(str(tmp_path / "runs" / "compile-run"))
    layout.ensure()
    try:
        result = run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
        assert result.status == "succeeded"
    except TorchBackendError:
        pytest.skip("torch.compile 当前环境不支持")


def test_matmul_precision_recorded(tmp_path):
    cfg = _cfg("matmul-run", max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "matmul-run"))
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    env = json.load(open(layout.environment_json, encoding="utf-8"))
    assert env  # environment manifest 已写


def test_runtime_environment_records_resources(tmp_path):
    """OSR-006：environment 记录 seed/确定性/matmul/compile/loader 资源。"""
    cfg = _cfg("env-res", max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "env-res"))
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    env = json.load(open(layout.environment_json, encoding="utf-8"))
    assert env["seed"] == cfg.run.seed
    assert env["torch"]["deterministic"] == "off"
    assert env["torch"]["matmul_precision"] == "high"
    assert env["torch"]["compile"] is False
    res = env["resources"]
    assert "num_processes" in res and "mixed_precision" in res
    assert "num_workers" in res and "pin_memory" in res
    assert "persistent_workers" in res and "prefetch_factor" in res
    assert "effective_batch_size" in res
    assert "loader_resources_applied" in res


def test_resumable_dm_loader_resources_applied(tmp_path):
    """OSR-006：可恢复 DataModule 应用解析后的 loader 资源。"""
    cfg = _cfg("env-res2", max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "env-res2"))
    layout.ensure()
    run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, layout, 0, 1, "none")
    env = json.load(open(layout.environment_json, encoding="utf-8"))
    assert env["resources"]["loader_resources_applied"] is True
    # 资源合同不按数据集大小静默改写。
    expected_workers = min(8, max(1, (os.cpu_count() or 1) // cfg.distributed.num_processes))
    assert env["resources"]["num_workers"] == expected_workers


def test_resolved_config_records_resources(tmp_path):
    """OSR-006：解析后的资源写入独立 resolved-resources.json，config.resolved.yaml 可严格重放。"""
    import yaml

    from dl_helper.training.config import parse_config

    cfg = _cfg("env-res3", max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "env-res3"))
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    # config.resolved.yaml 仍是合法配置（严格解析器可重放）
    resolved_yaml = open(layout.path("config.resolved.yaml"), encoding="utf-8").read()
    replayed = parse_config(yaml.safe_load(resolved_yaml))
    assert replayed.run.seed == cfg.run.seed
    # 资源记录在独立文件
    import json as _json
    rr = _json.load(open(layout.path("resolved-resources.json"), encoding="utf-8"))
    assert rr["num_processes"] == 1
    assert "effective_batch_size" in rr
    assert "num_workers" in rr


def test_loader_datamodule_records_applied(tmp_path):
    """OSR-006：LoaderDataModule 重建 DataLoader 并记录实际应用资源。"""
    cfg = _cfg("env-ldm", max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "env-ldm"))
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    env = json.load(open(layout.environment_json, encoding="utf-8"))
    assert env["resources"]["loader_resources_applied"] is True


def test_dynamic_batch_stats_recorded(tmp_path):
    """OSR-006：nominal 未知（dynamic）时记录实际 batch 范围。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
    from dl_helper.training.task import MulticlassClassificationTask

    exp = tmp_path / "dyn_exp.py"
    exp.write_text('''
def build_experiment(config):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
    from dl_helper.training.task import MulticlassClassificationTask

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            return self.fc(x)

    def model_factory():
        return M()

    def datamodule_factory():
        g = torch.Generator().manual_seed(3)
        x = torch.randn(130, 8, generator=g)
        y = torch.randint(0, 3, (130,), generator=g)
        train_ds = TensorDataset(x[:100], y[:100])
        val_ds = TensorDataset(x[100:], y[100:])
        loader = DataLoader(train_ds, batch_size=16)  # 100%16 → 最后 batch=4
        val_loader = DataLoader(val_ds, batch_size=16)
        return LoaderDataModule(DataIdentity("dyn-exp", "1", "fp"), loader,
                                val_dataloader=val_loader, nominal_train_batch_size=None)

    def task_factory():
        return MulticlassClassificationTask(num_classes=3)

    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=0.05)

    return TorchExperiment(name="dyn-exp", backend="torch",
                           model_factory=model_factory,
                           datamodule_factory=datamodule_factory,
                           task_factory=task_factory,
                           optimizer_factory=optimizer_factory,
                           scheduler_factory=lambda o: None,
                           model_config=dict(config))
''', encoding="utf-8")

    cfg = _cfg("env-dyn", max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "env-dyn"))
    layout.ensure()
    import sys as _sys
    _sys.path.insert(0, str(tmp_path))
    try:
        run_worker("dyn_exp:build_experiment", cfg, layout, 0, 1, "none")
    finally:
        _sys.path.remove(str(tmp_path))
    env = json.load(open(layout.environment_json, encoding="utf-8"))
    assert env["resources"]["effective_batch_size"] == "dynamic"
    stats = env["resources"].get("batch_stats")
    assert stats and stats["min_batch"] == 4 and stats["max_batch"] == 16
    assert stats["num_batches"] == 7  # 100 // 16 = 6 + 1 尾批
