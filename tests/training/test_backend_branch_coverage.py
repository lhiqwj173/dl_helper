"""补充 launcher / sklearn_backend / torch_backend 分支覆盖。"""
from __future__ import annotations

import os

import numpy as np
import pytest

from dl_helper.training.config import default_schema, parse_config


def _skl_cfg(**patch):
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096,
                                     "n_jobs": None, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = None
    for k, v in patch.items():
        schema[k] = v
    return parse_config(schema)


def test_launcher_spawn_entry_in_process(tmp_path, monkeypatch):
    """_spawn_entry 在进程内调用（不实际 spawn）。"""
    import os

    from dl_helper.training.launcher import _spawn_entry
    from dl_helper.training.backends.base import BackendResult

    # _spawn_entry 直接写入 RANK/WORLD_SIZE 等 env；monkeypatch.delenv 对原本
    # 不存在的 key 不追踪（pytest 仅在 key 已存在时记录），故用 try/finally 显式清理，
    # 避免污染后续进程内 Accelerator() 初始化（OSR-009 覆盖率门禁）。
    for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)

    calls = {}

    def fake_worker(ref, config, layout, rank, world, resume, publish_terminal=True, budget_monotonic=None):
        calls["ref"] = ref
        calls["rank"] = rank
        return BackendResult(status="succeeded")

    cfg = parse_config(default_schema())
    from dl_helper.training.config import config_to_dict
    try:
        _spawn_entry("exp:build", config_to_dict(cfg), str(tmp_path / "runs" / "spawn"),
                     0, 1, "none", fake_worker)
    finally:
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)
    assert calls["ref"] == "exp:build"
    assert calls["rank"] == 0
    # 环境已恢复：后续进程内 Accelerator() 不受影响
    assert "RANK" not in os.environ
    assert "WORLD_SIZE" not in os.environ


def test_sklearn_apply_params_n_jobs():
    from dl_helper.training.backends.sklearn_backend import apply_params
    from sklearn.linear_model import SGDClassifier
    est = SGDClassifier()
    cfg = _skl_cfg(backend={"type": "sklearn", "torch": None,
                            "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096,
                                        "n_jobs": None, "random_state": "require_explicit",
                                        "sample_weight_parameter": None}})
    # require_explicit 但 SGDClassifier random_state 为 None → 失败
    from dl_helper.training.backends.sklearn_backend import SklearnBackendError
    with pytest.raises(SklearnBackendError):
        apply_params(est, cfg)


def test_sklearn_apply_params_run_seed():
    from dl_helper.training.backends.sklearn_backend import apply_params
    from sklearn.linear_model import SGDClassifier
    est = SGDClassifier()
    cfg = _skl_cfg()
    apply_params(est, cfg)
    assert est.get_params()["random_state"] == cfg.run.seed


def test_torch_export_models_best_none(tmp_path):
    """best_model_state=None 时只导出 last。"""
    import torch
    from dl_helper.training.artifacts import RunLayout
    from dl_helper.training.backends.torch_backend import _export_models
    from dl_helper.training.config import default_schema, parse_config
    model = torch.nn.Linear(4, 3)
    cfg = parse_config(default_schema())
    from dl_helper.training.engine import EngineState
    layout = RunLayout(str(tmp_path / "runs" / "exp"))
    layout.ensure()
    state = EngineState("torch", "exp", "fp")
    artifact = _export_models(layout, model, None, {"class": "x", "num_parameters": 15, "params": {}},
                              cfg, state)
    assert artifact.format == "safetensors"
    assert os.path.exists(layout.path("models", "last", "model.safetensors"))
    # best 未设置 → best_path None
    assert artifact.best_path is None


def test_sklearn_resolve_fit_kwargs_no_weight_no_param():
    from dl_helper.training.backends.sklearn_backend import resolve_fit_kwargs
    from dl_helper.training.contracts import EstimatorBatch
    cfg = _skl_cfg()
    batch = EstimatorBatch(features=np.zeros((2, 2)), targets=np.array([0, 1]), sample_count=2)
    assert resolve_fit_kwargs(cfg, batch) == {}


def test_torch_unwrap_model_plain():
    import torch
    from dl_helper.training.backends.torch_backend import unwrap_model
    m = torch.nn.Linear(2, 2)
    assert unwrap_model(m) is m
