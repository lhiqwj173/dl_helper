"""任务 7.1：backend 资源解析（Torch GPU/worker/有效批量、sklearn n_jobs）。"""
from __future__ import annotations

import os

import pytest

from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import Platform, PlatformError

nominal = None


def _torch_cfg(num_procs="auto", mixed="auto", acc=1, nominal_batch_size=None):
    schema = default_schema()
    schema["backend"]["torch"]["gradient_accumulation_steps"] = acc
    schema["backend"]["torch"]["mixed_precision"] = mixed
    schema["distributed"]["num_processes"] = num_procs
    return parse_config(schema)


def test_torch_resources_cpu_or_gpu():
    """num_processes=auto：无 CUDA → 1；有 CUDA → 可见数。"""
    p = Platform("local")
    r = p.resolve_torch_resources(_torch_cfg(), None)
    assert r.num_processes >= 1
    assert r.mixed_precision in ("no", "fp16", "bf16")


def test_torch_explicit_num_processes():
    p = Platform("local")
    r = p.resolve_torch_resources(_torch_cfg(num_procs=1), nominal)
    assert r.num_processes == 1


def test_torch_effective_batch():
    p = Platform("local")
    r = p.resolve_torch_resources(_torch_cfg(num_procs=1, acc=2), nominal_batch_size=16)
    assert r.effective_batch_size == 32  # 16 * 1 * 2
    r2 = p.resolve_torch_resources(_torch_cfg(num_procs=1), nominal_batch_size=None)
    assert r2.effective_batch_size == "dynamic"


def test_torch_mixed_precision_no():
    p = Platform("local")
    r = p.resolve_torch_resources(_torch_cfg(mixed="no"), nominal)
    assert r.mixed_precision == "no"


def test_sklearn_resources():
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096,
                                     "n_jobs": "auto", "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = None
    cfg = parse_config(schema)
    p = Platform("local")
    r = p.resolve_sklearn_resources(cfg)
    assert r.n_jobs == (os.cpu_count() or 1)  # auto → 逻辑 CPU
    assert r.fit_mode == "batch"


def test_sklearn_requires_single_process():
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096,
                                     "n_jobs": 1, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 2}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = None
    with pytest.raises(Exception):
        parse_config(schema)  # sklearn 必须 num_processes=1
