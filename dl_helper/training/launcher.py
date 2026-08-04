"""1/多进程启动：只传实验引用与纯配置，不初始化 CUDA。

单进程在当前进程内运行；多进程使用 multiprocessing spawn 启动隔离 worker。
"""
from __future__ import annotations

import multiprocessing
import os
from typing import Any, Callable

from .config import Config


def _spawn_entry(
    experiment_ref: str,
    config_dict: dict[str, Any],
    run_dir: str,
    local_rank: int,
    world_size: int,
    resume: str,
    worker_fn: Callable,
    publish_terminal: bool = True,
    budget_monotonic=None,
) -> None:
    """spawn 子进程入口；由 launcher 注入 worker_fn 保持模块级可导入。"""
    import sys as _sys

    from .artifacts import RunLayout
    from .config import parse_config

    config = parse_config(config_dict)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    layout = RunLayout(run_dir)
    result = worker_fn(experiment_ref, config, layout, local_rank, world_size, resume,
                       publish_terminal=publish_terminal, budget_monotonic=budget_monotonic)
    # OSR-004：多进程 preempted 状态经退出码 75 传播给父进程
    if getattr(result, "status", "succeeded") == "preempted":
        _sys.exit(75)


def launch_torch(
    experiment_ref: str,
    config: Config,
    run_dir: str,
    num_processes: int,
    resume: str = "none",
    worker_fn: Callable | None = None,
    publish_terminal: bool = True,
    budget_monotonic=None,
) -> int:
    """启动 torch 训练。返回进程退出码。"""
    from .backends.torch_backend import run_worker

    worker_fn = worker_fn or run_worker
    if num_processes == 1:
        from .artifacts import RunLayout

        layout = RunLayout(run_dir)
        result = worker_fn(experiment_ref, config, layout, 0, 1, resume,
                           publish_terminal=publish_terminal, budget_monotonic=budget_monotonic)
        if getattr(result, "status", "succeeded") == "preempted":
            return 75
        return 0

    ctx = multiprocessing.get_context("spawn")
    processes = []
    config_dict = _config_to_dict(config)
    for rank in range(num_processes):
        p = ctx.Process(
            target=_spawn_entry,
            args=(experiment_ref, config_dict, run_dir, rank, num_processes, resume, worker_fn,
                  publish_terminal, budget_monotonic),
        )
        processes.append(p)
    for p in processes:
        p.start()
    # OSR-004：健康训练不受固定总时限限制；仅在 rank 异常时联动回收。
    while True:
        codes = [p.exitcode for p in processes]
        if all(code is not None for code in codes):
            if all(code == 0 for code in codes):
                return 0
            if all(code == 75 for code in codes):
                return 75
            failure = next((code for code in codes if code not in (0, 75)), None)
            return int(failure) if failure is not None else 1

        unexpected = next((code for code in codes if code not in (None, 0, 75)), None)
        if unexpected is not None:
            _terminate_and_join(processes)
            return int(unexpected)
        for process in processes:
            process.join(timeout=0.05)


def _terminate_and_join(processes: list[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)
    remaining = [process for process in processes if process.is_alive()]
    for process in remaining:
        process.kill()
    for process in remaining:
        process.join(timeout=5.0)
    if any(process.is_alive() for process in remaining):
        raise RuntimeError("多进程异常回收超时，仍有 worker 存活")


def _config_to_dict(config: Config) -> dict[str, Any]:
    from .config import config_to_dict

    return config_to_dict(config)
