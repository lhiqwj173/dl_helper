"""Local/Kaggle 平台检测、资源解析、路径合同与 Secret resolver。"""
from __future__ import annotations

import os
import platform as _platform
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from .config import Config

PlatformKind = Literal["local", "kaggle"]


class PlatformError(Exception):
    """平台合同违规。"""


def detect_platform() -> PlatformKind:
    """任一环境键以 KAGGLE 开头时识别 Kaggle。"""
    for key in os.environ:
        if key.startswith("KAGGLE"):
            return "kaggle"
    return "local"


def hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    total_memory_bytes: int | None
    compute_capability: str | None


@dataclass(frozen=True)
class TorchResources:
    num_processes: int
    devices: tuple[DeviceInfo, ...]
    mixed_precision: str  # 已解析 no/fp16/bf16
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    effective_batch_size: int | str  # 整数或 "dynamic"
    matmul_precision: str
    compile: bool
    deterministic: str
    find_unused_parameters: bool
    clip_grad_norm: float | None
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class SklearnResources:
    n_jobs: int | None
    logical_cpus: int
    fit_mode: str
    evaluation_batch_size: int


@dataclass(frozen=True)
class EnvironmentManifest:
    """复现所需的环境摘要。"""

    os: str
    hostname: str
    logical_cpus: int
    physical_cpus: int | None
    python: str
    platform: PlatformKind
    torch_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    devices: list[dict[str, Any]] = field(default_factory=list)
    sklearn_version: str | None = None
    numpy_version: str | None = None
    thread_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EpochBudgetForecast:
    """完成一个完整 epoch 后的预算预测。"""

    epoch_seconds: float
    average_epoch_seconds: float
    remaining_training_seconds: float
    should_preempt: bool


class Platform:
    """平台上下文：检测、路径、资源与 Secret。"""

    def __init__(self, kind: PlatformKind | None = None) -> None:
        self.kind = kind or detect_platform()

    @property
    def is_kaggle(self) -> bool:
        return self.kind == "kaggle"

    # ---- 路径 ----

    def resolve_output_root(self, config: Config) -> str:
        configured = config.run.output_root
        if configured is not None:
            if self.is_kaggle:
                resolved = os.path.realpath(configured)
                working = os.path.realpath("/kaggle/working")
                if not (resolved == working or resolved.startswith(working + os.sep)):
                    raise PlatformError(f"Kaggle output_root 必须位于 /kaggle/working 内: {configured!r}")
            return configured
        if self.is_kaggle:
            return "/kaggle/working/dl-helper-runs"
        return os.getcwd()

    def validate_kaggle_inputs(self, config: Config) -> None:
        """Kaggle 输入必须是显式 /kaggle/input/... 路径，禁止首目录选择。"""
        if not self.is_kaggle:
            return
        # 输入路径在 Experiment 的 model_config 中显式声明；此处校验其格式
        for key, value in config.experiment.items():
            if key.endswith("path") and isinstance(value, str):
                if not value.startswith("/kaggle/input/"):
                    raise PlatformError(f"Kaggle 输入路径必须是 /kaggle/input/...: {value!r}")
                if not os.path.exists(value):
                    raise PlatformError(f"Kaggle 输入路径不存在: {value!r}")
        # 禁止写 /kaggle/input
        out = self.resolve_output_root(config)
        if out.startswith("/kaggle/input"):
            raise PlatformError("Kaggle 禁止写 /kaggle/input 目录")

    # ---- Torch 资源 ----

    def resolve_torch_resources(self, config: Config, nominal_batch_size: int | None) -> TorchResources:
        import torch

        backend = config.backend.torch
        if backend is None:
            raise PlatformError("torch backend 配置缺失")

        num_procs = config.distributed.num_processes
        if num_procs == "auto":
            if torch.cuda.is_available():
                num_procs = max(1, torch.cuda.device_count())
            else:
                num_procs = 1
        if not isinstance(num_procs, int) or num_procs <= 0:
            raise PlatformError(f"torch num_processes 非法: {num_procs!r}")

        devices: list[DeviceInfo] = []
        if torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cc = f"{props.major}.{props.minor}"
                devices.append(DeviceInfo(name=props.name, total_memory_bytes=props.total_memory, compute_capability=cc))
        if num_procs > len(devices) and num_procs > 1 and len(devices) > 0:
            raise PlatformError(f"请求 {num_procs} 进程但仅 {len(devices)} 可见 CUDA 设备")

        mixed = backend.mixed_precision
        if mixed == "auto":
            if torch.cuda.device_count() > 0:
                major = torch.cuda.get_device_capability(0)[0]
                mixed = "bf16" if major >= 8 else "fp16"
            else:
                mixed = "no"

        logical_cpus = os.cpu_count() or 1
        # OSR-006：按设计公式跨平台解析（不按 OS 硬编码）
        num_workers = min(8, max(1, logical_cpus // max(num_procs, 1)))
        pin_memory = torch.cuda.device_count() > 0
        persistent_workers = num_workers > 0
        prefetch = 2 if num_workers > 0 else None

        if nominal_batch_size is not None:
            eff = nominal_batch_size * num_procs * backend.gradient_accumulation_steps
        else:
            eff = "dynamic"

        return TorchResources(
            num_processes=num_procs,
            devices=tuple(devices),
            mixed_precision=mixed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch,
            effective_batch_size=eff,
            matmul_precision=backend.matmul_precision,
            compile=backend.compile,
            deterministic=backend.deterministic,
            find_unused_parameters=backend.find_unused_parameters,
            clip_grad_norm=backend.clip_grad_norm,
            gradient_accumulation_steps=backend.gradient_accumulation_steps,
        )

    # ---- sklearn 资源 ----

    def resolve_sklearn_resources(self, config: Config) -> SklearnResources:
        backend = config.backend.sklearn
        if backend is None:
            raise PlatformError("sklearn backend 配置缺失")
        logical_cpus = os.cpu_count() or 1
        n_jobs = backend.n_jobs
        if n_jobs == "auto":
            n_jobs = logical_cpus
        return SklearnResources(
            n_jobs=n_jobs,
            logical_cpus=logical_cpus,
            fit_mode=backend.fit_mode,
            evaluation_batch_size=backend.evaluation_batch_size,
        )

    # ---- 环境 manifest ----

    def environment_manifest(self) -> dict[str, Any]:
        import numpy as np

        try:
            import torch
            torch_version = torch.__version__
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        except Exception:
            torch_version = cuda_version = cudnn_version = None
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except Exception:
            sklearn_version = None
        devices: list[dict[str, Any]] = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        "index": i,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "compute_capability": f"{props.major}.{props.minor}",
                    })
        except Exception:
            pass
        thread_env = {k: v for k, v in os.environ.items() if k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )}
        return {
            "os": _platform.system(),
            "release": _platform.release(),
            "hostname": hostname(),
            "logical_cpus": os.cpu_count() or 1,
            "python": _platform.python_version(),
            "platform": self.kind,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "cudnn_version": cudnn_version,
            "sklearn_version": sklearn_version,
            "numpy_version": np.__version__,
            "devices": devices,
            "thread_env": thread_env,
        }


# --------------------------------------------------------------------------
# Secret resolver（任务 5.1）
# --------------------------------------------------------------------------

class SecretError(Exception):
    """Secret 缺失或解析失败。"""


class SecretResolver:
    """从 Kaggle Secrets 或同名环境变量解析 Secret；值不入 repr/log/config。"""

    def __init__(self, platform: Platform, env: Mapping[str, str] | None = None) -> None:
        self._platform = platform
        self._env = dict(env) if env is not None else dict(os.environ)
        self._resolved: dict[str, str] = {}

    def resolve(self, key: str) -> str:
        """解析 Secret；缺失时报告 key 名而非值。"""
        if key in self._resolved:
            return self._resolved[key]
        value = self._lookup(key)
        if value is None or value == "":
            raise SecretError(f"Secret {key!r} 缺失（启用服务必须配置）")
        self._resolved[key] = value
        return value

    def _lookup(self, key: str) -> str | None:
        # Kaggle Notebook 也允许显式注入同名环境变量，便于调试和自托管运行。
        env_value = self._env.get(key)
        if env_value:
            return env_value
        if self._platform.is_kaggle:
            try:
                from kaggle_secrets import UserSecretsClient
            except ImportError as exc:
                raise SecretError("Kaggle Secret 客户端不可用") from exc
            return UserSecretsClient().get_secret(key)
        return self._env.get(key)

    def resolved_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._resolved))

    def redact(self, text: str) -> str:
        """把已解析 Secret 值从任意文本中替换为 [REDACTED]。"""
        if not self._resolved:
            return text
        for value in self._resolved.values():
            if value:
                text = text.replace(value, "[REDACTED]")
        return text


def resolve_source_revision(config: Config, cwd: str | None = None) -> str:
    """解析可审计的代码版本标识。

    版本标识可以是 tag、分支名、短 SHA 或任意非空字符串；平台不再强制
    使用完整 40 位 SHA。未显式配置时尽力读取当前 Git HEAD，Git 不可用时
    直接报错，避免把未知版本静默记录为可复现版本。
    """
    revision = config.run.source_revision
    if revision:
        if not revision.strip() or any(ch.isspace() for ch in revision):
            raise PlatformError(f"source_revision 必须是无空白的非空版本标识: {revision!r}")
        return revision.strip()

    import subprocess

    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    raise PlatformError("无法获取 Git revision；请显式提供 run.source_revision（tag/分支/短 SHA 均可）")


def free_disk_bytes(path: str) -> int:
    """目标路径所在分区的可用空间（字节）。"""
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            raise PlatformError(f"无法定位输出目录所在分区: {path!r}")
        probe = parent
    if not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    try:
        import ctypes

        if os.name == "nt":
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                probe, None, None, ctypes.byref(free_bytes)
            )
            return int(free_bytes.value)
        st = os.statvfs(probe)
        return st.f_bavail * st.f_frsize
    except Exception:
        raise PlatformError(f"无法获取磁盘空间: {path!r}")


# --------------------------------------------------------------------------
# 执行策略（D-003：平台独立，不可由 YAML 构造）
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionPolicy:
    """平台独立执行策略；只能由平台检测构造，不可写入用户配置 schema。

    max_minutes 为 None 表示不启用运行预算（Local）。
    """

    platform: PlatformKind
    max_minutes: float | None
    shutdown_grace_minutes: float


# 唯一 Kaggle 策略：660 分钟训练预算 + 10 分钟收尾窗口（720 平台上限内留约 60 分钟缓冲）
KAGGLE_TRAINING_BUDGET_MINUTES = 660.0
KAGGLE_SHUTDOWN_GRACE_MINUTES = 10.0
LOCAL_SHUTDOWN_GRACE_MINUTES = 10.0


def kaggle_execution_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        platform="kaggle",
        max_minutes=KAGGLE_TRAINING_BUDGET_MINUTES,
        shutdown_grace_minutes=KAGGLE_SHUTDOWN_GRACE_MINUTES,
    )


def local_execution_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        platform="local",
        max_minutes=None,
        shutdown_grace_minutes=LOCAL_SHUTDOWN_GRACE_MINUTES,
    )


def execution_policy_for(platform: Platform) -> ExecutionPolicy:
    """当前平台的唯一执行策略；Kaggle 恒为 660/10，Local 不启用预算。"""
    return kaggle_execution_policy() if platform.is_kaggle else local_execution_policy()


def execution_policy_to_dict(policy: ExecutionPolicy) -> dict[str, Any]:
    """策略序列化为纯 dict（spawn 子进程 / execution-policy.json 用）。"""
    return {
        "schema_version": 1,
        "platform": policy.platform,
        "max_minutes": policy.max_minutes,
        "shutdown_grace_minutes": policy.shutdown_grace_minutes,
    }


def execution_policy_from_dict(data: Mapping[str, Any]) -> ExecutionPolicy:
    """从纯 dict 严格重建策略；缺字段、未知字段或与平台不一致立即失败。"""
    if not isinstance(data, Mapping):
        raise PlatformError("execution-policy dict 必须是 mapping")
    unknown = [k for k in data if k not in (
        "schema_version", "platform", "max_minutes", "shutdown_grace_minutes"
    )]
    if unknown:
        raise PlatformError(f"execution-policy 含未知字段: {sorted(unknown)}")
    if data.get("schema_version") != 1:
        raise PlatformError(f"execution-policy schema_version 必须为 1: {data.get('schema_version')!r}")
    kind = data.get("platform")
    if kind not in ("local", "kaggle"):
        raise PlatformError(f"execution-policy platform 非法: {kind!r}")
    rebuilt = ExecutionPolicy(
        platform=kind,
        max_minutes=data.get("max_minutes"),
        shutdown_grace_minutes=data.get("shutdown_grace_minutes"),
    )
    expected = kaggle_execution_policy() if kind == "kaggle" else local_execution_policy()
    if rebuilt != expected:
        raise PlatformError(
            f"execution-policy 与平台 {kind!r} 不一致: "
            f"max_minutes={rebuilt.max_minutes!r}, grace={rebuilt.shutdown_grace_minutes!r}"
        )
    return rebuilt


# --------------------------------------------------------------------------
# 运行预算（任务 7.3）
# --------------------------------------------------------------------------

class RuntimeBudget:
    """monotonic 预算：硬截止检查 + 完整 epoch 均值预测。"""

    def __init__(self, max_minutes: float, grace_minutes: float, monotonic=None) -> None:
        if max_minutes <= 0 or grace_minutes < 0 or grace_minutes >= max_minutes:
            raise PlatformError("预算要求 grace < max 且均为正")
        self._now = monotonic or time.monotonic
        self._deadline = self._now() + (max_minutes - grace_minutes) * 60.0
        self._epoch_durations: list[float] = []
        self.max_minutes = max_minutes
        self.grace_minutes = grace_minutes

    def hit(self) -> bool:
        return self._now() >= self._deadline

    def begin_epoch(self) -> float:
        """返回 epoch 起始 monotonic 时间戳。"""
        return self._now()

    def complete_epoch(self, started_at: float) -> EpochBudgetForecast:
        """记录完整 epoch，并判断剩余时间能否容纳下一个平均 epoch。"""
        finished_at = self._now()
        duration = finished_at - started_at
        if duration < 0:
            raise PlatformError("monotonic 时钟倒退，无法估算 epoch 耗时")
        self._epoch_durations.append(duration)
        average = sum(self._epoch_durations) / len(self._epoch_durations)
        remaining = max(0.0, self._deadline - finished_at)
        return EpochBudgetForecast(
            epoch_seconds=duration,
            average_epoch_seconds=average,
            remaining_training_seconds=remaining,
            should_preempt=average > remaining,
        )
