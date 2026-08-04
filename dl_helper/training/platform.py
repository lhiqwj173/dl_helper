"""Local/Kaggle 平台检测、资源解析、路径合同与 Secret resolver。"""
from __future__ import annotations

import os
import platform as _platform
import socket
import string
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


def resolve_source_revision(config: Config) -> str:
    """解析 source revision：Kaggle 必须 40 位 SHA；本地显式提供或从 Git 获取。"""
    revision = config.run.source_revision
    if revision:
        if len(revision) != 40 or any(c not in string.hexdigits for c in revision):
            raise PlatformError(f"source_revision 必须为 40 位 Git SHA: {revision!r}")
        return revision
    try:
        import subprocess
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        if proc.returncode == 0:
            head = proc.stdout.strip()
            if len(head) == 40:
                return head
    except Exception:
        pass
    raise PlatformError("无法获取 Git revision；请显式提供 run.source_revision")


def free_disk_bytes(path: str) -> int:
    """目标路径所在分区的可用空间（字节）。"""
    try:
        import ctypes

        if os.name == "nt":
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                os.path.dirname(os.path.abspath(path)) or path, None, None, ctypes.byref(free_bytes)
            )
            return int(free_bytes.value)
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except Exception:
        raise PlatformError(f"无法获取磁盘空间: {path!r}")


# --------------------------------------------------------------------------
# 运行预算（任务 7.3）
# --------------------------------------------------------------------------

class RuntimeBudget:
    """monotonic 预算：elapsed >= max-grace 时停止新 step，不动态估算保存耗时。"""

    def __init__(self, max_minutes: float, grace_minutes: float, monotonic=None) -> None:
        if max_minutes <= 0 or grace_minutes < 0 or grace_minutes >= max_minutes:
            raise PlatformError("预算要求 grace < max 且均为正")
        self._now = monotonic or time.monotonic
        self._deadline = self._now() + (max_minutes - grace_minutes) * 60.0
        self.max_minutes = max_minutes
        self.grace_minutes = grace_minutes

    def hit(self) -> bool:
        return self._now() >= self._deadline
