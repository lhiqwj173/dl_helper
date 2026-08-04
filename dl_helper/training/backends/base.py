"""backend 结果、能力与生命周期协议。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Union

from ..contracts import JSONValue


@dataclass(frozen=True)
class ModelArtifact:
    """backend 产出的模型 Artifact 引用（相对 run 根目录）。"""

    format: Literal["safetensors", "joblib"]
    best_path: str | None = None
    last_path: str | None = None


@dataclass(frozen=True)
class BackendResult:
    """统一 backend worker 返回；engine 只消费其中的纯数据结构。"""

    status: Literal["succeeded", "preempted"]
    epoch: int = 0
    batch_in_epoch: int = 0
    global_step: int = 0
    model_artifact: ModelArtifact | None = None
    environment_stats: Mapping[str, JSONValue] = field(default_factory=dict)
    eval_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("succeeded", "preempted"):
            raise ValueError(f"BackendResult.status 非法: {self.status!r}")
        if self.epoch < 0 or self.batch_in_epoch < 0 or self.global_step < 0:
            raise ValueError("BackendResult 位置字段不得为负")


class Backend(Protocol):
    """backend worker 生命周期协议。"""

    name: Literal["torch", "sklearn"]

    def run(self, experiment: Any, config: Any, platform: Any, artifact_writer: Any) -> BackendResult: ...


def validate_backend_result(result: BackendResult) -> None:
    if not isinstance(result, BackendResult):
        raise TypeError(f"worker 必须返回 BackendResult，得到 {type(result).__name__}")
    result.__post_init__()
