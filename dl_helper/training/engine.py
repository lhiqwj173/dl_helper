"""backend-neutral 生命周期状态机、EngineState、selection 与编排。

engine 不导入领域代码或包含 torch/sklearn 训练细节；所有状态单向，
终态互斥，primary 异常向上传播。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .config import Config, SelectionConfig
from .contracts import JSONValue, MetricDefinition, validate_experiment
from .platform import Platform

STAGE_CREATED = "CREATED"
STAGE_PREFLIGHTED = "PREFLIGHTED"
STAGE_SERVICES_STARTED = "SERVICES_STARTED"
STAGE_PREPARED = "PREPARED"
STAGE_RESUMED = "RESUMED"
STAGE_FITTING = "FITTING"
STAGE_EVALUATING = "EVALUATING"
STAGE_CHECKPOINTING = "CHECKPOINTING"
STAGE_TESTING = "TESTING"
STAGE_FINALIZING = "FINALIZING"
STAGE_SERVICES_FINALIZED = "SERVICES_FINALIZED"
STAGE_SUCCEEDED = "SUCCEEDED"
STAGE_FAILED = "FAILED"
STAGE_PREEMPTED = "PREEMPTED"

TERMINAL_STAGES = (STAGE_SUCCEEDED, STAGE_FAILED, STAGE_PREEMPTED)

_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    STAGE_CREATED: frozenset({STAGE_PREFLIGHTED, STAGE_FAILED}),
    STAGE_PREFLIGHTED: frozenset({STAGE_SERVICES_STARTED, STAGE_FAILED}),
    STAGE_SERVICES_STARTED: frozenset({STAGE_PREPARED, STAGE_FAILED}),
    STAGE_PREPARED: frozenset({STAGE_RESUMED, STAGE_FITTING, STAGE_FAILED}),
    STAGE_RESUMED: frozenset({STAGE_FITTING, STAGE_FAILED}),
    STAGE_FITTING: frozenset({STAGE_EVALUATING, STAGE_CHECKPOINTING, STAGE_TESTING, STAGE_FAILED}),
    STAGE_EVALUATING: frozenset({STAGE_CHECKPOINTING, STAGE_FAILED}),
    STAGE_CHECKPOINTING: frozenset({STAGE_FITTING, STAGE_FINALIZING, STAGE_FAILED}),
    STAGE_TESTING: frozenset({STAGE_FINALIZING, STAGE_FAILED}),
    STAGE_FINALIZING: frozenset({STAGE_SERVICES_FINALIZED, STAGE_FAILED}),
    STAGE_SERVICES_FINALIZED: frozenset({STAGE_SUCCEEDED, STAGE_PREEMPTED, STAGE_FAILED}),
    STAGE_SUCCEEDED: frozenset(),
    STAGE_FAILED: frozenset(),
    STAGE_PREEMPTED: frozenset(),
}


class EngineStateError(Exception):
    """状态机或 selection 合同违规。"""


# --------------------------------------------------------------------------
# 状态机
# --------------------------------------------------------------------------

class StageMachine:
    """单向状态推进；终态互斥。"""

    def __init__(self, stage: str = STAGE_CREATED) -> None:
        if stage not in _ALLOWED_TRANSITIONS:
            raise EngineStateError(f"未知 stage: {stage!r}")
        self.stage = stage

    def transition(self, target: str) -> None:
        if self.stage in TERMINAL_STAGES:
            raise EngineStateError(f"终态 {self.stage} 不可再转移")
        allowed = _ALLOWED_TRANSITIONS.get(self.stage, frozenset())
        if target not in allowed:
            raise EngineStateError(f"非法状态转移: {self.stage} -> {target}")
        self.stage = target

    def snapshot(self) -> dict[str, Any]:
        return {"stage": self.stage}


# --------------------------------------------------------------------------
# EngineState
# --------------------------------------------------------------------------

class EngineState:
    """backend-neutral 版本化可序列化状态，恢复时先校验 schema/backend/定义。"""

    schema_version = 1

    def __init__(
        self,
        backend: str,
        run_id: str,
        config_fingerprint: str,
        metric_name: str | None = None,
        mode: str | None = None,
        patience: int | None = None,
        min_delta: float = 0.0,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.config_fingerprint = config_fingerprint
        self.stage_machine = StageMachine()
        self.epoch = 0
        self.batch_in_epoch = 0
        self.global_step = 0
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.best_step: int | None = None
        self.no_improve = 0
        self.selection_metric = metric_name
        self.selection_mode = mode
        self.selection_patience = patience
        self.selection_min_delta = min_delta
        self.partial_metric_states: dict[str, Any] = {}
        self.current_stage_metrics: dict[str, Any] = {}

    def transition(self, target: str) -> None:
        self.stage_machine.transition(target)

    @property
    def stage(self) -> str:
        return self.stage_machine.stage

    def advance_batch(self) -> None:
        self.batch_in_epoch += 1

    def advance_epoch(self) -> None:
        self.epoch += 1
        self.batch_in_epoch = 0

    def increment_global_step(self) -> None:
        self.global_step += 1

    # ---- selection ----

    def selection_update(self, value: float) -> bool:
        """更新 selection；返回是否改善。direction 由 self.selection_mode 决定。"""
        if not _finite(value):
            raise EngineStateError(f"selection 值非有限: {value!r}")
        mode = self.selection_mode
        min_delta = self.selection_min_delta
        if self.best_value is None or _improved(value, self.best_value, mode, min_delta):
            self.best_value = value
            self.best_epoch = self.epoch
            self.best_step = self.global_step
            self.no_improve = 0
            return True
        self.no_improve += 1
        return False

    def should_early_stop(self) -> bool:
        if self.selection_patience is None:
            return False
        return self.no_improve >= self.selection_patience

    def best(self) -> tuple[float | None, int | None, int | None]:
        return (self.best_value, self.best_epoch, self.best_step)

    # ---- 序列化 ----

    def state_dict(self) -> Mapping[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
            "stage": self.stage_machine.stage,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "global_step": self.global_step,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "no_improve": self.no_improve,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "selection_patience": self.selection_patience,
            "selection_min_delta": self.selection_min_delta,
            "partial_metric_states": self.partial_metric_states,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["schema_version"]) != self.schema_version:
            raise EngineStateError(f"EngineState schema 版本不兼容: {state['schema_version']}")
        if state["backend"] != self.backend:
            raise EngineStateError(
                f"EngineState backend 漂移: {state['backend']} != {self.backend}"
            )
        if state["run_id"] != self.run_id:
            raise EngineStateError(f"EngineState run_id 漂移: {state['run_id']} != {self.run_id}")
        if state["config_fingerprint"] != self.config_fingerprint:
            raise EngineStateError("EngineState 配置指纹漂移，拒绝恢复")
        self.stage_machine = StageMachine(str(state["stage"]))
        self.epoch = int(state["epoch"])
        self.batch_in_epoch = int(state["batch_in_epoch"])
        self.global_step = int(state["global_step"])
        self.best_value = state.get("best_value")
        self.best_epoch = state.get("best_epoch")
        self.best_step = state.get("best_step")
        self.no_improve = int(state.get("no_improve", 0))
        self.selection_metric = state.get("selection_metric")
        self.selection_mode = state.get("selection_mode")
        self.selection_patience = state.get("selection_patience")
        self.selection_min_delta = float(state.get("selection_min_delta", 0.0))
        self.partial_metric_states = dict(state.get("partial_metric_states", {}))


def _finite(v: float) -> bool:
    import math
    return isinstance(v, (int, float)) and math.isfinite(v)


def _improved(new: float, old: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return new < old - min_delta
    return new > old + min_delta


# --------------------------------------------------------------------------
# Selection 校验
# --------------------------------------------------------------------------

def resolve_definition(metric_name: str, definitions: Mapping[str, MetricDefinition]) -> MetricDefinition | None:
    """解析 selection metric（如 val/loss）对应的 Task 定义。"""
    if "/" in metric_name:
        stage, _, name = metric_name.partition("/")
        if stage != "val":
            return None
        return definitions.get(name)
    return definitions.get(metric_name)


def validate_selection(
    selection: SelectionConfig | None,
    metric_definitions: Mapping[str, MetricDefinition],
    has_val: bool,
) -> None:
    """校验 selection 与 val split / MetricDefinition 合同。"""
    if has_val and selection is None:
        raise EngineStateError("存在验证 split 时必须配置 selection")
    if not has_val and selection is not None:
        raise EngineStateError("不存在验证 split 时 selection 必须为 null")
    if selection is None:
        return
    defn = resolve_definition(selection.metric, metric_definitions)
    if defn is None:
        raise EngineStateError(f"selection metric {selection.metric!r} 未由 Task 产生")
    if not defn.exact or defn.evaluation_scope != "full":
        raise EngineStateError(f"selection metric {selection.metric!r} 必须 exact/full")
    if defn.direction != selection.mode:
        raise EngineStateError(
            f"selection.mode={selection.mode!r} 必须等于 MetricDefinition.direction={defn.direction!r}"
        )


# --------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------

@dataclass
class RunEngine:
    """共享编排：状态机、worker 消费、服务与终态发布。

    不包含 backend 细节；worker/services/report 通过注入协作。
    """

    config: Config
    experiment: Any
    platform: Platform
    artifact_writer: Any
    engine_state: EngineState
    worker: Any = None
    services: Any = None
    reporter: Any = None

    def run(self) -> Any:
        self.engine_state.transition(STAGE_PREFLIGHTED)
        self._preflight()
        if self.services is not None:
            self.engine_state.transition(STAGE_SERVICES_STARTED)
            self.services.start_run()
        self.engine_state.transition(STAGE_PREPARED)
        if self._should_resume():
            self.engine_state.transition(STAGE_RESUMED)
        else:
            self.engine_state.transition(STAGE_FITTING)

        if self.worker is None:
            raise EngineStateError("engine 缺少 backend worker")
        result = self.worker.run(self.experiment, self.config, self.platform, self.artifact_writer)
        if result.status == "preempted":
            self.engine_state.transition(STAGE_FINALIZING)
        else:
            self.engine_state.transition(STAGE_TESTING)
            self.engine_state.transition(STAGE_FINALIZING)
        if self.services is not None:
            self.engine_state.transition(STAGE_SERVICES_FINALIZED)
            self.services.finalize(result.status)
        if result.status == "preempted":
            self.engine_state.transition(STAGE_PREEMPTED)
        else:
            self.engine_state.transition(STAGE_SUCCEEDED)
        return result

    def _preflight(self) -> None:
        self.artifact_writer.ensure()
        validate_experiment(self.experiment)
        # 平台路径与资源校验
        self.platform.validate_kaggle_inputs(self.config)
        self.artifact_writer.write_config(self.config)

    def _should_resume(self) -> bool:
        return self.config.checkpoint.resume != "none"
