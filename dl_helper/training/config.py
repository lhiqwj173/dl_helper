"""严格配置 schema v1：YAML 解析、重复 key 检测、frozen typed config、跨字段校验与指纹。"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal, Mapping, Union

import yaml

from .contracts import JSONValue, validate_json_value

CONFIG_SCHEMA_VERSION = 1


class ConfigError(Exception):
    """严格配置 schema 违规。"""


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SPLITS = ("train", "val", "test", "predict")
_REJECTED_STR_PATTERNS = (("${", "环境变量插值"), ("{{", "模板表达式"), ("}}", "模板表达式"))
_UNSAFE_TAGS = ("tag:yaml.org,2002:merge",)


# --------------------------------------------------------------------------
# Frozen typed config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    name: str
    id: str | None
    output_root: str | None
    source_revision: str | None
    seed: int
    tags: Mapping[str, str]


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    log_every_steps: int


@dataclass(frozen=True)
class TorchBackendConfig:
    gradient_accumulation_steps: int
    mixed_precision: Literal["auto", "no", "fp16", "bf16"]
    compile: bool
    clip_grad_norm: float | None
    deterministic: Literal["strict", "warn", "off"]
    matmul_precision: Literal["highest", "high", "medium"]
    find_unused_parameters: bool


@dataclass(frozen=True)
class SklearnBackendConfig:
    fit_mode: Literal["batch", "incremental"]
    evaluation_batch_size: int
    n_jobs: Union[int, Literal["auto"], None]
    random_state: Literal["run_seed", "require_explicit"]
    sample_weight_parameter: str | None


@dataclass(frozen=True)
class BackendConfig:
    type: Literal["torch", "sklearn"]
    torch: TorchBackendConfig | None
    sklearn: SklearnBackendConfig | None


@dataclass(frozen=True)
class DistributedConfig:
    num_processes: Union[int, Literal["auto"]]


@dataclass(frozen=True)
class SelectionConfig:
    metric: str
    mode: Literal["min", "max"]
    patience: int
    min_delta: float


@dataclass(frozen=True)
class CheckpointConfig:
    every_epochs: int | None
    every_optimizer_steps: int | None
    keep_last: int | None
    resume: Literal["none", "auto", "required"]


@dataclass(frozen=True)
class RuntimeConfig:
    max_minutes: float | None
    shutdown_grace_minutes: float


@dataclass(frozen=True)
class ReportConfig:
    enabled: bool
    curve_sample_limit: int
    prediction_sample_limit: int
    prediction_splits: tuple[str, ...]


@dataclass(frozen=True)
class NoRemoteConfig:
    type: Literal["none"]


@dataclass(frozen=True)
class AListRemoteConfig:
    type: Literal["alist"]
    host: str
    base_path: str
    user_secret_key: str
    password_secret_key: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_attempts: int
    async_upload: bool
    failure_policy: Literal["required", "record"]


RemoteConfig = Union[NoRemoteConfig, AListRemoteConfig]


@dataclass(frozen=True)
class NoNotificationsConfig:
    type: Literal["none"]


@dataclass(frozen=True)
class WecomNotificationsConfig:
    type: Literal["wecom"]
    corp_id_secret_key: str
    corp_secret_key: str
    agent_id_secret_key: str
    to_user: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_attempts: int
    failure_policy: Literal["required", "record"]


NotificationsConfig = Union[NoNotificationsConfig, WecomNotificationsConfig]


@dataclass(frozen=True)
class Config:
    schema_version: int
    run: RunConfig
    experiment: Mapping[str, JSONValue]
    training: TrainingConfig
    backend: BackendConfig
    distributed: DistributedConfig
    selection: SelectionConfig | None
    checkpoint: CheckpointConfig
    runtime: RuntimeConfig
    report: ReportConfig
    remote: RemoteConfig
    notifications: NotificationsConfig


# --------------------------------------------------------------------------
# 严格 YAML loader
# --------------------------------------------------------------------------

class _StrictSafeLoader(yaml.SafeLoader):
    """拒绝重复 key、anchor/alias、merge key 的 YAML loader。"""


def _construct_mapping_strict(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(None, None, "期望 mapping 节点", node.start_mark)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag in _UNSAFE_TAGS:
            raise ConfigError("YAML merge key (<<) 不允许")
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ConfigError(f"配置 key 必须为字符串: {key!r}")
        if key in mapping:
            raise ConfigError(f"重复 YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_StrictSafeLoader.construct_mapping = _construct_mapping_strict


def _construct_yaml_map_strict(loader: yaml.SafeLoader, node: yaml.Node):
    if isinstance(node, yaml.MappingNode):
        for key_node, _value_node in node.value:
            if key_node.tag in _UNSAFE_TAGS:
                raise ConfigError("YAML merge key (<<) 不允许")
    data: dict[Any, Any] = {}
    yield data
    data.update(loader.construct_mapping(node))


_StrictSafeLoader.construct_yaml_map = _construct_yaml_map_strict


def yaml_load_strict(text: str) -> Mapping[str, Any]:
    """严格解析 UTF-8 YAML 文本为 mapping，拒绝重复 key/alias/merge。"""
    try:
        for token in yaml.scan(text, Loader=_StrictSafeLoader):
            if isinstance(token, yaml.tokens.AliasToken):
                raise ConfigError("YAML alias 不允许")
            if isinstance(token, yaml.tokens.AnchorToken):
                raise ConfigError("YAML anchor 不允许")
        data = yaml.load(text, Loader=_StrictSafeLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"配置根必须是 YAML mapping，得到 {type(data).__name__}")
    return data


def _reject_unsafe_strings(value: Any, path: str) -> None:
    if isinstance(value, str):
        for marker, name in _REJECTED_STR_PATTERNS:
            if marker in value:
                raise ConfigError(f"{path} 包含 {name}（不允许）: {value!r}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_unsafe_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _reject_unsafe_strings(item, f"{path}[{i}]")


# --------------------------------------------------------------------------
# 类型化构建
# --------------------------------------------------------------------------

def _require(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{path} 缺少必填字段 {key!r}")
    return mapping[key]


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """拒绝未知字段。"""
    unknown = [k for k in mapping if k not in allowed]
    if unknown:
        raise ConfigError(f"{path} 含未知字段: {sorted(unknown)}")


def _optional_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} 必须是整数或 null，得到 {value!r}")
    return value


def _optional_float(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{path} 必须是数值或 null，得到 {value!r}")
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{path} 含非有限数值: {value!r}")
        return value
    raise ConfigError(f"{path} 必须是数值或 null，得到 {value!r}")


def _int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} 必须是整数，得到 {value!r}")
    return value


def _positive_int(value: Any, path: str) -> int:
    v = _int(value, path)
    if v <= 0:
        raise ConfigError(f"{path} 必须为正整数: {v!r}")
    return v


def _non_negative_int(value: Any, path: str) -> int:
    v = _int(value, path)
    if v < 0:
        raise ConfigError(f"{path} 必须为非负整数: {v!r}")
    return v


def _float(value: Any, path: str) -> float:
    v = _optional_float(value, path)
    if v is None:
        raise ConfigError(f"{path} 必须为数值")
    return v


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} 必须是布尔值，得到 {value!r}（不接受字符串 truthiness）")
    return value


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path} 必须是字符串，得到 {value!r}")
    return value


def _enum(value: Any, allowed: tuple[Any, ...], path: str) -> Any:
    if value not in allowed:
        raise ConfigError(f"{path} 必须是 {allowed} 之一，得到 {value!r}")
    return value


def _build_run(raw: Mapping[str, Any], path: str) -> RunConfig:
    _check_keys(raw, {"name", "id", "output_root", "source_revision", "seed", "tags"}, path)
    name = _str(_require(raw, "name", path), f"{path}.name")
    if not name:
        raise ConfigError(f"{path}.name 必须是非空字符串")
    run_id = _require(raw, "id", path)
    if run_id is not None:
        run_id = _str(run_id, f"{path}.id")
        if not _RUN_ID_RE.match(run_id):
            raise ConfigError(f"{path}.id 不匹配运行 ID 字符集: {run_id!r}")
    seed = _int(_require(raw, "seed", path), f"{path}.seed")
    if seed < 0:
        raise ConfigError(f"{path}.seed 必须为非负整数")
    tags_raw = _require(raw, "tags", path)
    if not isinstance(tags_raw, Mapping):
        raise ConfigError(f"{path}.tags 必须是 mapping")
    tags: dict[str, str] = {}
    for k, v in tags_raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ConfigError(f"{path}.tags 键值必须都是字符串")
        tags[k] = v
    return RunConfig(
        name=name,
        id=run_id,
        output_root=_optional_str(_require(raw, "output_root", path), f"{path}.output_root"),
        source_revision=_optional_str(_require(raw, "source_revision", path), f"{path}.source_revision"),
        seed=seed,
        tags=tags,
    )


def _optional_str(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _str(value, path)


def _build_training(raw: Mapping[str, Any], path: str) -> TrainingConfig:
    _check_keys(raw, {"max_epochs", "log_every_steps"}, path)
    return TrainingConfig(
        max_epochs=_positive_int(_require(raw, "max_epochs", path), f"{path}.max_epochs"),
        log_every_steps=_positive_int(_require(raw, "log_every_steps", path), f"{path}.log_every_steps"),
    )


def _build_torch(raw: Mapping[str, Any], path: str) -> TorchBackendConfig:
    _check_keys(
        raw,
        {"gradient_accumulation_steps", "mixed_precision", "compile", "clip_grad_norm",
         "deterministic", "matmul_precision", "find_unused_parameters"},
        path,
    )
    mixed = _enum(_require(raw, "mixed_precision", path), ("auto", "no", "fp16", "bf16"), f"{path}.mixed_precision")
    determin = _enum(_require(raw, "deterministic", path), ("strict", "warn", "off"), f"{path}.deterministic")
    matmul = _enum(_require(raw, "matmul_precision", path), ("highest", "high", "medium"), f"{path}.matmul_precision")
    return TorchBackendConfig(
        gradient_accumulation_steps=_positive_int(
            _require(raw, "gradient_accumulation_steps", path), f"{path}.gradient_accumulation_steps"
        ),
        mixed_precision=mixed,
        compile=_bool(_require(raw, "compile", path), f"{path}.compile"),
        clip_grad_norm=_optional_float(_require(raw, "clip_grad_norm", path), f"{path}.clip_grad_norm"),
        deterministic=determin,
        matmul_precision=matmul,
        find_unused_parameters=_bool(_require(raw, "find_unused_parameters", path), f"{path}.find_unused_parameters"),
    )


def _build_sklearn(raw: Mapping[str, Any], path: str) -> SklearnBackendConfig:
    _check_keys(
        raw,
        {"fit_mode", "evaluation_batch_size", "n_jobs", "random_state", "sample_weight_parameter"},
        path,
    )
    fit_mode = _enum(_require(raw, "fit_mode", path), ("batch", "incremental"), f"{path}.fit_mode")
    n_jobs = _require(raw, "n_jobs", path)
    if n_jobs != "auto" and n_jobs is not None:
        n_jobs = _positive_int(n_jobs, f"{path}.n_jobs")
    random_state = _enum(_require(raw, "random_state", path), ("run_seed", "require_explicit"), f"{path}.random_state")
    sw = _require(raw, "sample_weight_parameter", path)
    if sw is not None:
        sw = _str(sw, f"{path}.sample_weight_parameter")
        if not sw:
            raise ConfigError(f"{path}.sample_weight_parameter 必须是非空参数路径")
    return SklearnBackendConfig(
        fit_mode=fit_mode,
        evaluation_batch_size=_positive_int(
            _require(raw, "evaluation_batch_size", path), f"{path}.evaluation_batch_size"
        ),
        n_jobs=n_jobs,
        random_state=random_state,
        sample_weight_parameter=sw,
    )


def _build_backend(raw: Mapping[str, Any], path: str) -> BackendConfig:
    _check_keys(raw, {"type", "torch", "sklearn"}, path)
    btype = _enum(_require(raw, "type", path), ("torch", "sklearn"), f"{path}.type")
    torch_raw = raw.get("torch")
    sklearn_raw = raw.get("sklearn")
    if btype == "torch":
        if torch_raw is None:
            raise ConfigError(f"{path}.torch 分支缺失")
        if sklearn_raw is not None:
            raise ConfigError(f"{path}.sklearn 分支必须为 null（未选 backend）")
        if not isinstance(torch_raw, Mapping):
            raise ConfigError(f"{path}.torch 必须是 mapping")
        return BackendConfig(type="torch", torch=_build_torch(torch_raw, f"{path}.torch"), sklearn=None)
    if sklearn_raw is None:
        raise ConfigError(f"{path}.sklearn 分支缺失")
    if torch_raw is not None:
        raise ConfigError(f"{path}.torch 分支必须为 null（未选 backend）")
    if not isinstance(sklearn_raw, Mapping):
        raise ConfigError(f"{path}.sklearn 必须是 mapping")
    return BackendConfig(type="sklearn", torch=None, sklearn=_build_sklearn(sklearn_raw, f"{path}.sklearn"))


def _build_distributed(raw: Mapping[str, Any], path: str) -> DistributedConfig:
    _check_keys(raw, {"num_processes"}, path)
    np_ = _require(raw, "num_processes", path)
    if np_ != "auto":
        np_ = _positive_int(np_, f"{path}.num_processes")
    return DistributedConfig(num_processes=np_)


def _build_selection(raw: Mapping[str, Any], path: str) -> SelectionConfig:
    _check_keys(raw, {"metric", "mode", "patience", "min_delta"}, path)
    metric = _str(_require(raw, "metric", path), f"{path}.metric")
    if not metric:
        raise ConfigError(f"{path}.metric 必须是非空字符串")
    mode = _enum(_require(raw, "mode", path), ("min", "max"), f"{path}.mode")
    patience = _non_negative_int(_require(raw, "patience", path), f"{path}.patience")
    min_delta = _float(_require(raw, "min_delta", path), f"{path}.min_delta")
    if min_delta < 0:
        raise ConfigError(f"{path}.min_delta 必须非负")
    return SelectionConfig(metric=metric, mode=mode, patience=patience, min_delta=min_delta)


def _build_checkpoint(raw: Mapping[str, Any], path: str) -> CheckpointConfig:
    _check_keys(raw, {"every_epochs", "every_optimizer_steps", "keep_last", "resume"}, path)
    resume = _enum(_require(raw, "resume", path), ("none", "auto", "required"), f"{path}.resume")
    every_epochs = _optional_int(_require(raw, "every_epochs", path), f"{path}.every_epochs")
    if every_epochs is not None and every_epochs <= 0:
        raise ConfigError(f"{path}.every_epochs 必须为正整数或 null")
    every_steps = _optional_int(_require(raw, "every_optimizer_steps", path), f"{path}.every_optimizer_steps")
    if every_steps is not None and every_steps <= 0:
        raise ConfigError(f"{path}.every_optimizer_steps 必须为正整数或 null")
    keep_last = _optional_int(_require(raw, "keep_last", path), f"{path}.keep_last")
    if keep_last is not None and keep_last <= 0:
        raise ConfigError(f"{path}.keep_last 必须为正整数或 null")
    return CheckpointConfig(
        every_epochs=every_epochs,
        every_optimizer_steps=every_steps,
        keep_last=keep_last,
        resume=resume,
    )


def _build_runtime(raw: Mapping[str, Any], path: str) -> RuntimeConfig:
    _check_keys(raw, {"max_minutes", "shutdown_grace_minutes"}, path)
    max_minutes = _optional_float(_require(raw, "max_minutes", path), f"{path}.max_minutes")
    if max_minutes is not None and max_minutes <= 0:
        raise ConfigError(f"{path}.max_minutes 必须为正数或 null")
    grace = _float(_require(raw, "shutdown_grace_minutes", path), f"{path}.shutdown_grace_minutes")
    if grace < 0:
        raise ConfigError(f"{path}.shutdown_grace_minutes 必须非负")
    return RuntimeConfig(max_minutes=max_minutes, shutdown_grace_minutes=grace)


def _build_report(raw: Mapping[str, Any], path: str) -> ReportConfig:
    _check_keys(
        raw, {"enabled", "curve_sample_limit", "prediction_sample_limit", "prediction_splits"}, path
    )
    enabled = _bool(_require(raw, "enabled", path), f"{path}.enabled")
    curve = _positive_int(_require(raw, "curve_sample_limit", path), f"{path}.curve_sample_limit")
    pred = _positive_int(_require(raw, "prediction_sample_limit", path), f"{path}.prediction_sample_limit")
    splits_raw = _require(raw, "prediction_splits", path)
    if not isinstance(splits_raw, list):
        raise ConfigError(f"{path}.prediction_splits 必须是列表")
    splits: list[str] = []
    for item in splits_raw:
        s = _str(item, f"{path}.prediction_splits")
        if s not in _SPLITS:
            raise ConfigError(f"{path}.prediction_splits 含非法 split: {s!r}")
        if s in splits:
            raise ConfigError(f"{path}.prediction_splits 含重复 split: {s!r}")
        splits.append(s)
    return ReportConfig(enabled=enabled, curve_sample_limit=curve, prediction_sample_limit=pred, prediction_splits=tuple(splits))


def _build_remote(raw: Mapping[str, Any], path: str) -> RemoteConfig:
    _check_keys(
        raw,
        {"type", "host", "base_path", "user_secret_key", "password_secret_key",
         "connect_timeout_seconds", "read_timeout_seconds", "max_attempts",
         "async_upload", "failure_policy"},
        path,
    )
    rtype = _enum(_require(raw, "type", path), ("none", "alist"), f"{path}.type")
    if rtype == "none":
        return NoRemoteConfig(type="none")
    host = _str(_require(raw, "host", path), f"{path}.host")
    if not host:
        raise ConfigError(f"{path}.host 必须显式配置（不提供默认 IP）")
    base_path = _str(_require(raw, "base_path", path), f"{path}.base_path")
    if not base_path:
        raise ConfigError(f"{path}.base_path 必须非空")
    user_key = _str(_require(raw, "user_secret_key", path), f"{path}.user_secret_key")
    password_key = _str(_require(raw, "password_secret_key", path), f"{path}.password_secret_key")
    if not user_key or not password_key:
        raise ConfigError(f"{path} 必须配置 user/password Secret key")
    policy = _enum(_require(raw, "failure_policy", path), ("required", "record"), f"{path}.failure_policy")
    return AListRemoteConfig(
        type="alist",
        host=host,
        base_path=base_path,
        user_secret_key=user_key,
        password_secret_key=password_key,
        connect_timeout_seconds=_float(_require(raw, "connect_timeout_seconds", path), f"{path}.connect_timeout_seconds"),
        read_timeout_seconds=_float(_require(raw, "read_timeout_seconds", path), f"{path}.read_timeout_seconds"),
        max_attempts=_positive_int(_require(raw, "max_attempts", path), f"{path}.max_attempts"),
        async_upload=_bool(_require(raw, "async_upload", path), f"{path}.async_upload"),
        failure_policy=policy,
    )


def _build_notifications(raw: Mapping[str, Any], path: str) -> NotificationsConfig:
    _check_keys(
        raw,
        {"type", "corp_id_secret_key", "corp_secret_key", "agent_id_secret_key",
         "to_user", "connect_timeout_seconds", "read_timeout_seconds",
         "max_attempts", "failure_policy"},
        path,
    )
    ntype = _enum(_require(raw, "type", path), ("none", "wecom"), f"{path}.type")
    if ntype == "none":
        return NoNotificationsConfig(type="none")
    to_user = _str(_require(raw, "to_user", path), f"{path}.to_user")
    if not to_user:
        raise ConfigError(f"{path}.to_user 必须是非空字符串")
    for key in ("corp_id_secret_key", "corp_secret_key", "agent_id_secret_key"):
        v = _str(_require(raw, key, path), f"{path}.{key}")
        if not v:
            raise ConfigError(f"{path}.{key} 必须非空")
    policy = _enum(_require(raw, "failure_policy", path), ("required", "record"), f"{path}.failure_policy")
    return WecomNotificationsConfig(
        type="wecom",
        corp_id_secret_key=_str(_require(raw, "corp_id_secret_key", path), f"{path}.corp_id_secret_key"),
        corp_secret_key=_str(_require(raw, "corp_secret_key", path), f"{path}.corp_secret_key"),
        agent_id_secret_key=_str(_require(raw, "agent_id_secret_key", path), f"{path}.agent_id_secret_key"),
        to_user=to_user,
        connect_timeout_seconds=_float(_require(raw, "connect_timeout_seconds", path), f"{path}.connect_timeout_seconds"),
        read_timeout_seconds=_float(_require(raw, "read_timeout_seconds", path), f"{path}.read_timeout_seconds"),
        max_attempts=_positive_int(_require(raw, "max_attempts", path), f"{path}.max_attempts"),
        failure_policy=policy,
    )


def parse_config(data: Mapping[str, Any]) -> Config:
    """从已解析的严格 mapping 构造 frozen Config，并执行跨字段校验。"""
    _check_keys(
        data,
        {"schema_version", "run", "experiment", "training", "backend", "distributed",
         "selection", "checkpoint", "runtime", "report", "remote", "notifications"},
        "$",
    )
    schema_version = _int(_require(data, "schema_version", "$"), "$.schema_version")
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"不支持的 schema_version: {schema_version!r}（期望 {CONFIG_SCHEMA_VERSION}）")

    experiment_raw = _require(data, "experiment", "$")
    if not isinstance(experiment_raw, Mapping):
        raise ConfigError("$.experiment 必须是 mapping")
    validate_json_value(dict(experiment_raw), "$.experiment")

    cfg = Config(
        schema_version=schema_version,
        run=_build_run(_require(data, "run", "$"), "$.run"),
        experiment=dict(experiment_raw),
        training=_build_training(_require(data, "training", "$"), "$.training"),
        backend=_build_backend(_require(data, "backend", "$"), "$.backend"),
        distributed=_build_distributed(_require(data, "distributed", "$"), "$.distributed"),
        selection=(_build_selection(data["selection"], "$.selection") if data.get("selection") is not None else None),
        checkpoint=_build_checkpoint(_require(data, "checkpoint", "$"), "$.checkpoint"),
        runtime=_build_runtime(_require(data, "runtime", "$"), "$.runtime"),
        report=_build_report(_require(data, "report", "$"), "$.report"),
        remote=_build_remote(_require(data, "remote", "$"), "$.remote"),
        notifications=_build_notifications(_require(data, "notifications", "$"), "$.notifications"),
    )
    _cross_validate(cfg)
    return cfg


def parse_config_text(text: str) -> Config:
    """从严格 UTF-8 YAML 文本解析配置。"""
    data = yaml_load_strict(text)
    _reject_unsafe_strings(data, "$")
    return parse_config(data)


def load_config_file(path: str) -> Config:
    """读取并解析 UTF-8 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_config_text(text)


def _cross_validate(cfg: Config) -> None:
    backend = cfg.backend
    if backend.type == "torch":
        if backend.torch is None or backend.sklearn is not None:
            raise ConfigError("torch backend 必须配置 torch 分支且 sklearn 分支为 null")
        np_ = cfg.distributed.num_processes
        if np_ != "auto" and not isinstance(np_, int):
            raise ConfigError("distributed.num_processes 在 torch backend 必须是正整数或 auto")
        if cfg.checkpoint.every_optimizer_steps is not None:
            # 需要 DataModule 支持中途恢复，由 preflight 进一步校验；此处至少要求非负已由 builder 保证
            pass
    else:
        if backend.sklearn is None or backend.torch is not None:
            raise ConfigError("sklearn backend 必须配置 sklearn 分支且 torch 分支为 null")
        if cfg.distributed.num_processes != 1:
            raise ConfigError("sklearn backend 的 distributed.num_processes 必须为 1")
        if cfg.checkpoint.every_optimizer_steps is not None:
            raise ConfigError("sklearn backend 不支持 every_optimizer_steps（incremental 使用 batch 边界检查点）")
        if backend.sklearn.fit_mode == "batch":
            if cfg.training.max_epochs != 1:
                raise ConfigError("sklearn batch fit_mode 要求 training.max_epochs=1")
            if cfg.checkpoint.resume != "none":
                raise ConfigError("sklearn batch fit_mode 要求 checkpoint.resume=none")
            if cfg.runtime.max_minutes is not None:
                raise ConfigError("sklearn batch fit_mode 不允许运行时预算（fit 无受控暂停点）")

    # runtime grace 关系
    if cfg.runtime.max_minutes is not None:
        if cfg.runtime.shutdown_grace_minutes >= cfg.runtime.max_minutes:
            raise ConfigError("runtime.shutdown_grace_minutes 必须小于 runtime.max_minutes")

    # selection
    if cfg.selection is not None:
        if cfg.selection.mode not in ("min", "max"):
            raise ConfigError("selection.mode 必须是 min/max")
        if cfg.selection.patience < 0 or cfg.selection.min_delta < 0:
            raise ConfigError("selection.patience/min_delta 必须非负")

    # report
    if "val" not in cfg.report.prediction_splits and "test" not in cfg.report.prediction_splits:
        # 允许，但无意义；不强制
        pass


# --------------------------------------------------------------------------
# 规范化序列化
# --------------------------------------------------------------------------

def config_to_dict(cfg: Config) -> dict[str, Any]:
    """把 frozen Config 规范化为 JSON-safe dict。"""

    def as_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: as_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, Mapping):
            return {str(k): as_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [as_dict(v) for v in obj]
        return obj

    data = as_dict(cfg)
    # 规范化：排序键、JSON-safe
    return _normalize(data)


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"配置含非有限数值: {value!r}")
        return value
    if isinstance(value, str):
        return value
    raise ConfigError(f"配置含不可序列化值: {type(value).__name__}")


def config_canonical_json(cfg: Config) -> str:
    return json.dumps(config_to_dict(cfg), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_fingerprint(cfg: Config, resume: bool = False) -> str:
    """配置指纹。

    resume=True 时排除允许恢复时变化的字段：training.max_epochs、runtime.*、
    checkpoint.every_*、checkpoint.keep_last、report.*、remote/notifications 超时与重试。
    """
    data = config_to_dict(cfg)
    if resume:
        data = dict(data)
        training = dict(data["training"])
        training.pop("max_epochs", None)
        data["training"] = training
        # runtime.* 全部允许恢复时变化（预算/grace 不影响恢复兼容）
        data.pop("runtime", None)
        ckpt = dict(data["checkpoint"])
        ckpt.pop("every_epochs", None)
        ckpt.pop("every_optimizer_steps", None)
        ckpt.pop("keep_last", None)
        data["checkpoint"] = ckpt
        report = dict(data["report"])
        data["report"] = {
            k: v for k, v in report.items()
            if k in ("enabled", "curve_sample_limit", "prediction_sample_limit", "prediction_splits")
        }
        if "remote" in data:
            remote = data["remote"]
            if isinstance(remote, Mapping) and remote.get("type") == "alist":
                data["remote"] = {k: v for k, v in remote.items()
                                  if k not in ("connect_timeout_seconds", "read_timeout_seconds", "max_attempts")}
        if "notifications" in data:
            notif = data["notifications"]
            if isinstance(notif, Mapping) and notif.get("type") == "wecom":
                data["notifications"] = {k: v for k, v in notif.items()
                                         if k not in ("connect_timeout_seconds", "read_timeout_seconds", "max_attempts")}
    return _sha256_text(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


_TUNING_TOP_KEYS = ("experiment", "training", "backend", "selection")


def tuning_fingerprint(cfg: Config) -> str:
    """tuning 指纹：只覆盖会改变拟合结果或停止位置的字段。"""
    data = config_to_dict(cfg)
    subset: dict[str, Any] = {}
    for key in _TUNING_TOP_KEYS:
        if key in data:
            subset[key] = data[key]
    return _sha256_text(json.dumps(subset, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


# --------------------------------------------------------------------------
# Variant resolver（任务 1.3）
# --------------------------------------------------------------------------

VARIANT_ALLOWED_TOP = {
    "run": {"name", "tags"},
    "experiment": None,
    "training": None,
    "backend": None,
    "selection": None,
    "checkpoint": None,
    "runtime": None,
    "report": None,
    "remote": None,
    "notifications": None,
}

VARIANT_FORBIDDEN_TOP = {"schema_version", "distributed"}

# checkpoint 只允许频率/保留参数
_VARIANT_CHECKPOINT_ALLOWED = {"every_epochs", "every_optimizer_steps", "keep_last"}
# backend 分支内允许（由 base 决定分支，不允许改 type / 分支选择）
_VARIANT_BACKEND_FORBIDDEN = {"type", "torch", "sklearn"}


def _resolve_variant_path(base_dir: str, raw_path: str, label: str) -> str:
    """解析 base/variant 文件路径，拒绝 URL 与 symlink 逃逸。

    允许调用方传入已解析的绝对路径（如 sweep 子进程），但仍强制位于 base_dir 树内。
    """
    if not isinstance(raw_path, str) or not raw_path:
        raise ConfigError(f"{label} 路径必须是非空字符串")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_path):
        raise ConfigError(f"{label} 不允许 URL 路径: {raw_path!r}")
    candidate = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    resolved = os.path.realpath(candidate)
    base_real = os.path.realpath(base_dir)
    if not (resolved == base_real or resolved.startswith(base_real + os.sep)):
        raise ConfigError(f"{label} 路径逃逸目录边界: {raw_path!r}")
    return resolved


def _merge_mapping(base: Mapping[str, Any], patch: Mapping[str, Any], path: str) -> dict[str, Any]:
    """mapping 递归合并，scalar/list/null 整体替换。"""
    result: dict[str, Any] = {}
    for key, bval in base.items():
        if key in patch:
            pval = patch[key]
            if isinstance(bval, Mapping) and isinstance(pval, Mapping):
                result[key] = _merge_mapping(bval, pval, f"{path}.{key}")
            else:
                result[key] = pval
        else:
            result[key] = bval
    for key, pval in patch.items():
        if key not in base:
            result[key] = pval
    return result


def _check_variant_allowed(patch: Mapping[str, Any], path: str) -> None:
    """校验 variant 顶层只覆盖允许字段且不触及禁止基础设施。"""
    for key in patch:
        if key in VARIANT_FORBIDDEN_TOP:
            raise ConfigError(f"variant 禁止覆盖 {path}.{key}")
        if key not in VARIANT_ALLOWED_TOP:
            raise ConfigError(f"variant 覆盖了不允许的顶层字段 {path}.{key}")
    # run 只允许 name/tags
    run_patch = patch.get("run")
    if run_patch is not None:
        if not isinstance(run_patch, Mapping):
            raise ConfigError(f"{path}.run 必须是 mapping")
        for key in run_patch:
            if key not in VARIANT_ALLOWED_TOP["run"]:
                raise ConfigError(f"variant 禁止覆盖 run.{key}")
    # checkpoint 只允许频率/保留
    ckpt_patch = patch.get("checkpoint")
    if ckpt_patch is not None:
        if not isinstance(ckpt_patch, Mapping):
            raise ConfigError(f"{path}.checkpoint 必须是 mapping")
        for key in ckpt_patch:
            if key not in _VARIANT_CHECKPOINT_ALLOWED:
                raise ConfigError(f"variant 禁止覆盖 checkpoint.{key}")
    # backend 不允许改 type / 分支切换
    backend_patch = patch.get("backend")
    if backend_patch is not None:
        if not isinstance(backend_patch, Mapping):
            raise ConfigError(f"{path}.backend 必须是 mapping")
        for key in backend_patch:
            if key in _VARIANT_BACKEND_FORBIDDEN:
                raise ConfigError(f"variant 禁止覆盖 backend.{key}")


def resolve_variant_text(base_text: str, variant_text: str, label: str = "variant") -> Config:
    """合并完整 base 与严格 variant 文本，重跑完整 schema。

    variant 不含 schema_version。
    """
    base_data = yaml_load_strict(base_text)
    variant_data = yaml_load_strict(variant_text)
    _reject_unsafe_strings(base_data, "base")
    _reject_unsafe_strings(variant_data, "variant")
    _check_variant_allowed(variant_data, "variant")
    merged = _merge_mapping(base_data, variant_data, "merged")
    merged.pop("schema_version", None)
    merged["schema_version"] = base_data.get("schema_version", CONFIG_SCHEMA_VERSION)
    return parse_config(merged)


def resolve_variant_files(base_path: str, variant_path: str) -> Config:
    """从文件解析并合并 base/variant，执行路径边界检查。"""
    base_path = os.path.realpath(base_path)
    variant_path = _resolve_variant_path(os.path.dirname(base_path), variant_path, "variant")
    with open(base_path, "r", encoding="utf-8") as f:
        base_text = f.read()
    with open(variant_path, "r", encoding="utf-8") as f:
        variant_text = f.read()
    return resolve_variant_text(base_text, variant_text)


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# 默认 schema（供示例与测试参考）
# --------------------------------------------------------------------------

def default_schema() -> dict[str, Any]:
    """返回可复制的默认 base schema dict。"""
    return {
        "schema_version": 1,
        "run": {
            "name": "example",
            "id": None,
            "output_root": None,
            "source_revision": None,
            "seed": 42,
            "tags": {},
        },
        "experiment": {},
        "training": {"max_epochs": 20, "log_every_steps": 20},
        "backend": {
            "type": "torch",
            "torch": {
                "gradient_accumulation_steps": 1,
                "mixed_precision": "auto",
                "compile": False,
                "clip_grad_norm": 1.0,
                "deterministic": "off",
                "matmul_precision": "high",
                "find_unused_parameters": False,
            },
            "sklearn": None,
        },
        "distributed": {"num_processes": "auto"},
        "selection": {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0},
        "checkpoint": {"every_epochs": 1, "every_optimizer_steps": None, "keep_last": 2, "resume": "none"},
        "runtime": {"max_minutes": None, "shutdown_grace_minutes": 10},
        "report": {
            "enabled": True,
            "curve_sample_limit": 100000,
            "prediction_sample_limit": 10000,
            "prediction_splits": ["val", "test"],
        },
        "remote": {"type": "none"},
        "notifications": {"type": "none"},
    }
