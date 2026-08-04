"""doctor：不训练的后端感知预检，聚合多错误并输出 evaluation contract。

doctor 不执行拟合/检查点/远程目录/通知；Secret 只显示 key。
"""
from __future__ import annotations

import json
import os
from typing import Any

from .config import Config
from .platform import Platform, free_disk_bytes, resolve_source_revision

MIN_FREE_DISK_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB


def run_doctor(
    config: Config,
    platform: Platform,
    experiment_ref: str,
    emit_contract: bool = False,
) -> list[str]:
    """返回聚合错误列表；空列表表示通过。"""
    errors: list[str] = []
    try:
        _check_disk(config, platform)
    except Exception as exc:
        errors.append(str(exc))
    try:
        _check_revision(config, platform)
    except Exception as exc:
        errors.append(str(exc))
    try:
        _check_paths(config, platform)
    except Exception as exc:
        errors.append(str(exc))
    try:
        _check_services(config)
    except Exception as exc:
        errors.append(str(exc))
    # backend 专属
    try:
        _check_backend(config, experiment_ref)
    except Exception as exc:
        errors.append(str(exc))
    if emit_contract:
        _emit_contract(config, platform, experiment_ref, errors)
    return errors


def _check_disk(config: Config, platform: Platform) -> None:
    output_root = platform.resolve_output_root(config)
    free = free_disk_bytes(output_root)
    if free < MIN_FREE_DISK_BYTES:
        raise RuntimeError(f"输出目录可用空间不足 5 GiB: {free / 1e9:.2f} GiB at {output_root}")


def _check_revision(config: Config, platform: Platform) -> None:
    if platform.is_kaggle:
        resolve_source_revision(config)
    elif config.run.source_revision is None:
        try:
            resolve_source_revision(config)
        except Exception as exc:
            raise RuntimeError(f"本地无 Git revision 且未显式提供: {exc}") from exc


def _check_paths(config: Config, platform: Platform) -> None:
    platform.validate_kaggle_inputs(config)
    if config.run.id:
        import re
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", config.run.id):
            raise RuntimeError(f"run.id 不匹配字符集: {config.run.id!r}")


def _check_services(config: Config) -> None:
    # 服务配置存在性检查（Secret 解析在启用服务时进行）
    if config.remote.type == "alist":
        remote = config.remote
        if not (remote.user_secret_key and remote.password_secret_key):
            raise RuntimeError("alist remote 缺少 user/password Secret key")
    if config.notifications.type == "wecom":
        notif = config.notifications
        for key in ("corp_id_secret_key", "corp_secret_key", "agent_id_secret_key"):
            if not getattr(notif, key):
                raise RuntimeError(f"wecom notifications 缺少 {key}")
        if not notif.to_user:
            raise RuntimeError("wecom notifications 缺少 to_user")


def _check_backend(config: Config, experiment_ref: str) -> None:
    from .backends.torch_backend import TorchBackendError, build_torch_components, validate_fresh_components
    from .backends.sklearn_backend import (
        SklearnBackendError,
        apply_params,
        build_sklearn_experiment,
        clone_estimator,
        validate_estimator_contract,
    )

    if config.backend.type == "torch":
        if not experiment_ref:
            raise RuntimeError("torch doctor 需要 --experiment 引用")
        from .backends.torch_backend import build_experiment_from_ref
        try:
            experiment = build_experiment_from_ref(experiment_ref, config.experiment)
            model, datamodule, task, optimizer, scheduler = build_torch_components(experiment, config)
            validate_fresh_components(model, datamodule, task, optimizer, scheduler, config)
        except (TorchBackendError, ImportError, Exception) as exc:
            raise RuntimeError(f"torch 组件预检失败: {exc}") from exc
    else:
        try:
            experiment = build_sklearn_experiment(experiment_ref, config.experiment)
            estimator = clone_estimator(experiment)
            task = experiment.task_factory()
            validate_estimator_contract(estimator, task, config)
            apply_params(estimator, config)
        except (SklearnBackendError, Exception) as exc:
            raise RuntimeError(f"sklearn 组件预检失败: {exc}") from exc


def _emit_contract(config: Config, platform: Platform, experiment_ref: str, errors: list[str]) -> None:
    """输出 evaluation contract JSON 到 stdout（sweep 可比性预检）。"""
    contract: dict[str, Any] = {
        "schema_version": 1,
        "backend": config.backend.type,
        "experiment_ref": experiment_ref,
        "valid": not errors,
        "errors": errors,
        "config_fingerprint": _config_fp(config),
        "platform": platform.kind,
    }
    if not errors:
        try:
            info = _extract_contract_info(config, experiment_ref)
            contract.update(info)
        except Exception as exc:
            # OSR-008：预检合同提取失败 → invalid，不静默通过
            contract["valid"] = False
            contract["errors"] = contract.get("errors", []) + [f"contract 提取失败: {exc}"]
    print(json.dumps(contract, ensure_ascii=False))


def _extract_contract_info(config: Config, experiment_ref: str) -> dict[str, Any]:
    """提取 DataIdentity / Task / MetricDefinition / splits / model signature 用于 sweep 可比性。"""
    model = None
    if config.backend.type == "torch":
        from .backends.torch_backend import build_experiment_from_ref, build_torch_components
        experiment = build_experiment_from_ref(experiment_ref, config.experiment)
        model, datamodule, task, optimizer, scheduler = build_torch_components(experiment, config)
    else:
        from .backends.sklearn_backend import build_sklearn_experiment
        experiment = build_sklearn_experiment(experiment_ref, config.experiment)
        datamodule = experiment.datamodule_factory()
        task = experiment.task_factory()
    # model signature
    model_sig: dict[str, Any] | None = None
    if config.backend.type == "torch" and model is not None:
        from .backends.torch_backend import model_signature
        model_sig = model_signature(model)
    else:
        estimator = experiment.estimator_factory()
        model_sig = {"class": f"{type(estimator).__module__}.{type(estimator).__name__}"}
    from .contracts import build_evaluation_contract
    return build_evaluation_contract(datamodule, task, config.backend.type, model_sig)


def _config_fp(config: Config) -> str:
    from .config import config_fingerprint
    return config_fingerprint(config)
