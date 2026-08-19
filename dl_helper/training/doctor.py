"""doctor：不训练的后端感知预检，聚合多错误并输出 evaluation contract。

doctor 不执行拟合/检查点/远程目录/通知；Secret 只显示 key。
"""
from __future__ import annotations

import json
import os
from typing import Any

from .config import RESUME_AUTO, Config, ConfigError
from .platform import Platform, SecretError, SecretResolver, execution_policy_for, free_disk_bytes, resolve_source_revision

MIN_FREE_DISK_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB


def validate_training_start(
    config: Config,
    platform: Platform,
    experiment_ref: str,
    *,
    emit_contract: bool = False,
    resume: str = RESUME_AUTO,
    execution_policy=None,
) -> None:
    """训练入口统一调用的前置检查。

    该函数是库内部运行时合同，不作为 CLI 命令暴露。检查失败聚合为一个
    `ConfigError`，在创建 run 目录或启动训练进程前抛出。
    """
    collected: list[BaseException] = []
    errors = run_doctor(config, platform, experiment_ref, emit_contract=False,
                        resume=resume, execution_policy=execution_policy,
                        collect_exceptions=collected)
    if platform.is_kaggle:
        errors.extend(_check_kaggle_requirements(config, platform, execution_policy))
    if errors:
        # preflight 可比性模式（sweep）聚合为 ConfigError 列出全部问题；
        # 真实训练中实验导入失败是运行前致命错误，直接传播原始
        # ModuleNotFoundError/ImportError，不被 ConfigError 掩盖（OSR-003）。
        cause = _find_import_cause(collected)
        if cause is not None and not emit_contract:
            raise cause
        raise ConfigError("训练启动预检失败:\n" + "\n".join(f"- {item}" for item in errors))
    if emit_contract:
        _emit_contract(config, platform, experiment_ref, [])


def _find_import_cause(exceptions: list) -> BaseException | None:
    """沿异常链追踪收集到的异常，返回最深的 ImportError/ModuleNotFoundError。"""
    seen: set[int] = set()
    stack = list(exceptions)
    while stack:
        exc = stack.pop()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, (ImportError, ModuleNotFoundError)):
                return cur
            if cur.__cause__ is not None:
                cur = cur.__cause__
            elif cur.__context__ is not None:
                cur = cur.__context__
            else:
                break
    return None


def _check_kaggle_requirements(config: Config, platform: Platform, execution_policy=None) -> list[str]:
    """Kaggle 必须应用独立执行策略（660/10），并启用 AList、企业微信与预解析全部 Secret。"""
    errors: list[str] = []
    expected = execution_policy_for(platform)
    if execution_policy is None:
        errors.append("Kaggle 必须应用独立 ExecutionPolicy（660 分钟预算 + 10 分钟收尾，720 上限内）")
    elif execution_policy != expected:
        errors.append(
            f"Kaggle ExecutionPolicy 必须为 "
            f"{expected.max_minutes:g}/{expected.shutdown_grace_minutes:g}，"
            f"得到 {execution_policy.max_minutes or 0:g}/{execution_policy.shutdown_grace_minutes:g}"
        )
    if config.remote.type != "alist":
        errors.append("Kaggle 必须启用 remote.type=alist")
    elif config.remote.failure_policy != "required":
        errors.append("Kaggle AList failure_policy 必须为 required")
    if config.notifications.type != "wecom":
        errors.append("Kaggle 必须启用 notifications.type=wecom")
    elif config.notifications.failure_policy != "required":
        errors.append("Kaggle 企业微信 failure_policy 必须为 required")

    resolver = SecretResolver(platform)
    keys: list[str] = []
    if config.remote.type == "alist":
        keys.extend([config.remote.user_secret_key, config.remote.password_secret_key])
    if config.notifications.type == "wecom":
        keys.extend([
            config.notifications.corp_id_secret_key,
            config.notifications.corp_secret_key,
            config.notifications.agent_id_secret_key,
        ])
    for key in keys:
        try:
            resolver.resolve(key)
        except SecretError as exc:
            errors.append(f"Secret {key!r} 不可用: {exc}")
    return errors


def run_doctor(
    config: Config,
    platform: Platform,
    experiment_ref: str,
    emit_contract: bool = False,
    resume: str = RESUME_AUTO,
    execution_policy=None,
    collect_exceptions: list | None = None,
) -> list[str]:
    """返回聚合错误列表；空列表表示通过。

    collect_exceptions 若给定，则在捕获每个检查异常时收集原始异常对象，
    供调用方保留失败根因（如实验导入的 ImportError）。
    """
    errors: list[str] = []

    def _record(exc: BaseException) -> str:
        if collect_exceptions is not None and exc not in collect_exceptions:
            collect_exceptions.append(exc)
        return str(exc)

    try:
        _check_disk(config, platform)
    except Exception as exc:
        errors.append(_record(exc))
    try:
        _check_revision(config, platform)
    except Exception as exc:
        errors.append(_record(exc))
    try:
        _check_paths(config, platform)
    except Exception as exc:
        errors.append(_record(exc))
    try:
        _check_services(config)
    except Exception as exc:
        errors.append(_record(exc))
    # backend 专属
    try:
        _check_backend(config, experiment_ref, resume=resume, execution_policy=execution_policy)
    except Exception as exc:
        errors.append(_record(exc))
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


def _check_backend(config: Config, experiment_ref: str, resume: str = RESUME_AUTO,
                   execution_policy=None) -> None:
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
            validate_fresh_components(model, datamodule, task, optimizer, scheduler, config,
                                      execution_policy=execution_policy)
        except (TorchBackendError, ImportError, Exception) as exc:
            raise RuntimeError(f"torch 组件预检失败: {exc}") from exc
    else:
        try:
            experiment = build_sklearn_experiment(experiment_ref, config.experiment)
            estimator = clone_estimator(experiment)
            task = experiment.task_factory()
            datamodule = experiment.datamodule_factory()
            validate_estimator_contract(estimator, task, config)
            apply_params(estimator, config)
            # D-002：显式 required 对 sklearn batch 无效，在预检阶段失败
            if resume == "required":
                if config.backend.sklearn is not None and config.backend.sklearn.fit_mode == "batch":
                    raise RuntimeError("sklearn batch fit_mode 不支持受控恢复，resume=required 无效")
                if not datamodule.incremental_train_data().supports_mid_fit_resume:
                    raise RuntimeError("sklearn incremental 数据源不支持中途恢复，resume=required 无效")
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
