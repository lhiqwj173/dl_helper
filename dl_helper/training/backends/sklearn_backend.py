"""sklearn backend worker：clone/预检、batch fit、incremental partial_fit 与可信 joblib。"""
from __future__ import annotations

import math
import os
from typing import Any, Mapping

import numpy as np

from ..artifacts import RunLayout, append_jsonl, sha256_file, write_json, write_prediction_manifest, write_prediction_shard
from ..checkpoint import CheckpointError, apply_retention, validate_sklearn_checkpoint_source, write_model_manifest, write_sklearn_checkpoint
from ..config import Config
from ..contracts import (
    EstimatorBatch,
    SklearnExperiment,
    validate_backend_match,
    validate_experiment,
    validate_sklearn_task,
)
from ..engine import EngineState, validate_selection
from ..metrics import StageMetricState
from .base import BackendResult, ModelArtifact

_SHARD_BATCH = 4096


class SklearnBackendError(Exception):
    """sklearn backend 合同违规。"""


def import_experiment_ref(ref: str):
    if ":" not in ref:
        raise SklearnBackendError(f"实验引用必须为 module:function: {ref!r}")
    module_path, _, func_name = ref.partition(":")
    import importlib
    module = importlib.import_module(module_path)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise SklearnBackendError(f"实验工厂不可调用: {ref!r}")
    return func


def build_sklearn_experiment(ref: str, experiment_config: Mapping[str, Any]) -> SklearnExperiment:
    func = import_experiment_ref(ref)
    experiment = func(dict(experiment_config))
    validate_experiment(experiment)
    if not isinstance(experiment, SklearnExperiment):
        raise SklearnBackendError(f"实验引用必须返回 SklearnExperiment: {type(experiment).__name__}")
    return experiment


def clone_estimator(experiment: SklearnExperiment):
    """clone(safe=True) 并确认原始 estimator 未拟合、clone 可拟合。"""
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    estimator = experiment.estimator_factory()
    # 原始对象已拟合 → 拒绝（clone 会剥离拟合状态，但用户意图是复用 fitted 模型）
    try:
        check_is_fitted(estimator)
        raise SklearnBackendError("estimator 已拟合，必须返回未拟合对象")
    except NotFittedError:
        pass
    try:
        cloned = clone(estimator, safe=True)
    except TypeError as exc:
        raise SklearnBackendError(f"estimator 无法 clone(safe=True): {exc}") from exc
    try:
        check_is_fitted(cloned)
        raise SklearnBackendError("clone 产物意外已拟合")
    except NotFittedError:
        pass
    return cloned


def validate_estimator_contract(estimator, task, config: Config) -> None:
    """estimator kind / prediction capability / fit mode / 参数预检。"""
    from sklearn.base import is_classifier, is_regressor

    if task.estimator_kind == "classifier" and not is_classifier(estimator):
        raise SklearnBackendError(f"Task 声明 classifier，但 estimator 不是 classifier")
    if task.estimator_kind == "regressor" and not is_regressor(estimator):
        raise SklearnBackendError(f"Task 声明 regressor，但 estimator 不是 regressor")
    if not hasattr(estimator, "predict"):
        raise SklearnBackendError("estimator 缺少 predict")
    required = task.required_prediction
    if required not in ("predict", "decision_function", "predict_proba"):
        raise SklearnBackendError(f"Task required_prediction 非法: {required!r}")
    if required != "predict" and not hasattr(estimator, required):
        raise SklearnBackendError(
            f"Task 要求 {required}，但 estimator 未实现（不允回归退到其他输出）"
        )

    backend = config.backend.sklearn
    fit_mode = backend.fit_mode
    if fit_mode == "batch" and not hasattr(estimator, "fit"):
        raise SklearnBackendError("batch fit_mode 要求 estimator 实现 fit")
    if fit_mode == "incremental":
        if not hasattr(estimator, "partial_fit"):
            raise SklearnBackendError("incremental fit_mode 要求 estimator 实现 partial_fit")
        if task.estimator_kind == "classifier" and task.classes is None:
            raise SklearnBackendError("incremental classifier 必须由 Task 声明完整 classes")


def apply_params(estimator, config: Config) -> None:
    """递归 random_state、顶层 n_jobs 参数解析。"""
    backend = config.backend.sklearn
    params = estimator.get_params(deep=True)
    # n_jobs
    if backend.n_jobs is not None:
        if "n_jobs" not in params:
            raise SklearnBackendError("n_jobs 非 null 时 estimator 必须暴露顶层 n_jobs")
        n_jobs = backend.n_jobs
        if n_jobs == "auto":
            n_jobs = os.cpu_count() or 1
        estimator.set_params(n_jobs=n_jobs)
    # random_state
    if backend.random_state == "run_seed":
        for name, value in params.items():
            if name.endswith("random_state") and value is None:
                estimator.set_params(**{name: config.run.seed})
    elif backend.random_state == "require_explicit":
        for name, value in params.items():
            if name.endswith("random_state") and value is None:
                raise SklearnBackendError(
                    f"random_state=require_explicit 但参数 {name!r} 为 null"
                )


def resolve_fit_kwargs(config: Config, batch: EstimatorBatch) -> dict[str, Any]:
    """解析 fit/partial_fit 的 sample_weight 参数路径。"""
    backend = config.backend.sklearn
    sw_param = backend.sample_weight_parameter
    kwargs: dict[str, Any] = {}
    has_weight = batch.sample_weight is not None
    if has_weight:
        if sw_param is None:
            raise SklearnBackendError("训练数据带 sample_weight 但未配置 sample_weight_parameter")
        kwargs[sw_param] = batch.sample_weight
    else:
        if sw_param is not None:
            raise SklearnBackendError("配置了 sample_weight_parameter 但训练数据无权重")
    return kwargs


def evaluate_estimator(estimator, task, datamodule, stage, layout, config) -> StageMetricState:
    """按 evaluation batch 生成 PredictedBatch 并更新指标状态。"""
    state = task.metric_state(stage)
    shard_arrays: list[dict[str, np.ndarray]] = []
    shard_n = 0
    shard_count = 0
    for batch in datamodule.evaluation_batches(stage):
        predicted = task.predict_batch(estimator, batch)
        state.update_predicted(predicted)
        arrays = task.prediction_arrays(predicted)
        shard_arrays.append(arrays)
        shard_n += predicted.sample_count
        if shard_n >= _SHARD_BATCH:
            _write_shard_batch(layout, stage, 0, shard_count, shard_arrays, shard_n)
            shard_count += 1
            shard_arrays = []
            shard_n = 0
    if shard_arrays:
        _write_shard_batch(layout, stage, 0, shard_count, shard_arrays, shard_n)
    return state


def _write_shard_batch(layout, stage, rank, shard_count, arrays_list, total_n):
    entries = []
    for i, arrays in enumerate(arrays_list):
        sample_count = arrays[list(arrays)[0]].shape[0]
        entry = write_prediction_shard(layout.predictions_dir(stage), rank, shard_count + i, arrays, sample_count)
        entries.append(entry)
    write_prediction_manifest(layout.predictions_dir(stage), stage, entries, total_n,
                              sampled=True, total_sample_count=total_n,
                              sampling_notes="分片存储全部样本；曲线抽样在报告阶段进行")


class SklearnBackend:
    """sklearn backend worker。"""

    name = "sklearn"

    def run(self, experiment, config, platform, layout) -> BackendResult:
        return run_sklearn_worker_experiment(experiment, config, platform, layout)


def run_sklearn_worker_experiment(
    experiment: SklearnExperiment, config: Config, platform: Any, layout: RunLayout,
    resume: str = "none", budget_monotonic=None, services=None, publish_terminal=True,
    execution_policy=None,
) -> BackendResult:
    from sklearn.base import clone as sklearn_clone

    validate_experiment(experiment)
    validate_backend_match(experiment, "sklearn")
    task = experiment.task_factory()
    validate_sklearn_task(task)

    datamodule = experiment.datamodule_factory()
    backend = config.backend.sklearn
    fit_mode = backend.fit_mode
    if fit_mode == "batch":
        from ..contracts import validate_sklearn_batch_datamodule
        validate_sklearn_batch_datamodule(datamodule)
    else:
        from ..contracts import validate_sklearn_incremental_datamodule
        validate_sklearn_incremental_datamodule(datamodule)
        validate_selection(config.selection, task.metric_definitions,
                           any(True for _ in datamodule.evaluation_batches("val")))

    estimator = clone_estimator(experiment)
    validate_estimator_contract(estimator, task, config)
    apply_params(estimator, config)

    datamodule.setup("fit")
    data_fp = datamodule.identity().fingerprint
    model_sig = _sklearn_model_signature(estimator, config)
    engine_state = EngineState(
        backend="sklearn", run_id=config.run.id or "unknown",
        config_fingerprint=_config_fingerprint(config),
        metric_name=config.selection.metric if config.selection else None,
        mode=config.selection.mode if config.selection else None,
        patience=config.selection.patience if config.selection else None,
        min_delta=config.selection.min_delta if config.selection else 0.0,
    )

    from ..artifacts import existing_terminal
    if existing_terminal(layout.run_dir) == "run-manifest.json":
        raise SklearnBackendError(f"run 已成功完成，禁止重跑改写: {engine_state.run_id}")
    if services is not None:
        services.start_run(engine_state.run_id)
    if fit_mode == "batch":
        result = _run_batch(estimator, task, datamodule, config, layout, engine_state, data_fp, model_sig,
                            services=services, publish_terminal=publish_terminal)
    else:
        result = _run_incremental(estimator, task, datamodule, config, layout, engine_state, data_fp, model_sig,
                                  resume=resume, budget_monotonic=budget_monotonic, services=services,
                                  publish_terminal=publish_terminal, execution_policy=execution_policy)
    return result


def _config_fingerprint(config: Config) -> str:
    from ..config import config_fingerprint
    return config_fingerprint(config, resume=True)


def _sklearn_model_signature(estimator, config: Config) -> dict[str, Any]:
    from ..config import config_to_dict
    return {
        "class": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "params": config_to_dict(config) if False else _params_json(estimator),
        "library_versions": _library_versions(),
    }


def _params_json(estimator) -> dict[str, Any]:
    import json
    params = estimator.get_params(deep=True)
    out: dict[str, Any] = {}
    for k, v in sorted(params.items()):
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) for x in v]
        else:
            out[k] = str(v)
    return out


def _library_versions() -> dict[str, str]:
    import numpy, scipy, sklearn
    try:
        import joblib
        joblib_v = joblib.__version__
    except Exception:
        joblib_v = "unknown"
    return {"sklearn": sklearn.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "joblib": joblib_v}


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------

def _run_batch(estimator, task, datamodule, config, layout, engine_state, data_fp, model_sig,
                services=None, publish_terminal=True) -> BackendResult:
    train = datamodule.full_train_data()
    fit_kwargs = resolve_fit_kwargs(config, train)
    layout.log(f"sklearn batch fit start estimator={model_sig['class']}")
    estimator.fit(train.features, train.targets, **fit_kwargs)
    layout.log("sklearn batch fit done")

    metric_states: dict[str, StageMetricState] = {}
    for stage in ("train", "val", "test", "predict"):
        try:
            has = any(True for _ in datamodule.evaluation_batches(stage))
        except Exception:
            has = False
        if has:
            metric_states[stage] = evaluate_estimator(estimator, task, datamodule, stage, layout, config)
            _persist_stage_metrics(layout, metric_states[stage], stage, 0, 0)

    # best = last
    model_artifact = _export_joblib_model(layout, estimator, model_sig, config, engine_state)

    if config.selection is not None and "val" in metric_states:
        _apply_selection(engine_state, metric_states["val"], config)
    _write_summary(layout, config, "sklearn", engine_state, model_artifact, model_sig, data_fp)
    # OSR-002：核心 Artifact（eval contract/report）先于服务 bundle 完成
    report_path = _write_sklearn_core_artifacts(layout, config, task, datamodule, estimator=estimator)
    if services is not None:
        services.finalize_run(engine_state.run_id, "succeeded")
    if publish_terminal:
        _publish_sklearn_terminal(layout, "succeeded", config, engine_state, model_sig, data_fp,
                                  task=task, datamodule=datamodule, services=services,
                                  report_path=report_path)
    return BackendResult(status="succeeded", epoch=0, batch_in_epoch=0, global_step=0,
                         model_artifact=model_artifact, environment_stats={})


# --------------------------------------------------------------------------
# incremental
# --------------------------------------------------------------------------

def _run_incremental(estimator, task, datamodule, config, layout, engine_state, data_fp, model_sig,
                     resume: str = "none", budget_monotonic=None, services=None, publish_terminal=True,
                     execution_policy=None) -> BackendResult:
    source = datamodule.incremental_train_data()
    if source.supports_mid_fit_resume and resume in ("auto", "required"):
        latest = _read_latest_ckpt(layout)
        if latest is None:
            if resume == "required":
                raise CheckpointError("required 恢复但无 latest 检查点")
        else:
            ckpt_dir = validate_sklearn_checkpoint_source(
                layout.path("checkpoints"), latest["path"], engine_state.run_id,
                _config_fingerprint(config), data_fp, model_sig,
            )
            _load_sklearn_checkpoint(ckpt_dir, estimator, source, engine_state, task, layout, config)

    classes = None
    global_step = engine_state.global_step
    batch_in_epoch = engine_state.batch_in_epoch
    epoch = engine_state.epoch
    budget_hit = False
    metric_states: dict[str, StageMetricState] = {}

    from ..platform import RuntimeBudget
    budget = (
        RuntimeBudget(execution_policy.max_minutes, execution_policy.shutdown_grace_minutes,
                      monotonic=budget_monotonic)
        if execution_policy is not None and execution_policy.max_minutes is not None
        else None
    )
    resumed_mid_epoch = batch_in_epoch > 0
    first_loop_iter = True

    while epoch < config.training.max_epochs and not budget_hit:
        resumed_partial_this_epoch = first_loop_iter and resumed_mid_epoch
        first_loop_iter = False
        epoch_started_at = (
            budget.begin_epoch()
            if budget is not None and not resumed_partial_this_epoch
            else None
        )
        for idx, batch in enumerate(source.iter_epoch(epoch)):
            if idx < batch_in_epoch:
                continue  # 跳过已消费 batch
            if task.estimator_kind == "classifier" and classes is None:
                classes = task.classes
                if classes is None:
                    raise SklearnBackendError("incremental classifier 必须提供 classes")
                fit_kwargs = resolve_fit_kwargs(config, batch)
                fit_kwargs["classes"] = classes
                estimator.partial_fit(batch.features, batch.targets, **fit_kwargs)
            else:
                fit_kwargs = resolve_fit_kwargs(config, batch)
                estimator.partial_fit(batch.features, batch.targets, **fit_kwargs)
            engine_state.advance_batch()
            engine_state.increment_global_step()
            global_step = engine_state.global_step
            if budget is not None and budget.hit():
                _save_sklearn_checkpoint(
                    layout, estimator, source, engine_state, task, config, data_fp, model_sig,
                    services=services,
                )
                budget_hit = True
                break
            # batch 边界检查点
            if config.checkpoint.every_epochs is not None or config.checkpoint.keep_last is not None:
                _save_sklearn_checkpoint(
                    layout, estimator, source, engine_state, task, config, data_fp, model_sig,
                    services=services,
                )
        if budget_hit:
            break  # 跳过 epoch 评价，直接进入 PREEMPTED 终态
        # epoch 评价
        engine_state.advance_epoch()
        epoch = engine_state.epoch
        batch_in_epoch = 0
        for stage in ("train", "val"):
            try:
                has = any(True for _ in datamodule.evaluation_batches(stage))
            except Exception:
                has = False
            if has:
                metric_states[stage] = evaluate_estimator(estimator, task, datamodule, stage, layout, config)
                _persist_stage_metrics(layout, metric_states[stage], stage, epoch, global_step)
        if "val" in metric_states and config.selection is not None:
            _apply_selection(engine_state, metric_states["val"], config)
            if engine_state.should_early_stop():
                layout.log(f"early stop at epoch {epoch}")
                break

        if budget is not None and epoch_started_at is not None and epoch < config.training.max_epochs:
            forecast = budget.complete_epoch(epoch_started_at)
            if forecast.should_preempt:
                layout.log(
                    "预算预测暂停: "
                    f"average_epoch={forecast.average_epoch_seconds:.3f}s, "
                    f"remaining_training={forecast.remaining_training_seconds:.3f}s"
                )
                _save_sklearn_checkpoint(
                    layout, estimator, source, engine_state, task, config, data_fp, model_sig,
                    services=services,
                )
                budget_hit = True
                break

    model_artifact = _export_joblib_model(layout, estimator, model_sig, config, engine_state)
    status = "preempted" if budget_hit else "succeeded"
    _write_summary(layout, config, "sklearn", engine_state, model_artifact, model_sig, data_fp,
                   status=status)
    # OSR-002：核心 Artifact（eval contract/report）先于服务 bundle 完成
    report_path = _write_sklearn_core_artifacts(layout, config, task, datamodule, estimator=estimator)
    def publish_local_terminal() -> None:
        _publish_sklearn_terminal(layout, status, config, engine_state, model_sig, data_fp,
                                  task=task, datamodule=datamodule, services=services,
                                  report_path=report_path)
    if services is not None:
        services.finalize_run(
            engine_state.run_id,
            status,
            prepare_terminal=publish_local_terminal if publish_terminal else None,
        )
    elif publish_terminal:
        publish_local_terminal()
    return BackendResult(
        status=status, epoch=epoch, batch_in_epoch=batch_in_epoch, global_step=global_step,
        model_artifact=model_artifact, environment_stats={},
    )


def _write_sklearn_core_artifacts(layout, config, task, datamodule, estimator=None) -> str | None:
    """写 evaluation contract + report（OSR-002：服务 bundle 前完成；OSR-005：完整合同）。"""
    from ..artifacts import write_json
    from ..reporting import generate_run_report
    if datamodule is not None:
        from ..contracts import build_evaluation_contract
        if task is None:
            raise SklearnBackendError("sklearn evaluation contract 缺少 Task")
        model_signature = None
        if estimator is not None:
            model_signature = {"class": f"{type(estimator).__module__}.{type(estimator).__name__}"}
        contract = build_evaluation_contract(datamodule, task, "sklearn", model_signature)
        write_json(layout.evaluation_contract_json, contract)
    report_path = None
    if config.report.enabled:
        try:
            report_path = generate_run_report(layout.run_dir)
        except Exception as exc:
            raise SklearnBackendError(f"报告生成失败: {exc}") from exc
    return report_path


def _publish_sklearn_terminal(layout, status, config, engine_state, model_sig, data_fp,
                              task=None, datamodule=None, services=None, report_path=None):
    """发布唯一终态；包含 checksum/MetricDefinition/模型引用/服务状态/报告。"""
    from ..artifacts import existing_terminal, publish_terminal, sha256_manifest, write_json
    from ..config import config_fingerprint, tuning_fingerprint
    from ..checkpoint import read_latest
    from ..platform import resolve_source_revision

    existing = existing_terminal(layout.run_dir)
    desired = "pause-manifest.json" if status == "preempted" else "run-manifest.json"
    if existing == desired:
        return
    if existing is not None and existing != "pause-manifest.json":
        return
    if existing == "pause-manifest.json":
        os.remove(os.path.join(layout.run_dir, "pause-manifest.json"))
    if services is not None:
        services_result = {
            "degraded": list(services.result.degraded),
            "audit": "services/service-audit.jsonl",
        }
    else:
        services_result = {"degraded": [], "audit": None}
    source_revision = config.run.source_revision
    if source_revision is None:
        try:
            source_revision = resolve_source_revision(config)
        except Exception:
            source_revision = None
    manifest = {
        "schema_version": 1,
        "run_id": engine_state.run_id,
        "backend": "sklearn",
        "status": status,
        "created_utc": _utc_now(),
        "epoch": engine_state.epoch,
        "global_step": engine_state.global_step,
        "source_revision": source_revision,
        "config_fingerprint": config_fingerprint(config),
        "tuning_fingerprint": tuning_fingerprint(config),
        "data_fingerprint": data_fp,
        "model_signature": model_sig,
        "selection": {
            "best_value": engine_state.best_value,
            "best_epoch": engine_state.best_epoch,
            "best_step": engine_state.best_step,
        } if config.selection is not None else None,
        "metric_definitions": {
            name: d.__dict__ for name, d in (task.metric_definitions.items() if task else [])
        },
        "model_artifact": {
            "format": "joblib",
            "best_path": "models/best/model.joblib",
            "last_path": "models/last/model.joblib",
        },
        "report": os.path.relpath(report_path, layout.run_dir).replace(os.sep, "/") if report_path else None,
        "services": services_result,
        "artifacts": sha256_manifest(layout.run_dir),
    }
    if status == "preempted":
        latest = read_latest(layout.path("checkpoints"))
        manifest["resume_checkpoint"] = latest["checkpoint_id"] if latest else None
    publish_terminal(layout.run_dir, "preempted" if status == "preempted" else "success", manifest)



def _read_latest_ckpt(layout):
    from ..checkpoint import read_latest
    return read_latest(layout.path("checkpoints"))


def _load_sklearn_checkpoint(ckpt_dir, estimator, source, engine_state, task, layout, config):
    import joblib
    est_path = os.path.join(ckpt_dir, "estimator.joblib")
    loaded = joblib.load(est_path)
    source.load_state_dict(_read_json(os.path.join(ckpt_dir, "source-state.json")))
    engine_state.load_state_dict(_read_json(os.path.join(ckpt_dir, "engine-state.json")))
    metric_payload = _read_json(os.path.join(ckpt_dir, "metric-states.json"))
    # 复核 fitted kind / signature
    from sklearn.base import is_classifier
    if task.estimator_kind == "classifier" and not is_classifier(loaded):
        raise CheckpointError("加载后 estimator kind 与 Task 不符")
    # 替换 estimator 引用
    estimator.__dict__.clear()
    estimator.__dict__.update(loaded.__dict__)
    # 恢复 metric states
    for stage, payload in metric_payload.items():
        st = task.metric_state(stage)
        st.load_state_dict(payload)
    return engine_state


def _save_sklearn_checkpoint(layout, estimator, source, engine_state, task, config, data_fp, model_sig,
                              services=None):
    import joblib
    metric_payload = {}
    ckpt_id = write_sklearn_checkpoint(
        estimator, source.state_dict(), engine_state, metric_payload,
        layout.path("checkpoints"), engine_state.run_id,
        _config_fingerprint(config), data_fp, model_sig,
        engine_state.epoch, engine_state.global_step, engine_state.batch_in_epoch,
        joblib,
    )
    if services is not None:
        services.submit_checkpoint(engine_state.run_id, ckpt_id)
    apply_retention(layout.path("checkpoints"), config.checkpoint.keep_last)


def _read_json(path):
    from ..artifacts import read_json
    return read_json(path)


# --------------------------------------------------------------------------
# 导出与 summary
# --------------------------------------------------------------------------

def _export_joblib_model(layout, estimator, model_sig, config, engine_state):
    import joblib
    for sub in ("best", "last"):
        target_dir = layout.path("models", sub)
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, "model.joblib")
        joblib.dump(estimator, path)
        files = {"model.joblib": {"size": os.path.getsize(path), "sha256": sha256_file(path)}}
        write_model_manifest(target_dir, "sklearn", model_sig, engine_state.run_id, files)
    return ModelArtifact(format="joblib",
                         best_path="models/best/model.joblib",
                         last_path="models/last/model.joblib")


def _persist_stage_metrics(layout, state, stage, epoch, global_step):
    record = {
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "computed_utc": _utc_now(),
        "metrics": state.compute(),
        "extended": state.extended_compute(),
    }
    append_jsonl(layout.metrics_jsonl, record)


def _apply_selection(engine_state, val_state, config):
    from ..engine import EngineStateError
    scalars = val_state.compute()
    metric = config.selection.metric
    if metric not in scalars:
        raise EngineStateError(f"selection metric {metric!r} 缺失")
    engine_state.selection_update(scalars[metric])


def _write_summary(layout, config, backend, engine_state, model_artifact, model_sig, data_fp,
                   metric_states=None, status="succeeded"):
    from ..config import config_fingerprint, tuning_fingerprint
    summary = {
        "schema_version": 1,
        "run_id": engine_state.run_id,
        "backend": backend,
        "status": status,
        "epoch": engine_state.epoch,
        "global_step": engine_state.global_step,
        "config_fingerprint": config_fingerprint(config),
        "tuning_fingerprint": tuning_fingerprint(config),
        "data_fingerprint": data_fp,
        "model_signature": model_sig,
        "selection": {
            "best_value": engine_state.best_value,
            "best_epoch": engine_state.best_epoch,
            "best_step": engine_state.best_step,
        } if config.selection is not None else None,
        "model_artifact": model_artifact.__dict__ if model_artifact else None,
        "stage_metrics": {
            stage: _safe_stage_compute(st) for stage, st in metric_states.items()
        } if metric_states else None,
        "metric_definitions": {
            name: d.__dict__
            for name, d in (next(iter(metric_states.values())).definitions.items()
                            if metric_states else [])
        },
    }
    write_json(layout.summary_json, summary)



def _safe_stage_compute(state) -> dict:
    """空 stage（如预算预占时未评价的 val）返回空 dict，避免 summary 失败。"""
    from ..metrics import MetricStateError
    try:
        return state.compute()
    except MetricStateError:
        return {}

def _utc_now():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
