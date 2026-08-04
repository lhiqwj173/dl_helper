"""PyTorch backend worker：DDP/AMP、精确梯度归一化、评价与导出。

worker 在独立子进程运行并延迟导入实验；父进程不初始化 CUDA。
"""
from __future__ import annotations

import hashlib
import math
import os
import pickle
from typing import Any, Mapping

import numpy as np

from ..artifacts import RunLayout, append_jsonl, sha256_file, write_json, write_prediction_manifest, write_prediction_shard
from ..checkpoint import (
    CheckpointError,
    apply_retention,
    load_torch_checkpoint,
    write_model_manifest,
    write_torch_checkpoint,
)
from ..config import Config
from ..contracts import LossResult, TorchExperiment, validate_backend_match, validate_experiment, validate_loss_result, validate_torch_task
from ..engine import EngineStateError, resolve_definition, validate_selection
from ..metrics import StageMetricState, combine_reduction_states
from .base import BackendResult, ModelArtifact

# 预测分片每片样本数
_SHARD_BATCH = 4096


class TorchBackendError(Exception):
    """Torch backend 合同违规。"""


def import_experiment_ref(ref: str) -> Any:
    """module.path:build_experiment 引用，返回 build_experiment 函数。"""
    if ":" not in ref:
        raise TorchBackendError(f"实验引用必须为 module:function: {ref!r}")
    module_path, _, func_name = ref.partition(":")
    import importlib
    module = importlib.import_module(module_path)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise TorchBackendError(f"实验工厂不可调用: {ref!r}")
    return func


def build_experiment_from_ref(ref: str, experiment_config: Mapping[str, Any]) -> TorchExperiment:
    func = import_experiment_ref(ref)
    experiment = func(dict(experiment_config))
    validate_experiment(experiment)
    if not isinstance(experiment, TorchExperiment):
        raise TorchBackendError(f"实验引用必须返回 TorchExperiment: {type(experiment).__name__}")
    return experiment


def build_torch_components(experiment: TorchExperiment, config: Config):
    """在 worker 内构造全新组件。"""
    model = experiment.model_factory()
    datamodule = experiment.datamodule_factory()
    task = experiment.task_factory()
    validate_torch_task(task)
    optimizer = experiment.optimizer_factory(model.parameters())
    scheduler_binding = experiment.scheduler_factory(optimizer)
    return model, datamodule, task, optimizer, scheduler_binding


def validate_fresh_components(model, datamodule, task, optimizer, scheduler_binding, config: Config) -> None:
    """预检：模型未上设备/DDP/训练过，scheduler 可序列化，DataModule 结构完整。"""
    import torch
    import torch.nn as nn

    for p in model.parameters():
        if p.is_cuda:
            raise TorchBackendError("模型参数已上 CUDA，必须返回 CPU 全新模型")
    from torch.nn.parallel import DistributedDataParallel
    if isinstance(model, (nn.DataParallel, DistributedDataParallel)):
        raise TorchBackendError("模型已由并行包装，必须返回未包装模型")
    for name, buf in model.named_buffers():
        if getattr(buf, "grad", None) is not None:
            raise TorchBackendError(f"模型 buffer 已带梯度: {name}")

    for member in ("supports_mid_epoch_resume", "nominal_train_batch_size", "setup",
                   "train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader",
                   "identity", "state_dict", "load_state_dict"):
        if not hasattr(datamodule, member):
            raise TorchBackendError(f"DataModule 缺少成员 {member}")
    if not isinstance(getattr(datamodule, "supports_mid_epoch_resume"), bool):
        raise TorchBackendError("DataModule.supports_mid_epoch_resume 必须为 bool")

    if scheduler_binding is not None:
        try:
            pickle.dumps(scheduler_binding.scheduler)
        except Exception as exc:
            raise TorchBackendError("scheduler 不可序列化") from exc
        if scheduler_binding.interval == "validation_metric":
            if resolve_definition(scheduler_binding.monitor, task.metric_definitions) is None:
                raise TorchBackendError(f"scheduler monitor 未由 Task 产生: {scheduler_binding.monitor!r}")

    if config.checkpoint.every_optimizer_steps is not None and not datamodule.supports_mid_epoch_resume:
        raise TorchBackendError("every_optimizer_steps 要求 DataModule 支持中途恢复")
    if config.runtime.max_minutes is not None and not datamodule.supports_mid_epoch_resume:
        raise TorchBackendError("运行时预算要求 DataModule 支持中途恢复")

    validate_selection(config.selection, task.metric_definitions, datamodule.val_dataloader() is not None)


def model_signature(model) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, p in model.named_parameters():
        params[name] = {"shape": list(p.shape), "dtype": str(p.dtype)}
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "params": params,
    }


def data_fingerprint(datamodule) -> str:
    identity = datamodule.identity()
    text = f"{identity.name}:{identity.version}:{identity.fingerprint}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unwrap_model(model):
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel
    while isinstance(model, (nn.DataParallel, DistributedDataParallel)):
        model = model.module
    return model


def _detach_scalar(loss: LossResult) -> tuple[float, float]:
    import torch
    num = float(loss.numerator.detach())
    den = float(loss.denominator.detach()) if isinstance(loss.denominator, torch.Tensor) else float(loss.denominator)
    return num, den


def reduce_stage_metrics(accelerator, state: StageMetricState) -> None:
    """跨 rank 归约 stage 指标（sum + rank 顺序 moment merge）。"""
    if accelerator.num_processes == 1:
        return
    import torch.distributed as dist

    world = accelerator.num_processes
    local = {k: (v[0].cpu().clone(), v[1]) for k, v in state.reduction_state().items()}
    tensors = {k: v[0] for k, v in local.items()}
    ops = {k: v[1] for k, v in local.items()}
    gathered: list[dict[str, Any]] = [{} for _ in range(world)]
    dist.all_gather_object(gathered, tensors)
    per_rank = [{k: (v, ops[k]) for k, v in rank_t.items()} for rank_t in gathered]
    combined = combine_reduction_states(per_rank)
    state.load_reduced_state(combined)


class TorchBackend:
    """torch backend worker。"""

    name = "torch"

    def run(self, experiment: TorchExperiment, config: Config, platform: Any, layout: RunLayout) -> BackendResult:
        raise TorchBackendError("Torch worker 必须经 launcher 在子进程运行")


# --------------------------------------------------------------------------
# 主 worker 入口（spawn 子进程 / 单进程）
# --------------------------------------------------------------------------

def run_worker(
    experiment_ref: str,
    config: Config,
    layout: RunLayout,
    local_rank: int = 0,
    world_size: int = 1,
    resume: str = "none",
    budget_monotonic=None,
    services=None,
    publish_terminal=True,
) -> BackendResult:
    """在 worker 内执行完整 torch 训练并返回 BackendResult。"""
    # strict 确定性需 CuBLAS workspace 配置，必须在 torch 导入/CUDA 初始化前设置
    if config.backend.torch is not None and config.backend.torch.deterministic == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator

    from ..platform import Platform

    platform = Platform()

    # torch 2.7 quirk：CUDA_VISIBLE_DEVICES="" 时 device_count()==0 但 is_available()==True
    if torch.cuda.device_count() == 0 and torch.cuda.is_available():
        torch.cuda.is_available = lambda: False

    backend = config.backend.torch
    if backend is None:
        raise TorchBackendError("torch backend 配置缺失")

    # 运行时控制：seed/确定性/matmul/compile 在组件构造前应用（OSR-006）
    _apply_runtime_controls(config)

    experiment = build_experiment_from_ref(experiment_ref, config.experiment)
    model, datamodule, task, optimizer, scheduler_binding = build_torch_components(experiment, config)
    validate_fresh_components(model, datamodule, task, optimizer, scheduler_binding, config)

    # compile 显式：失败不回退
    if backend.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            raise TorchBackendError(f"torch.compile 不兼容，拒绝回退 eager: {exc}") from exc

    datamodule.setup("fit")
    has_val = datamodule.val_dataloader() is not None
    model_sig = model_signature(model)
    data_fp = data_fingerprint(datamodule)

    # 预检：模型 param 未含 Inf/NaN
    for p in model.parameters():
        if not torch.isfinite(p.detach()).all():
            raise TorchBackendError("模型初始参数含 NaN/Inf")

    resources = platform.resolve_torch_resources(config, datamodule.nominal_train_batch_size)
    # OSR-006：应用解析的 DataLoader 资源到实际 loader（在创建 loader 前）；小数据集钳制
    loader_resources_applied = False
    applied_loader = None
    if hasattr(datamodule, "configure_resources"):
        applied_loader = datamodule.configure_resources(
            num_workers=resources.num_workers,
            pin_memory=resources.pin_memory,
            persistent_workers=resources.persistent_workers,
            prefetch_factor=resources.prefetch_factor,
        )
        loader_resources_applied = True
    accelerator = Accelerator(
        mixed_precision=resources.mixed_precision,
        gradient_accumulation_steps=backend.gradient_accumulation_steps,
    )
    engine_state = EngineState_from_config(config, model_sig, data_fp)

    # OSR-006：把解析后的资源写回 resolved config（复现信息；写入失败不阻止训练）
    _write_resolved_resources(layout, config, resources, applied_loader)

    # 已完成 run 不可重跑/改写（OSR-005 幂等性）
    from ..artifacts import existing_terminal
    if existing_terminal(layout.run_dir) == "run-manifest.json":
        raise TorchBackendError(f"run 已成功完成，禁止重跑改写: {engine_state.run_id}")

    # 服务启动（OSR-002）：Secret 预检 + RUN_STARTED 通知
    if services is not None and accelerator.is_main_process:
        services.start_run(engine_state.run_id)

    # 先 prepare model/optimizer，恢复后再创建 loader（OSR-004：恢复状态须在 loader 前生效）
    model, optimizer = accelerator.prepare(model, optimizer)

    # 注册 scheduler 到 Accelerate checkpoint
    if scheduler_binding is not None:
        accelerator.register_for_checkpointing(scheduler_binding.scheduler)

    metric_states = _build_stage_states(task, has_val, datamodule)

    # 恢复：auto 只在确实无 latest 时从零开始；latest 存在但损坏/漂移必须失败（OSR-004）
    from ..checkpoint import read_latest

    resumed_position = None
    best_model_state = None
    if resume in ("auto", "required"):
        latest = read_latest(layout.path("checkpoints"))
        if latest is None:
            if resume == "required":
                raise CheckpointError("required 恢复但无 latest 检查点")
            resumed_position = None
        else:
            loaded = load_torch_checkpoint(
                accelerator, layout.path("checkpoints"), engine_state, datamodule,
                lambda: metric_states, config_fingerprint_resume(config), data_fp, model_sig,
            )
            resumed_position = loaded
            best_model_state = loaded["best_model_state"]  # OSR-007：恢复检查点中 best 权重

    # train loader 每 epoch 重建（DataModule 按 epoch 确定性种子，保证连续/恢复 shuffle 一致；
    # OSR-004）；val/test 只构造一次（确定性、非按 epoch 分片）。
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if test_loader is not None:
        test_loader = accelerator.prepare(test_loader)

    epoch = engine_state.epoch
    best_value = None

    budget = None
    if config.runtime.max_minutes is not None:
        from ..platform import RuntimeBudget
        budget = RuntimeBudget(config.runtime.max_minutes, config.runtime.shutdown_grace_minutes,
                               monotonic=budget_monotonic)

    layout.log(f"worker rank={local_rank}/{world_size} started, model={model_sig['class']}")

    budget_hit = False
    batch_sizes: list[int] = []  # OSR-006：动态批量统计（跨全部 epoch）
    # OSR-010：每 epoch 独立清零阶段状态；中途恢复本 epoch 时保留已恢复的部分状态
    resumed_mid_epoch = resumed_position is not None and resumed_position["batch_in_epoch"] > 0
    first_loop_iter = True
    try:  # OSR-003：训练失败时记录精确位置后重抛
        while epoch < config.training.max_epochs and not budget_hit:
            if not (first_loop_iter and resumed_mid_epoch):
                metric_states["train"].reset()
            first_loop_iter = False
            train_loader = accelerator.prepare(datamodule.train_dataloader())
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)  # DDP：每 epoch 不同 shuffle
            model.train()
            train_state = metric_states["train"]
            window_denom = 0.0
            skip_count = engine_state.batch_in_epoch if engine_state.epoch == epoch else 0
            for batch_index, raw_batch in enumerate(train_loader):
                if batch_index < skip_count:
                    # 恢复时跳过本 epoch 已消费 batch，不重复优化/指标（位置以 engine_state 为唯一来源）
                    continue
                # 每个成功消费的 micro-batch 原子推进位置（OSR-004）
                engine_state.advance_batch()
                if hasattr(datamodule, "advance_batch"):
                    datamodule.advance_batch()
                with accelerator.accumulate(model):
                    prepared = task.prepare_batch(raw_batch, "train")
                    batch_sizes.append(int(getattr(prepared, "sample_count", 0)))  # OSR-006
                    outputs = task.forward(model, prepared)
                    loss = task.loss(outputs, prepared)
                    validate_loss_result(loss)
                    accelerator.backward(loss.numerator * backend.gradient_accumulation_steps)
                    num, den = _detach_scalar(loss)
                    window_denom += den
                    train_state.update_loss(num, den)
                    predicted = task.to_predicted_batch(outputs, prepared)
                    train_state.update_predicted(predicted)
                    if accelerator.sync_gradients:
                        global_denom = _all_reduce_sum(accelerator, window_denom)
                        if global_denom <= 0:
                            raise TorchBackendError(f"accumulation window denominator <= 0: {global_denom}")
                        accelerator.unscale_gradients(optimizer)
                        _normalize_gradients(model, accelerator, global_denom)
                        if backend.clip_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), backend.clip_grad_norm)
                        _validate_gradients_finite(model)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler_binding is not None and scheduler_binding.interval == "optimizer_step":
                            scheduler_binding.scheduler.step()
                        engine_state.increment_global_step()
                        window_denom = 0.0
                        budget_hit_any = False
                        if budget is not None:
                            budget_hit_any = budget.hit()
                            if accelerator.num_processes > 1:
                                # OSR-004：预算决策跨 rank 同步（任一命中 → 全部命中），避免
                                # 部分 rank 进入 checkpoint 屏障而死锁
                                import torch as _torch
                                _t = _torch.tensor([1.0 if budget_hit_any else 0.0],
                                                   device=accelerator.device)
                                accelerator.reduce(_t, reduction="sum")
                                budget_hit_any = _t.item() > 0
                        if budget_hit_any:
                            # 停止新 step；保存完整检查点后进入 PREEMPTED
                            _save_torch_checkpoint(accelerator, layout, engine_state, datamodule, metric_states,
                                                   config, model_sig, data_fp, best_model_state=best_model_state, services=services)
                            budget_hit = True
                            break
                        if (config.checkpoint.every_optimizer_steps is not None
                                and engine_state.global_step % config.checkpoint.every_optimizer_steps == 0):
                            _save_torch_checkpoint(accelerator, layout, engine_state, datamodule, metric_states,
                                                   config, model_sig, data_fp, best_model_state=best_model_state, services=services)
            if budget_hit:
                break  # 跳过 epoch 结束代码，直接进入 PREEMPTED 终态
            # epoch 结束
            reduce_stage_metrics(accelerator, train_state)
            if accelerator.is_main_process:
                _persist_stage_metrics(layout, train_state, "train", epoch, engine_state.global_step, config)
                _persist_extended(layout, train_state, "train", epoch, engine_state.global_step)
            _write_epoch_log(layout, engine_state, train_state, accelerator, epoch, config)

            # val + selection（所有 rank 决策一致；OSR-007）
            if has_val:
                metric_states["val"].reset()  # OSR-010：每 epoch val 独立清零
                val_state = _evaluate(accelerator, model, task, val_loader, "val", layout, config,
                                      target_state=metric_states["val"])
                reduce_stage_metrics(accelerator, val_state)
                improved = False
                if config.selection is not None:
                    improved = _apply_selection(engine_state, val_state, config)
                if accelerator.is_main_process:
                    _persist_stage_metrics(layout, val_state, "val", epoch, engine_state.global_step, config)
                    if improved:
                        best_model_state = _extract_state_dict(model)
                        best_value = engine_state.best_value
                # scheduler validation_metric：所有 rank 同步 step（OSR-007），避免学习率分叉
                if scheduler_binding is not None and scheduler_binding.interval == "validation_metric":
                    _step_metric_scheduler(scheduler_binding, val_state)
                if engine_state.should_early_stop():
                    layout.log(f"early stop at epoch {epoch}")
                    # OSR-004：所有 rank 参与 checkpoint（屏障一致），避免 DDP 死锁；主 rank 负责写入
                    _save_torch_checkpoint(accelerator, layout, engine_state, datamodule, metric_states,
                                           config, model_sig, data_fp, best_model_state=best_model_state,
                                           services=services)
                    break
            if scheduler_binding is not None and scheduler_binding.interval == "epoch":
                scheduler_binding.scheduler.step()
            if hasattr(datamodule, "advance_epoch"):
                datamodule.advance_epoch()  # OSR-004：DataModule epoch 状态同步推进
            engine_state.advance_epoch()
            epoch = engine_state.epoch
            if (config.checkpoint.every_epochs is not None
                    and engine_state.epoch % config.checkpoint.every_epochs == 0):
                _save_torch_checkpoint(accelerator, layout, engine_state, datamodule, metric_states,
                                       config, model_sig, data_fp, best_model_state=best_model_state, services=services)

    except BaseException:
        try:
            _write_failure_position(layout, engine_state, stage="train")
        except BaseException:
            pass  # OSR-003：位置记录失败不得替换原训练异常
        raise

    # test
    if test_loader is not None:
        test_state = _evaluate(accelerator, model, task, test_loader, "test", layout, config,
                               target_state=metric_states["test"])
        reduce_stage_metrics(accelerator, test_state)
        if accelerator.is_main_process:
            _persist_stage_metrics(layout, test_state, "test", engine_state.epoch, engine_state.global_step, config)

    # OSR-006：动态批量（nominal 未知）时统计实际 sample_count 范围（跨 rank 汇总）
    batch_stats = None
    if isinstance(resources.effective_batch_size, str) and resources.effective_batch_size == "dynamic":
        per_rank_sizes = [list(batch_sizes)]
        if accelerator.num_processes > 1:
            import torch.distributed as dist
            gathered: list[list[int] | None] = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(gathered, list(batch_sizes))
            per_rank_sizes = [list(values or []) for values in gathered]
        sizes = [value for values in per_rank_sizes for value in values]
        batch_stats = {
            "dynamic": True,
            "min_batch": min(sizes) if sizes else None,
            "max_batch": max(sizes) if sizes else None,
            "num_batches": len(sizes),
            "per_rank": {
                str(rank): {
                    "min_batch": min(values) if values else None,
                    "max_batch": max(values) if values else None,
                    "num_batches": len(values),
                }
                for rank, values in enumerate(per_rank_sizes)
            },
        }

    # 导出 best/last safetensors
    model_artifact = None
    if accelerator.is_main_process:
        model_artifact = _export_models(layout, model, best_model_state, model_sig, config, engine_state)
        _write_summary(layout, config, model_sig, data_fp, engine_state, metric_states, model_artifact,
                       platform, resources=resources, loader_resources_applied=loader_resources_applied,
                       applied_loader=applied_loader, batch_stats=batch_stats,
                       status="preempted" if budget_hit else "succeeded")
        # OSR-002：核心 Artifact（eval contract/environment/report）先于服务 bundle 完成
        report_path = _write_run_core_artifacts(layout, config, task, platform, resources, applied_loader,
                                                loader_resources_applied, batch_stats,
                                                datamodule=datamodule, model_sig=model_sig)
        # 服务终结前先写候选终态，使远端 bundle 携带完整 run 终态；required
        # 服务失败会向上抛出，CLI 负责清理候选并写 FAILED。
        def publish_local_terminal() -> None:
            _publish_terminal(layout, "preempted" if budget_hit else "succeeded", config, engine_state,
                              model_sig, data_fp, task, platform, metric_states, model_artifact,
                              resources=resources, loader_resources_applied=loader_resources_applied,
                              services=services, applied_loader=applied_loader, batch_stats=batch_stats,
                              report_path=report_path)
        if services is not None:
            services.finalize_run(
                engine_state.run_id,
                "preempted" if budget_hit else "succeeded",
                prepare_terminal=publish_local_terminal if publish_terminal else None,
            )
        elif publish_terminal:
            publish_local_terminal()

    status = "preempted" if budget_hit else "succeeded"
    return BackendResult(
        status=status,
        epoch=engine_state.epoch,
        batch_in_epoch=engine_state.batch_in_epoch,
        global_step=engine_state.global_step,
        model_artifact=model_artifact,
        environment_stats=platform.environment_manifest(),
    )


def _write_run_core_artifacts(layout, config, task, platform, resources, applied_loader,
                              loader_resources_applied, batch_stats,
                              datamodule=None, model_sig=None) -> str | None:
    """写 evaluation-contract / environment / HTML 报告（OSR-002：服务 bundle 前完成）。"""
    from ..artifacts import write_json
    from ..reporting import generate_run_report

    if task is not None:
        if datamodule is None or not hasattr(datamodule, "identity"):
            raise TorchBackendError("DataModule 必须提供 identity() 以生成 evaluation contract")
        from ..contracts import build_evaluation_contract
        contract = build_evaluation_contract(datamodule, task, "torch", model_sig)
        write_json(layout.evaluation_contract_json, contract)
    if platform is not None:
        write_json(layout.environment_json, _runtime_environment(platform, config, resources,
                                                                 loader_resources_applied, applied_loader,
                                                                 batch_stats))
    report_path = None
    if config.report.enabled:
        try:
            report_path = generate_run_report(layout.run_dir)
        except Exception as exc:
            raise TorchBackendError(f"报告生成失败: {exc}") from exc
    return report_path


def _publish_terminal(layout, status, config, engine_state, model_sig, data_fp,
                      task=None, platform=None, metric_states=None, model_artifact=None,
                      resources=None, loader_resources_applied=False, services=None,
                      applied_loader=None, batch_stats=None, report_path=None):
    """发布唯一终态：success → run-manifest.json；preempted → pause-manifest.json。

    终态最后写入且包含全量 Artifact checksum、MetricDefinition、模型引用、服务状态与报告路径。
    """
    from ..artifacts import (
        existing_terminal,
        publish_terminal,
        sha256_manifest,
    )
    from ..config import config_fingerprint, tuning_fingerprint

    existing = existing_terminal(layout.run_dir)
    desired = "pause-manifest.json" if status == "preempted" else "run-manifest.json"
    if existing == desired:
        return
    if existing is not None and existing != "pause-manifest.json":
        return
    if existing == "pause-manifest.json":
        os.remove(os.path.join(layout.run_dir, "pause-manifest.json"))

    # 核心 Artifact（eval contract/environment/report）已在 services finalize 前写入；
    # 此处只生成终态 manifest 并发布本地终态（OSR-002 时序）。

    # 真实服务结果（OSR-005：不再固定伪造 degraded=[]）
    if services is not None:
        services_result = {
            "degraded": list(services.result.degraded),
            "audit": "services/service-audit.jsonl",
        }
    else:
        services_result = {"degraded": [], "audit": None}

    manifest = {
        "schema_version": 1,
        "run_id": engine_state.run_id,
        "backend": "torch",
        "status": status,
        "created_utc": _utc_now(),
        "epoch": engine_state.epoch,
        "batch_in_epoch": engine_state.batch_in_epoch,
        "global_step": engine_state.global_step,
        "source_revision": _source_revision(platform, config),
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
            name: d.__dict__ for name, d in (task.metric_definitions.items() if task is not None else [])
        },
        "model_artifact": model_artifact.__dict__ if model_artifact else None,
        "report": os.path.relpath(report_path, layout.run_dir).replace(os.sep, "/") if report_path else None,
        "environment": _runtime_environment(platform, config, resources, loader_resources_applied, applied_loader, batch_stats),
        "services": services_result,
        "artifacts": sha256_manifest(layout.run_dir),
    }
    if status == "preempted":
        from ..checkpoint import read_latest
        latest = read_latest(layout.path("checkpoints"))
        manifest["resume_checkpoint"] = latest["checkpoint_id"] if latest else None
    publish_terminal(layout.run_dir, "preempted" if status == "preempted" else "success", manifest)


def _source_revision(platform, config):
    """manifest 记录 source revision；本地无 Git 且未显式提供时为 None。"""
    from ..platform import resolve_source_revision
    if config.run.source_revision:
        return config.run.source_revision
    try:
        return resolve_source_revision(config)
    except Exception:
        return None



def read_latest_checkpoint(layout):
    from ..checkpoint import read_latest
    latest = read_latest(layout.path("checkpoints"))
    return latest["checkpoint_id"] if latest else None


def EngineState_from_config(config: Config, model_sig: Mapping[str, Any], data_fp: str):
    from ..engine import EngineState
    sel = config.selection
    run_id = config.run.id or "unknown"
    state = EngineState(
        backend="torch",
        run_id=run_id,
        config_fingerprint=config_fingerprint_resume(config),
        metric_name=sel.metric if sel else None,
        mode=sel.mode if sel else None,
        patience=sel.patience if sel else None,
        min_delta=sel.min_delta if sel else 0.0,
    )
    return state


def config_fingerprint_resume(config: Config) -> str:
    from ..config import config_fingerprint
    return config_fingerprint(config, resume=True)


def _apply_runtime_controls(config: Config) -> None:
    """在组件构造前应用 seed / 确定性 / matmul 精度（OSR-006）。"""
    import numpy as np
    import torch

    torch.manual_seed(config.run.seed)
    np.random.seed(config.run.seed)
    backend = config.backend.torch
    if backend is None:
        raise TorchBackendError("torch backend 配置缺失")
    deterministic = backend.deterministic
    if deterministic == "strict":
        # CuBLAS 确定性需 workspace 配置；平台自动设置，避免 strict 在 CUDA 上误报
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
    elif deterministic == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif deterministic == "off":
        torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision(backend.matmul_precision)


def _build_stage_states(task, has_val, datamodule):
    states = {"train": task.metric_state("train")}
    if has_val:
        states["val"] = task.metric_state("val")
    if datamodule.test_dataloader() is not None:
        states["test"] = task.metric_state("test")
    return states


def _all_reduce_sum(accelerator, value: float) -> float:
    if accelerator.num_processes == 1:
        return value
    import torch
    t = torch.tensor(float(value), dtype=torch.float64, device=accelerator.device)
    accelerator.reduce(t, reduction="sum")
    return float(t.item())


def _normalize_gradients(model, accelerator, global_denom: float) -> None:
    import torch
    world = accelerator.num_processes
    scale = world / global_denom
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


def _validate_gradients_finite(model) -> None:
    import torch
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise TorchBackendError("非有限梯度，终止")


def _save_torch_checkpoint(accelerator, layout, engine_state, datamodule, metric_states,
                           config, model_sig, data_fp, best_model_state=None, services=None) -> None:
    # OSR-004：所有 rank 参与 write_torch_checkpoint（内部屏障 + 各 rank save + 主 rank 写 manifest）
    metric_states_payload = {stage: st.state_dict() for stage, st in metric_states.items()}
    from ..checkpoint import write_torch_checkpoint
    ckpt_id = write_torch_checkpoint(
        accelerator, layout.path("checkpoints"), engine_state.run_id, engine_state,
        datamodule.state_dict(), metric_states_payload,
        config_fingerprint_resume(config), data_fp, model_sig,
        engine_state.epoch, engine_state.global_step, engine_state.batch_in_epoch,
        best_model_state=best_model_state,
    )
    # OSR-002：主 rank 提交 checkpoint 到有界异步同步器
    if services is not None and accelerator.is_main_process:
        services.submit_checkpoint(engine_state.run_id, ckpt_id)
    if accelerator.is_main_process:
        apply_retention(layout.path("checkpoints"), config.checkpoint.keep_last)
        layout.log(f"checkpoint saved {ckpt_id} (step={engine_state.global_step})")


def _evaluate(accelerator, model, task, loader, stage, layout, config, target_state=None):
    model.eval()
    state = target_state if target_state is not None else task.metric_state(stage)
    shard_arrays: list[dict[str, np.ndarray]] = []
    shard_n = 0
    shard_count = 0
    rank = accelerator.process_index
    import torch
    with torch.no_grad():
        for raw_batch in loader:
            prepared = task.prepare_batch(raw_batch, stage)
            outputs = task.forward(model, prepared)
            predicted = task.to_predicted_batch(outputs, prepared)
            state.update_predicted(predicted)
            # val/test loss 数值（不要求梯度）
            loss = task.loss(outputs, prepared)
            num, den = _detach_scalar(loss)
            state.update_loss(num, den)
            arrays = task.prediction_arrays(predicted)
            shard_arrays.append(arrays)
            shard_n += predicted.sample_count
            if shard_n >= _SHARD_BATCH:
                _write_shard_batch(layout, stage, rank, shard_count, shard_arrays, shard_n)
                shard_count += 1
                shard_arrays = []
                shard_n = 0
    if shard_arrays:
        _write_shard_batch(layout, stage, rank, shard_count, shard_arrays, shard_n)
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


def _persist_stage_metrics(layout, state, stage, epoch, global_step, config):
    record = {
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "computed_utc": _utc_now(),
        "metrics": state.compute(),
        "extended": state.extended_compute(),
    }
    append_jsonl(layout.metrics_jsonl, record)


def _persist_extended(layout, state, stage, epoch, global_step):
    pass  # extended 已随 persist 记录


def _write_epoch_log(layout, engine_state, train_state, accelerator, epoch, config):
    import torch
    if accelerator.is_main_process:
        scalars = train_state.compute()
        loss = scalars.get("train/loss", float("nan"))
        lr = None
        layout.log(f"epoch={epoch} loss={loss:.6f} step={engine_state.global_step}")


def _apply_selection(engine_state, val_state, config) -> bool:
    scalars = val_state.compute()
    metric = config.selection.metric
    if metric not in scalars:
        raise EngineStateError(f"selection metric {metric!r} 缺失于 val summary")
    value = scalars[metric]
    return engine_state.selection_update(value)


def _step_metric_scheduler(scheduler_binding, val_state):
    scalars = val_state.compute()
    if scheduler_binding.monitor in scalars:
        scheduler_binding.scheduler.step(scalars[scheduler_binding.monitor])


def _extract_state_dict(model):
    import torch
    return {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()}


def _export_models(layout, model, best_state, model_sig, config, engine_state):
    from ..checkpoint import write_model_manifest
    unwrapped = unwrap_model(model)
    last_state = {k: v.detach().cpu().contiguous() for k, v in unwrapped.state_dict().items()}
    model_artifact = ModelArtifact(format="safetensors")

    if best_state is not None:
        best_dir = layout.path("models", "best")
        _write_safetensors(best_dir, best_state, model_sig, engine_state.run_id)
        model_artifact = ModelArtifact(format="safetensors",
                                       best_path="models/best/model.safetensors")
    last_dir = layout.path("models", "last")
    _write_safetensors(last_dir, last_state, model_sig, engine_state.run_id)
    model_artifact = ModelArtifact(
        format="safetensors",
        best_path=model_artifact.best_path,
        last_path="models/last/model.safetensors",
    )
    return model_artifact


def _write_safetensors(target_dir, state, model_sig, run_id):
    import safetensors.torch
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, "model.safetensors")
    state = {k: v.contiguous() for k, v in state.items()}
    safetensors.torch.save_file(state, path)
    files = {"model.safetensors": {"size": os.path.getsize(path), "sha256": sha256_file(path)}}
    from ..checkpoint import write_model_manifest
    write_model_manifest(target_dir, "torch", model_sig, run_id, files)


def _write_summary(layout, config, model_sig, data_fp, engine_state, metric_states, model_artifact, platform,
                   resources=None, loader_resources_applied=False, applied_loader=None, batch_stats=None,
                   status="succeeded"):
    from ..config import config_canonical_json, config_fingerprint, tuning_fingerprint
    summary = {
        "schema_version": 1,
        "run_id": engine_state.run_id,
        "backend": "torch",
        "status": status,
        "created_utc": _utc_now(),
        "epoch": engine_state.epoch,
        "global_step": engine_state.global_step,
        "source_revision": _source_revision(platform, config),
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
        "environment": _runtime_environment(platform, config, resources, loader_resources_applied, applied_loader, batch_stats),
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


def _runtime_environment(platform, config, resources=None, loader_resources_applied=False,
                         applied_loader=None, batch_stats=None) -> dict:
    """复现所需环境：平台摘要 + seed/确定性/matmul/compile + 实际应用的 loader 资源。"""
    env = platform.environment_manifest()
    env["seed"] = config.run.seed
    if config.backend.torch is not None:
        env["torch"] = {
            "deterministic": config.backend.torch.deterministic,
            "matmul_precision": config.backend.torch.matmul_precision,
            "compile": config.backend.torch.compile,
        }
    if resources is not None:
        env["resources"] = {
            "num_processes": resources.num_processes,
            "mixed_precision": resources.mixed_precision,
            "num_workers": (applied_loader or {}).get("num_workers", resources.num_workers),
            "pin_memory": (applied_loader or {}).get("pin_memory", resources.pin_memory),
            "persistent_workers": (applied_loader or {}).get("persistent_workers", resources.persistent_workers),
            "prefetch_factor": (applied_loader or {}).get("prefetch_factor", resources.prefetch_factor),
            "effective_batch_size": resources.effective_batch_size,
            "matmul_precision": resources.matmul_precision,
            "compile": resources.compile,
            "deterministic": resources.deterministic,
            "loader_resources_applied": loader_resources_applied,
        }
        if batch_stats is not None:
            env["resources"]["batch_stats"] = batch_stats
    return env



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


def _write_failure_position(layout, engine_state, stage="train") -> None:
    """训练失败时记录精确位置（OSR-003），供 CLI failure.json 引用。

    写入失败形成独立 failure-position-error.json（secondary audit），不替换原训练异常。
    """
    from ..artifacts import write_json
    try:
        write_json(layout.path("failure-position.json"), {
            "stage": stage,
            "epoch": engine_state.epoch,
            "batch_in_epoch": engine_state.batch_in_epoch,
            "global_step": engine_state.global_step,
        })
    except Exception as exc:
        try:
            write_json(layout.path("failure-position-error.json"),
                       {"error_type": type(exc).__name__, "message": str(exc)})
        except Exception:
            pass


def _write_resolved_resources(layout, config, resources, applied_loader) -> None:
    """OSR-006：保证 config.resolved.yaml 存在且严格可重放，资源写入独立 resolved-resources.json。"""
    from ..artifacts import write_json
    from ..config import config_to_dict
    import yaml
    # config.resolved.yaml 为纯配置（严格解析器可重放），不注入派生字段
    if not os.path.exists(layout.path("config.resolved.yaml")):
        layout.write_text("config.resolved.yaml",
                          yaml.safe_dump(config_to_dict(config), allow_unicode=True, sort_keys=False))
    write_json(layout.path("resolved-resources.json"), {
        "schema_version": 1,
        "num_processes": resources.num_processes,
        "mixed_precision": resources.mixed_precision,
        "num_workers": (applied_loader or {}).get("num_workers", resources.num_workers),
        "pin_memory": (applied_loader or {}).get("pin_memory", resources.pin_memory),
        "persistent_workers": (applied_loader or {}).get("persistent_workers", resources.persistent_workers),
        "prefetch_factor": (applied_loader or {}).get("prefetch_factor", resources.prefetch_factor),
        "effective_batch_size": resources.effective_batch_size,
        "matmul_precision": resources.matmul_precision,
        "compile": resources.compile,
        "deterministic": resources.deterministic,
    })
