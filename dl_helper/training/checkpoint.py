"""不可变检查点、manifest、latest 指针与可信恢复。

Torch 使用 staging -> Accelerate state -> manifest -> immutable dir -> latest；
sklearn incremental 使用可信 joblib + source state，校验必须先于反序列化。
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .artifacts import (
    ArtifactError,
    atomic_write_text,
    ensure_within,
    json_safe,
    move_tree,
    list_relative_files,
    read_json,
    remove_tree,
    sha256_file,
    sha256_manifest,
    write_json,
)

CHECKPOINT_MANIFEST = "checkpoint-manifest.json"
LATEST_FILE = "latest.json"
HISTORY_MANIFEST = "run-history-manifest.json"
HISTORY_DIR = "run-history"
HISTORY_FILES = ("metrics/metrics.jsonl", "logs/train.log")


class CheckpointError(Exception):
    """检查点不可恢复或校验失败。"""


class HistoryRecoveryError(CheckpointError):
    """历史恢复未完成；禁止把旧 run 改写成失败终态。"""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def checkpoint_id(epoch: int, global_step: int, checkpoint_kind: str | None = None) -> str:
    if checkpoint_kind not in (None, "early-stop", "final"):
        raise ValueError(f"不支持的 checkpoint_kind: {checkpoint_kind!r}")
    checkpoint = f"epoch-{epoch:06d}-step-{global_step:08d}"
    if checkpoint_kind is None:
        return checkpoint
    return f"{checkpoint}-{checkpoint_kind}"


def runtime_versions(backend: str) -> dict[str, str]:
    import platform as _p

    out: dict[str, str] = {"python": _p.python_version()}
    if backend == "torch":
        import torch, accelerate, numpy

        out.update({"torch": torch.__version__, "accelerate": accelerate.__version__,
                    "numpy": numpy.__version__})
    else:
        import sklearn, numpy, scipy, joblib

        out.update({"sklearn": sklearn.__version__, "numpy": numpy.__version__,
                    "scipy": scipy.__version__, "joblib": joblib.__version__})
    return out


def verify_runtime_versions(backend: str, recorded: Mapping[str, str]) -> None:
    """校验检查点记录的运行时版本。

    Torch 允许跨镜像漂移：Kaggle 镜像升级（Python/torch/accelerate/numpy）不应让旧
    检查点永久不可恢复；漂移输出告警后继续，新检查点记录当前版本。
    sklearn/joblib 反序列化是代码执行边界，仍要求精确匹配。
    """
    current = runtime_versions(backend)
    mismatches = [
        f"{key} recorded={value} current={current.get(key)}"
        for key, value in recorded.items()
        if current.get(key) != value
    ]
    if not mismatches:
        return
    detail = "; ".join(mismatches)
    if backend == "sklearn":
        raise CheckpointError(f"runtime 版本不精确匹配: {detail}")
    print(f"[checkpoint] runtime 版本漂移，继续恢复: {detail}", flush=True)


# --------------------------------------------------------------------------
# 通用 manifest / latest
# --------------------------------------------------------------------------

def write_manifest(checkpoint_root: str, manifest: Mapping[str, Any]) -> str:
    path = os.path.join(checkpoint_root, CHECKPOINT_MANIFEST)
    write_json(path, manifest)
    return path


def validate_manifest_complete(manifest: Mapping[str, Any], checkpoint_root: str) -> None:
    if not manifest.get("complete"):
        raise CheckpointError("checkpoint manifest 标记 incomplete")
    for rel, meta in manifest.get("files", {}).items():
        full = ensure_within(checkpoint_root, os.path.join(checkpoint_root, rel), "checkpoint 文件")
        if not os.path.exists(full):
            raise CheckpointError(f"checkpoint 文件缺失: {rel}")
        if sha256_file(full) != meta.get("sha256"):
            raise CheckpointError(f"checkpoint 文件 checksum 不匹配: {rel}")
        if os.path.getsize(full) != meta.get("size"):
            raise CheckpointError(f"checkpoint 文件 size 不匹配: {rel}")


def update_latest(checkpoint_root: str, checkpoint_dir_name: str, checkpoint_id_value: str) -> None:
    """原子更新 latest.json；损坏时不尝试旧项。"""
    data = {"schema_version": 1, "checkpoint_id": checkpoint_id_value, "path": checkpoint_dir_name}
    write_json(os.path.join(checkpoint_root, LATEST_FILE), data)


def read_latest(checkpoint_root: str) -> dict[str, Any] | None:
    path = os.path.join(checkpoint_root, LATEST_FILE)
    if not os.path.exists(path):
        return None
    try:
        data = read_json(path)
    except Exception:
        raise CheckpointError("latest.json 损坏")
    if not isinstance(data, Mapping) or not data.get("path"):
        raise CheckpointError("latest.json 内容非法")
    return dict(data)


def _stage_and_finalize(staging: str, final_dir: str) -> None:
    """staging -> 不可变目录，禁止覆盖。"""
    if not os.path.isdir(staging):
        raise CheckpointError(f"staging 目录不存在: {staging}")
    move_tree(staging, final_dir)


def _snapshot_run_history(staging: str, run_dir: str, *, position_epoch: int,
                          require_metrics: bool = True) -> None:
    """将检查点位置之前的机器可读历史纳入不可变检查点。"""
    history_dir = os.path.join(staging, HISTORY_DIR)
    os.makedirs(history_dir)
    if require_metrics and position_epoch > 0 and not os.path.isfile(os.path.join(run_dir, "metrics", "metrics.jsonl")):
        raise CheckpointError("已完成轮次缺少 metrics.jsonl，拒绝提交不完整检查点")
    paths = [rel for rel in HISTORY_FILES if os.path.isfile(os.path.join(run_dir, *rel.split("/")))]
    predictions_dir = os.path.join(run_dir, "predictions")
    if os.path.islink(predictions_dir):
        raise CheckpointError("预测历史目录为符号链接")
    if os.path.isdir(predictions_dir):
        for directory, subdirectories, _files in os.walk(predictions_dir):
            if any(os.path.islink(os.path.join(directory, name)) for name in subdirectories):
                raise CheckpointError("预测历史包含符号链接目录")
        paths.extend(f"predictions/{rel.replace(os.sep, '/')}"
                     for rel in list_relative_files(predictions_dir))
    for rel in paths:
        raw_source = os.path.join(run_dir, *rel.split("/"))
        if os.path.islink(raw_source):
            raise CheckpointError(f"运行历史不是普通文件: {rel}")
        source = ensure_within(run_dir, raw_source, "运行历史")
        if not os.path.isfile(source):
            raise CheckpointError(f"运行历史不是普通文件: {rel}")
        target = os.path.join(history_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
    write_json(os.path.join(staging, HISTORY_MANIFEST), {
        "schema_version": 1,
        "files": sha256_manifest(history_dir),
    })


def restore_run_history(ckpt_dir: str, run_dir: str, *, position_epoch: int,
                        require_metrics: bool = True) -> None:
    """按检查点回滚历史文件；旧检查点若无法证明历史完整则拒绝继续。"""
    try:
        _restore_run_history(ckpt_dir, run_dir, position_epoch=position_epoch,
                             require_metrics=require_metrics)
    except HistoryRecoveryError:
        raise
    except (CheckpointError, ArtifactError, OSError, ValueError, TypeError) as exc:
        raise HistoryRecoveryError(f"运行历史恢复失败: {exc}") from exc


def _restore_run_history(ckpt_dir: str, run_dir: str, *, position_epoch: int,
                         require_metrics: bool) -> None:
    manifest_path = os.path.join(ckpt_dir, HISTORY_MANIFEST)
    if not os.path.isfile(manifest_path):
        if position_epoch > 0 and not os.path.isfile(os.path.join(run_dir, "metrics", "metrics.jsonl")):
            raise HistoryRecoveryError("旧检查点未包含运行历史，当前会话也无逐轮指标；拒绝生成缺曲线的成果")
        return
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise HistoryRecoveryError("运行历史清单非法")
    expected = manifest.get("files")
    if not isinstance(expected, Mapping):
        raise HistoryRecoveryError("运行历史文件清单非法")
    history_dir = os.path.join(ckpt_dir, HISTORY_DIR)
    actual = sha256_manifest(history_dir)
    if actual != expected:
        raise HistoryRecoveryError("运行历史与检查点清单不一致")
    if require_metrics and position_epoch > 0 and "metrics/metrics.jsonl" not in {rel.replace("\\", "/") for rel in expected}:
        raise HistoryRecoveryError("检查点缺少已完成轮次的逐轮指标")
    allowed = set(HISTORY_FILES)
    for rel in expected:
        normalized = rel.replace("\\", "/")
        if normalized not in allowed and not normalized.startswith("predictions/"):
            raise HistoryRecoveryError(f"运行历史包含不允许的路径: {rel}")
        if normalized.startswith("/") or any(part in ("", ".", "..") for part in normalized.split("/")):
            raise HistoryRecoveryError(f"运行历史路径非法: {rel}")
    staging = os.path.join(run_dir, f".history-restore-{os.getpid()}")
    if os.path.exists(staging):
        raise HistoryRecoveryError(f"运行历史恢复暂存目录已存在: {staging}")
    try:
        shutil.copytree(history_dir, staging)
        if sha256_manifest(staging) != expected:
            raise HistoryRecoveryError("运行历史复制后校验失败")
        for rel in HISTORY_FILES:
            destination = ensure_within(run_dir, os.path.join(run_dir, *rel.split("/")), "运行历史")
            staged = os.path.join(staging, *rel.split("/"))
            if os.path.isfile(staged):
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                os.replace(staged, destination)
            elif os.path.exists(destination):
                os.remove(destination)
        destination = ensure_within(run_dir, os.path.join(run_dir, "predictions"), "预测历史")
        staged = os.path.join(staging, "predictions")
        if os.path.exists(destination):
            remove_tree(destination)
        if os.path.isdir(staged):
            move_tree(staged, destination)
    finally:
        remove_tree(staging)


# --------------------------------------------------------------------------
# 保留策略
# --------------------------------------------------------------------------

def apply_retention(checkpoint_root: str, keep_last: int | None) -> None:
    """只删除当前 run 的、manifest 完整且非 latest 引用的旧检查点。"""
    latest = read_latest(checkpoint_root)
    latest_path = latest["path"] if latest else None
    if keep_last is None:
        return
    entries = []
    for name in sorted(os.listdir(checkpoint_root)):
        if not name.startswith("epoch-"):
            continue
        manifest_path = os.path.join(checkpoint_root, name, CHECKPOINT_MANIFEST)
        if os.path.exists(manifest_path):
            entries.append(name)
    # 按名字排序（epoch-step 字典序 == 时间序）
    keep = set(entries[-keep_last:])
    if latest_path:
        keep.add(latest_path)
    for name in entries:
        if name not in keep:
            remove_tree(os.path.join(checkpoint_root, name))


# --------------------------------------------------------------------------
# Torch checkpoint
# --------------------------------------------------------------------------

def write_torch_checkpoint(
    accelerator: Any,
    checkpoints_dir: str,
    run_id: str,
    engine_state: Any,
    datamodule_state: Mapping[str, Any],
    metric_states: Mapping[str, Any],
    config_fingerprint: str,
    data_fingerprint: str,
    model_signature: Mapping[str, Any],
    epoch: int,
    global_step: int,
    batch_in_epoch: int,
    best_model_state: Mapping[str, Any] | None = None,
    progress_snapshot: Mapping[str, Any] | None = None,
    run_dir: str | None = None,
    checkpoint_kind: str | None = None,
    archive_callback: Callable[[str], None] | None = None,
) -> str:
    """保存 torch 不可变检查点并返回 checkpoint_id。

    OSR-004：所有 rank 共同参与 Accelerate save 协议（各自保存 RNG/state 到 rank 子目录），
    仅主 rank 写 metadata/manifest 并 move 到不可变目录。
    progress_snapshot + run_dir 提供时，主 rank 在 manifest 计算前生成内嵌进度报告
    （progress/index.html + progress/progress-snapshot.json，随 sha256 manifest 校验）；
    报告生成失败直接中止 checkpoint（fail-fast）。
    """
    ckpt_id = checkpoint_id(epoch, global_step, checkpoint_kind)
    os.makedirs(checkpoints_dir, exist_ok=True)
    final_dir = os.path.join(checkpoints_dir, ckpt_id)
    if os.path.exists(final_dir):
        raise CheckpointError(f"检查点已存在（不可变，禁止覆盖）: {ckpt_id}")
    staging = os.path.join(checkpoints_dir, f".staging-{ckpt_id}")
    try:
        if accelerator.is_main_process:
            remove_tree(staging)
            os.makedirs(os.path.join(staging, "accelerator-state"), exist_ok=True)
        accelerator.wait_for_everyone()
    except BaseException:
        _abort_distributed(accelerator)
        raise
    # OSR-004：所有 rank 对同一 Accelerate state 目录执行 save（主 rank 写模型/优化器，
    # 各 rank 写自身 RNG）；Accelerate 1.6 要求共享目录。
    try:
        accelerator.save_state(os.path.join(staging, "accelerator-state"))
        accelerator.wait_for_everyone()
    except BaseException:
        _abort_distributed(accelerator)
        raise
    if accelerator.is_main_process:
        try:
            write_json(os.path.join(staging, "engine-state.json"), engine_state.state_dict())
            write_json(os.path.join(staging, "datamodule-state.json"), dict(datamodule_state))
            write_json(os.path.join(staging, "metric-states.json"), json_safe(metric_states))
            if best_model_state is not None:
                import torch
                torch.save(best_model_state, os.path.join(staging, "best-model-state.pt"))
            if progress_snapshot is not None:
                if run_dir is None:
                    raise CheckpointError("提供 progress_snapshot 时必须同时提供 run_dir")
                from .reporting import generate_checkpoint_progress_report
                generate_checkpoint_progress_report(
                    run_dir, progress_snapshot, os.path.join(staging, "progress")
                )
            if run_dir is not None:
                _snapshot_run_history(staging, run_dir, position_epoch=epoch)
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "checkpoint_id": ckpt_id,
                "created_utc": _utc_now(),
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "global_step": global_step,
                "config_fingerprint": config_fingerprint,
                "backend": "torch",
                "data_fingerprint": data_fingerprint,
                "model_signature": model_signature,
                "runtime_versions": runtime_versions("torch"),
                "files": sha256_manifest(staging),
                "complete": True,
            }
            write_manifest(staging, manifest)
            # fsync 全部文件
            for dirpath, _dirs, filenames in os.walk(staging):
                for name in filenames:
                    _fsync_file(os.path.join(dirpath, name))
            _stage_and_finalize(staging, final_dir)
            if archive_callback is not None:
                try:
                    archive_callback(final_dir)
                except BaseException:
                    # latest 尚未提交：撤销本次目录，避免重试同一训练位置发生 ID 冲突。
                    # 若 ZIP 已提交，它仍是完整恢复来源，下一启动会自动发现。
                    remove_tree(final_dir)
                    raise
            update_latest(checkpoints_dir, ckpt_id, ckpt_id)
        except Exception:
            remove_tree(staging)
            _abort_distributed(accelerator)
            raise
    # OSR-004：所有 rank 等主 rank 原子提交 manifest/latest 后再统一离开
    try:
        accelerator.wait_for_everyone()
    except BaseException:
        _abort_distributed(accelerator)
        raise
    return ckpt_id


def load_torch_checkpoint(
    accelerator: Any,
    checkpoints_dir: str,
    engine_state: Any,
    datamodule: Any,
    metric_states_builder: Any,
    config_fingerprint: str,
    data_fingerprint: str,
    model_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """从 latest 恢复 torch 检查点；返回恢复位置。metric_states_builder 返回 {stage: StageMetricState}。"""
    latest = read_latest(checkpoints_dir)
    if latest is None:
        raise CheckpointError("无可用 latest 检查点")
    ckpt_dir = os.path.join(checkpoints_dir, latest["path"])
    manifest = read_json(os.path.join(ckpt_dir, CHECKPOINT_MANIFEST))
    validate_manifest_complete(manifest, ckpt_dir)
    if manifest["run_id"] != engine_state.run_id:
        raise CheckpointError("checkpoint run_id 不匹配")
    if manifest["config_fingerprint"] != config_fingerprint:
        raise CheckpointError("checkpoint 配置指纹不匹配，拒绝恢复")
    if manifest["data_fingerprint"] != data_fingerprint:
        raise CheckpointError("checkpoint 数据指纹不匹配")
    if manifest["model_signature"] != model_signature:
        raise CheckpointError("checkpoint 模型签名不匹配")
    verify_runtime_versions("torch", manifest["runtime_versions"])
    try:
        if accelerator.is_main_process:
            restore_run_history(ckpt_dir, os.path.dirname(checkpoints_dir),
                                position_epoch=manifest["epoch"])
        accelerator.wait_for_everyone()
    except BaseException:
        _abort_distributed(accelerator)
        raise

    # OSR-004：各 rank 从共享 Accelerate state 目录加载（各 rank 载入自身 RNG）；
    # 兼容旧的非共享 rank-N 结构。
    shared_accel = os.path.join(ckpt_dir, "accelerator-state")
    if os.path.isdir(shared_accel) and any(
        f.startswith("random_states") for f in os.listdir(shared_accel)
    ):
        accelerator.load_state(shared_accel)
    else:
        rank_accel = os.path.join(shared_accel, f"rank-{accelerator.process_index}")
        accelerator.load_state(rank_accel if os.path.isdir(rank_accel) else shared_accel)
    engine_state.load_state_dict(read_json(os.path.join(ckpt_dir, "engine-state.json")))
    datamodule.load_state_dict(read_json(os.path.join(ckpt_dir, "datamodule-state.json")))
    metric_states = metric_states_builder()
    metric_payload = read_json(os.path.join(ckpt_dir, "metric-states.json"))
    for stage, st_state in metric_payload.items():
        if stage not in metric_states:
            raise CheckpointError(f"检查点含未声明 stage: {stage!r}")
        metric_states[stage].load_state_dict(st_state)
    best_model_state = None
    best_path = os.path.join(ckpt_dir, "best-model-state.pt")
    if os.path.exists(best_path):
        import torch
        best_model_state = torch.load(best_path, weights_only=True, map_location="cpu")
    return {
        "checkpoint_id": manifest["checkpoint_id"],
        "epoch": manifest["epoch"],
        "batch_in_epoch": manifest["batch_in_epoch"],
        "global_step": manifest["global_step"],
        "metric_states": metric_states,
        "best_model_state": best_model_state,
    }


def _fsync_file(path: str) -> None:
    with open(path, "ab") as f:
        os.fsync(f.fileno())


def _abort_distributed(accelerator: Any) -> None:
    """checkpoint 任一 rank 失败时主动断开进程组，唤醒其余 rank。"""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------
# 模型 Artifact（best/last）
# --------------------------------------------------------------------------

def write_model_manifest(
    model_dir: str,
    backend: str,
    model_signature: Mapping[str, Any],
    origin_run_id: str,
    files: Mapping[str, Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "backend": backend,
        "format": "safetensors" if backend == "torch" else "joblib",
        "format_version": 1,
        "model_signature": model_signature,
        "origin_run_id": origin_run_id,
        "created_utc": _utc_now(),
        "files": files,
        "runtime_versions": runtime_versions(backend),
    }
    if extra:
        manifest.update(extra)
    write_json(os.path.join(model_dir, "model-manifest.json"), manifest)
    return manifest


# --------------------------------------------------------------------------
# sklearn checkpoint（可信 joblib）
# --------------------------------------------------------------------------

def write_sklearn_checkpoint(
    estimator: Any,
    source_state: Mapping[str, Any],
    engine_state: Any,
    metric_states: Mapping[str, Any],
    checkpoints_dir: str,
    run_id: str,
    config_fingerprint: str,
    data_fingerprint: str,
    model_signature: Mapping[str, Any],
    epoch: int,
    global_step: int,
    batch_in_epoch: int,
    joblib: Any,
    progress_snapshot: Mapping[str, Any] | None = None,
    run_dir: str | None = None,
) -> str:
    """保存 sklearn incremental 可信检查点并返回 checkpoint_id。

    progress_snapshot + run_dir 提供时在 manifest 计算前生成内嵌进度报告
    （progress/index.html + progress/progress-snapshot.json，随 sha256 manifest 校验）；
    报告生成失败直接中止 checkpoint（fail-fast）。
    """
    ckpt_id = checkpoint_id(epoch, global_step)
    os.makedirs(checkpoints_dir, exist_ok=True)
    final_dir = os.path.join(checkpoints_dir, ckpt_id)
    if os.path.exists(final_dir):
        raise CheckpointError(f"检查点已存在: {ckpt_id}")
    staging = os.path.join(checkpoints_dir, f".staging-{ckpt_id}-{os.getpid()}")
    remove_tree(staging)
    os.makedirs(staging, exist_ok=True)
    try:
        joblib.dump(estimator, os.path.join(staging, "estimator.joblib"))
        write_json(os.path.join(staging, "engine-state.json"), engine_state.state_dict())
        write_json(os.path.join(staging, "source-state.json"), dict(source_state))
        write_json(os.path.join(staging, "metric-states.json"), json_safe(metric_states))
        if progress_snapshot is not None:
            if run_dir is None:
                raise CheckpointError("提供 progress_snapshot 时必须同时提供 run_dir")
            from .reporting import generate_checkpoint_progress_report
            generate_checkpoint_progress_report(
                run_dir, progress_snapshot, os.path.join(staging, "progress")
            )
        if run_dir is not None:
            _snapshot_run_history(staging, run_dir, position_epoch=epoch,
                                  require_metrics=False)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "checkpoint_id": ckpt_id,
            "created_utc": _utc_now(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "config_fingerprint": config_fingerprint,
            "backend": "sklearn",
            "data_fingerprint": data_fingerprint,
            "model_signature": model_signature,
            "runtime_versions": runtime_versions("sklearn"),
            "files": sha256_manifest(staging),
            "complete": True,
        }
        write_manifest(staging, manifest)
        _stage_and_finalize(staging, final_dir)
    except Exception:
        remove_tree(staging)
        raise
    update_latest(checkpoints_dir, ckpt_id, ckpt_id)
    return ckpt_id


def validate_sklearn_checkpoint_source(
    checkpoints_dir: str,
    latest_path: str,
    run_id: str,
    config_fingerprint: str,
    data_fingerprint: str,
    model_signature: Mapping[str, Any],
) -> str:
    """在 joblib.load 之前校验可信来源；返回检查点目录。"""
    ckpt_dir = os.path.join(checkpoints_dir, latest_path)
    try:
        ckpt_dir = ensure_within(checkpoints_dir, ckpt_dir, "checkpoint")
        manifest = read_json(os.path.join(ckpt_dir, CHECKPOINT_MANIFEST))
        validate_manifest_complete(manifest, ckpt_dir)
        if manifest["run_id"] != run_id:
            raise CheckpointError("joblib 来源 run_id 不匹配（只加载当前 run 自产模型）")
        if manifest["config_fingerprint"] != config_fingerprint:
            raise CheckpointError("joblib 配置指纹不匹配")
        if manifest["data_fingerprint"] != data_fingerprint:
            raise CheckpointError("joblib 数据指纹不匹配")
        if manifest["model_signature"] != model_signature:
            raise CheckpointError("joblib 模型签名不匹配")
        verify_runtime_versions("sklearn", manifest["runtime_versions"])
        est_path = os.path.join(ckpt_dir, "estimator.joblib")
        if os.path.islink(est_path):
            raise CheckpointError("joblib 为符号链接，拒绝加载")
        return ckpt_dir
    except (ArtifactError, OSError, ValueError) as exc:
        raise CheckpointError(f"joblib 校验失败，拒绝加载: {exc}") from exc
