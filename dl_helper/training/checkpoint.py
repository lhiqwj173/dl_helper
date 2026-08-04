"""不可变检查点、manifest、latest 指针与可信恢复。

Torch 使用 staging -> Accelerate state -> manifest -> immutable dir -> latest；
sklearn incremental 使用可信 joblib + source state，校验必须先于反序列化。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .artifacts import (
    ArtifactError,
    atomic_write_text,
    ensure_within,
    json_safe,
    move_tree,
    read_json,
    remove_tree,
    sha256_file,
    sha256_manifest,
    write_json,
)

CHECKPOINT_MANIFEST = "checkpoint-manifest.json"
LATEST_FILE = "latest.json"


class CheckpointError(Exception):
    """检查点不可恢复或校验失败。"""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def checkpoint_id(epoch: int, global_step: int) -> str:
    return f"epoch-{epoch:06d}-step-{global_step:08d}"


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
    current = runtime_versions(backend)
    for key, value in recorded.items():
        if current.get(key) != value:
            raise CheckpointError(f"runtime 版本不精确匹配: {key} recorded={value} current={current.get(key)}")


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
) -> str:
    """保存 torch 不可变检查点并返回 checkpoint_id。

    OSR-004：所有 rank 共同参与 Accelerate save 协议（各自保存 RNG/state 到 rank 子目录），
    仅主 rank 写 metadata/manifest 并 move 到不可变目录。
    """
    ckpt_id = checkpoint_id(epoch, global_step)
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
) -> str:
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
