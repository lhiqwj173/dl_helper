"""固定 run/sweep schema、原子写入、SHA256 清单与预测分片。

所有文本 I/O 显式 UTF-8；原子写入使用同目录 tmp + flush + fsync + os.replace；
只有主进程写共享文件；success/pause/failure 终态互斥且最后发布。
"""
from __future__ import annotations

import hashlib
import json
import os
import string
import shutil
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

TERMINAL_SUCCESS = "run-manifest.json"
TERMINAL_PREEMPTED = "pause-manifest.json"
TERMINAL_FAILED = "failure.json"
TERMINAL_FILES = (TERMINAL_SUCCESS, TERMINAL_PREEMPTED, TERMINAL_FAILED)


class ArtifactError(Exception):
    """Artifact schema 或写入违规。"""


# --------------------------------------------------------------------------
# 原子文本写入
# --------------------------------------------------------------------------

def atomic_write_text(path: str, text: str) -> None:
    """同目录 tmp + flush + fsync + os.replace 原子写。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def write_json(path: str, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n")


def write_yaml(path: str, data: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    """JSONL 追加（不原子；仅主进程写）。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def json_safe(value: Any) -> Any:
    """递归把 numpy 标量/数组转换为 JSON-safe Python 类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: str) -> int:
    return os.path.getsize(path)


def list_relative_files(root: str) -> list[str]:
    """返回 root 下全部相对文件路径（排序），拒绝 symlink。"""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                raise ArtifactError(f"Artifact 含符号链接: {full}")
            rel = os.path.relpath(full, root)
            out.append(rel)
    return out


def sha256_manifest(root: str) -> dict[str, dict[str, Any]]:
    """生成 root 下每个文件的 size/SHA256。"""
    manifest: dict[str, dict[str, Any]] = {}
    for rel in list_relative_files(root):
        full = os.path.join(root, rel)
        manifest[rel] = {"size": file_size(full), "sha256": sha256_file(full)}
    return manifest


# --------------------------------------------------------------------------
# 路径边界
# --------------------------------------------------------------------------

def ensure_within(root: str, candidate: str, label: str) -> str:
    """确保 candidate（解析后）位于 root 目录树内，否则失败。"""
    root_real = os.path.normpath(os.path.realpath(root))
    cand_real = os.path.normpath(os.path.realpath(candidate))
    root_key = os.path.normcase(root_real)
    cand_key = os.path.normcase(cand_real)
    root_prefix = root_key if root_key.endswith(os.sep) else root_key + os.sep
    if not (cand_key == root_key or cand_key.startswith(root_prefix)):
        raise ArtifactError(f"{label} 路径逃逸目录边界: {candidate!r}")
    return cand_real


def safe_join(root: str, *parts: str, label: str = "path") -> str:
    candidate = os.path.join(root, *parts)
    return ensure_within(root, candidate, label)


# --------------------------------------------------------------------------
# 终态互斥
# --------------------------------------------------------------------------

def existing_terminal(run_dir: str) -> str | None:
    """返回唯一终态文件名；多终态并存视为损坏并显式失败（OSR-003）。"""
    found = [name for name in TERMINAL_FILES if os.path.exists(os.path.join(run_dir, name))]
    if len(found) > 1:
        raise ArtifactError(f"多终态并存（不合法）: {found}")
    return found[0] if found else None


def publish_terminal(run_dir: str, kind: str, data: Mapping[str, Any]) -> str:
    """原子发布唯一终态；允许恢复失败时从 pause 过渡为 FAILED。"""
    if kind == "success":
        filename = TERMINAL_SUCCESS
    elif kind == "preempted":
        filename = TERMINAL_PREEMPTED
    elif kind == "failed":
        filename = TERMINAL_FAILED
    else:
        raise ArtifactError(f"未知终态: {kind!r}")
    existing = existing_terminal(run_dir)
    path = os.path.join(run_dir, filename)
    if existing is None or existing == filename:
        write_json(path, data)
        return path
    if not (existing == TERMINAL_PREEMPTED and filename == TERMINAL_FAILED):
        raise ArtifactError(f"终态互斥违规: 已存在 {existing}，尝试写 {filename}")

    old_path = os.path.join(run_dir, existing)
    transition_path = os.path.join(run_dir, f".{existing}.transitioning-{os.getpid()}")
    temp_failure = os.path.join(run_dir, f".{filename}.transitioning-{os.getpid()}")
    write_json(temp_failure, data)
    moved_old = False
    try:
        os.replace(old_path, transition_path)
        moved_old = True
        os.replace(temp_failure, path)
        try:
            os.remove(transition_path)
        except OSError as cleanup_exc:
            # 不能留下 failure + pause 双终态；先把新 failure 移出终态集合，
            # 再恢复旧 pause。临时文件清理失败也只能留下隐藏诊断文件。
            try:
                os.replace(path, temp_failure)
                os.replace(transition_path, old_path)
                if os.path.exists(temp_failure):
                    os.remove(temp_failure)
            except BaseException as rollback_exc:
                raise ArtifactError("FAILED 过渡清理失败，无法恢复唯一终态") from rollback_exc
            raise ArtifactError(f"FAILED 已提交但旧 pause 清理失败: {transition_path!r}") from cleanup_exc
        return path
    except BaseException:
        if moved_old and os.path.exists(transition_path) and not os.path.exists(old_path):
            os.replace(transition_path, old_path)
        if os.path.exists(temp_failure):
            os.remove(temp_failure)
        raise


# --------------------------------------------------------------------------
# Run 目录布局
# --------------------------------------------------------------------------

@dataclass
class RunLayout:
    """固定 run 目录 schema 的路径布局与常用写入。"""

    run_dir: str

    def ensure(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)

    def path(self, *parts: str) -> str:
        return safe_join(self.run_dir, *parts)

    @property
    def logs(self) -> str:
        return self.path("logs", "train.log")

    @property
    def metrics_jsonl(self) -> str:
        return self.path("metrics", "metrics.jsonl")

    @property
    def summary_json(self) -> str:
        return self.path("metrics", "summary.json")

    @property
    def config_resolved(self) -> str:
        return self.path("config.resolved.yaml")

    @property
    def environment_json(self) -> str:
        return self.path("environment.json")

    @property
    def evaluation_contract_json(self) -> str:
        return self.path("evaluation-contract.json")

    @property
    def service_audit_jsonl(self) -> str:
        return self.path("services", "service-audit.jsonl")

    @property
    def checkpoints_latest(self) -> str:
        return self.path("checkpoints", "latest.json")

    @property
    def report_index(self) -> str:
        return self.path("report", "index.html")

    def predictions_dir(self, split: str) -> str:
        return self.path("predictions", split)

    def write_text(self, rel: str, text: str) -> None:
        atomic_write_text(self.path(rel), text)

    def write_json(self, rel: str, data: Any) -> None:
        write_json(self.path(rel), data)

    def log(self, message: str) -> None:
        line = f"[{_utc_now()}] {message}"
        append_jsonl(self.logs, {"ts": _utc_now(), "message": message})
        print(line)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# 确定性优先级抽样
# --------------------------------------------------------------------------

def _stable_hash(seed: int, split: str, sample_id: Any) -> int:
    """64-bit hash：用于曲线抽样的稳定 sample key。"""
    if isinstance(sample_id, (np.integer,)):
        sample_id = int(sample_id)
    text = f"{seed}:{split}:{sample_id}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def priority_sample(
    n: int,
    seed: int,
    split: str,
    limit: int,
    sample_ids: Sequence[Any] | None = None,
) -> np.ndarray:
    """返回保留的样本索引（hash 最小的前 limit 个）。

    sample_ids 为 None 时使用全局递增位置作为 key（记录位置抽样限制）。
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if limit <= 0:
        raise ArtifactError("抽样上限必须为正")
    if sample_ids is None:
        keys = np.array([_stable_hash(seed, split, i) for i in range(n)], dtype=np.uint64)
        order = np.argsort(keys, kind="stable")
    else:
        if len(sample_ids) != n:
            raise ArtifactError("sample_ids 长度必须等于样本数")
        keys = np.array([_stable_hash(seed, split, sid) for sid in sample_ids], dtype=np.uint64)
        order = np.argsort(keys, kind="stable")
    return order[:limit].astype(np.int64)


def merge_topk_candidates(
    per_rank: Sequence[np.ndarray],
    limit: int,
) -> np.ndarray:
    """合并各 rank 候选后保留全局最小 hash 的索引。"""
    all_idx = np.concatenate(per_rank) if per_rank else np.empty(0, dtype=np.int64)
    return all_idx[:limit]


# --------------------------------------------------------------------------
# 预测分片
# --------------------------------------------------------------------------

_FIELD_NAME_CHARS = string.ascii_letters + string.digits + "_.-"

_OBJECT_KIND = "O"
_COMPLEX_KIND = "c"


def _validate_field_name(name: str) -> None:
    if not name or not all(c in _FIELD_NAME_CHARS for c in name) or name[0].isdigit():
        raise ArtifactError(f"预测字段名非法: {name!r}")


def _validate_field_array(name: str, arr: np.ndarray, sample_count: int) -> None:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        raise ArtifactError(f"预测字段 {name!r} 必须至少一维")
    if arr.shape[0] != sample_count:
        raise ArtifactError(f"预测字段 {name!r} 样本维 {arr.shape[0]} != {sample_count}")
    if arr.dtype.kind in "O":
        raise ArtifactError(f"预测字段 {name!r} 禁止 object dtype")
    if arr.dtype.kind == "U":
        # 固定宽度 Unicode 允许
        pass
    if arr.dtype.kind == "c":
        raise ArtifactError(f"预测字段 {name!r} 禁止 complex dtype")
    if arr.dtype.kind in "fc":
        if not np.all(np.isfinite(arr)):
            raise ArtifactError(f"预测字段 {name!r} 含非有限值")


def write_prediction_shard(
    dir_path: str,
    rank: int,
    index: int,
    arrays: Mapping[str, np.ndarray],
    sample_count: int,
) -> dict[str, Any]:
    """写一个无 pickle NPZ 分片并返回其 manifest 项。

    文件名固定 part-rank{rank:05d}-{index:06d}.npz。
    """
    os.makedirs(dir_path, exist_ok=True)
    filename = f"part-rank{rank:05d}-{index:06d}.npz"
    path = os.path.join(dir_path, filename)
    npz: dict[str, np.ndarray] = {}
    field_meta: dict[str, dict[str, Any]] = {}
    for name, arr in arrays.items():
        _validate_field_name(name)
        arr = np.asarray(arr)
        _validate_field_array(name, arr, sample_count)
        npz[name] = arr
        field_meta[name] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    np.savez_compressed(path, **npz)
    return {
        "file": filename,
        "rank": rank,
        "index": index,
        "sample_count": sample_count,
        "fields": field_meta,
        "size": file_size(path),
        "sha256": sha256_file(path),
    }


def write_prediction_manifest(
    dir_path: str,
    split: str,
    shard_entries: Sequence[Mapping[str, Any]],
    sample_count: int,
    sampled: bool,
    total_sample_count: int | None = None,
    sampling_notes: str | None = None,
) -> dict[str, Any]:
    """写预测 manifest。"""
    manifest = {
        "schema_version": 1,
        "split": split,
        "shards": list(shard_entries),
        "sample_count": sample_count,
        "sampled": sampled,
    }
    if total_sample_count is not None:
        manifest["total_sample_count"] = total_sample_count
    if sampling_notes:
        manifest["sampling_notes"] = sampling_notes
    manifest_path = os.path.join(dir_path, "prediction-manifest.json")
    write_json(manifest_path, manifest)
    return manifest


def remove_tree(path: str) -> None:
    """删除目录树（仅用于 staging/tmp 清理，不得删除终态）。"""
    if os.path.exists(path):
        shutil.rmtree(path)


def move_tree(src: str, dst: str) -> None:
    """原子移动目录（用于 staging -> 不可变目录）。"""
    if os.path.exists(dst):
        raise ArtifactError(f"目标已存在: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(src, dst)
