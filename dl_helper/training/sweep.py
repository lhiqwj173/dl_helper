"""多 variant sweep：严格 manifest、顺序隔离 trial、恢复与未舍入排名。

coordinator 自身不导入实验/torch/sklearn/CUDA；trial 经子进程隔离运行。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import yaml

from .artifacts import append_jsonl, atomic_write_text, write_json
from .config import (
    ConfigError,
    config_fingerprint,
    file_sha256,
    load_config_file,
    resolve_variant_files,
    tuning_fingerprint,
)
from .contracts import MetricDefinition

SWEEP_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SweepError(Exception):
    """sweep 合同违规。"""


@dataclass(frozen=True)
class TrialSpec:
    name: str
    variant: str
    resolved_config: str  # 解析后的 variant 绝对路径
    tuning_fingerprint: str = ""


@dataclass(frozen=True)
class SweepManifest:
    schema_version: int
    sweep_id: str
    experiment: str
    base_config: str
    comparison_metric: str
    mode: str
    trials: Sequence[TrialSpec]
    manifest_path: str = ""

    def derived_run_id(self, trial_name: str) -> str:
        return f"{self.sweep_id}--{trial_name}"


def _require_str(raw: Mapping[str, Any], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SweepError(f"{label} 必须为非空字符串: {key!r}")
    return value


def parse_sweep_manifest(path: str) -> SweepManifest:
    """解析并严格校验 sweep manifest。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise SweepError("sweep manifest 根必须是 mapping")
    if int(raw.get("schema_version", 0)) != 1:
        raise SweepError("sweep manifest schema_version 必须为 1")
    sweep_raw = raw.get("sweep")
    if not isinstance(sweep_raw, Mapping):
        raise SweepError("sweep manifest 缺少 sweep 块")
    sweep_id = _require_str(sweep_raw, "id", "sweep")
    if not SWEEP_RUN_ID_RE.match(sweep_id):
        raise SweepError(f"sweep.id 不匹配字符集: {sweep_id!r}")
    experiment = _require_str(sweep_raw, "experiment", "sweep")
    base_config = _require_str(sweep_raw, "base_config", "sweep")
    comparison_metric = _require_str(sweep_raw, "comparison_metric", "sweep")
    if not comparison_metric.startswith("val/"):
        raise SweepError("comparison_metric 必须是 val/ 前缀")
    mode = sweep_raw.get("mode")
    if mode not in ("min", "max"):
        raise SweepError("sweep.mode 必须是 min/max")

    base_dir = os.path.dirname(os.path.abspath(path))
    base_path = _resolve_local_path(base_dir, base_config, "base_config")

    trials_raw = sweep_raw.get("trials")
    if not isinstance(trials_raw, list) or len(trials_raw) < 2:
        raise SweepError("sweep 至少需要两个 trial")
    seen_names: set[str] = set()
    trials: list[TrialSpec] = []
    for item in trials_raw:
        if not isinstance(item, Mapping):
            raise SweepError("trial 必须是 mapping")
        name = _require_str(item, "name", "trial")
        if not SWEEP_RUN_ID_RE.match(name):
            raise SweepError(f"trial.name 不匹配字符集: {name!r}")
        if name in seen_names:
            raise SweepError(f"trial 名称重复: {name!r}")
        seen_names.add(name)
        variant = _require_str(item, "variant", "trial")
        variant_path = _resolve_local_path(base_dir, variant, "variant")
        trials.append(TrialSpec(name=name, variant=variant, resolved_config=variant_path))
    return SweepManifest(
        schema_version=1, sweep_id=sweep_id, experiment=experiment,
        base_config=base_path, comparison_metric=comparison_metric, mode=mode,
        trials=trials, manifest_path=os.path.abspath(path),
    )


def _resolve_local_path(base_dir: str, raw_path: str, label: str) -> str:
    if not raw_path or raw_path.startswith(("http://", "https://")):
        raise SweepError(f"{label} 不允许 URL: {raw_path!r}")
    if os.path.isabs(raw_path):
        raise SweepError(f"{label} 不允许绝对路径: {raw_path!r}")
    resolved = os.path.realpath(os.path.join(base_dir, raw_path))
    base_real = os.path.realpath(base_dir)
    if not (resolved == base_real or resolved.startswith(base_real + os.sep)):
        raise SweepError(f"{label} 路径逃逸 manifest 目录: {raw_path!r}")
    if os.path.islink(resolved):
        raise SweepError(f"{label} 不允许 symlink: {raw_path!r}")
    if not os.path.exists(resolved):
        raise SweepError(f"{label} 不存在: {resolved}")
    return resolved


def resolve_trial_configs(manifest: SweepManifest) -> list[tuple[TrialSpec, Any]]:
    """逐 trial 解析 base+variant 配置，校验 tuning fingerprint 唯一。"""
    base = load_config_file(manifest.base_config)
    out: list[tuple[TrialSpec, Any]] = []
    fps: dict[str, str] = {}
    for trial in manifest.trials:
        config = resolve_variant_files(manifest.base_config, trial.resolved_config)
        tfp = tuning_fingerprint(config)
        if tfp in fps:
            raise SweepError(
                f"trial {trial.name!r} 与 {fps[tfp]!r} tuning fingerprint 相同（仅基础设施差异伪装）"
            )
        fps[tfp] = trial.name
        out.append((trial, config))
    return out


def sweep_checksum(manifest: SweepManifest) -> str:
    """sweep manifest/base/variant checksum。"""
    parts = [manifest.manifest_path, manifest.base_config]
    parts += [t.resolved_config for t in manifest.trials]
    import hashlib
    h = hashlib.sha256()
    for p in parts:
        h.update(os.path.basename(p).encode("utf-8"))
        h.update(file_sha256(p).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------
# coordinator
# --------------------------------------------------------------------------

def _sweep_dir(output_root: str, sweep_id: str) -> str:
    return os.path.join(output_root, "sweeps", sweep_id)


def _acquire_lock(sweep_dir: str) -> int:
    """原子独占 lock；冲突即失败。返回需保持打开的 fd。"""
    lock_path = os.path.join(sweep_dir, ".sweep.lock")
    os.makedirs(sweep_dir, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                raise SweepError(f"sweep 已被其他 coordinator 锁定")
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise SweepError(f"sweep 已被其他 coordinator 锁定")
    except Exception:
        os.close(fd)
        raise
    return fd


def run_sweep(manifest_path: str, resume: bool = False, project_dir: str | None = None) -> int:
    """顺序运行 sweep trial。返回退出码（0 成功 / 75 暂停 / 其他失败）。"""
    # 省略 project-dir 时按调用者当前目录解析（CLI 已解析；此处兜底保证直接调用一致）
    if project_dir is None:
        project_dir = os.getcwd()
    project_dir = os.path.realpath(project_dir)
    if not os.path.isdir(project_dir):
        raise SweepError(f"project-dir 不存在或不是目录: {project_dir}")
    manifest = parse_sweep_manifest(manifest_path)
    base_config = load_config_file(manifest.base_config)
    if project_dir is not None and project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    pythonpath_parts = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    if project_dir is not None and project_dir not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([project_dir, *pythonpath_parts])
    output_root = _resolve_output_root(manifest)
    sweep_dir = _sweep_dir(output_root, manifest.sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    lock_fd = _acquire_lock(sweep_dir)
    layout = _SweepLayout(sweep_dir)
    # OSR-002：sweep 级生命周期服务（启用时构造）
    services = _build_sweep_services(base_config, sweep_dir)
    try:
        return _run_sweep_locked(manifest, sweep_dir, layout, resume, output_root, services, project_dir)
    except BaseException as exc:
        # 恢复/预检/服务异常均把 PREEMPTED 原子转换为 FAILED，禁止双终态。
        _publish_sweep_failure(sweep_dir, {
            "sweep_id": manifest.sweep_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    finally:
        os.close(lock_fd)


def _build_sweep_services(base_config, sweep_dir):
    """从 base 配置构造 sweep 级 LifecycleServices；全部禁用时返回 None。"""
    from .notifications import WecomClient
    from .platform import Platform, SecretResolver
    from .remote import build_artifact_stores
    from .services import LifecycleServices, ServiceAudit

    enabled = base_config.remote.type == "alist" or base_config.notifications.type == "wecom"
    if not enabled:
        return None

    class _Layout:
        def __init__(self, dirpath):
            self.run_dir = dirpath
            self.service_audit_jsonl = os.path.join(dirpath, "services", "service-audit.jsonl")

    layout = _Layout(sweep_dir)
    platform = Platform()
    resolver = SecretResolver(platform)
    stores, async_sync = build_artifact_stores(base_config, platform, resolver, layout)
    wecom = None
    store_policy = "record"
    wecom_policy = "record"
    if base_config.notifications.type == "wecom":
        n = base_config.notifications
        wecom = WecomClient(n.corp_id_secret_key, n.corp_secret_key, n.agent_id_secret_key,
                            n.to_user, resolver, n.connect_timeout_seconds,
                            n.read_timeout_seconds, n.max_attempts)
        wecom_policy = n.failure_policy
    if base_config.remote.type == "alist":
        store_policy = base_config.remote.failure_policy
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=resolver.redact)
    return LifecycleServices(layout=layout, secret_resolver=resolver, stores=stores,
                             async_sync=async_sync, wecom_client=wecom, audit=audit,
                             failure_policy=store_policy, wecom_policy=wecom_policy)


def _resolve_output_root(manifest: SweepManifest) -> str:
    base = load_config_file(manifest.base_config)
    from .platform import Platform
    return Platform().resolve_output_root(base)


def _run_sweep_locked(manifest: SweepManifest, sweep_dir: str, layout: _SweepLayout,
                      resume: bool, output_root: str, services=None,
                      project_dir: str | None = None) -> int:

    # OSR-002：sweep 生命周期事件
    if services is not None:
        services.start_sweep(manifest.sweep_id)

    # 恢复校验
    if resume:
        _validate_resume(manifest, sweep_dir)

    trial_configs = resolve_trial_configs(manifest)

    # evaluation contract 预检（零拟合 step）+ 跨 trial 可比性
    contracts: list[dict[str, Any]] = []
    contract_dir = os.path.join(sweep_dir, "contracts")
    os.makedirs(contract_dir, exist_ok=True)
    for trial, config in trial_configs:
        contract = _emit_evaluation_contract(trial, config, manifest, project_dir)
        if contract is None or contract.get("valid") is not True:
            _publish_sweep_failure(sweep_dir,
                       {"sweep_id": manifest.sweep_id,
                        "preflight_errors": contract.get("errors") if contract else ["contract 无效"]})
            if services is not None:
                services.finalize_sweep(manifest.sweep_id, "failed")
            return 1
        # OSR-008：contract 落盘，供暂停/恢复 checksum 校验
        write_json(os.path.join(contract_dir, f"{trial.name}.json"), contract)
        contracts.append(contract)
    try:
        _compare_contracts(manifest, contracts)
    except SweepError as exc:
        _publish_sweep_failure(sweep_dir,
                   {"sweep_id": manifest.sweep_id, "comparability_errors": [str(exc)]})
        if services is not None:
            services.finalize_sweep(manifest.sweep_id, "failed")
        return 1

    statuses: list[dict[str, Any]] = []
    for trial, config in trial_configs:
        run_id = manifest.derived_run_id(trial.name)
        if resume and _trial_completed(layout, run_id):
            statuses.append({"trial": trial.name, "run_id": run_id, "status": "succeeded"})
            continue
        append_jsonl(layout.trials_jsonl, {"trial": trial.name, "run_id": run_id, "status": "started"})
        if services is not None:
            services.trial_event(manifest.sweep_id, trial.name, "started")
        code = _run_trial_subprocess(manifest, trial, config, run_id, sweep_dir, project_dir)
        if code == 0:
            append_jsonl(layout.trials_jsonl, {"trial": trial.name, "run_id": run_id, "status": "succeeded"})
            if services is not None:
                services.trial_event(manifest.sweep_id, trial.name, "succeeded")
            statuses.append({"trial": trial.name, "run_id": run_id, "status": "succeeded"})
        elif code == 75:
            if services is not None:
                services.trial_event(manifest.sweep_id, trial.name, "preempted")
            # 先形成可校验的 sweep pause，让远端 bundle 与本地终态一致。
            _write_pause_manifest(manifest, sweep_dir, run_id, statuses, output_root)
            if services is not None:
                try:
                    services.finalize_sweep(manifest.sweep_id, "preempted")
                except BaseException:
                    os.remove(os.path.join(sweep_dir, "pause-manifest.json"))
                    raise
            return 75
        else:
            _publish_sweep_failure(sweep_dir,
                       {"sweep_id": manifest.sweep_id, "failed_trial": trial.name,
                        "run_id": run_id, "exit_code": code})
            if services is not None:
                services.trial_event(manifest.sweep_id, trial.name, "failed")
                services.finalize_sweep(manifest.sweep_id, "failed")
            return code

    # 全部成功 → 排名 + best + 报告 + manifest
    ranking = _compute_ranking(manifest, trial_configs, output_root)
    best_trial = ranking[0]["trial"] if ranking else None
    write_json(os.path.join(sweep_dir, "best-trial.json"),
               {"sweep_id": manifest.sweep_id, "best_trial": best_trial,
                "best_run_id": manifest.derived_run_id(best_trial) if best_trial else None})
    from .reporting import generate_sweep_report
    generate_sweep_report(sweep_dir)
    success = {
        "schema_version": 1,
        "sweep_id": manifest.sweep_id,
        "comparison_metric": manifest.comparison_metric,
        "mode": manifest.mode,
        "ranking": ranking,
        "best_trial": best_trial,
        "checksum": sweep_checksum(manifest),
        "trials": statuses,
    }
    # 先移除旧 pause 再写 success，保证任意时刻不并存 pause/success 双终态（OSR-008）
    pause_path = os.path.join(sweep_dir, "pause-manifest.json")
    if os.path.exists(pause_path):
        os.remove(pause_path)
    write_json(os.path.join(sweep_dir, "sweep-manifest.json"), success)
    # 终态文件先落盘，服务 bundle 才能携带完整 sweep 终态。
    if services is not None:
        try:
            services.finalize_sweep(manifest.sweep_id, "succeeded", best=best_trial)
        except BaseException:
            os.remove(os.path.join(sweep_dir, "sweep-manifest.json"))
            raise
    return 0


def _publish_sweep_failure(sweep_dir: str, data: Mapping[str, Any]) -> str:
    """原子发布 sweep FAILED，允许且只允许从 pause 过渡。"""
    pause_path = os.path.join(sweep_dir, "pause-manifest.json")
    success_path = os.path.join(sweep_dir, "sweep-manifest.json")
    failure_path = os.path.join(sweep_dir, "failure.json")
    if os.path.exists(success_path):
        raise SweepError("已存在成功 sweep 终态，拒绝追加 FAILED")

    token = f"{os.getpid()}"
    temp_failure = os.path.join(sweep_dir, f".failure.transitioning-{token}")
    transition_pause = os.path.join(sweep_dir, f".pause.transitioning-{token}")
    write_json(temp_failure, data)
    moved_pause = False
    try:
        if os.path.exists(pause_path):
            os.replace(pause_path, transition_pause)
            moved_pause = True
        os.replace(temp_failure, failure_path)
        if moved_pause:
            try:
                os.remove(transition_pause)
            except OSError as cleanup_exc:
                os.replace(failure_path, temp_failure)
                os.replace(transition_pause, pause_path)
                if os.path.exists(temp_failure):
                    os.remove(temp_failure)
                raise SweepError("FAILED 已提交但旧 pause 清理失败") from cleanup_exc
        return failure_path
    except BaseException:
        if moved_pause and os.path.exists(transition_pause) and not os.path.exists(pause_path):
            os.replace(transition_pause, pause_path)
        if os.path.exists(temp_failure):
            os.remove(temp_failure)
        raise


class _SweepLayout:
    def __init__(self, sweep_dir: str) -> None:
        self.sweep_dir = sweep_dir
        self.trials_jsonl = os.path.join(sweep_dir, "trials.jsonl")
        self.pause_manifest = os.path.join(sweep_dir, "pause-manifest.json")


def _trial_completed(layout: _SweepLayout, run_id: str) -> bool:
    """校验 run 终态（run-manifest.json 存在且完整），而非只信 trials.jsonl 文本（OSR-008）。"""
    if not os.path.exists(layout.trials_jsonl):
        return False
    found = False
    with open(layout.trials_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("run_id") == run_id and rec.get("status") == "succeeded":
                found = True
                break
    if not found:
        return False
    # 校验 run 终态存在（sweep_dir = output_root/sweeps/<id>）
    output_root = os.path.dirname(os.path.dirname(layout.sweep_dir))
    run_dir = os.path.join(output_root, "runs", run_id)
    terminal = os.path.join(run_dir, "run-manifest.json")
    return os.path.exists(terminal)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _run_trial_subprocess(manifest: SweepManifest, trial: TrialSpec, config: Any,
                          run_id: str, sweep_dir: str,
                          project_dir: str | None = None) -> int:
    """以 sys.executable -m ... train 在全新子进程运行 trial。

    暂停的 trial 省略 --resume（内部 auto 自动恢复）；未暂停的 trial 显式
    --resume none 禁止误恢复旧 checkpoint。
    """
    args = [
        sys.executable, "-m", "dl_helper.training.cli", "train",
        "--config", manifest.base_config,
        "--variant", trial.resolved_config,
        "--experiment", manifest.experiment,
        "--run-id", run_id,
    ]
    if not _trial_paused(sweep_dir, run_id):
        args.extend(["--resume", "none"])
    if project_dir is not None:
        args.extend(["--project-dir", project_dir])
    proc = subprocess.run(args, cwd=_repo_root(), check=False, encoding="utf-8",
                          env=_subprocess_env())
    return proc.returncode


def _trial_paused(sweep_dir: str, run_id: str) -> bool:
    pause = os.path.join(sweep_dir, "pause-manifest.json")
    if not os.path.exists(pause):
        return False
    with open(pause, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("current_run_id") == run_id


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_pause_manifest(manifest: SweepManifest, sweep_dir: str, current_run_id: str,
                          completed: list[dict[str, Any]], output_root: str) -> None:
    """暂停清单绑定 checkpoint/remaining/contract/run checksum（OSR-008）。"""
    remaining = []
    found = False
    for trial in manifest.trials:
        if trial.name == current_run_id.split("--")[-1]:
            found = True
        elif found:
            remaining.append(trial.name)
    run_checksums = {}
    for entry in completed:
        run_id = entry.get("run_id")
        if run_id:
            run_checksums[run_id] = _run_terminal_checksum(output_root, run_id)
    # OSR-008：绑定当前被暂停 run 的 resume checkpoint（从 run 级 pause manifest 读取）
    current_checkpoint = None
    run_terminal = os.path.join(output_root, "runs", current_run_id, "pause-manifest.json")
    if os.path.exists(run_terminal):
        with open(run_terminal, "r", encoding="utf-8") as f:
            current_checkpoint = json.load(f).get("resume_checkpoint")
    # OSR-008：code 75 只有当前 run 有完整 pause/checkpoint 才能发布 sweep pause
    if current_checkpoint is None:
        raise SweepError(f"当前 run {current_run_id} 无 pause/resume_checkpoint，无法发布 sweep pause")
    # OSR-008：绑定 checkpoint 的 manifest checksum（防篡改）；缺失必须失败
    ckpt_manifest = os.path.join(output_root, "runs", current_run_id,
                                 "checkpoints", current_checkpoint, "checkpoint-manifest.json")
    if not os.path.exists(ckpt_manifest):
        raise SweepError(f"当前 run checkpoint manifest 缺失，无法绑定 pause: {current_checkpoint}")
    from .artifacts import sha256_file
    current_checkpoint_checksum = sha256_file(ckpt_manifest)
    # OSR-008：已完成 run 必须都有非空 checksum
    for entry in completed:
        run_id = entry.get("run_id")
        if run_id and _run_terminal_checksum(output_root, run_id) is None:
            raise SweepError(f"已完成 run {run_id} 缺终态，无法绑定 pause")
    pause = {
        "schema_version": 1,
        "sweep_id": manifest.sweep_id,
        "current_run_id": current_run_id,
        "current_checkpoint": current_checkpoint,
        "current_checkpoint_checksum": current_checkpoint_checksum,
        "completed": completed,
        "remaining": remaining,
        "checksum": sweep_checksum(manifest),
        "contract_checksum": _contracts_checksum(manifest, output_root),
        "run_checksums": run_checksums,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(os.path.join(sweep_dir, "pause-manifest.json"), pause)


def _run_terminal_checksum(output_root: str, run_id: str) -> str | None:
    from .artifacts import sha256_file
    terminal = os.path.join(output_root, "runs", run_id, "run-manifest.json")
    if not os.path.exists(terminal):
        return None
    return sha256_file(terminal)


def _contracts_checksum(manifest: SweepManifest, output_root: str) -> str:
    import hashlib
    from .config import file_sha256
    h = hashlib.sha256()
    for trial in manifest.trials:
        contract_path = os.path.join(output_root, "sweeps", manifest.sweep_id,
                                     "contracts", f"{trial.name}.json")
        if os.path.exists(contract_path):
            h.update(file_sha256(contract_path).encode("utf-8"))
    return h.hexdigest()


def _validate_resume(manifest: SweepManifest, sweep_dir: str) -> None:
    pause_path = os.path.join(sweep_dir, "pause-manifest.json")
    if not os.path.exists(pause_path):
        raise SweepError("resume 需要完整 PREEMPTED sweep（pause-manifest 缺失）")
    if os.path.exists(os.path.join(sweep_dir, "failure.json")):
        raise SweepError("FAILED sweep 不允许原 ID 续跑")
    with open(pause_path, "r", encoding="utf-8") as f:
        pause = json.load(f)
    if pause.get("sweep_id") != manifest.sweep_id:
        raise SweepError("pause manifest sweep_id 漂移")
    if pause.get("checksum") != sweep_checksum(manifest):
        raise SweepError("manifest/base/variant checksum 漂移，拒绝恢复")
    output_root = os.path.dirname(os.path.dirname(sweep_dir))
    if pause.get("contract_checksum") != _contracts_checksum(manifest, output_root):
        raise SweepError("contract checksum 漂移，拒绝恢复")
    # OSR-008：字段缺失/漂移立即失败
    if not pause.get("current_run_id"):
        raise SweepError("pause manifest 缺 current_run_id，拒绝恢复")
    if not pause.get("current_checkpoint"):
        raise SweepError("pause manifest 缺 current_checkpoint，拒绝恢复")
    if not pause.get("current_checkpoint_checksum"):
        raise SweepError("pause manifest 缺 current_checkpoint_checksum，拒绝恢复")
    current_run = pause["current_run_id"]
    run_pause = os.path.join(output_root, "runs", current_run, "pause-manifest.json")
    if not os.path.exists(run_pause):
        raise SweepError(f"当前 run {current_run} 缺少暂停终态，拒绝恢复")
    ckpt_dir = os.path.join(output_root, "runs", current_run, "checkpoints",
                            pause["current_checkpoint"])
    ckpt_manifest = os.path.join(ckpt_dir, "checkpoint-manifest.json")
    if not os.path.isdir(ckpt_dir) or not os.path.exists(ckpt_manifest):
        raise SweepError(f"当前 run checkpoint {pause['current_checkpoint']} manifest 缺失，拒绝恢复")
    if pause.get("current_checkpoint_checksum"):
        from .artifacts import sha256_file
        if pause["current_checkpoint_checksum"] != sha256_file(ckpt_manifest):
            raise SweepError("当前 checkpoint manifest checksum 漂移，拒绝恢复")
    recorded = pause.get("run_checksums", {})
    if not isinstance(recorded, dict) or "run_checksums" not in pause:
        raise SweepError("pause manifest 缺 run_checksums，拒绝恢复")
    for entry in pause.get("completed", []):
        run_id = entry.get("run_id")
        if run_id and run_id not in recorded:
            raise SweepError(f"已完成 run {run_id} 缺 checksum，拒绝恢复")
    for run_id, cksum in recorded.items():
        if cksum is not None and cksum != _run_terminal_checksum(output_root, run_id):
            raise SweepError(f"已完成 run {run_id} checksum 漂移，拒绝恢复")


# --------------------------------------------------------------------------
# 零优化步可比性预检（任务 6.2）
# --------------------------------------------------------------------------

def _emit_evaluation_contract(trial: TrialSpec, config: Any, manifest: SweepManifest,
                              project_dir: str | None = None) -> dict[str, Any] | None:
    """通过隐藏的 train preflight 子进程获取零拟合 evaluation contract。"""
    args = [
        sys.executable, "-m", "dl_helper.training.cli", "train",
        "--config", manifest.base_config,
        "--variant", trial.resolved_config,
        "--experiment", manifest.experiment,
        "--preflight-only",
    ]
    if project_dir is not None:
        args.extend(["--project-dir", project_dir])
    proc = subprocess.run(args, cwd=_repo_root(), check=False, encoding="utf-8",
                          capture_output=True, env=_subprocess_env())
    if proc.returncode != 0:
        return None
    # 解析 stdout 中最后一行 JSON contract
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _compare_contracts(manifest: SweepManifest, contracts: list[dict[str, Any]]) -> None:
    """跨 trial 比较评价合同：Experiment/backend/DataIdentity/Task/MetricDefinition 必须一致。"""
    if len(contracts) < 2:
        return
    reference = contracts[0]
    for required in ("data_identity", "splits", "label_schema", "metric_definitions"):
        if not reference.get(required):
            raise SweepError(f"evaluation contract 缺少 {required}，零拟合 step 拒绝")
    identity = reference["data_identity"]
    if not identity.get("fingerprint"):
        raise SweepError("evaluation contract 缺少 data_identity fingerprint")
    for split, split_info in reference["splits"].items():
        if not isinstance(split_info, Mapping) or not split_info.get("fingerprint"):
            raise SweepError(f"evaluation contract 缺少 {split} split fingerprint")
    ref_fields = ("backend", "experiment_ref", "data_identity", "task_name", "metric_definitions")
    # OSR-008：split 集合与 label/target schema 必须一致；model_signature 允许跨 trial 变化（design）
    ref_fields = ref_fields + ("splits", "label_schema")
    for trial, contract in zip(manifest.trials[1:], contracts[1:]):
        if not contract.get("label_schema") or not contract.get("splits"):
            raise SweepError(f"trial {trial.name} evaluation contract 声明不完整")
        for field in ref_fields:
            if contract.get(field) != reference.get(field):
                raise SweepError(
                    f"trial {trial.name} 与首 trial 的 {field} 不一致（零拟合 step 拒绝）"
                )
    # comparison metric 必须由 Task 产生且 full/exact
    defs = reference.get("metric_definitions", {})
    metric_name = manifest.comparison_metric.removeprefix("val/")
    if metric_name not in defs:
        raise SweepError(f"comparison metric {manifest.comparison_metric!r} 未由 Task 产生")
    defn = defs[metric_name]
    if not defn.get("exact") or defn.get("evaluation_scope") != "full":
        raise SweepError("comparison metric 必须 exact/full")
    if defn.get("direction") != manifest.mode:
        raise SweepError("comparison metric direction 必须等于 sweep.mode")


# --------------------------------------------------------------------------
# 排名与 best（任务 6.5）
# --------------------------------------------------------------------------

def _compute_ranking(manifest: SweepManifest, trial_configs: list[tuple[TrialSpec, Any]],
                     output_root: str) -> list[dict[str, Any]]:
    """从各 run summary 读取未舍入 comparison 值并复核定义后排名（OSR-008）。"""
    rows: list[dict[str, Any]] = []
    for order, (trial, _config) in enumerate(trial_configs):
        run_id = manifest.derived_run_id(trial.name)
        entry = _read_summary_comparison(output_root, run_id, manifest.comparison_metric, manifest.mode)
        if entry is None:
            raise SweepError(f"trial {trial.name} comparison 值缺失/非有限: {manifest.comparison_metric!r}")
        value, defn = entry
        rows.append({"rank": 0, "trial": trial.name, "run_id": run_id,
                     "value": value, "order": order})
    # 按 mode 方向排序；并列按 YAML order 稳定
    direction = -1 if manifest.mode == "max" else 1
    rows.sort(key=lambda r: (direction * r["value"], r["order"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
        row.pop("order", None)
    return rows


def _read_summary_comparison(output_root: str, run_id: str, metric: str, mode: str):
    """从 summary.json 读取最终 val 全量指标并复核 MetricDefinition（exact/full/direction）。"""
    runs_dir = os.path.join(output_root, "runs", run_id)
    summary_path = os.path.join(runs_dir, "metrics", "summary.json")
    if not os.path.exists(summary_path):
        raise SweepError(f"run summary 缺失: {run_id}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    defs = summary.get("metric_definitions", {})
    stage_metrics = summary.get("stage_metrics", {})
    if metric not in stage_metrics.get("val", {}):
        return None
    value = float(stage_metrics["val"][metric])
    if not _finite(value):
        raise SweepError(f"run {run_id} comparison 值非有限: {value!r}")
    metric_name = metric.removeprefix("val/")
    defn = defs.get(metric_name)
    if defn is None:
        raise SweepError(f"run {run_id} 缺 comparison metric 定义: {metric_name!r}")
    if not defn.get("exact") or defn.get("evaluation_scope") != "full":
        raise SweepError(f"run {run_id} comparison metric 必须 exact/full")
    if defn.get("direction") != mode:
        raise SweepError(f"run {run_id} comparison direction {defn.get('direction')} != mode {mode}")
    return value, defn


def _finite(v: float) -> bool:
    import math
    return math.isfinite(v)


def generate_sweep_report(sweep_dir: str, out_dir: str | None = None) -> str:
    from .reporting import generate_sweep_report as _gen
    return _gen(sweep_dir, out_dir)
