"""CLI：train/report/sweep/sweep-report 命令与退出码。

退出码：0 成功、75 PREEMPTED、其他非零为失败。顶层只写脱敏失败证据后原样 raise。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from typing import Any, Sequence

from .config import RESUME_AUTO, Config, config_to_dict, load_config_file, resolve_variant_files

EXIT_OK = 0
EXIT_PREEMPTED = 75

# 实际导入的 dl_helper 包目录（realpath）；训练内容不得进入库模块
_DL_HELPER_PACKAGE_REALPATH = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class CliError(Exception):
    """CLI 参数或执行错误。"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dl_helper.training", description="Kaggle 通用训练平台")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="运行一次训练")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--variant")
    p_train.add_argument("--experiment", required=True)
    p_train.add_argument("--project-dir", help="外部训练项目根目录（默认当前目录）")
    p_train.add_argument("--output-root", help="覆盖配置中的 run.output_root")
    p_train.add_argument("--resume", choices=["none", "required"], default=None,
                         help="显式恢复策略：none 禁止恢复；required 无兼容 checkpoint 即失败；省略时内部自动恢复")
    p_train.add_argument("--run-id")
    # sweep 的零拟合合同使用；隐藏以免形成公共命令面。
    p_train.add_argument("--preflight-only", action="store_true", help=argparse.SUPPRESS)

    p_report = sub.add_parser("report", help="从 Artifact 生成离线报告")
    p_report.add_argument("--run", required=True)
    p_report.add_argument("--out")

    p_sweep = sub.add_parser("sweep", help="顺序运行多 variant sweep")
    p_sweep.add_argument("--sweep", required=True)
    p_sweep.add_argument("--project-dir", help="外部训练项目根目录（默认当前目录）")
    p_sweep.add_argument("--resume", action="store_true")

    p_sr = sub.add_parser("sweep-report", help="生成 sweep 聚合报告")
    p_sr.add_argument("--sweep-dir", required=True)
    p_sr.add_argument("--out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except Exception as exc:
        if not getattr(args, "_failure_evidence_attempted", False):
            try:
                _write_failure_evidence(args, exc)
            except Exception as evidence_exc:
                # OSR-003：证据写入问题进入 secondary_errors（可审计），不替换 primary
                if not hasattr(args, "_secondary_errors"):
                    args._secondary_errors = []
                args._secondary_errors.append({
                    "service": "evidence",
                    "event": "WRITE_FAILURE",
                    "error_type": type(evidence_exc).__name__,
                    "message": str(evidence_exc),
                })
                print(f"failure evidence write failed: {evidence_exc}", file=sys.stderr)
        raise


def _dispatch(args: argparse.Namespace) -> int:
    cmd = args.command
    if cmd == "train":
        return _cmd_train(args)
    if cmd == "report":
        return _cmd_report(args)
    if cmd == "sweep":
        return _cmd_sweep(args)
    if cmd == "sweep-report":
        return _cmd_sweep_report(args)
    raise CliError(f"未知命令: {cmd!r}")


def _load_config(args: argparse.Namespace) -> Config:
    if args.variant:
        return resolve_variant_files(args.config, args.variant)
    return load_config_file(args.config)


def _prepare_project_dir(path: str | None) -> str:
    """把外部训练项目加入 import 路径，库本身不携带任何训练项目。"""
    project_dir = os.path.realpath(path or os.getcwd())
    if not os.path.isdir(project_dir):
        raise CliError(f"project-dir 不存在或不是目录: {project_dir}")
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [item for item in current_pythonpath.split(os.pathsep) if item]
    if project_dir not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([project_dir, *pythonpath_parts])
    return project_dir


def _reject_inside_package(path: str, label: str, package_root: str) -> None:
    """realpath + normcase 拒绝包内路径，处理大小写与路径逃逸。"""
    normalized = os.path.normcase(os.path.realpath(path))
    root = os.path.normcase(package_root)
    if normalized == root or normalized.startswith(root + os.sep):
        raise CliError(f"{label} {path!r} 位于库包 dl_helper 目录内；训练内容必须属于库外训练项目")


def _check_library_boundaries(args: argparse.Namespace, config: Config, platform: Any) -> None:
    """库模块边界校验：Experiment 引用、配置与 output root 不得进入 dl_helper。

    在导入 Experiment 或创建任何 run 目录前执行（D-001）。
    """
    module_path = args.experiment.partition(":")[0]
    if module_path == "dl_helper" or module_path.startswith("dl_helper."):
        raise CliError(
            f"experiment 模块 {module_path!r} 位于库包 dl_helper 内；训练内容必须属于库外训练项目"
        )
    package_root = _DL_HELPER_PACKAGE_REALPATH
    _reject_inside_package(args.config, "配置文件", package_root)
    _reject_inside_package(platform.resolve_output_root(config), "output root", package_root)


def _compute_run_dir(config: Config, platform) -> tuple[str, str]:
    """计算 run 目录（不创建产物）：供预检前确定失败证据落盘位置。"""
    run_id = config.run.id
    if run_id is None:
        run_id = _generate_run_id(config)
    output_root = platform.resolve_output_root(config)
    run_dir = os.path.join(output_root, "runs", run_id)
    if platform.is_kaggle:
        if not run_dir.startswith("/kaggle/working"):
            raise CliError(f"Kaggle 输出必须位于 /kaggle/working: {run_dir!r}")
    return run_id, run_dir


def _resolve_run_layout(config: Config, platform) -> tuple[str, Any]:
    from .artifacts import RunLayout

    run_id, run_dir = _compute_run_dir(config, platform)
    layout = RunLayout(run_dir)
    layout.ensure()
    return run_id, layout


def _generate_run_id(config: Config) -> str:
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{config.run.name}-{stamp}"


def _cmd_train(args: argparse.Namespace) -> int:
    from .platform import Platform, execution_policy_for
    from .doctor import validate_training_start

    project_dir = _prepare_project_dir(args.project_dir)
    config = _load_config(args)
    if config.run.source_revision is None:
        from .platform import resolve_source_revision
        config = replace(
            config,
            run=replace(config.run, source_revision=resolve_source_revision(config, cwd=project_dir)),
        )
    if args.output_root is not None:
        config = replace(config, run=replace(config.run, output_root=os.path.abspath(args.output_root)))
    if getattr(args, "run_id", None):
        config = replace(config, run=replace(config.run, id=args.run_id))
    platform = Platform()
    # D-001：库模块边界校验必须在导入 Experiment 或创建任何产物前完成
    _check_library_boundaries(args, config, platform)
    execution_policy = execution_policy_for(platform)
    resume = args.resume if args.resume is not None else RESUME_AUTO
    # 预检前先确定 run 目录（不创建产物）：预检/导入失败也必须落 failure.json 失败证据
    run_id, run_dir = _compute_run_dir(config, platform)
    args._run_dir = run_dir  # OSR-003：受控 run_dir 在预检前即确定
    validate_training_start(config, platform, args.experiment, resume=resume,
                            execution_policy=execution_policy, emit_contract=args.preflight_only)
    if args.preflight_only:
        return EXIT_OK
    from .artifacts import RunLayout
    layout = RunLayout(run_dir)
    status = "succeeded"
    services = None
    try:
        layout.ensure()
        # OSR-005：任何写入前拒绝已完成 run（暂停 run 可 resume）
        from .artifacts import existing_terminal
        if existing_terminal(layout.run_dir) == "run-manifest.json":
            raise CliError(f"run {run_id} 已成功完成，禁止重跑改写")
        layout.write_text("config.resolved.yaml", _yaml_dump(config_to_dict(config)))
        # D-003：独立执行策略 Artifact 记录平台、resume policy、预算与收尾窗口
        layout.write_text("execution-policy.json", json.dumps({
            "schema_version": 1,
            "platform": platform.kind,
            "resume": resume,
            "max_minutes": execution_policy.max_minutes,
            "shutdown_grace_minutes": execution_policy.shutdown_grace_minutes,
        }, ensure_ascii=False, sort_keys=True))
        args._run_dir = layout.run_dir  # 布局后确认受控 run_dir（与预检前一致）

        services = _build_services(config, platform, layout)
        # OSR-003：记录同一 SecretResolver 与启用服务的 Secret key，供失败证据全链路脱敏
        args._secret_resolver = getattr(services, "_resolver", None) if services else None
        args._secret_keys = _configured_secret_keys(config)
        args._secondary_errors = []
        # sklearn batch 无受控恢复点：内部 auto 不查询本地或远程 checkpoint
        batch_no_resume = (
            config.backend.type == "sklearn"
            and config.backend.sklearn is not None
            and config.backend.sklearn.fit_mode == "batch"
        )
        if resume in (RESUME_AUTO, "required") and not batch_no_resume:
            from .checkpoint import read_latest

            if read_latest(layout.path("checkpoints")) is None and services is not None:
                services.restore_latest_checkpoint(run_id)
        if config.backend.type == "sklearn":
            from .backends.sklearn_backend import build_sklearn_experiment, run_sklearn_worker_experiment
            experiment = build_sklearn_experiment(args.experiment, config.experiment)
            result = run_sklearn_worker_experiment(experiment, config, platform, layout, resume=resume,
                                                   execution_policy=execution_policy, services=services)
            status = result.status
        else:
            from .backends.torch_backend import run_worker
            num_procs = platform.resolve_torch_resources(config, None).num_processes
            if num_procs == 1:
                result = run_worker(args.experiment, config, layout, 0, 1, resume,
                                    execution_policy=execution_policy, services=services)
                status = result.status
            else:
                # 多进程：CLI 父进程处理服务与唯一终态（OSR-002）
                from .launcher import launch_torch
                if services is not None:
                    services.start_run(run_id)
                code = launch_torch(args.experiment, config, layout.run_dir, num_procs, resume,
                                    execution_policy=execution_policy, publish_terminal=False)
                if code in (EXIT_OK, EXIT_PREEMPTED) and services is not None:
                    from .checkpoint import read_latest

                    latest = read_latest(layout.path("checkpoints"))
                    if latest is None and code == EXIT_PREEMPTED:
                        raise CliError("多进程暂停缺少本地 latest checkpoint，拒绝发布 PREEMPTED")
                    if latest is not None:
                        services.submit_checkpoint(run_id, latest["checkpoint_id"])
                if code == EXIT_PREEMPTED:
                    status = "preempted"
                elif code != EXIT_OK:
                    raise CliError(f"torch worker 退出码 {code}")
                if services is not None:
                    services.finalize_run(
                        run_id,
                        status,
                        prepare_terminal=lambda: _publish_cli_terminal(
                            layout, status, config, run_id, services=services
                        ),
                    )
                else:
                    _publish_cli_terminal(layout, status, config, run_id, services=services)
    except Exception as exc:
        # OSR-003：先原子写脱敏 failure.json，再执行 FAILED finalization
        # 服务终结可能已写入候选 success/pause，required 服务随后失败时撤销候选，
        # 让 failure evidence 能完成唯一 FAILED 过渡。
        from .artifacts import existing_terminal
        if services is not None and existing_terminal(layout.run_dir) == "run-manifest.json":
            os.remove(os.path.join(layout.run_dir, "run-manifest.json"))
        evidence_ok = False
        args._failure_evidence_attempted = True
        try:
            _write_failure_evidence(args, exc)
            evidence_ok = os.path.exists(os.path.join(layout.run_dir, "failure.json"))
        except Exception as evidence_exc:
            args._secondary_errors.append({
                "service": "evidence",
                "event": "WRITE_FAILURE",
                "error_type": type(evidence_exc).__name__,
                "message": str(evidence_exc),
            })
        # OSR-003：failure.json 未成功持久化时不得发布声称完整的 FAILED bundle
        if services is not None and evidence_ok:
            try:
                services.finalize_run(run_id, "failed")
            except Exception as sec:
                # OSR-003：服务终结失败作为可审计 secondary，不吞掉原训练异常
                resolver = getattr(args, "_secret_resolver", None)
                msg = str(sec)
                if resolver is not None:
                    msg = resolver.redact(msg)
                args._secondary_errors.append({
                    "service": "finalize",
                    "event": "RUN_FAILED",
                    "error_type": type(sec).__name__,
                    "message": msg,
                })
        raise

    if status == "preempted":
        return EXIT_PREEMPTED
    return EXIT_OK


def _configured_secret_keys(config: Config) -> list[str]:
    """启用服务声明的全部 Secret key（用于失败证据脱敏）。"""
    keys: list[str] = []
    if config.remote.type == "alist":
        keys.append(config.remote.user_secret_key)
        keys.append(config.remote.password_secret_key)
    if config.notifications.type == "wecom":
        n = config.notifications
        keys.extend([n.corp_id_secret_key, n.corp_secret_key, n.agent_id_secret_key])
    return keys


def _build_services(config: Config, platform: Platform, layout) -> Any:
    """构造启用服务的 LifecycleServices；全部禁用时返回 None。"""
    from .notifications import WecomClient
    from .platform import SecretResolver
    from .remote import build_artifact_stores
    from .services import LifecycleServices, ServiceAudit

    enabled = config.remote.type == "alist" or config.notifications.type == "wecom"
    if not enabled:
        return None
    resolver = SecretResolver(platform)
    stores, async_sync = build_artifact_stores(config, platform, resolver, layout)
    wecom = None
    store_policy = "record"
    wecom_policy = "record"
    if config.notifications.type == "wecom":
        n = config.notifications
        wecom = WecomClient(
            n.corp_id_secret_key, n.corp_secret_key, n.agent_id_secret_key, n.to_user,
            resolver, n.connect_timeout_seconds, n.read_timeout_seconds, n.max_attempts,
        )
        wecom_policy = n.failure_policy
    if config.remote.type == "alist":
        store_policy = config.remote.failure_policy
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=resolver.redact)
    return LifecycleServices(
        layout=layout, secret_resolver=resolver, stores=stores, async_sync=async_sync,
        wecom_client=wecom, audit=audit, failure_policy=store_policy,
        wecom_policy=wecom_policy,
    )


def _publish_cli_terminal(layout, status: str, config: Config, run_id: str, services=None) -> None:
    """多进程路径由 CLI 发布唯一终态（OSR-005：读取 worker summary 补全字段）。"""
    import json as _json

    from .artifacts import existing_terminal, publish_terminal, sha256_manifest
    from .config import config_fingerprint, tuning_fingerprint
    from .platform import resolve_source_revision

    existing = existing_terminal(layout.run_dir)
    desired = "pause-manifest.json" if status == "preempted" else "run-manifest.json"
    if existing == desired:
        return
    if existing == "run-manifest.json":
        return  # 已成功终态
    if existing == "pause-manifest.json" and status == "succeeded":
        # OSR-005：恢复成功原子替换旧 pause
        os.remove(os.path.join(layout.run_dir, "pause-manifest.json"))
    # 读取 rank0 worker 写入的 summary/environment 以补全 manifest；缺失或损坏必须失败（OSR-005）
    if not os.path.exists(layout.summary_json):
        raise CliError("worker 未生成 summary.json，无法发布多进程终态")
    with open(layout.summary_json, encoding="utf-8") as f:
        summary = _json.load(f)
    if not os.path.exists(layout.environment_json):
        raise CliError("worker 未生成 environment.json，无法发布多进程终态")
    with open(layout.environment_json, encoding="utf-8") as f:
        environment = _json.load(f)
    report_path = os.path.join(layout.run_dir, "report", "index.html")
    report_rel = "report/index.html" if os.path.exists(report_path) else None
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
        "run_id": run_id,
        "backend": "torch",
        "status": status,
        "created_utc": summary.get("created_utc"),
        "epoch": summary.get("epoch"),
        "global_step": summary.get("global_step"),
        "source_revision": source_revision,
        "config_fingerprint": config_fingerprint(config),
        "tuning_fingerprint": tuning_fingerprint(config),
        "data_fingerprint": summary.get("data_fingerprint"),
        "model_signature": summary.get("model_signature"),
        "selection": summary.get("selection"),
        "metric_definitions": summary.get("metric_definitions"),
        "model_artifact": summary.get("model_artifact"),
        "report": report_rel,
        "environment": environment,
        "services": services_result,
        "artifacts": sha256_manifest(layout.run_dir),
    }
    publish_terminal(layout.run_dir, "preempted" if status == "preempted" else "success", manifest)


def _yaml_dump(data: Any) -> str:
    import yaml
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def _cmd_report(args: argparse.Namespace) -> int:
    from .reporting import generate_run_report

    out = args.out
    index_path = generate_run_report(args.run, out)
    print(f"report written: {index_path}")
    return EXIT_OK


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import run_sweep
    # 省略 --project-dir 时按调用者当前目录解析（与 train 一致），
    # 并把绝对路径传给预检/子进程，避免以仓库根为 cwd 而丢失外部项目导入
    project_dir = _prepare_project_dir(args.project_dir)
    return run_sweep(args.sweep, resume=args.resume, project_dir=project_dir)


def _cmd_sweep_report(args: argparse.Namespace) -> int:
    from .sweep import generate_sweep_report

    index_path = generate_sweep_report(args.sweep_dir, args.out)
    print(f"sweep report written: {index_path}")
    return EXIT_OK


def _root_exception_type(exc: BaseException) -> str:
    """沿 cause/context 链取最深层根因异常的类型名（OSR-003：不被包装类掩盖）。"""
    root: BaseException = exc
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if cur.__cause__ is not None:
            root = cur.__cause__
            cur = cur.__cause__
        elif cur.__context__ is not None and cur.__context__ is not exc:
            root = cur.__context__
            cur = cur.__context__
        else:
            break
    return type(root).__name__


def _write_failure_evidence(args: argparse.Namespace, exc: Exception) -> None:
    """写脱敏 failure artifact（OSR-003）。

    使用与训练相同的 SecretResolver 全链路脱敏；不做静默回退到明文；
    包含 stage/训练位置与可审计 secondary 错误。写入异常由 main 捕获并记录，
    不替换原训练异常。
    """
    import json as _json

    from .artifacts import existing_terminal

    run_dir = getattr(args, "_run_dir", None)
    if run_dir is None:
        return
    if existing_terminal(run_dir) == "run-manifest.json":
        return  # 成功终态不可改写
    # OSR-003：publish_terminal 负责 pause → FAILED 的原子过渡；
    # 证据写失败时保留旧 pause，禁止留下声称 FAILED 的不完整 bundle。
    tb = traceback.format_exc()
    resolver = getattr(args, "_secret_resolver", None)
    if resolver is None:
        from .platform import Platform, SecretResolver
        resolver = SecretResolver(Platform())
    # 解析配置的 Secret key；解析失败显式进入 secondary（OSR-003：不静默 pass）
    secondary_errors = []
    for key in getattr(args, "_secret_keys", ()) or ():
        try:
            resolver.resolve(key)
        except Exception as sec:
            secondary_errors.append({"service": "secrets", "event": "RESOLVE_FAILURE",
                                     "error_type": type(sec).__name__, "message": str(sec)})
    # 脱敏不静默回退：redact 为纯字符串替换，失败会向上抛并由 main 记录
    redacted_message = resolver.redact(str(exc))
    redacted_tb = resolver.redact(tb)
    # secondary 错误消息同样脱敏（纵深防御）
    for s in (getattr(args, "_secondary_errors", []) or []):
        sec = dict(s)
        if sec.get("message"):
            sec["message"] = resolver.redact(str(sec["message"]))
        secondary_errors.append(sec)
    # OSR-003：位置写入失败形成的 secondary audit（若有）
    pos_err = os.path.join(run_dir, "failure-position-error.json")
    if os.path.exists(pos_err):
        try:
            with open(pos_err, encoding="utf-8") as _f:
                _pe = _json.load(_f)
            secondary_errors.append({"service": "evidence", "event": "POSITION_WRITE_FAILURE",
                                     "error_type": _pe.get("error_type", "unknown")})
        except Exception:
            secondary_errors.append({"service": "evidence", "event": "POSITION_WRITE_FAILURE",
                                     "error_type": "unknown"})
    # 训练位置（run_worker 在失败时写入 failure-position.json；损坏显式失败进 secondary）
    position: dict = {}
    pos_path = os.path.join(run_dir, "failure-position.json")
    if os.path.exists(pos_path):
        with open(pos_path, encoding="utf-8") as _f:
            position = _json.load(_f)
    failure = {
        "schema_version": 1,
        "exception_type": _root_exception_type(exc),
        "message": redacted_message,
        "traceback": redacted_tb,
        "primary_exception": type(exc).__name__,
        "secondary_errors": secondary_errors,
        "stage": position.get("stage", "cli"),
        "epoch": position.get("epoch"),
        "batch_in_epoch": position.get("batch_in_epoch"),
        "global_step": position.get("global_step"),
    }
    # OSR-003：经 publish_terminal 原子发布唯一 FAILED 终态（禁止多终态）
    # 预检失败可能尚未创建 run 目录：失败证据仍须落盘，先确保目录存在
    os.makedirs(run_dir, exist_ok=True)
    from .artifacts import publish_terminal
    publish_terminal(run_dir, "failed", failure)


def entry() -> None:
    """console 脚本入口；以异常非零退出。"""
    try:
        code = main(sys.argv[1:])
    except SystemExit:
        raise
    except BaseException:
        raise
    else:
        sys.exit(code)


if __name__ == "__main__":
    entry()
