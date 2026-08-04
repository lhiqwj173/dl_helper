"""CLI：doctor/train/report/sweep/sweep-report 命令与退出码。

退出码：0 成功、75 PREEMPTED、其他非零为失败。顶层只写脱敏失败证据后原样 raise。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Sequence

from .config import Config, ConfigError, config_to_dict, load_config_file, resolve_variant_files

EXIT_OK = 0
EXIT_PREEMPTED = 75


class CliError(Exception):
    """CLI 参数或执行错误。"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dl_helper.training", description="Kaggle 通用训练平台")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="运行后端感知预检")
    p_doctor.add_argument("--profile", choices=["local", "kaggle"], default=None)
    p_doctor.add_argument("--config", required=True)
    p_doctor.add_argument("--variant")
    p_doctor.add_argument("--experiment", required=True)
    p_doctor.add_argument("--emit-evaluation-contract", action="store_true")

    p_train = sub.add_parser("train", help="运行一次训练")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--variant")
    p_train.add_argument("--experiment", required=True)
    p_train.add_argument("--resume", choices=["none", "auto", "required"], default=None)
    p_train.add_argument("--run-id")

    p_report = sub.add_parser("report", help="从 Artifact 生成离线报告")
    p_report.add_argument("--run", required=True)
    p_report.add_argument("--out")

    p_sweep = sub.add_parser("sweep", help="顺序运行多 variant sweep")
    p_sweep.add_argument("--sweep", required=True)
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
    if cmd == "doctor":
        return _cmd_doctor(args)
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


def _resolve_run_layout(config: Config, platform) -> tuple[str, Any]:
    from .artifacts import RunLayout
    from .platform import resolve_source_revision

    run_id = config.run.id
    if run_id is None:
        run_id = _generate_run_id(config)
    output_root = platform.resolve_output_root(config)
    run_dir = os.path.join(output_root, "runs", run_id)
    if platform.is_kaggle:
        if not run_dir.startswith("/kaggle/working"):
            raise CliError(f"Kaggle 输出必须位于 /kaggle/working: {run_dir!r}")
    layout = RunLayout(run_dir)
    layout.ensure()
    return run_id, layout


def _generate_run_id(config: Config) -> str:
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{config.run.name}-{stamp}"


def _cmd_train(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from .platform import Platform

    config = _load_config(args)
    if getattr(args, "run_id", None):
        config = replace(config, run=replace(config.run, id=args.run_id))
    platform = Platform()
    run_id, layout = _resolve_run_layout(config, platform)
    # OSR-005：任何写入前拒绝已完成 run（暂停 run 可 resume）
    from .artifacts import existing_terminal
    if existing_terminal(layout.run_dir) == "run-manifest.json":
        raise CliError(f"run {run_id} 已成功完成，禁止重跑改写")
    layout.write_text("config.resolved.yaml", _yaml_dump(config_to_dict(config)))
    args._run_dir = layout.run_dir  # OSR-003：布局后立即保存受控 run_dir

    services = _build_services(config, platform, layout)
    # OSR-003：记录同一 SecretResolver 与启用服务的 Secret key，供失败证据全链路脱敏
    args._secret_resolver = getattr(services, "_resolver", None) if services else None
    args._secret_keys = _configured_secret_keys(config)
    args._secondary_errors = []
    resume = args.resume or config.checkpoint.resume
    status = "succeeded"
    try:
        if config.backend.type == "sklearn":
            from .backends.sklearn_backend import build_sklearn_experiment, run_sklearn_worker_experiment
            experiment = build_sklearn_experiment(args.experiment, config.experiment)
            result = run_sklearn_worker_experiment(experiment, config, platform, layout, services=services)
            status = result.status
        else:
            from .backends.torch_backend import run_worker
            num_procs = platform.resolve_torch_resources(config, None).num_processes
            if num_procs == 1:
                result = run_worker(args.experiment, config, layout, 0, 1, resume, services=services)
                status = result.status
            else:
                # 多进程：CLI 父进程处理服务与唯一终态（OSR-002）
                from .launcher import launch_torch
                if services is not None:
                    services.start_run(run_id)
                code = launch_torch(args.experiment, config, layout.run_dir, num_procs, resume,
                                    publish_terminal=False)
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


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .platform import Platform
    from .doctor import run_doctor

    config = _load_config(args)
    platform = Platform(args.profile) if args.profile else Platform()
    errors = run_doctor(config, platform, args.experiment, emit_contract=args.emit_evaluation_contract)
    if errors:
        for err in errors:
            print(f"doctor error: {err}", file=sys.stderr)
        return 1
    return EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    from .reporting import generate_run_report

    out = args.out
    index_path = generate_run_report(args.run, out)
    print(f"report written: {index_path}")
    return EXIT_OK


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import run_sweep

    return run_sweep(args.sweep, resume=args.resume)


def _cmd_sweep_report(args: argparse.Namespace) -> int:
    from .sweep import generate_sweep_report

    index_path = generate_sweep_report(args.sweep_dir, args.out)
    print(f"sweep report written: {index_path}")
    return EXIT_OK


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
        "exception_type": type(exc).__name__,
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
