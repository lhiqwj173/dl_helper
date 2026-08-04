"""服务基础设施：Secret resolver 集成、service audit、required/record policy 与错误模型。"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Mapping

from .artifacts import append_jsonl, write_json
from .notifications import render_event_template
from .platform import SecretResolver


class ServiceDeliveryError(Exception):
    """服务投递失败（重试耗尽或不可重试错误）。"""


class SecondaryServiceError(Exception):
    """secondary 服务错误；不覆盖 primary 训练异常。"""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


def stable_event_id(scope: str, event: str, attempt: str = "1") -> str:
    text = f"{scope}:{event}:{attempt}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ServiceAudit:
    """结构化 UTF-8 JSONL 服务审计；不含 token URL/认证参数/Secret/response body。"""

    def __init__(self, path: str, redactor=None) -> None:
        self._path = path
        self._redactor = redactor or (lambda t: t)

    def record(
        self,
        scope: str,
        service: str,
        event: str,
        attempt: int,
        outcome: str,
        *,
        started_utc: str,
        finished_utc: str,
        duration_ms: int,
        http_status: int | None = None,
        error_type: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": 1,
            "event_id": stable_event_id(scope, event, str(attempt)),
            "scope": scope,
            "service": service,
            "event": event,
            "attempt": attempt,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "duration_ms": duration_ms,
            "outcome": outcome,
            "http_status": http_status,
            "error_type": error_type,
        }
        if extra:
            for k, v in extra.items():
                if not k.startswith("_"):
                    record[k] = v
        # 脱敏全部字符串字段
        redacted = {k: (self._redactor(v) if isinstance(v, str) else v) for k, v in record.items()}
        append_jsonl(self._path, redacted)
        return redacted


class ServiceErrorPolicy:
    """required/record 失败策略。required 阻止成功/暂停终态；record 标 degraded。"""

    def __init__(self, policy: str) -> None:
        if policy not in ("required", "record"):
            raise ValueError(f"failure_policy 非法: {policy!r}")
        self.policy = policy

    @property
    def is_required(self) -> bool:
        return self.policy == "required"

    def handle_failure(self, service: str, exc: Exception) -> None:
        """按策略处理；required 立即抛出，record 记录 degraded 后继续。"""
        if self.policy == "required":
            raise ServiceDeliveryError(f"{service} required 失败") from exc
        # record：不抛，由调用方写 degraded 状态


class ServiceResult:
    """服务投递结果（用于终态 manifest 的 degraded 汇总）。"""

    def __init__(self) -> None:
        self.degraded: list[dict[str, str]] = []

    def mark_degraded(self, service: str, error_type: str) -> None:
        self.degraded.append({"service": service, "error_type": error_type})

    @property
    def has_degraded(self) -> bool:
        return bool(self.degraded)

    def snapshot(self) -> dict[str, Any]:
        return {"degraded": list(self.degraded)}


# --------------------------------------------------------------------------
# 可重入服务终结（任务 5.5）
# --------------------------------------------------------------------------

class LifecycleServices:
    """run/sweep 生命周期服务编排：审计、通知、AList 与可重入 finalization。

    以 stable event_id / checksum 幂等复核已成功动作；required 失败阻止成功/暂停终态。
    """

    def __init__(
        self,
        layout: Any,
        secret_resolver: Any,
        stores: list[Any],
        async_sync: Any,
        wecom_client: Any,
        audit: ServiceAudit,
        failure_policy: str,
        wecom_policy: str | None = None,
    ) -> None:
        self._layout = layout
        self._resolver = secret_resolver
        self._stores = stores
        self._async_sync = async_sync
        self._wecom = wecom_client
        self._audit = audit
        # OSR-002：分服务策略 —— store（AList）与 wecom 各自独立，不折叠为全局最严
        self._policy = ServiceErrorPolicy(failure_policy)
        self._wecom_policy = ServiceErrorPolicy(wecom_policy if wecom_policy is not None else failure_policy)
        self.result = ServiceResult()
        self._started_events: set[str] = set()
        # OSR-002：从持久 service audit 恢复已成功 event_id，避免重启后重复通知
        self._restore_seen_events()
        self._restore_degraded()

    def _restore_seen_events(self) -> None:
        """从持久 service audit JSONL 恢复已成功投递的 event_id（outcome=success）。"""
        import json as _json
        audit_path = getattr(self._audit, "_path", None)
        if not audit_path or not os.path.exists(audit_path):
            return
        with open(audit_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                scope = rec.get("scope")
                event = rec.get("event")
                attempt = rec.get("attempt")
                if not scope or not event or not isinstance(attempt, int):
                    raise ServiceDeliveryError("service audit 记录损坏")
                if rec.get("event_id") != stable_event_id(scope, event, str(attempt)):
                    raise ServiceDeliveryError("service audit event_id checksum 漂移")
                if rec.get("scope") and rec.get("event") and rec.get("outcome") == "success":
                    self._started_events.add(f"{rec['scope']}:{rec['event']}")

    def _restore_degraded(self) -> None:
        import json as _json
        path = os.path.join(self._layout.run_dir, "services", "service-manifest.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        degraded = data.get("degraded", [])
        if not isinstance(degraded, list):
            raise ServiceDeliveryError("service-manifest degraded 字段损坏")
        for item in degraded:
            if not isinstance(item, dict) or not item.get("service") or not item.get("error_type"):
                raise ServiceDeliveryError("service-manifest degraded 记录损坏")
            self.result.mark_degraded(str(item["service"]), str(item["error_type"]))

    # ---- 内部 ----

    def _audit_and_notify(self, scope: str, event: str, policy: str,
                          fields: Mapping[str, Any], attempt: int = 1) -> bool:
        """投递 + 审计；返回投递是否成功（OSR-002：仅在成功后由调用方标记 event_id）。"""
        started = utc_now()
        last_exc: Exception | None = None
        try:
            self._wecom.send_text(render_event_template(event, **fields), redactor=self._resolver.redact)
            outcome = "success"
            error_type = None
        except Exception as exc:
            # OSR-002：先审计再抛出；record 标记 degraded（失败事件可重试）
            outcome = "failed"
            error_type = type(exc).__name__
            last_exc = exc
            if policy != "required":
                self.result.mark_degraded("wecom", error_type)
        self._audit.record(
            scope, "wecom", event, attempt, outcome,
            started_utc=started, finished_utc=utc_now(),
            duration_ms=0, error_type=error_type,
        )
        if outcome == "failed" and policy == "required":
            raise ServiceDeliveryError(f"企业微信 {event} required 失败") from last_exc
        return outcome == "success"

    def _event_seen(self, scope: str, event: str) -> bool:
        return f"{scope}:{event}" in self._started_events

    def _mark_event_seen(self, scope: str, event: str) -> None:
        self._started_events.add(f"{scope}:{event}")

    def _ensure_finalize_audit(self, scope: str, kind: str, scope_id: str, status: str) -> None:
        """确保终态 service manifest 总有一个稳定、可复核的最终 audit 记录。"""
        event = f"{kind.upper()}_FINALIZE_{status.upper()}"
        if self._event_seen(scope, event):
            return
        now = utc_now()
        self._audit.record(
            scope, "lifecycle", event, 1, "success",
            started_utc=now, finished_utc=now, duration_ms=0,
        )
        self._mark_event_seen(scope, event)

    # ---- run ----

    def start_run(self, run_id: str, platform: str = "local") -> None:
        scope = f"run/{run_id}"
        if self._wecom is not None and not self._event_seen(scope, "RUN_STARTED"):
            if self._audit_and_notify(scope, "RUN_STARTED", self._wecom_policy.policy,
                                      {"run_id": run_id, "platform": platform, "utc": utc_now()}):
                self._mark_event_seen(scope, "RUN_STARTED")
        # OSR-002：启用服务的全部 Secret 在首个拟合 step 前解析（含 WeCom，非空操作）
        self._preflight_secrets()

    def _preflight_secrets(self) -> None:
        """解析启用 store 与 WeCom 声明的全部 Secret；缺失立即失败。"""
        for store in self._stores:
            if hasattr(store, "secret_keys"):
                for key in store.secret_keys():
                    self._resolver.resolve(key)
        if self._wecom is not None and hasattr(self._wecom, "secret_keys"):
            for key in self._wecom.secret_keys():
                self._resolver.resolve(key)

    def finalize_run(self, run_id: str, status: str, *, prepare_terminal=None, **fields: Any) -> None:
        """终态 finalization：通知+audit → service manifest → bundle。可重入（bundle 幂等）。"""
        scope = f"run/{run_id}"
        if self._async_sync is not None:
            self._async_sync.flush(terminal=True)
        event = f"RUN_{status.upper()}"
        if self._wecom is not None and not self._event_seen(scope, event):
            if self._audit_and_notify(scope, event, self._wecom_policy.policy,
                                      {"run_id": run_id, "utc": utc_now(), **fields}):
                self._mark_event_seen(scope, event)
        # OSR-002：终态通知后写最终 service manifest（含最终 audit 引用），再发布 bundle（远端含完整终态）
        self._ensure_finalize_audit(scope, "run", run_id, status)
        self._write_service_manifest("run", run_id, status)
        if prepare_terminal is not None:
            prepare_terminal()
        for store in self._stores:
            if type(store).__name__ == "LocalArtifactStore":
                continue
            try:
                self._final_service_manifest("run", run_id, status)
                # OSR-002：按状态跳过已成功发布的 bundle（幂等重入）
                if not self._bundle_published("run", run_id, status):
                    store.publish_run_bundle(self._layout.run_dir, run_id)
                    self._mark_bundle_published("run", run_id, status)
            except Exception as exc:
                if isinstance(exc, ServiceDeliveryError):
                    raise
                self._handle_store_failure(f"run/{run_id}", "alist", exc)
                self._write_service_manifest("run", run_id, status)

    def _write_service_manifest(self, kind: str, scope_id: str, status: str) -> None:
        """把最终服务状态与 audit checksum 原子落盘。"""
        import json as _json
        from .artifacts import sha256_file

        audit_path = os.path.join(self._layout.run_dir, "services", "service-audit.jsonl")
        audit_checksum = sha256_file(audit_path) if os.path.exists(audit_path) else None
        existing_path = os.path.join(self._layout.run_dir, "services", "service-manifest.json")
        created_utc = utc_now()
        if os.path.exists(existing_path):
            with open(existing_path, "r", encoding="utf-8") as f:
                existing = _json.load(f)
            if (existing.get("kind"), existing.get("id"), existing.get("status")) != (kind, scope_id, status):
                raise ServiceDeliveryError("service-manifest 状态漂移")
            created_utc = existing.get("created_utc") or created_utc
        payload = {
            "schema_version": 1,
            "kind": kind,
            "id": scope_id,
            "status": status,
            "audit": "services/service-audit.jsonl",
            "audit_checksum": audit_checksum,
            "degraded": [dict(d) for d in self.result.degraded],
            "created_utc": created_utc,
        }
        canonical = _json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["manifest_checksum"] = hashlib.sha256(canonical).hexdigest()
        write_json(os.path.join(self._layout.run_dir, "services", "service-manifest.json"), payload)

    def _final_service_manifest(self, kind: str, scope_id: str, status: str) -> dict[str, Any]:
        """发布前严格复核最终 service manifest，禁止远端收到半成品。"""
        import json as _json
        from .artifacts import sha256_file

        path = os.path.join(self._layout.run_dir, "services", "service-manifest.json")
        if not os.path.isfile(path):
            raise ServiceDeliveryError("最终 service-manifest.json 缺失")
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        if data.get("kind") != kind or data.get("id") != scope_id or data.get("status") != status:
            raise ServiceDeliveryError("service-manifest 状态漂移")
        recorded_manifest_checksum = data.get("manifest_checksum")
        if not recorded_manifest_checksum:
            raise ServiceDeliveryError("service-manifest 缺少 manifest_checksum")
        unsigned = dict(data)
        unsigned.pop("manifest_checksum", None)
        canonical = _json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if recorded_manifest_checksum != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
            raise ServiceDeliveryError("service-manifest manifest_checksum 不匹配")
        audit_path = os.path.join(self._layout.run_dir, "services", "service-audit.jsonl")
        actual_audit = sha256_file(audit_path) if os.path.exists(audit_path) else None
        if data.get("audit_checksum") != actual_audit:
            raise ServiceDeliveryError("service-manifest audit checksum 漂移")
        return {
            "service_manifest_sha256": sha256_file(path),
            "audit_sha256": actual_audit,
        }

    def _bundle_published(self, kind: str, scope_id: str, status: str) -> bool:
        """严格复核 immutable bundle marker；损坏或漂移不得静默当作未发布。"""
        import json as _json
        from .artifacts import sha256_file
        from .remote import bundle_checksum_for_directory

        marker = os.path.join(self._layout.run_dir, "services", "bundle-published.json")
        if not os.path.exists(marker):
            return False
        with open(marker, "r", encoding="utf-8") as f:
            data = _json.load(f)
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ServiceDeliveryError("bundle-published.json 损坏")
        entry = data.get("entries", {}).get(f"{kind}:{scope_id}")
        if not isinstance(entry, dict):
            raise ServiceDeliveryError("bundle-published.json 缺少 immutable entry")
        expected_manifest = self._final_service_manifest(kind, scope_id, status)
        expected_bundle = bundle_checksum_for_directory(
            self._layout.run_dir, ("checkpoints",)
        )
        if entry.get("status") != status:
            raise ServiceDeliveryError("bundle marker status 漂移")
        if entry.get("bundle_checksum") != expected_bundle:
            raise ServiceDeliveryError("bundle checksum 漂移")
        if entry.get("service_manifest_sha256") != expected_manifest["service_manifest_sha256"]:
            raise ServiceDeliveryError("service-manifest checksum 漂移")
        if entry.get("audit_sha256") != expected_manifest["audit_sha256"]:
            raise ServiceDeliveryError("audit checksum 漂移")
        return True

    def _mark_bundle_published(self, kind: str, scope_id: str, status: str) -> None:
        """持久记录该 run/sweep 已按指定状态发布 bundle。"""
        import json as _json
        from .artifacts import sha256_file
        from .remote import bundle_checksum_for_directory

        marker_dir = os.path.join(self._layout.run_dir, "services")
        os.makedirs(marker_dir, exist_ok=True)
        marker = os.path.join(marker_dir, "bundle-published.json")
        data: dict[str, Any] = {"schema_version": 1, "entries": {}}
        if os.path.exists(marker):
            with open(marker, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise ServiceDeliveryError("bundle-published.json 损坏")
            if not isinstance(data.get("entries"), dict):
                raise ServiceDeliveryError("bundle-published.json entries 损坏")
        service_manifest = os.path.join(marker_dir, "service-manifest.json")
        audit_path = os.path.join(marker_dir, "service-audit.jsonl")
        if not os.path.isfile(service_manifest):
            raise ServiceDeliveryError("发布成功后缺少 service-manifest.json")
        data["entries"][f"{kind}:{scope_id}"] = {
            "status": status,
            "bundle_checksum": bundle_checksum_for_directory(self._layout.run_dir, ("checkpoints",)),
            "service_manifest_sha256": sha256_file(service_manifest),
            "audit_sha256": sha256_file(audit_path) if os.path.exists(audit_path) else None,
        }
        write_json(marker, data)

    # ---- sweep/trial ----

    def start_sweep(self, sweep_id: str, platform: str = "local") -> None:
        scope = f"sweep/{sweep_id}"
        if self._wecom is not None and not self._event_seen(scope, "SWEEP_STARTED"):
            if self._audit_and_notify(scope, "SWEEP_STARTED", self._wecom_policy.policy,
                                      {"sweep_id": sweep_id, "platform": platform, "utc": utc_now()}):
                self._mark_event_seen(scope, "SWEEP_STARTED")
        self._preflight_secrets()  # OSR-002：sweep 级 Secret 预检

    def trial_event(self, sweep_id: str, trial: str, status: str, **fields: Any) -> None:
        event = f"TRIAL_{status.upper()}"
        # OSR-002：trial event scope 含 trial，避免跨 trial 去重导致事件被跳过
        scope = f"sweep/{sweep_id}/trial/{trial}"
        if self._wecom is not None and not self._event_seen(scope, event):
            if self._audit_and_notify(scope, event, self._wecom_policy.policy,
                                      {"sweep_id": sweep_id, "trial": trial, "utc": utc_now(), **fields}):
                self._mark_event_seen(scope, event)

    def finalize_sweep(self, sweep_id: str, status: str, **fields: Any) -> None:
        scope = f"sweep/{sweep_id}"
        if self._async_sync is not None:
            self._async_sync.flush(terminal=True)
        event = f"SWEEP_{status.upper()}"
        if self._wecom is not None and not self._event_seen(scope, event):
            if self._audit_and_notify(scope, event, self._wecom_policy.policy,
                                      {"sweep_id": sweep_id, "utc": utc_now(), **fields}):
                self._mark_event_seen(scope, event)
        # OSR-002：终态通知后写最终 service manifest，再发布 sweep bundle（远端含完整终态）
        self._ensure_finalize_audit(scope, "sweep", sweep_id, status)
        self._write_service_manifest("sweep", sweep_id, status)
        for store in self._stores:
            if type(store).__name__ == "LocalArtifactStore":
                continue
            try:
                self._final_service_manifest("sweep", sweep_id, status)
                # OSR-002：按状态跳过已成功发布的 sweep bundle（幂等重入）
                if not self._bundle_published("sweep", sweep_id, status):
                    store.publish_sweep_bundle(self._layout.run_dir, sweep_id)
                    self._mark_bundle_published("sweep", sweep_id, status)
            except Exception as exc:
                if isinstance(exc, ServiceDeliveryError):
                    raise
                self._handle_store_failure(f"sweep/{sweep_id}", "alist", exc)
                self._write_service_manifest("sweep", sweep_id, status)

    def _handle_store_failure(self, scope: str, service: str, exc: Exception) -> None:
        """按策略处理 store 发布失败；required 阻止终态，record 审计并标记 degraded。"""
        started = utc_now()
        if self._policy.is_required:
            self._audit.record(scope, service, "PUBLISH", 1, "failed",
                               started_utc=started, finished_utc=utc_now(), duration_ms=0,
                               error_type=type(exc).__name__)
            raise ServiceDeliveryError(f"{service} 终态发布 required 失败") from exc
        self._audit.record(scope, service, "PUBLISH", 1, "failed",
                           started_utc=started, finished_utc=utc_now(), duration_ms=0,
                           error_type=type(exc).__name__)
        self.result.mark_degraded(service, type(exc).__name__)

    def submit_checkpoint(self, run_id: str, checkpoint_id: str) -> None:
        """OSR-002：把新 checkpoint 提交到有界异步同步器（容量 1，合并 pending）。"""
        if self._async_sync is not None:
            self._async_sync.submit_checkpoint(self._layout.run_dir, run_id, checkpoint_id)
