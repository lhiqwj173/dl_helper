"""任务 5.5：run/sweep 服务顺序与可重入 finalization。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.services import (
    LifecycleServices,
    ServiceAudit,
    ServiceDeliveryError,
    utc_now,
)


class _FakeWecom:
    def __init__(self, fail_content=()):
        self.sent: list[str] = []
        self._fail = tuple(fail_content)

    def send_text(self, content, redactor=None):
        if any(m in content for m in self._fail):
            raise ConnectionError("net down")
        self.sent.append(content)


class _FakeStore:
    def __init__(self):
        self.bundles: list[str] = []
        self.checkpoints: list[tuple[str, str, str]] = []

    def publish_run_bundle(self, local_dir, run_id):
        self.bundles.append(("run", run_id))

    def publish_sweep_bundle(self, local_dir, sweep_id):
        self.bundles.append(("sweep", sweep_id))

    def publish_checkpoint(self, *a):
        self.checkpoints.append(a)


class _NoAsync:
    def flush(self, terminal=False):
        return None


def _resolver():
    class R:
        def redact(self, t):
            return t
    return R()


def _services(tmp_path, wecom, policy="record"):
    layout = RunLayout(str(tmp_path / "runs" / "sr"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    return LifecycleServices(
        layout=layout, secret_resolver=_resolver(),
        stores=[_FakeStore()], async_sync=_NoAsync(),
        wecom_client=wecom, audit=audit, failure_policy=policy,
    )


def test_run_lifecycle_events(tmp_path):
    wecom = _FakeWecom()
    svc = _services(tmp_path, wecom)
    svc.start_run("run-1")
    svc.finalize_run("run-1", "succeeded", elapsed="5m", summary="acc=0.9")
    assert any("训练开始" in s for s in wecom.sent)
    assert any("训练成功" in s for s in wecom.sent)
    # 审计记录
    lines = [json.loads(l) for l in open(os.path.join(tmp_path, "runs", "sr", "services", "service-audit.jsonl"), encoding="utf-8")]
    assert len(lines) >= 2


def test_sync_checkpoint_upload_when_async_disabled(tmp_path):
    layout = RunLayout(str(tmp_path / "runs" / "sync-checkpoint"))
    layout.ensure()
    store = _FakeStore()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    svc = LifecycleServices(
        layout=layout, secret_resolver=_resolver(), stores=[store], async_sync=None,
        wecom_client=None, audit=audit, failure_policy="required",
    )

    svc.submit_checkpoint("run-1", "ck-1")

    assert store.checkpoints == [(layout.run_dir, "run-1", "ck-1")]


def test_reentrant_finalize_does_not_duplicate(tmp_path):
    wecom = _FakeWecom()
    svc = _services(tmp_path, wecom)
    svc.start_run("run-1")
    svc.finalize_run("run-1", "succeeded")
    svc.finalize_run("run-1", "succeeded")  # 重入
    sent_success = [s for s in wecom.sent if "训练成功" in s]
    assert len(sent_success) == 1


def test_trial_and_sweep_events(tmp_path):
    wecom = _FakeWecom()
    svc = _services(tmp_path, wecom)
    svc.start_sweep("sweep-1")
    svc.trial_event("sweep-1", "lr-1e-3", "started")
    svc.trial_event("sweep-1", "lr-1e-3", "succeeded")
    svc.finalize_sweep("sweep-1", "succeeded", best="lr-1e-3")
    events = set()
    for s in wecom.sent:
        for ev in ("Sweep开始", "Trial开始", "Trial成功", "Sweep成功"):
            if ev in s:
                events.add(ev)
    assert events == {"Sweep开始", "Trial开始", "Trial成功", "Sweep成功"}


def test_required_failure_blocks_terminal(tmp_path):
    wecom = _FakeWecom(fail_content=("[训练成功]",))
    svc = _services(tmp_path, wecom, policy="required")
    svc.start_run("run-1")
    with pytest.raises(ServiceDeliveryError):
        svc.finalize_run("run-1", "succeeded")


def test_record_failure_marks_degraded(tmp_path):
    wecom = _FakeWecom(fail_content=("[训练成功]",))
    svc = _services(tmp_path, wecom, policy="record")
    svc.start_run("run-1")
    svc.finalize_run("run-1", "succeeded")  # 不抛
    assert svc.result.has_degraded
    degraded = svc.result.snapshot()["degraded"]
    assert any(d["service"] == "wecom" for d in degraded)


def test_audit_does_not_contain_secret(tmp_path):
    class SecretResolver:
        def redact(self, t):
            return t.replace("secret-token-xyz", "[REDACTED]")

    wecom = _FakeWecom()
    layout = RunLayout(str(tmp_path / "runs" / "sec"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=SecretResolver().redact)
    svc = LifecycleServices(layout=layout, secret_resolver=SecretResolver(),
                            stores=[_FakeStore()], async_sync=_NoAsync(),
                            wecom_client=wecom, audit=audit, failure_policy="record")
    svc._audit.record("run/1", "alist", "X", 1, "failed",
                      started_utc=utc_now(), finished_utc=utc_now(), duration_ms=1,
                      error_type="secret-token-xyz")
    content = open(layout.service_audit_jsonl, encoding="utf-8").read()
    assert "secret-token-xyz" not in content


def test_start_run_preflights_store_secrets(tmp_path):
    """OSR-002：start_run 真实解析启用 store 的 Secret（非空操作）。"""
    class _StoreWithSecrets:
        def secret_keys(self):
            return ["ALIST_USER", "ALIST_PWD"]

        def publish_run_bundle(self, *a):
            pass

        def publish_sweep_bundle(self, *a):
            pass

        def publish_checkpoint(self, *a):
            pass

    resolved: list[str] = []

    class _SpyResolver:
        def resolve(self, key):
            resolved.append(key)
            return "secret-value"

        def redact(self, t):
            return t

    layout = RunLayout(str(tmp_path / "runs" / "sr-sec"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    svc = LifecycleServices(layout=layout, secret_resolver=_SpyResolver(),
                            stores=[_StoreWithSecrets()], async_sync=_NoAsync(),
                            wecom_client=None, audit=audit, failure_policy="record")
    svc.start_run("run-sec")
    assert "ALIST_USER" in resolved and "ALIST_PWD" in resolved


def test_submit_checkpoint_forwards_to_async(tmp_path):
    """OSR-002：checkpoint 提交到有界异步同步器。"""
    submitted: list[tuple] = []

    class _Async:
        def submit_checkpoint(self, local_dir, run_id, ckpt_id):
            submitted.append((run_id, ckpt_id))

        def flush(self, terminal=False):
            pass

    layout = RunLayout(str(tmp_path / "runs" / "sr-ck"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    svc = LifecycleServices(layout=layout, secret_resolver=_resolver(),
                            stores=[_FakeStore()], async_sync=_Async(),
                            wecom_client=None, audit=audit, failure_policy="record")
    svc.submit_checkpoint("run-1", "epoch-000000-step-00000008")
    assert submitted == [("run-1", "epoch-000000-step-00000008")]


def test_sweep_services_built_when_enabled(tmp_path):
    """OSR-002：启用服务时 sweep 构造 LifecycleServices 并调用生命周期事件。"""
    from dl_helper.training.config import default_schema, parse_config
    from dl_helper.training.sweep import _build_sweep_services

    schema = default_schema()
    schema["notifications"] = {"type": "wecom", "corp_id_secret_key": "A",
                               "corp_secret_key": "B", "agent_id_secret_key": "C",
                               "to_user": "u", "connect_timeout_seconds": 1,
                               "read_timeout_seconds": 1, "max_attempts": 2,
                               "failure_policy": "record"}
    cfg = parse_config(schema)
    sweep_dir = str(tmp_path / "sweeps" / "s")
    os.makedirs(sweep_dir, exist_ok=True)
    svc = _build_sweep_services(cfg, sweep_dir)
    assert svc is not None
    assert svc._wecom is not None
    # 未启用服务 → None
    assert _build_sweep_services(parse_config(default_schema()), sweep_dir) is None


def test_trial_events_per_trial_not_deduped(tmp_path):
    """OSR-002：trial event scope 含 trial，多 trial 事件不被去重跳过。"""
    wecom = _FakeWecom()
    svc = _services(tmp_path, wecom)
    svc.start_sweep("s1")
    svc.trial_event("s1", "trial-a", "started")
    svc.trial_event("s1", "trial-b", "started")  # 旧实现会被去重跳过
    svc.trial_event("s1", "trial-a", "succeeded")
    svc.trial_event("s1", "trial-b", "succeeded")
    trial_starts = [s for s in wecom.sent if "Trial开始" in s]
    assert len(trial_starts) == 2


def test_required_notify_audits_before_raise(tmp_path):
    """OSR-002：required 通知失败先审计再抛出。"""
    from dl_helper.training.services import ServiceDeliveryError

    wecom = _FakeWecom(fail_content=("[训练成功]",))
    layout = RunLayout(str(tmp_path / "runs" / "audit-first"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    svc = LifecycleServices(layout=layout, secret_resolver=_resolver(),
                            stores=[_FakeStore()], async_sync=_NoAsync(),
                            wecom_client=wecom, audit=audit, failure_policy="record",
                            wecom_policy="required")
    svc.start_run("r1")
    with pytest.raises(ServiceDeliveryError):
        svc.finalize_run("r1", "succeeded")
    # 审计已写入（失败也被审计）
    content = open(layout.service_audit_jsonl, encoding="utf-8").read()
    assert "RUN_SUCCEEDED" in content


def test_per_service_policies_independent(tmp_path):
    """OSR-002：AList/WeCom 策略独立，required 不折叠为全局。"""
    from dl_helper.training.services import ServiceDeliveryError

    # wecom required + store record：wecom 失败阻止，store 失败仅 degraded
    wecom = _FakeWecom(fail_content=("[训练成功]",))
    layout = RunLayout(str(tmp_path / "runs" / "policies"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    svc = LifecycleServices(layout=layout, secret_resolver=_resolver(),
                            stores=[_FakeStore()], async_sync=_NoAsync(),
                            wecom_client=wecom, audit=audit, failure_policy="record",
                            wecom_policy="required")
    svc.start_run("r1")
    with pytest.raises(ServiceDeliveryError):
        svc.finalize_run("r1", "succeeded")
    # store 为 record → bundle 发布失败不阻止；此处 _FakeStore 不抛 → 无 degraded
    assert not any(d["service"] == "alist" for d in svc.result.degraded)


def test_event_dedup_restored_from_audit(tmp_path):
    """OSR-002：重启后从持久 audit 恢复已成功 event_id，终态通知不重复。"""
    layout = RunLayout(str(tmp_path / "runs" / "dedup"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    wecom1 = _FakeWecom()
    svc1 = LifecycleServices(layout=layout, secret_resolver=_resolver(),
                             stores=[_FakeStore()], async_sync=_NoAsync(),
                             wecom_client=wecom1, audit=audit, failure_policy="record")
    svc1.start_run("r1")
    svc1.finalize_run("r1", "succeeded")
    assert len([s for s in wecom1.sent if "训练成功" in s]) == 1

    # 重启：新 LifecycleServices 从 audit 恢复 seen event
    wecom2 = _FakeWecom()
    svc2 = LifecycleServices(layout=layout, secret_resolver=_resolver(),
                             stores=[_FakeStore()], async_sync=_NoAsync(),
                             wecom_client=wecom2, audit=audit, failure_policy="record")
    svc2.finalize_run("r1", "succeeded")
    assert len([s for s in wecom2.sent if "训练成功" in s]) == 0  # 不重复

    # 终态通知后发布最终 service manifest（含 audit 引用）
    manifest_path = os.path.join(layout.run_dir, "services", "service-manifest.json")
    assert os.path.exists(manifest_path)
    sm = json.load(open(manifest_path, encoding="utf-8"))
    assert sm["status"] == "succeeded"
    assert sm["audit"] == "services/service-audit.jsonl"


def test_final_bundle_snapshot_contains_service_manifest_and_stable_marker(tmp_path):
    class RecordingStore(_FakeStore):
        def __init__(self):
            super().__init__()
            self.snapshots = []

        def publish_run_bundle(self, local_dir, run_id):
            manifest = os.path.join(local_dir, "services", "service-manifest.json")
            self.snapshots.append(json.load(open(manifest, encoding="utf-8")))
            self.bundles.append(("run", run_id))

    store = RecordingStore()
    layout = RunLayout(str(tmp_path / "runs" / "bundle-final"))
    layout.ensure()
    audit = ServiceAudit(layout.service_audit_jsonl, redactor=lambda t: t)
    svc = LifecycleServices(layout=layout, secret_resolver=_resolver(), stores=[store],
                             async_sync=_NoAsync(), wecom_client=_FakeWecom(), audit=audit,
                             failure_policy="record")
    svc.finalize_run("run-1", "succeeded")
    marker_path = os.path.join(layout.run_dir, "services", "bundle-published.json")
    marker = json.load(open(marker_path, encoding="utf-8"))
    checksum = marker["entries"]["run:run-1"]["bundle_checksum"]
    assert len(store.snapshots) == 1
    assert store.snapshots[0]["audit_checksum"]
    svc2 = LifecycleServices(layout=layout, secret_resolver=_resolver(), stores=[store],
                              async_sync=_NoAsync(), wecom_client=_FakeWecom(), audit=audit,
                              failure_policy="record")
    svc2.finalize_run("run-1", "succeeded")
    marker2 = json.load(open(marker_path, encoding="utf-8"))
    assert len(store.snapshots) == 1
    assert marker2["entries"]["run:run-1"]["bundle_checksum"] == checksum

    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("{}")
    with pytest.raises(ServiceDeliveryError):
        svc2.finalize_run("run-1", "succeeded")
