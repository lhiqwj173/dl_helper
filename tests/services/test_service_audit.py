"""任务 5.1：service audit JSONL 与脱敏。"""
from __future__ import annotations

import json

from dl_helper.training.services import ServiceAudit, stable_event_id, utc_now


def test_audit_record_utf8(tmp_path):
    path = str(tmp_path / "service-audit.jsonl")
    audit = ServiceAudit(path, redactor=lambda t: t.replace("REDACTED", "[R]"))
    record = audit.record(
        "run/r1", "wecom", "RUN_STARTED", 1, "success",
        started_utc=utc_now(), finished_utc=utc_now(), duration_ms=5,
    )
    assert record["event_id"] == stable_event_id("run/r1", "RUN_STARTED", "1")
    assert record["service"] == "wecom"
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(lines) == 1


def test_audit_redacts_secret(tmp_path):
    path = str(tmp_path / "a.jsonl")
    audit = ServiceAudit(path, redactor=lambda t: t.replace("leaky", "[REDACTED]"))
    audit.record("run/r", "alist", "PUBLISH", 1, "failed",
                 started_utc=utc_now(), finished_utc=utc_now(), duration_ms=1,
                 error_type="leaky-value")
    content = open(path, encoding="utf-8").read()
    assert "leaky" not in content
    assert "[REDACTED]" in content
