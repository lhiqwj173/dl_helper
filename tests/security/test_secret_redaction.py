"""任务 9.3：全链路脱敏 —— HTTP/joblib/traceback/通知/audit。"""
from __future__ import annotations

from dl_helper.training.platform import SecretError, SecretResolver
from dl_helper.training.services import ServiceAudit, utc_now


class _FakePlatform:
    def __init__(self, env):
        self._env = env
        self.is_kaggle = False


def test_traceback_redaction():
    resolver = SecretResolver(_FakePlatform({"TOKEN": "leaky-secret-xyz"}), {"TOKEN": "leaky-secret-xyz"})
    resolver.resolve("TOKEN")
    tb = "Error: failed with leaky-secret-xyz in request"
    assert "leaky-secret-xyz" not in resolver.redact(tb)
    assert "[REDACTED]" in resolver.redact(tb)


def test_audit_redaction():
    resolver = SecretResolver(_FakePlatform({"PWD": "super-secret-pw"}), {"PWD": "super-secret-pw"})
    resolver.resolve("PWD")

    class Audit(ServiceAudit):
        def __init__(self, path):
            super().__init__(path, redactor=resolver.redact)
            self.records = []

        def record(self, *a, **kw):
            rec = super().record(*a, **kw)
            self.records.append(rec)
            return rec

    import tempfile
    audit = Audit(tempfile.mktemp())
    audit.record("run/r", "wecom", "E", 1, "failed",
                 started_utc=utc_now(), finished_utc=utc_now(), duration_ms=1,
                 error_type="super-secret-pw")
    assert "super-secret-pw" not in audit.records[0]["error_type"]


def test_notification_redaction():
    resolver = SecretResolver(_FakePlatform({"AGENT": "corp-secret-42"}), {"AGENT": "corp-secret-42"})
    resolver.resolve("AGENT")
    content = "企业微信通知含 corp-secret-42"
    redacted = resolver.redact(content)
    assert "corp-secret-42" not in redacted
    assert "[REDACTED]" in redacted


def test_secret_not_in_error_message():
    resolver = SecretResolver(_FakePlatform({}), {})
    try:
        resolver.resolve("MISSING_KEY")
        raise AssertionError("应抛出 SecretError")
    except SecretError as exc:
        assert "MISSING_KEY" in str(exc)
        assert "value" not in str(exc)
