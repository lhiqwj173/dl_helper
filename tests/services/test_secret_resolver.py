"""任务 5.1：Secret resolver 与脱敏。"""
from __future__ import annotations

import pytest

from dl_helper.training.platform import SecretError, SecretResolver


class _FakePlatform:
    def __init__(self, env=None):
        self._env = env or {}
        self.kind = "local"
        self.is_kaggle = False


def test_secret_resolver_local_env():
    env = {"ALIST_USER": "alice", "ALIST_PWD": "s3cret"}
    r = SecretResolver(_FakePlatform(), env)
    assert r.resolve("ALIST_USER") == "alice"
    assert r.resolve("ALIST_PWD") == "s3cret"
    assert "ALIST_USER" in r.resolved_keys()


def test_secret_missing_reports_key_only():
    r = SecretResolver(_FakePlatform(), {})
    with pytest.raises(SecretError) as ei:
        r.resolve("MISSING_SECRET")
    assert "MISSING_SECRET" in str(ei.value)


def test_redaction_replaces_values():
    r = SecretResolver(_FakePlatform(), {"TOKEN": "secret-value-123"})
    r.resolve("TOKEN")
    assert r.redact("my secret-value-123 here") == "my [REDACTED] here"


def test_secret_never_in_repr():
    r = SecretResolver(_FakePlatform(), {"PWD": "super-secret"})
    r.resolve("PWD")
    assert "super-secret" not in repr(r)
    assert "super-secret" not in str(r.resolved_keys())


def test_kaggle_secret_missing_client():
    class KaggleFake:
        is_kaggle = True

    r = SecretResolver(KaggleFake())
    with pytest.raises(SecretError):
        r.resolve("ANY")
