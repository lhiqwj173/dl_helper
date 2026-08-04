"""补充 services / notifications / base 分支覆盖（OSR-009 覆盖率门禁）。

覆盖 LifecycleServices 禁用路径、ServiceAudit extra 脱敏、WecomClient 重试边界。
"""
from __future__ import annotations

import json

import pytest

from dl_helper.training.notifications import (
    WecomClient,
    WecomError,
    render_event_template,
)
from dl_helper.training.services import ServiceAudit, utc_now


class _Layout:
    def __init__(self, tmp_path):
        self.run_dir = str(tmp_path / "runs" / "x")


class _Resolver:
    def resolve(self, key):
        return {"WECOM_CORP_ID": "c", "WECOM_CORP_SECRET": "s", "WECOM_AGENT_ID": "1000002"}[key]

    def redact(self, text):
        return text


class _Resp:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _client():
    return WecomClient("WECOM_CORP_ID", "WECOM_CORP_SECRET", "WECOM_AGENT_ID",
                       "user1", _Resolver(), 1.0, 1.0, 3)


# ---------- services：LifecycleServices 全禁用路径 ----------
def test_lifecycle_all_disabled_paths(tmp_path):
    """wecom=None、async_sync=None、含 resolved/无 resolved store 时各入口不崩。"""
    from dl_helper.training.services import LifecycleServices

    class _StoreWithResolved:
        resolved = True

        def publish_run_bundle(self, local_dir, run_id):
            pass

        def publish_sweep_bundle(self, local_dir, sweep_id):
            pass

    class _StorePlain:
        def publish_run_bundle(self, local_dir, run_id):
            pass

        def publish_sweep_bundle(self, local_dir, sweep_id):
            pass

    audit = ServiceAudit(str(tmp_path / "audit.jsonl"), redactor=lambda t: t)
    ls = LifecycleServices(layout=_Layout(tmp_path), secret_resolver=_Resolver(),
                           stores=[_StoreWithResolved(), _StorePlain()],
                           async_sync=None, wecom_client=None, audit=audit,
                           failure_policy="record")
    ls.start_run("r1")
    ls.finalize_run("r1", "succeeded")
    ls.start_sweep("s1")
    ls.trial_event("s1", "t1", "succeeded")
    ls.finalize_sweep("s1", "succeeded")
    assert ls.result.has_degraded is False


def test_service_audit_extra_with_private_key(tmp_path):
    """extra 含 _ 前缀私有键时不写入记录，普通键透传。"""
    audit = ServiceAudit(str(tmp_path / "a.jsonl"), redactor=lambda t: t)
    rec = audit.record("run/r1", "alist", "PUBLISH", 1, "success",
                       started_utc=utc_now(), finished_utc=utc_now(), duration_ms=1,
                       extra={"host": "https://alist.example", "_private": "leak",
                              "size": 10})
    assert rec["host"] == "https://alist.example"
    assert rec["size"] == 10
    assert "_private" not in rec
    lines = [json.loads(l) for l in open(str(tmp_path / "a.jsonl"), encoding="utf-8")]
    assert len(lines) == 1
    assert "_private" not in lines[0]


# ---------- base：validate_backend_result 负向 ----------
def test_validate_backend_result_wrong_type():
    from dl_helper.training.backends.base import validate_backend_result
    with pytest.raises(TypeError):
        validate_backend_result("not-a-result")


# ---------- notifications：构造与重试边界 ----------
def test_wecom_empty_to_user_raises():
    with pytest.raises(WecomError):
        WecomClient("k1", "k2", "k3", "", _Resolver(), 1.0, 1.0, 2)


def test_wecom_send_text_redactor_applied(monkeypatch):
    sent = {}

    def fake_post(url, **kwargs):
        if "/cgi-bin/gettoken" in url:
            return _Resp(200, {"errcode": 0, "access_token": "tok", "expires_in": 7200})
        sent["content"] = kwargs["json"]["text"]["content"]
        return _Resp(200, {"errcode": 0})

    monkeypatch.setattr("dl_helper.training.notifications.requests.get", fake_post)
    monkeypatch.setattr("dl_helper.training.notifications.requests.post", fake_post)
    c = _client()
    c.send_text("hello secret", redactor=lambda t: t.replace("secret", "[R]"))
    assert sent["content"] == "hello [R]"


def test_wecom_retryable_code_retries_then_succeeds(monkeypatch):
    seq = iter([45009, 0])  # 系统繁忙可重试，随后成功
    n = {"n": 0}

    def fake_post(url, **kwargs):
        if "/cgi-bin/gettoken" in url:
            return _Resp(200, {"errcode": 0, "access_token": "tok", "expires_in": 7200})
        n["n"] += 1
        return _Resp(200, {"errcode": next(seq)})

    monkeypatch.setattr("dl_helper.training.notifications.requests.get", fake_post)
    monkeypatch.setattr("dl_helper.training.notifications.requests.post", fake_post)
    c = _client()
    c.send_text("msg")
    assert n["n"] == 2


def test_wecom_connection_error_retries_exhausted(monkeypatch):
    import requests

    def boom(url, **kwargs):
        if "/cgi-bin/gettoken" in url:
            return _Resp(200, {"errcode": 0, "access_token": "tok", "expires_in": 7200})
        raise requests.ConnectionError("net down")

    monkeypatch.setattr("dl_helper.training.notifications.requests.get", boom)
    monkeypatch.setattr("dl_helper.training.notifications.requests.post", boom)
    c = _client()
    with pytest.raises(WecomError):
        c.send_text("msg")


def test_render_sweep_template_requires_sweep_id():
    with pytest.raises(WecomError):
        render_event_template("SWEEP_STARTED", platform="kaggle", utc="t")


def test_render_sweep_template_ok():
    out = render_event_template("SWEEP_STARTED", sweep_id="s1", platform="kaggle", utc="t")
    assert "s1" in out
