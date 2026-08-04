"""任务 5.4：企业微信客户端 token/cache/刷新/业务码/重试。"""
from __future__ import annotations

import json

import pytest

from dl_helper.training.notifications import (
    CONTENT_MAX_BYTES,
    WecomClient,
    WecomError,
    render_event_template,
)


class _Resolver:
    def __init__(self, env=None):
        self._env = env or {"WECOM_CORP_ID": "corp", "WECOM_CORP_SECRET": "sec",
                            "WECOM_AGENT_ID": "1000002"}

    def resolve(self, key):
        if key not in self._env:
            raise WecomError(f"Secret {key!r} 缺失")
        return self._env[key]


def _client(**kw):
    return WecomClient(
        corp_id_secret_key="WECOM_CORP_ID",
        corp_secret_key="WECOM_CORP_SECRET",
        agent_id_secret_key="WECOM_AGENT_ID",
        to_user="user1",
        secret_resolver=_Resolver(),
        connect_timeout=1.0,
        read_timeout=1.0,
        max_attempts=3,
    )


def test_token_get_and_cache(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _Resp(200, {"errcode": 0, "access_token": "tok1", "expires_in": 7200})

    monkeypatch.setattr("dl_helper.training.notifications.requests.get", fake_get)
    c = _client()
    t1 = c.get_token()
    t2 = c.get_token()
    assert t1 == t2 == "tok1"
    assert calls["n"] == 1  # 缓存


def test_send_message_success(monkeypatch):
    sent = {}

    def fake_post(url, **kwargs):
        if "/cgi-bin/gettoken" in url:
            return _Resp(200, {"errcode": 0, "access_token": "tok", "expires_in": 7200})
        sent["payload"] = kwargs.get("json")
        return _Resp(200, {"errcode": 0})

    monkeypatch.setattr("dl_helper.training.notifications.requests.get", fake_post)
    monkeypatch.setattr("dl_helper.training.notifications.requests.post", fake_post)
    c = _client()
    c.send_text("训练成功")
    p = sent["payload"]
    assert p["touser"] == "user1"
    assert p["agentid"] == 1000002
    assert p["msgtype"] == "text"
    assert p["text"]["content"] == "训练成功"


def test_token_invalidation_refreshes_and_replays_once(monkeypatch):
    sequence = iter([40014, 0])  # 第一次 send 报 token 失效，第二次成功
    sent_count = {"n": 0}
    token_count = {"n": 0}

    def fake_post(url, **kwargs):
        if "/cgi-bin/gettoken" in url:
            token_count["n"] += 1
            return _Resp(200, {"errcode": 0, "access_token": f"tok{token_count['n']}", "expires_in": 7200})
        sent_count["n"] += 1
        errcode = next(sequence)
        return _Resp(200, {"errcode": errcode})

    monkeypatch.setattr("dl_helper.training.notifications.requests.get", fake_post)
    monkeypatch.setattr("dl_helper.training.notifications.requests.post", fake_post)
    c = _client()
    c.send_text("msg")
    assert sent_count["n"] == 2  # 重放一次
    assert token_count["n"] == 2  # 刷新一次


def test_business_error_not_retried(monkeypatch):
    sent = {"n": 0}

    def fake_post(url, **kwargs):
        if "/cgi-bin/gettoken" in url:
            return _Resp(200, {"errcode": 0, "access_token": "t", "expires_in": 7200})
        sent["n"] += 1
        return _Resp(200, {"errcode": 93000})  # 业务错误

    monkeypatch.setattr("dl_helper.training.notifications.requests.get", fake_post)
    monkeypatch.setattr("dl_helper.training.notifications.requests.post", fake_post)
    c = _client()
    with pytest.raises(WecomError):
        c.send_text("msg")
    assert sent["n"] == 1  # 业务错误不重试


def test_2048_utf8_bytes_limit():
    c = _client()
    content = "训练" * 2000  # 远超 2048 bytes
    trimmed = c._trim_utf8(content)
    assert len(trimmed.encode("utf-8")) <= CONTENT_MAX_BYTES
    # 不切多字节字符
    assert trimmed.endswith("训练") or trimmed.endswith("训")


def test_agent_id_must_be_positive_int(monkeypatch):
    resolver = _Resolver({"WECOM_CORP_ID": "c", "WECOM_CORP_SECRET": "s", "WECOM_AGENT_ID": "abc"})

    def fake_get(url, **kwargs):
        return _Resp(200, {"errcode": 0, "access_token": "t", "expires_in": 7200})

    monkeypatch.setattr("dl_helper.training.notifications.requests.get", fake_get)
    c = WecomClient("k1", "k2", "k3", "u", resolver, 1, 1, 2)
    with pytest.raises(WecomError):
        c.send_text("msg")


class _Resp:
    def __init__(self, status_code, json_data, content=b""):
        self.status_code = status_code
        self._json = json_data
        self.content = content

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")
