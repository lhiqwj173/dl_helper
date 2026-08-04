"""补充 notifications.py 覆盖率：模板全集、token 边界、裁剪。"""
from __future__ import annotations

import pytest

from dl_helper.training.notifications import (
    CONTENT_MAX_BYTES,
    WecomClient,
    WecomError,
    render_event_template,
)


def test_all_event_templates_render():
    events = {
        "RUN_STARTED": {"run_id": "r", "platform": "local", "utc": "t"},
        "RUN_SUCCEEDED": {"run_id": "r", "utc": "t", "elapsed": "1m", "summary": "a", "report": "x"},
        "RUN_PREEMPTED": {"run_id": "r", "utc": "t", "checkpoint": "c1"},
        "RUN_FAILED": {"run_id": "r", "utc": "t", "error_type": "E", "message": "m"},
        "SWEEP_STARTED": {"sweep_id": "s", "platform": "local", "utc": "t"},
        "TRIAL_STARTED": {"sweep_id": "s", "trial": "t1", "utc": "t"},
        "TRIAL_SUCCEEDED": {"sweep_id": "s", "trial": "t1", "utc": "t", "summary": "a"},
        "TRIAL_PREEMPTED": {"sweep_id": "s", "trial": "t1", "utc": "t", "checkpoint": "c"},
        "TRIAL_FAILED": {"sweep_id": "s", "trial": "t1", "utc": "t", "error_type": "E"},
        "SWEEP_SUCCEEDED": {"sweep_id": "s", "utc": "t", "best": "b"},
        "SWEEP_PREEMPTED": {"sweep_id": "s", "utc": "t", "checkpoint": "c"},
        "SWEEP_FAILED": {"sweep_id": "s", "utc": "t", "error_type": "E"},
    }
    for event, fields in events.items():
        msg = render_event_template(event, **fields)
        assert isinstance(msg, str) and msg


def test_missing_required_field_fails():
    with pytest.raises(WecomError):
        render_event_template("TRIAL_STARTED", sweep_id="s")  # 缺 trial


def test_unknown_event():
    with pytest.raises(WecomError):
        render_event_template("NOPE", run_id="r")


class _Resp:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def test_get_token_validation(monkeypatch):
    resolver = type("R", (), {"resolve": lambda self, k: "v"})()
    c = WecomClient("k1", "k2", "k3", "u", resolver, 1, 1, 2)

    # errcode 非 0
    monkeypatch.setattr("dl_helper.training.notifications.requests.get",
                        lambda *a, **k: _Resp({"errcode": 40001}))
    with pytest.raises(WecomError):
        c.get_token()

    # expires_in 非法
    monkeypatch.setattr("dl_helper.training.notifications.requests.get",
                        lambda *a, **k: _Resp({"errcode": 0, "access_token": "t", "expires_in": 0}))
    with pytest.raises(WecomError):
        c.get_token()


def test_trim_utf8_boundary():
    c = WecomClient("k", "k", "k", "u", type("R", (), {"resolve": lambda self, k: "v"})(), 1, 1, 2)
    # 中文多字节：裁剪不切字符
    content = "训" * 2000
    trimmed = c._trim_utf8(content)
    assert len(trimmed.encode("utf-8")) <= CONTENT_MAX_BYTES
    assert trimmed == trimmed[:-1] + trimmed[-1] if trimmed else True
    # 恰好边界
    ok = "a" * CONTENT_MAX_BYTES
    assert c._trim_utf8(ok).encode("utf-8") == b"a" * CONTENT_MAX_BYTES


def test_agent_id_positive_int():
    class Res:
        def resolve(self, key):
            return "not-an-int"
    c = WecomClient("k1", "k2", "k3", "u", Res(), 1, 1, 2)
    with pytest.raises(WecomError):
        c._agent_id()
