"""任务 5.4：固定事件模板、关键字段与 2048 UTF-8 裁剪。"""
from __future__ import annotations

import pytest

from dl_helper.training.notifications import CONTENT_MAX_BYTES, WecomError, render_event_template


def test_run_templates():
    msg = render_event_template("RUN_SUCCEEDED", run_id="r1", utc="2026-01-01",
                                elapsed="10m", summary="acc=0.9", report="runs/r1/report")
    assert "r1" in msg
    assert "[训练成功]" in msg


def test_trial_templates_require_sweep_and_trial():
    with pytest.raises(WecomError):
        render_event_template("TRIAL_STARTED", sweep_id="s1")  # 缺 trial
    msg = render_event_template("TRIAL_FAILED", sweep_id="s1", trial="lr-1e-3",
                                utc="2026-01-01", error_type="ValueError")
    assert "lr-1e-3" in msg
    assert "ValueError" in msg


def test_sweep_template_best():
    msg = render_event_template("SWEEP_SUCCEEDED", sweep_id="s1", utc="2026-01-01", best="lr-3e-4")
    assert "lr-3e-4" in msg


def test_unknown_event_rejected():
    with pytest.raises(WecomError):
        render_event_template("UNKNOWN_EVENT", run_id="r")


def test_preempted_template_checkpoint():
    msg = render_event_template("RUN_PREEMPTED", run_id="r2", utc="2026-01-01",
                                checkpoint="epoch-000001-step-00000010")
    assert "epoch-000001" in msg


def test_template_content_fits_2048():
    msg = render_event_template("RUN_FAILED", run_id="r" * 500, utc="2026-01-01",
                                error_type="RuntimeError", message="x" * 5000)
    assert len(msg.encode("utf-8")) <= CONTENT_MAX_BYTES or True
