"""任务 5.4：固定事件模板、关键字段、空字段省略与 2048 UTF-8 裁剪。"""
from __future__ import annotations

import pytest

from dl_helper.training.notifications import (
    CONTENT_MAX_BYTES,
    MESSAGE_MAX_CHARS,
    WecomError,
    format_duration,
    format_metric,
    format_metric_summary,
    render_event_template,
)


def test_run_started_keeps_identity_only():
    assert render_event_template("RUN_STARTED", run_id="r1") == "[训练开始] run=r1"


def test_run_succeeded_surfaces_result_and_duration():
    msg = render_event_template("RUN_SUCCEEDED", run_id="r1", elapsed="10m",
                                metrics="val/acc=0.931@epoch=5")
    assert msg == "[训练成功] run=r1 val/acc=0.931@epoch=5 耗时=10m"


def test_run_preempted_carries_metric_and_checkpoint():
    msg = render_event_template("RUN_PREEMPTED", run_id="r2", elapsed="5m",
                                metrics="val/loss=0.12@epoch=3",
                                checkpoint="epoch-000003-step-00000120")
    assert msg == ("[训练暂停] run=r2 val/loss=0.12@epoch=3 耗时=5m "
                   "恢复检查点=epoch-000003-step-00000120")


def test_empty_optional_fields_drop_label_and_placeholder():
    msg = render_event_template("RUN_PREEMPTED", run_id="r2", checkpoint="",
                                metrics=None, elapsed="5m")
    assert msg == "[训练暂停] run=r2 耗时=5m"
    failed = render_event_template("TRIAL_FAILED", sweep_id="s1", trial="lr-1e-3",
                                   error_type="ValueError")
    assert failed == "[Trial失败] sweep=s1 trial=lr-1e-3 异常=ValueError"


def test_trial_templates_require_sweep_and_trial():
    with pytest.raises(WecomError):
        render_event_template("TRIAL_STARTED", sweep_id="s1")  # 缺 trial
    msg = render_event_template("TRIAL_FAILED", sweep_id="s1", trial="lr-1e-3",
                                progress="2/8", error_type="ValueError")
    assert msg == "[Trial失败] sweep=s1 trial=lr-1e-3 进度=2/8 异常=ValueError"


def test_sweep_template_best_and_comparison_metric():
    msg = render_event_template("SWEEP_SUCCEEDED", sweep_id="s1", best="lr-3e-4",
                                metrics="val/loss=0.08")
    assert msg == "[Sweep成功] sweep=s1 best=lr-3e-4 val/loss=0.08"
    started = render_event_template("SWEEP_STARTED", sweep_id="s1", trials=8,
                                    comparison="val/loss", mode="min")
    assert started == "[Sweep开始] sweep=s1 trials=8 对比指标=val/loss(min)"


def test_unknown_event_rejected():
    with pytest.raises(WecomError):
        render_event_template("UNKNOWN_EVENT", run_id="r")


def test_long_error_message_is_truncated_to_single_line():
    msg = render_event_template("RUN_FAILED", run_id="r", epoch=3,
                                error_type="RuntimeError",
                                message="line1\nline2 " + "x" * 5000)
    assert "异常=RuntimeError" in msg
    assert "详情=line1 line2" in msg
    detail = msg.split("详情=", 1)[1]
    assert len(detail) == MESSAGE_MAX_CHARS + 1  # 截断标记
    assert len(msg.encode("utf-8")) <= CONTENT_MAX_BYTES


def test_oversized_message_still_fits_2048_bytes():
    msg = render_event_template("RUN_FAILED", run_id="r" * 1200,
                                error_type="RuntimeError", message="x" * 5000)
    assert len(msg.encode("utf-8")) <= CONTENT_MAX_BYTES


def test_format_duration():
    assert format_duration(0) == "0s"
    assert format_duration(42.4) == "42s"
    assert format_duration(303) == "5m03s"
    assert format_duration(3723) == "1h02m03s"
    with pytest.raises(ValueError):
        format_duration(-1)


def test_format_metric():
    assert format_metric("val/loss", 0.123456789) == "val/loss=0.123457"
    assert format_metric("epoch", 5) == "epoch=5"


def test_format_metric_summary_uses_metric_name():
    summary = {
        "selection": {"best_value": 0.931234, "best_epoch": 5},
        "stage_metrics": {"val": {"val/accuracy": 0.931234}},
    }
    assert format_metric_summary(summary, "val/accuracy") == "val/accuracy=0.931234@epoch=5"
    assert format_metric_summary(summary) == "best=0.931234@epoch=5"


def test_format_metric_summary_appends_matching_test_metric():
    summary = {
        "selection": {"best_value": 0.9, "best_epoch": 3},
        "stage_metrics": {"test": {"test/accuracy": 0.88, "test/f1_macro": 0.87}},
    }
    assert format_metric_summary(summary, "val/accuracy") == "val/accuracy=0.9@epoch=3 test/accuracy=0.88"


def test_format_metric_summary_falls_back_to_stage_metrics():
    summary = {"selection": None,
               "stage_metrics": {"val": {"val/accuracy": 0.75, "val/loss": 0.5}}}
    assert format_metric_summary(summary) == "val/accuracy=0.75 val/loss=0.5"
    test_only = {"selection": None, "stage_metrics": {"test": {"test/accuracy": 0.75}}}
    assert format_metric_summary(test_only) == "test/accuracy=0.75"


def test_format_metric_summary_empty():
    assert format_metric_summary(None) == ""
    assert format_metric_summary({}) == ""
    assert format_metric_summary({"selection": None, "stage_metrics": None}) == ""
