"""任务 5.1：required/record 策略与原异常保护。"""
from __future__ import annotations

import pytest

from dl_helper.training.services import (
    SecondaryServiceError,
    ServiceDeliveryError,
    ServiceErrorPolicy,
    ServiceResult,
)


def test_required_raises():
    p = ServiceErrorPolicy("required")
    with pytest.raises(ServiceDeliveryError):
        p.handle_failure("alist", RuntimeError("boom"))


def test_record_continues_and_marks_degraded():
    p = ServiceErrorPolicy("record")
    result = ServiceResult()
    p.handle_failure("wecom", RuntimeError("boom"))  # 不抛
    result.mark_degraded("wecom", "RuntimeError")
    assert result.has_degraded
    assert result.snapshot()["degraded"] == [{"service": "wecom", "error_type": "RuntimeError"}]


def test_invalid_policy():
    with pytest.raises(ValueError):
        ServiceErrorPolicy("sometimes")


def test_secondary_error_does_not_override_primary():
    secondary = SecondaryServiceError("notify failed", cause=ConnectionError("net"))
    assert secondary.cause is not None
