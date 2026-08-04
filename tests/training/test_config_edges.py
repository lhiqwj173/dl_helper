"""补充 config.py 校验辅助负向分支覆盖（OSR-009 覆盖率门禁）。"""
from __future__ import annotations

import pytest

from dl_helper.training.config import (
    ConfigError,
    _bool,
    _float,
    _int,
    _non_negative_int,
    _optional_float,
    _optional_int,
    _positive_int,
    _str,
)


def test_optional_int_negative():
    with pytest.raises(ConfigError):
        _optional_int(1.5, "$.x")
    with pytest.raises(ConfigError):
        _optional_int(True, "$.x")


def test_optional_int_none_ok():
    assert _optional_int(None, "$.x") is None
    assert _optional_int(3, "$.x") == 3


def test_optional_float_negative():
    with pytest.raises(ConfigError):
        _optional_float(True, "$.x")
    with pytest.raises(ConfigError):
        _optional_float("abc", "$.x")


def test_optional_float_positive():
    assert _optional_float(None, "$.x") is None
    assert _optional_float(5, "$.x") == 5.0
    assert _optional_float(2.5, "$.x") == 2.5


def test_optional_float_nan_negative():
    import math
    with pytest.raises(ConfigError):
        _optional_float(math.nan, "$.x")


def test_int_negative():
    with pytest.raises(ConfigError):
        _int(1.5, "$.x")
    with pytest.raises(ConfigError):
        _int(True, "$.x")


def test_positive_int_negative():
    with pytest.raises(ConfigError):
        _positive_int(0, "$.x")
    with pytest.raises(ConfigError):
        _positive_int(-2, "$.x")


def test_non_negative_int_negative():
    with pytest.raises(ConfigError):
        _non_negative_int(-1, "$.x")


def test_float_negative():
    with pytest.raises(ConfigError):
        _float(True, "$.x")
    with pytest.raises(ConfigError):
        _float("x", "$.x")


def test_bool_negative():
    with pytest.raises(ConfigError):
        _bool(1, "$.x")


def test_str_negative():
    with pytest.raises(ConfigError):
        _str(123, "$.x")
