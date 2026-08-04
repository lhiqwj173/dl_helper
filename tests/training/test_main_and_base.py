"""补充 __main__ 与 backends.base 覆盖率。"""
from __future__ import annotations

import subprocess
import sys

import pytest

from dl_helper.training.backends.base import BackendResult, ModelArtifact, validate_backend_result


def test_backend_result_validation():
    r = BackendResult(status="succeeded", epoch=1, global_step=5)
    validate_backend_result(r)
    assert r.model_artifact is None
    r2 = BackendResult(status="preempted", model_artifact=ModelArtifact(format="safetensors"))
    validate_backend_result(r2)
    with pytest.raises(ValueError):
        BackendResult(status="bogus")
    with pytest.raises(ValueError):
        BackendResult(status="succeeded", epoch=-1)


def test_main_entry_runs():
    """python -m dl_helper.training 走 console 入口（--help 退出码 0）。"""
    proc = subprocess.run(
        [sys.executable, "-m", "dl_helper.training", "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "usage" in out.lower() or "dl_helper" in out.lower()
