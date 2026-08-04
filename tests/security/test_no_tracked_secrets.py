"""任务 9.3：Git 受跟踪源码零明文凭证。"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_no_tracked_secrets():
    proc = subprocess.run(
        [sys.executable, os.path.join("tools", "scan_secrets.py")],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_scan_tool_finds_synthetic_violation(tmp_path):
    """扫描器能发现合成敏感字面量。"""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import scan_secrets

    line = 'my_password = "hunter2s3cretvalue"'
    violations = scan_secrets._scan_line(line, "fake.py")
    assert violations, "应发现敏感变量字面量"


def test_scan_exempts_sha_url_path():
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import scan_secrets

    assert scan_secrets._exempt("a" * 40)  # SHA
    assert scan_secrets._exempt("https://alist.example.invalid")
    assert scan_secrets._exempt("/kaggle/input/ds/data.csv")
    assert scan_secrets._exempt("${SECRET_KEY}")
    assert not scan_secrets._exempt("hunter2s3cretvalue")
