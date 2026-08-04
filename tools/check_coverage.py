#!/usr/bin/env python
"""独立覆盖率门禁：training 包 line>=85%、branch>=75%，分开计算。

先预导入重依赖（torch 等），避免 coverage tracer 与 torch._add_docstr 冲突；
branch% 由 covered_branches / num_branches 计算（coverage 7.x JSON 无 percent_covered_branches）。

用法：python tools/check_coverage.py [coverage_json]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE_MIN = 85.0
BRANCH_MIN = 75.0


def measure(tmp_path: str) -> str:
    cov_path = os.path.join(tmp_path, "coverage.json")
    code = (
        "import os, sys\n"
        "os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')\n"
        "import torch, numpy, sklearn, yaml, matplotlib, joblib, requests, safetensors\n"
        "import coverage\n"
        "cov = coverage.Coverage(source=['dl_helper.training'], omit=['*/site-packages/*'], branch=True)\n"
        "cov.start()\n"
        "import pytest\n"
        "args = ['-q', 'tests/training', 'tests/integration', 'tests/services', 'tests/sweeps',\n"
        "        'tests/reporting', 'tests/kaggle', 'tests/ci', 'tests/docs', 'tests/security',\n"
        "        '-m', 'not slow',\n"
        "        '--ignore=tests/training/test_cli_exit_codes.py',\n"
        "        '--ignore=tests/training/test_launcher.py',\n"
        "        '--ignore=tests/repository']\n"
        "rc = int(pytest.main(args))\n"
        "cov.stop()\n"
        "cov.save()\n"
        "cov.json_report(outfile=%r)\n"
        "sys.exit(rc)\n"
    ) % cov_path
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": REPO},
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"pytest 失败: {proc.returncode}")
    return cov_path


def check(cov_json: str) -> int:
    with open(cov_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = data["totals"]
    # coverage 7.x 的 percent_covered 是 line+branch 合并比率，不满足"分开计算"；
    # line 覆盖率必须以纯行数 covered_lines/num_statements 计算。
    num_lines = total.get("num_statements", 0)
    covered_lines = total.get("covered_lines", 0)
    if num_lines:
        line = covered_lines / num_lines * 100.0
    else:
        line = total["percent_covered"]
    num_branches = total.get("num_branches", 0)
    covered_branches = total.get("covered_branches", 0)
    branch = (covered_branches / num_branches * 100.0) if num_branches else 100.0
    ok = True
    if line < LINE_MIN:
        print(f"training line coverage {line:.2f}% < {LINE_MIN}%", file=sys.stderr)
        ok = False
    if branch < BRANCH_MIN:
        print(f"training branch coverage {branch:.2f}% < {BRANCH_MIN}%", file=sys.stderr)
        ok = False
    if ok:
        print(f"coverage OK: line={line:.2f}% branch={branch:.2f}%")
    return 0 if ok else 1


if __name__ == "__main__":
    import tempfile
    cov_json = sys.argv[1] if len(sys.argv) > 1 else measure(tempfile.mkdtemp())
    sys.exit(check(cov_json))
