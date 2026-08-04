"""任务 10.4：覆盖率双阈值门禁。"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tools"))

import check_coverage  # noqa: E402


def _cov_json(tmp_path, line, branch):
    path = tmp_path / "cov.json"
    num_statements = 200
    num_branches = 100
    covered_lines = int(num_statements * line / 100.0)
    covered_branches = int(num_branches * branch / 100.0)
    # coverage 7.x percent_covered 为 line+branch 合并比率；门禁须分开计算
    combined = 100.0 * (covered_lines + covered_branches) / (num_statements + num_branches)
    json.dump({"totals": {"percent_covered": combined, "num_statements": num_statements,
                          "covered_lines": covered_lines, "num_branches": num_branches,
                          "covered_branches": covered_branches}},
              open(path, "w", encoding="utf-8"))
    return str(path)


def test_line_gate(tmp_path):
    assert check_coverage.check(_cov_json(tmp_path, 90.0, 80.0)) == 0
    assert check_coverage.check(_cov_json(tmp_path, 80.0, 80.0)) == 1  # line 不足


def test_branch_gate(tmp_path):
    assert check_coverage.check(_cov_json(tmp_path, 90.0, 70.0)) == 1  # branch 不足
    assert check_coverage.check(_cov_json(tmp_path, 90.0, 75.0)) == 0  # 临界通过


def test_line_not_combined_percent(tmp_path):
    """OSR-009：percent_covered 是 line+branch 合并值，line 门禁必须用纯行数。

    line=90%、branch=50% 时合并 percent_covered=80%；若误读 percent_covered
    会把已达标 line 判为失败，本用例锁定纯行数语义。
    """
    assert check_coverage.check(_cov_json(tmp_path, 90.0, 50.0)) == 1  # branch 不足
    # 纯 line=90>=85 且 branch=50<75：返回 1 只因为 branch，line 不再误报
    assert check_coverage.check(_cov_json(tmp_path, 90.0, 76.0)) == 0


def test_thresholds_constants():
    assert check_coverage.LINE_MIN == 85.0
    assert check_coverage.BRANCH_MIN == 75.0
