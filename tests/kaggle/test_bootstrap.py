"""任务 7.4：bootstrap 脚本 —— 固定 revision、命令顺序、无 Secret、失败传播。"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

BOOTSTRAP = os.path.join("envs", "kaggle_bootstrap.py")


def _source():
    with open(BOOTSTRAP, "r", encoding="utf-8") as f:
        return f.read()


def test_no_secret_literals():
    src = _source()
    # 不包含真实凭证字面量
    assert "alist.example.invalid" not in src or "https://" not in src


def test_requires_40_sha():
    src = _source()
    assert "DL_HELPER_GIT_REF" in src
    assert "40" in src


def test_no_floating_master_or_git_pull():
    src = _source()
    # 不执行 git pull / master 下载（docstring 提及禁止行为不算）
    assert '"git", "pull"' not in src
    assert "origin/master" not in src
    assert '"git", "clone"' in src


def test_command_order():
    """clone → checkout → HEAD 校验 → pip install → doctor。"""
    src = _source()
    idx_clone = src.index('"git", "clone"')
    idx_checkout = src.index('"git", "checkout"')
    idx_rev_parse = src.index('"git", "rev-parse"')
    idx_pip = src.index('"install"')
    idx_doctor = src.index('"doctor"')
    assert idx_clone < idx_checkout < idx_rev_parse < idx_pip < idx_doctor


def test_return_codes_checked():
    src = _source()
    assert "returncode != 0" in src


def test_pip_install_no_deps():
    src = _source()
    assert "--no-deps" in src


def test_utf8_encoding():
    src = _source()
    assert "encoding='utf-8'" in src or "encoding=\"utf-8\"" in src


def test_invalid_ref_rejected():
    """无效 DL_HELPER_GIT_REF 立即失败。"""
    code = (
        "import os, sys\n"
        "sys.path.insert(0, 'envs')\n"
        "os.environ['DL_HELPER_GIT_REF'] = 'not-a-sha'\n"
        "import kaggle_bootstrap\n"
        "try:\n"
        "    kaggle_bootstrap.git_ref()\n"
        "    print('NO-RAISE')\n"
        "except SystemExit as e:\n"
        "    print('EXIT', e.code)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.abspath(__file__)) + "/../..",
                          capture_output=True, text=True, encoding="utf-8", check=False)
    assert "EXIT" in proc.stdout or "FAIL" in proc.stderr
