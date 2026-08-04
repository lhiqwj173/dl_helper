"""任务 7.4：Kaggle Notebook 模板 —— 结构、命令顺序、SHA、无 Secret。"""
from __future__ import annotations

import json
import os

import nbformat

NOTEBOOK = os.path.join("notebook", "kaggle_training_template.ipynb")


def _notebook():
    with open(NOTEBOOK, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def _all_code():
    nb = _notebook()
    return "\n".join(c["source"] for c in nb.cells if c.cell_type == "code")


def test_valid_notebook_structure():
    nb = _notebook()
    assert nb.nbformat == 4
    assert len([c for c in nb.cells if c.cell_type == "code"]) >= 2


def test_pinned_revision_sha():
    code = _all_code()
    assert "DL_HELPER_GIT_REF" in code
    # 占位 SHA 为 40 位十六进制
    assert "0000000000000000000000000000000000000000" in code


def test_no_floating_master_or_pull():
    code = _all_code()
    assert "git pull" not in code
    assert "origin/master" not in code


def test_bootstrap_and_doctor_in_order():
    code = _all_code()
    idx_bootstrap = code.index("kaggle_bootstrap.py")
    idx_doctor = code.index("doctor")
    assert idx_bootstrap < idx_doctor


def test_subprocess_return_code_checked():
    code = _all_code()
    assert "returncode" in code


def test_no_secret_values():
    code = _all_code()
    for secret in ("ALIST_PWD=", "WECOM_CORP_SECRET=", "password="):
        assert secret not in code


def test_utf8_encoding():
    code = _all_code()
    assert "encoding='utf-8'" in code or "encoding=\"utf-8\"" in code
