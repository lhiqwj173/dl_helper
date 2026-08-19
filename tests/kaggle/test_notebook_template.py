"""Kaggle Notebook 模板：结构、自动预检、无明文 Secret。"""
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


def test_revision_can_be_human_readable():
    code = _all_code()
    assert "DL_HELPER_GIT_REF" in code


def test_no_floating_master_or_pull():
    code = _all_code()
    assert "git pull" not in code
    assert "origin/master" not in code


def test_bootstrap_and_preflight_in_order():
    code = _all_code()
    idx_clone = code.index("'clone'")
    idx_checkout = code.index("'checkout'")
    idx_head = code.index("'rev-parse'")
    idx_bootstrap = code.index("kaggle_bootstrap.py")
    idx_preflight = code.index("--preflight-only")
    assert idx_clone < idx_checkout < idx_head < idx_bootstrap < idx_preflight
    assert "DL_HELPER_REPO_DIR" in code
    assert "/kaggle/working/dl-helper-kaggle.yaml" in code
    assert "'doctor'" not in code


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
