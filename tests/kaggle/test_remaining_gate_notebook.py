"""Kaggle 剩余发布门禁 Notebook 的固定 revision 与证据合同。"""
from __future__ import annotations

import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTEBOOK = os.path.join(ROOT, "notebook", "kaggle_remaining_release_gate.ipynb")


def _code() -> str:
    with open(NOTEBOOK, encoding="utf-8") as file:
        notebook = json.load(file)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_remaining_gate_is_pinned_and_complete():
    code = _code()
    assert "65f1063ffe3ebef746d080c503bfc27b29999829" in code
    assert "https://tmsatws.kdns.fr" in code
    assert "run_stamp = datetime.now(timezone.utc)" in code
    assert "DL_HELPER_RUN_STAMP" in code
    assert "f'kaggle-toy-sweep-{run_stamp}'" in code
    assert "f'sklearn-incremental-smoke-{run_stamp}'" in code
    for command in ("doctor", "sweep", "train"):
        assert f"'{command}'" in code
    for artifact in ("sweep-manifest.json", "run-manifest.json", "service-audit.jsonl",
                     "report/index.html"):
        assert artifact in code
    assert "f'/kaggle/working/kaggle-sweep-evidence-{evidence_stamp}.zip'" in code
    assert "f'/kaggle/working/sklearn-incremental-evidence-{evidence_stamp}.zip'" in code
    assert "sweep-report/index.html" in code


def test_remaining_gate_only_references_secret_names():
    code = _code()
    for name in ("ALIST_USER", "ALIST_PWD", "WECOM_CORP_ID", "WECOM_CORP_SECRET",
                 "WECOM_AGENT_ID"):
        assert name in code
    for forbidden in ("ALIST_PWD=", "WECOM_CORP_SECRET=", "password="):
        assert forbidden not in code
