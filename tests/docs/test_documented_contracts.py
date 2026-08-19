"""任务 10.5：文档合同 —— 路径/CLI/schema/formula/服务字段存在且无 Secret。"""
from __future__ import annotations

import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOC_DIR = os.path.join(REPO, "docs", "training")


def _all_docs():
    return glob.glob(os.path.join(DOC_DIR, "*.md")) + [os.path.join(REPO, "README.md")]


def test_doc_files_exist():
    expected = [
        "configuration.md", "metrics.md", "custom-task.md", "sklearn.md",
        "services.md", "sweeps.md", "kaggle.md", "artifacts.md", "breaking-removal.md",
    ]
    for name in expected:
        assert os.path.exists(os.path.join(DOC_DIR, name)), f"缺少文档 {name}"


def test_docs_reference_real_commands():
    config_doc = open(os.path.join(DOC_DIR, "configuration.md"), encoding="utf-8").read()
    assert "schema_version: 1" in config_doc
    assert "mixed_precision" in config_doc
    metrics_doc = open(os.path.join(DOC_DIR, "metrics.md"), encoding="utf-8").read()
    assert "formula_id/version" in metrics_doc or "formula_version" in metrics_doc
    assert "1e-6" in metrics_doc or "金标" in metrics_doc
    sweep_doc = open(os.path.join(DOC_DIR, "sweeps.md"), encoding="utf-8").read()
    assert "comparison_metric" in sweep_doc
    assert "val/" in sweep_doc


def test_docs_no_secrets():
    import subprocess, sys
    # 复用扫描器
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import scan_secrets

    violations = []
    for path in _all_docs():
        rel = os.path.relpath(path, REPO)
        with open(path, "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                violations.extend(scan_secrets._scan_line(line, rel))
    assert violations == [], f"文档含潜在凭证: {violations}"


def test_breaking_removal_doc():
    doc = open(os.path.join(DOC_DIR, "breaking-removal.md"), encoding="utf-8").read()
    assert "dl_helper.training" in doc
    assert "1.0.0" in doc
    assert "checkpoint.resume" in doc
    assert "runtime" in doc
    assert "--resume" in doc
    assert "execution-policy.json" in doc
