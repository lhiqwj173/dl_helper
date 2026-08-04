"""补充 cli/platform/reporting/remote 分支覆盖。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.config import default_schema, parse_config


def test_cli_generate_run_id():
    from dl_helper.training.cli import _generate_run_id
    cfg = parse_config(default_schema())
    rid = _generate_run_id(cfg)
    assert rid.startswith(cfg.run.name + "-")


def test_cli_failure_evidence_redacts(tmp_path):
    import argparse
    from dl_helper.training.cli import _write_failure_evidence
    run_dir = tmp_path / "runs" / "f"
    os.makedirs(run_dir, exist_ok=True)
    args = argparse.Namespace(command="train", _run_dir=str(run_dir))
    _write_failure_evidence(args, RuntimeError("training failed"))
    failure = json.load(open(os.path.join(run_dir, "failure.json"), encoding="utf-8"))
    assert failure["exception_type"] == "RuntimeError"
    assert "primary_exception" in failure


def test_cli_failure_evidence_skips_completed(tmp_path):
    import argparse
    from dl_helper.training.artifacts import publish_terminal
    from dl_helper.training.cli import _write_failure_evidence
    run_dir = tmp_path / "runs" / "f2"
    os.makedirs(run_dir, exist_ok=True)
    publish_terminal(str(run_dir), "success", {"status": "ok"})
    args = argparse.Namespace(command="train", _run_dir=str(run_dir))
    _write_failure_evidence(args, RuntimeError("x"))  # 已有终态，不覆盖
    assert not os.path.exists(os.path.join(run_dir, "failure.json"))


def test_platform_hostname_and_revision():
    from dl_helper.training.platform import hostname, resolve_source_revision
    assert hostname() != ""
    cfg = parse_config(default_schema())
    rev = resolve_source_revision(cfg)  # 仓库是 git → HEAD
    assert len(rev) == 40


def test_platform_invalid_revision():
    import yaml
    from dl_helper.training.platform import PlatformError, resolve_source_revision
    schema = default_schema()
    schema["run"]["source_revision"] = "short"  # 非 40 位
    cfg = parse_config(schema)
    with pytest.raises(PlatformError):
        resolve_source_revision(cfg)


def test_reporting_stage_metrics_bad_lines(tmp_path):
    from dl_helper.training.reporting import _stage_metrics
    path = tmp_path / "metrics.jsonl"
    path.write_text("not json\n{}\n", encoding="utf-8")
    out = _stage_metrics(str(path))
    assert isinstance(out, dict)
