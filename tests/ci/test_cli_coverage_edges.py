"""补充 cli.py 分支覆盖（OSR-009 覆盖率门禁）：dispatch、失败证据、doctor 输出。"""
from __future__ import annotations

import os

import pytest
import yaml

from dl_helper.training.config import default_schema, parse_config


def _write_cfg(tmp_path, run_id="cli-edge", **patch):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["run"]["output_root"] = str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    for k, v in patch.items():
        schema[k] = v
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_dispatch_unknown_command():
    import argparse

    from dl_helper.training.cli import CliError, _dispatch
    with pytest.raises(CliError):
        _dispatch(argparse.Namespace(command="bogus-command"))


def test_resolve_run_layout_generates_id(tmp_path):
    import dl_helper.training.cli as cli
    from dl_helper.training.platform import Platform
    schema = default_schema()
    schema["run"]["id"] = None
    schema["run"]["name"] = "gen"
    schema["run"]["output_root"] = str(tmp_path)
    cfg = parse_config(schema)
    run_id, layout = cli._resolve_run_layout(cfg, Platform("local"))
    assert run_id.startswith("gen-")
    assert os.path.isdir(layout.run_dir)


def test_publish_cli_terminal_idempotent(tmp_path):
    import json as _json
    import dl_helper.training.cli as cli
    from dl_helper.training.artifacts import RunLayout
    layout = RunLayout(str(tmp_path / "runs" / "p"))
    layout.ensure()
    # OSR-005：多进程 publisher 要求 summary/environment 存在
    os.makedirs(os.path.dirname(layout.summary_json), exist_ok=True)
    with open(layout.summary_json, "w", encoding="utf-8") as f:
        _json.dump({"status": "succeeded", "epoch": 1, "global_step": 8}, f)
    with open(layout.environment_json, "w", encoding="utf-8") as f:
        _json.dump({"platform": "local"}, f)
    cfg = parse_config(default_schema())
    cli._publish_cli_terminal(layout, "succeeded", cfg, "p")  # 首次发布
    cli._publish_cli_terminal(layout, "succeeded", cfg, "p")  # 已存在 → 直接返回
    assert os.path.exists(os.path.join(layout.run_dir, "run-manifest.json"))


def test_write_failure_evidence_with_secret_keys(tmp_path):
    import dl_helper.training.cli as cli
    from dl_helper.training.artifacts import RunLayout
    layout = RunLayout(str(tmp_path / "runs" / "f"))
    layout.ensure()

    class Args:
        _run_dir = layout.run_dir
        _secret_keys = ("NO_SUCH_SECRET_12345",)

    cli._write_failure_evidence(Args(), RuntimeError("boom"))
    assert os.path.exists(os.path.join(layout.run_dir, "failure.json"))


def test_cmd_doctor_prints_errors(tmp_path):
    from dl_helper.training.cli import main
    cfg = _write_cfg(tmp_path, run_id="doc-err")
    code = main(["doctor", "--config", cfg, "--experiment", "nonexistent_module:build"])
    assert code == 1
