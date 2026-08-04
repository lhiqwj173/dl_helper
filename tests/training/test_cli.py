"""任务 3.7：CLI 参数矩阵与命令分派。"""
from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from dl_helper.training.cli import build_parser, main
from dl_helper.training.config import default_schema


def test_parser_has_five_commands():
    parser = build_parser()
    for cmd in ("doctor", "train", "report", "sweep", "sweep-report"):
        sub = next(a for a in parser._actions if getattr(a, "dest", None) == "command")
        assert cmd in sub.choices


def test_train_success(tmp_path):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 10, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = "cli-train"
    schema["run"]["output_root"] = str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")

    code = main([
        "train",
        "--config", str(cfg_path),
        "--experiment", "experiments.toy_multiclass:build_experiment",
    ])
    assert code == 0


def test_train_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["nonexistent"])


def test_train_missing_config_raises(tmp_path):
    """缺失配置文件：main 原样 raise，由入口以非零退出。"""
    with pytest.raises(Exception):
        main([
            "train",
            "--config", str(tmp_path / "missing.yaml"),
            "--experiment", "experiments.toy_multiclass:build_experiment",
        ])


def test_doctor_success(tmp_path):
    schema = default_schema()
    schema["run"]["id"] = "cli-doctor"
    schema["run"]["output_root"] = str(tmp_path)
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    cfg_path = tmp_path / "doctor.yaml"
    cfg_path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    code = main([
        "doctor",
        "--config", str(cfg_path),
        "--experiment", "experiments.toy_multiclass:build_experiment",
    ])
    assert code == 0
