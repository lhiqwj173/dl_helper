"""任务 4.1：sweep 目录 schema 与终态互斥。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import ArtifactError, RunLayout, existing_terminal, publish_terminal
from dl_helper.training.sweep import _publish_sweep_failure


def test_sweep_layout_structure(tmp_path):
    sweep_dir = str(tmp_path / "sweeps" / "s1")
    os.makedirs(sweep_dir, exist_ok=True)
    layout = RunLayout(sweep_dir)
    layout.ensure()
    layout.write_json("sweep-manifest.json", {"id": "s1"})
    assert os.path.exists(os.path.join(sweep_dir, "sweep-manifest.json"))


def test_sweep_terminal_mutual_exclusion(tmp_path):
    sweep_dir = str(tmp_path / "sweeps" / "s2")
    os.makedirs(sweep_dir, exist_ok=True)
    publish_terminal(sweep_dir, "success", {"status": "ok"})
    with pytest.raises(ArtifactError):
        publish_terminal(sweep_dir, "failed", {"error": "x"})
    assert existing_terminal(sweep_dir) == "run-manifest.json"


def test_sweep_trials_jsonl(tmp_path):
    sweep_dir = str(tmp_path / "sweeps" / "s3")
    layout = RunLayout(sweep_dir)
    layout.ensure()
    trials = layout.path("trials.jsonl")
    for i in range(3):
        from dl_helper.training.artifacts import append_jsonl
        append_jsonl(trials, {"trial": i, "status": "ok"})
    lines = [json.loads(l) for l in open(trials, encoding="utf-8")]
    assert len(lines) == 3


def test_sweep_failure_replaces_pause_terminal(tmp_path):
    sweep_dir = str(tmp_path / "sweeps" / "s4")
    os.makedirs(sweep_dir, exist_ok=True)
    RunLayout(sweep_dir).write_json("pause-manifest.json", {"status": "preempted"})
    _publish_sweep_failure(sweep_dir, {"status": "failed"})
    assert os.path.exists(os.path.join(sweep_dir, "failure.json"))
    assert not os.path.exists(os.path.join(sweep_dir, "pause-manifest.json"))
    assert not os.path.exists(os.path.join(sweep_dir, "sweep-manifest.json"))
