"""任务 4.1：固定 run schema、原子写入、终态互斥与路径边界。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import (
    ArtifactError,
    RunLayout,
    atomic_write_text,
    existing_terminal,
    list_relative_files,
    publish_terminal,
    read_json,
    sha256_file,
    sha256_manifest,
    ensure_within,
    write_json,
)


def test_atomic_write_and_checksum(tmp_path):
    path = str(tmp_path / "a" / "b" / "f.txt")
    atomic_write_text(path, "中文内容\n")
    assert read_text(path) == "中文内容\n"
    assert len(sha256_file(path)) == 64
    # 覆盖原子
    atomic_write_text(path, "new")
    assert read_text(path) == "new"
    # 无残留 tmp
    assert not [f for f in os.listdir(os.path.dirname(path)) if f.startswith(".f.txt")]


def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def test_run_layout_structure(tmp_path):
    layout = RunLayout(str(tmp_path / "runs" / "r1"))
    layout.ensure()
    layout.write_text("config.resolved.yaml", "schema_version: 1\n")
    layout.write_json("environment.json", {"os": "test"})
    assert os.path.exists(layout.config_resolved)
    assert os.path.exists(layout.environment_json)
    assert layout.summary_json.endswith(os.path.join("metrics", "summary.json"))


def test_json_utf8(tmp_path):
    path = str(tmp_path / "x.json")
    write_json(path, {"标签": "中文", "值": 1.5})
    data = read_json(path)
    assert data["标签"] == "中文"


def test_terminal_mutual_exclusion(tmp_path):
    layout = RunLayout(str(tmp_path / "runs" / "r2"))
    layout.ensure()
    publish_terminal(layout.run_dir, "success", {"status": "ok"})
    assert existing_terminal(layout.run_dir) == "run-manifest.json"
    # 不同终态互斥
    with pytest.raises(ArtifactError):
        publish_terminal(layout.run_dir, "failed", {"error": "x"})
    # 相同终态幂等（覆盖）
    publish_terminal(layout.run_dir, "success", {"status": "ok2"})
    assert read_json(layout.path("run-manifest.json"))["status"] == "ok2"


def test_path_escape_rejected(tmp_path):
    layout = RunLayout(str(tmp_path / "runs" / "r3"))
    layout.ensure()
    with pytest.raises(ArtifactError):
        layout.path("..", "escape.txt")
    with pytest.raises(ArtifactError):
        layout.path("../../outside.txt")


@pytest.mark.skipif(os.name != "nt", reason="仅验证 Windows 大小写不敏感路径语义")
def test_path_boundary_is_case_insensitive_on_windows(tmp_path):
    root = str(tmp_path / "Runs")
    os.makedirs(root, exist_ok=True)
    candidate = str(tmp_path / "runs" / "child")
    assert ensure_within(root.upper(), candidate.upper(), "case")


def test_sha256_manifest(tmp_path):
    root = tmp_path / "tree"
    os.makedirs(root / "sub", exist_ok=True)
    (root / "a.txt").write_text("hello", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("world", encoding="utf-8")
    manifest = sha256_manifest(str(root))
    assert set(manifest) == {"a.txt", os.path.join("sub", "b.txt")}
    for meta in manifest.values():
        assert "size" in meta and "sha256" in meta
    assert len(list_relative_files(str(root))) == 2


def test_run_layout_log_utf8(tmp_path):
    layout = RunLayout(str(tmp_path / "runs" / "r4"))
    layout.ensure()
    layout.log("训练开始 中文")
    lines = [json.loads(l) for l in open(layout.logs, encoding="utf-8")]
    assert any("训练开始" in l["message"] for l in lines)
