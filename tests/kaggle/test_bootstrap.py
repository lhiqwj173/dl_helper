"""Kaggle bootstrap：库安装与可选 Git ref，不耦合训练项目。"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

BOOTSTRAP = os.path.join("envs", "kaggle_bootstrap.py")


def _source() -> str:
    with open(BOOTSTRAP, "r", encoding="utf-8") as file:
        return file.read()


def test_ref_is_optional_and_not_limited_to_full_sha(monkeypatch):
    sys.path.insert(0, "envs")
    import kaggle_bootstrap

    monkeypatch.delenv("DL_HELPER_GIT_REF", raising=False)
    assert kaggle_bootstrap.git_ref() is None
    monkeypatch.setenv("DL_HELPER_GIT_REF", "v1.2.0")
    assert kaggle_bootstrap.git_ref() == "v1.2.0"


def test_ref_with_whitespace_rejected(monkeypatch):
    sys.path.insert(0, "envs")
    import kaggle_bootstrap

    monkeypatch.setenv("DL_HELPER_GIT_REF", "bad ref")
    with pytest.raises(SystemExit):
        kaggle_bootstrap.git_ref()


def test_command_order_with_optional_ref(monkeypatch):
    sys.path.insert(0, "envs")
    import kaggle_bootstrap

    commands: list[list[str]] = []

    def fake_run(command, cwd=None):
        commands.append(command)
        return SimpleNamespace(stdout="abc123\n")

    monkeypatch.setenv("DL_HELPER_GIT_REPO", "https://repo.example.invalid/dl-helper.git")
    monkeypatch.setenv("DL_HELPER_GIT_REF", "v1.2.0")
    monkeypatch.delenv("DL_HELPER_REPO_DIR", raising=False)
    monkeypatch.setattr(kaggle_bootstrap, "run", fake_run)
    monkeypatch.setattr(kaggle_bootstrap.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(kaggle_bootstrap.os.path, "exists", lambda path: False)

    assert kaggle_bootstrap.main() == 0
    repo_dir = os.path.join("/kaggle/working", "dl-helper")
    assert commands == [
        ["git", "clone", "https://repo.example.invalid/dl-helper.git", repo_dir],
        ["git", "checkout", "v1.2.0"],
        ["git", "rev-parse", "--short", "HEAD"],
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
    ]


def test_bootstrap_does_not_run_training_project():
    source = _source()
    assert "experiments.mnist" not in source
    assert '"doctor"' not in source
    assert "--no-deps" in source
    assert 'encoding="utf-8"' in source


def test_bootstrap_reuses_supplied_repo_dir(monkeypatch, tmp_path):
    """OSR-001：文档安装单元复用已 clone 仓库（DL_HELPER_REPO_DIR=repo_dir）+ URL（DL_HELPER_GIT_REPO）。"""
    sys.path.insert(0, "envs")
    import kaggle_bootstrap

    repo_dir = tmp_path / "dl-helper"
    repo_dir.mkdir()
    import subprocess

    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True,
                   text=True, encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, cwd=None):
        commands.append(command)
        return SimpleNamespace(stdout=f"{repo_dir}\n")

    monkeypatch.setenv("DL_HELPER_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("DL_HELPER_GIT_REPO", "https://repo.example.invalid/dl-helper.git")
    monkeypatch.delenv("DL_HELPER_GIT_REF", raising=False)
    monkeypatch.setattr(kaggle_bootstrap, "run", fake_run)

    assert kaggle_bootstrap.main() == 0
    clone_cmds = [c for c in commands if c and c[0:2] == ["git", "clone"]]
    assert clone_cmds == [], "复用已 clone 仓库时 bootstrap 不得再次 clone"
    assert commands[-1] == [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"]


def test_doc_install_unit_sets_bootstrap_env():
    """OSR-001：kaggle.md 安装单元在调用 bootstrap 前必须设置其强制要求的两个环境变量。"""
    import os

    doc = open(os.path.join("docs", "training", "kaggle.md"), "r", encoding="utf-8").read()
    install_block, *_ = doc.split("print(\"dl-helper 安装完成\")", 1)
    assert "DL_HELPER_REPO_DIR" in install_block
    assert "DL_HELPER_GIT_REPO" in install_block
    # bootstrap 调用前已设置（赋值在调用之前出现）
    assert install_block.index("os.environ[\"DL_HELPER_REPO_DIR\"]") < install_block.index(
        "kaggle_bootstrap.py"
    )
    assert install_block.index("os.environ[\"DL_HELPER_GIT_REPO\"]") < install_block.index(
        "kaggle_bootstrap.py"
    )
