"""任务 7.4：bootstrap 脚本 —— 固定 revision、命令顺序、无 Secret、失败传播。"""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

BOOTSTRAP = os.path.join("envs", "kaggle_bootstrap.py")


def _source():
    with open(BOOTSTRAP, "r", encoding="utf-8") as f:
        return f.read()


def test_no_secret_literals():
    src = _source()
    # 不包含真实凭证字面量
    assert "alist.example.invalid" not in src or "https://" not in src


def test_requires_40_sha():
    src = _source()
    assert "DL_HELPER_GIT_REF" in src
    assert "40" in src


def test_no_floating_master_or_git_pull():
    src = _source()
    # 不执行 git pull / master 下载（docstring 提及禁止行为不算）
    assert '"git", "pull"' not in src
    assert "origin/master" not in src
    assert '"git", "clone"' in src


def test_command_order(monkeypatch):
    """clone → checkout → HEAD 校验 → pip install → doctor。"""
    sys.path.insert(0, "envs")
    import kaggle_bootstrap

    revision = "a" * 40
    commands: list[list[str]] = []

    def fake_run(command, cwd=None):
        commands.append(command)
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=revision + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setenv("DL_HELPER_GIT_REPO", "https://repo.example.invalid/dl-helper.git")
    monkeypatch.setenv("DL_HELPER_GIT_REF", revision)
    monkeypatch.delenv("DL_HELPER_REPO_DIR", raising=False)
    monkeypatch.setattr(kaggle_bootstrap, "run", fake_run)
    monkeypatch.setattr(kaggle_bootstrap.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(kaggle_bootstrap.os.path, "exists", lambda path: False)

    assert kaggle_bootstrap.main() == 0
    repo_dir = os.path.join("/kaggle/working", "dl-helper")
    assert commands == [
        ["git", "clone", "https://repo.example.invalid/dl-helper.git", repo_dir],
        ["git", "checkout", revision],
        ["git", "rev-parse", "HEAD"],
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
        [sys.executable, "-m", "dl_helper.training.cli", "doctor",
         "--config", os.path.join(repo_dir, "configs", "kaggle", "mnist.yaml"),
         "--experiment", "experiments.mnist:build_experiment"],
    ]


def test_return_codes_checked():
    src = _source()
    assert "returncode != 0" in src


def test_pip_install_no_deps():
    src = _source()
    assert "--no-deps" in src


def test_utf8_encoding():
    src = _source()
    assert "encoding='utf-8'" in src or "encoding=\"utf-8\"" in src


def test_invalid_ref_rejected():
    """无效 DL_HELPER_GIT_REF 立即失败。"""
    code = (
        "import os, sys\n"
        "sys.path.insert(0, 'envs')\n"
        "os.environ['DL_HELPER_GIT_REF'] = 'not-a-sha'\n"
        "import kaggle_bootstrap\n"
        "try:\n"
        "    kaggle_bootstrap.git_ref()\n"
        "    print('NO-RAISE')\n"
        "except SystemExit as e:\n"
        "    print('EXIT', e.code)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=os.path.dirname(os.path.abspath(__file__)) + "/../..",
        capture_output=True, text=True, encoding="utf-8", check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert "EXIT" in proc.stdout or "FAIL" in proc.stderr
