#!/usr/bin/env python
"""Kaggle 固定 revision 启动脚本：clone/checkout/HEAD 校验/安装/doctor。

不包含 Secret，不下载浮动 master，不执行 git pull 或静默升级框架。
所有子进程返回码被检查；文本 I/O 显式 UTF-8。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def fail(message: str) -> None:
    print(f"[bootstrap] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    """运行子进程并检查返回码；UTF-8 输出。"""
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        fail(f"命令失败 ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def git_ref() -> str:
    """从环境读取 40 位 commit SHA；拒绝其他值。"""
    ref = os.environ.get("DL_HELPER_GIT_REF", "").strip()
    if not re.match(r"^[0-9a-fA-F]{40}$", ref):
        fail("DL_HELPER_GIT_REF 必须是 40 位 commit SHA")
    return ref


def main() -> int:
    repo_url = os.environ.get("DL_HELPER_GIT_REPO", "").strip()
    if not repo_url:
        fail("缺少 DL_HELPER_GIT_REPO（仓库 URL）")
    ref = git_ref()

    working = "/kaggle/working"
    os.makedirs(working, exist_ok=True)
    repo_dir = os.path.join(working, "dl-helper")

    # 1. clone（浅取固定 revision）
    run(["git", "clone", repo_url, repo_dir])
    run(["git", "checkout", ref], cwd=repo_dir)
    # 2. 校验 HEAD 与固定 revision 一致
    head = run(["git", "rev-parse", "HEAD"], cwd=repo_dir).stdout.strip()
    if head.lower() != ref.lower():
        fail(f"HEAD {head} 与固定 revision {ref} 不一致")
    # 3. 安装（不升级框架）
    run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=repo_dir)
    # 4. doctor
    run([sys.executable, "-m", "dl_helper.training.cli", "doctor",
         "--config", os.path.join(repo_dir, "configs", "kaggle", "mnist.yaml"),
         "--experiment", "experiments.mnist:build_experiment"],
        cwd=repo_dir)
    print("[bootstrap] OK: 固定 revision 就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
