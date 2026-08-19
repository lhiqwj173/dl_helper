#!/usr/bin/env python
"""Kaggle 启动脚本：clone（可选 ref）/安装 dl-helper。

训练项目不属于 dl-helper；配置、项目目录和输出由调用方显式提供。
所有子进程返回码被检查；文本 I/O 显式 UTF-8。
"""
from __future__ import annotations

import os
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


def git_ref() -> str | None:
    """读取可选的 tag/分支/短 SHA；不再限制为 40 位 SHA。"""
    ref = os.environ.get("DL_HELPER_GIT_REF", "").strip()
    if ref and any(ch.isspace() for ch in ref):
        fail("DL_HELPER_GIT_REF 不得包含空白字符")
    return ref or None


def resolve_repo_dir(repo_url: str) -> str:
    """返回模板已校验目录，或为脚本独立执行时创建新的 checkout。"""
    supplied = os.environ.get("DL_HELPER_REPO_DIR", "").strip()
    if not supplied:
        working = "/kaggle/working"
        os.makedirs(working, exist_ok=True)
        repo_dir = os.path.join(working, "dl-helper")
        if os.path.exists(repo_dir):
            fail(f"默认 checkout 目录已存在，拒绝复用未经校验的目录: {repo_dir}")
        run(["git", "clone", repo_url, repo_dir])
        return repo_dir

    repo_dir = os.path.abspath(supplied)
    if not os.path.isdir(repo_dir):
        fail(f"DL_HELPER_REPO_DIR 不存在或不是目录: {repo_dir}")
    root = run(["git", "rev-parse", "--show-toplevel"], cwd=repo_dir).stdout.strip()
    if os.path.normcase(os.path.abspath(root)) != os.path.normcase(repo_dir):
        fail("DL_HELPER_REPO_DIR 必须指向 Git 工作树根目录")
    return repo_dir


def main() -> int:
    repo_url = os.environ.get("DL_HELPER_GIT_REPO", "").strip()
    if not repo_url:
        fail("缺少 DL_HELPER_GIT_REPO（仓库 URL）")
    ref = git_ref()

    # 模板可先完成 clone/checkout；独立执行时此处创建全新 checkout。
    repo_dir = resolve_repo_dir(repo_url)
    if ref:
        run(["git", "checkout", ref], cwd=repo_dir)
        head = run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir).stdout.strip()
        if not head:
            fail("无法读取 checkout 后的 Git 版本")
    # 安装库本身，不安装依赖、不运行任何训练项目。
    run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=repo_dir)
    print("[bootstrap] OK: dl-helper 已安装；请使用自己的 --project-dir/--config/--experiment 运行 train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
