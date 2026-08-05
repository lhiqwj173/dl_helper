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


def resolve_doctor_config(repo_dir: str, working_dir: str = "/kaggle/working") -> str:
    """为显式 Kaggle 数据路径生成 working 目录中的临时 doctor 配置。"""
    source = os.path.join(repo_dir, "configs", "kaggle", "mnist.yaml")
    data_path = os.environ.get("DL_HELPER_MNIST_PATH", "").strip()
    if not data_path:
        return source
    if not os.path.isabs(data_path):
        fail("DL_HELPER_MNIST_PATH 必须是绝对路径")
    if not os.path.isfile(data_path):
        fail(f"DL_HELPER_MNIST_PATH 不存在或不是文件: {data_path}")

    import yaml

    with open(source, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        fail("Kaggle MNIST 基础配置必须包含 experiment mapping")
    config["experiment"]["data_path"] = data_path
    output = os.path.join(working_dir, "dl-helper-doctor.yaml")
    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return output


def main() -> int:
    repo_url = os.environ.get("DL_HELPER_GIT_REPO", "").strip()
    if not repo_url:
        fail("缺少 DL_HELPER_GIT_REPO（仓库 URL）")
    ref = git_ref()

    # 模板可先完成 clone/checkout；独立执行时此处创建全新 checkout。
    repo_dir = resolve_repo_dir(repo_url)
    # 固定到请求 revision，不依赖当前分支或远端默认分支。
    run(["git", "checkout", ref], cwd=repo_dir)
    # 2. 校验 HEAD 与固定 revision 一致
    head = run(["git", "rev-parse", "HEAD"], cwd=repo_dir).stdout.strip()
    if head.lower() != ref.lower():
        fail(f"HEAD {head} 与固定 revision {ref} 不一致")
    # 3. 安装（不升级框架）
    run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=repo_dir)
    # 4. doctor
    doctor_config = resolve_doctor_config(repo_dir)
    run([sys.executable, "-m", "dl_helper.training.cli", "doctor",
         "--config", doctor_config,
         "--experiment", "experiments.mnist:build_experiment"],
        cwd=repo_dir)
    print("[bootstrap] OK: 固定 revision 就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
