"""任务 9.2：git 跟踪清单 —— 无禁止路径、无旧元数据与二进制。"""
from __future__ import annotations

import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FORBIDDEN_PREFIXES = (
    "cpp/",
    "参考/",
    "lob/",
    "dl_helper/acc/",
    "dl_helper/models/",
    "dl_helper/other_tests/",
    "dl_helper/rl/",
    "dl_helper/tests/",
    "dl_helper/transforms/",
)

FORBIDDEN_OLD_MODULES = (
    "dl_helper/trainer.py",
    "dl_helper/tester.py",
    "dl_helper/tracker.py",
    "dl_helper/train_param.py",
    "dl_helper/data.py",
    "dl_helper/scheduler.py",
    "dl_helper/tool.py",
    "dl_helper/ag_trainer.py",
    "dl_helper/param_compression.py",
    "dl_helper/deep_gradient_compression.py",
    "dl_helper/idx_manager.py",
    "dl_helper/Transfer_alist_to_kaggle.py",
)

FORBIDDEN_FILES = (
    "setup.py",
    "requirements.txt",
)


def _tracked():
    proc = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                          text=True, encoding="utf-8", check=False)
    return proc.stdout.splitlines()


def test_no_forbidden_paths():
    tracked = _tracked()
    for path in tracked:
        for prefix in FORBIDDEN_PREFIXES:
            assert not path.startswith(prefix), f"禁止路径被跟踪: {path}"
        for name in FORBIDDEN_OLD_MODULES + FORBIDDEN_FILES:
            assert path != name, f"旧文件被跟踪: {path}"


def test_dl_helper_only_new_files():
    tracked = _tracked()
    dl_files = [p for p in tracked if p.startswith("dl_helper/")]
    assert dl_files, "dl_helper 无跟踪文件"
    for path in dl_files:
        assert path == "dl_helper/__init__.py" or path.startswith("dl_helper/training/"), (
            f"dl_helper 含非新平台文件: {path}"
        )


def test_no_tracked_wheels_or_binaries():
    tracked = _tracked()
    for path in tracked:
        if path.endswith((".whl", ".tar.gz", ".pyd", ".pyc", ".so", ".dll", ".dwg")):
            assert False, f"仓库跟踪了二进制/包: {path}"


def test_new_files_present():
    """新平台文件在工作树存在（未提交前不要求 git 跟踪）。"""
    for expected in (
        "pyproject.toml",
        "envs/kaggle_bootstrap.py",
        "notebook/kaggle_training_template.ipynb",
        "dl_helper/training/__init__.py",
        "dl_helper/training/config.py",
        "dl_helper/training/cli.py",
    ):
        assert os.path.exists(os.path.join(REPO, expected)), f"新文件缺失: {expected}"
