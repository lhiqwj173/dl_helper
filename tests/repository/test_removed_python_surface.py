"""任务 9.1：旧 Python 训练体系导入面负向测试。"""
from __future__ import annotations

import subprocess
import sys

OLD_MODULES = [
    "dl_helper.trainer",
    "dl_helper.tester",
    "dl_helper.tracker",
    "dl_helper.train_param",
    "dl_helper.data",
    "dl_helper.scheduler",
    "dl_helper.tool",
    "dl_helper.ag_trainer",
    "dl_helper.models",
    "dl_helper.rl",
    "dl_helper.transforms",
    "dl_helper.tests",
    "dl_helper.acc",
    "dl_helper.param_compression",
    "dl_helper.deep_gradient_compression",
    "dl_helper.idx_manager",
    "dl_helper.Transfer_alist_to_kaggle",
    "dl_helper.simple_target",
    "dl_helper.train_folder_manager",
]

NEW_MODULES = [
    "dl_helper",
    "dl_helper.training",
    "dl_helper.training.config",
    "dl_helper.training.contracts",
    "dl_helper.training.task",
    "dl_helper.training.metrics",
    "dl_helper.training.engine",
    "dl_helper.training.backends.torch_backend",
    "dl_helper.training.backends.sklearn_backend",
    "dl_helper.training.cli",
]


def test_old_module_imports_fail():
    """所有旧模块导入必须 ModuleNotFoundError。"""
    for mod in OLD_MODULES:
        code = f"import {mod}\n"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, encoding="utf-8", check=False)
        assert proc.returncode != 0, f"{mod} 应导入失败但成功"


def test_new_module_imports_succeed():
    """新平台全部模块导入成功。"""
    for mod in NEW_MODULES:
        code = f"import {mod}\n"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, encoding="utf-8", check=False)
        assert proc.returncode == 0, f"{mod} 导入失败: {proc.stderr}"


def test_no_compat_shims():
    """不存在重导出/__getattr__ 兼容层。"""
    import dl_helper
    assert not hasattr(dl_helper, "trainer")
    assert not hasattr(dl_helper, "tester")
    assert not hasattr(dl_helper, "tracker")
