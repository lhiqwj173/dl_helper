"""任务 9.4：构建 wheel，检查 RECORD/METADATA 与旧包。"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_wheel(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(tmp_path)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = [f for f in os.listdir(str(tmp_path)) if f.endswith(".whl")]
    assert len(wheels) == 1
    return os.path.join(str(tmp_path), wheels[0])


def test_wheel_version_and_contents(tmp_path):
    wheel = _build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        # 元数据版本
        metadata = [n for n in names if n.endswith("METADATA")]
        assert metadata
        content = zf.read(metadata[0]).decode("utf-8")
        assert "Version: 1.0.0" in content
        # 只含新包代码
        assert any("dl_helper/training/" in n for n in names)
        # 不含旧包
        for bad in ("dl_helper/trainer.py", "dl_helper/tester.py", "dl_helper/models/",
                    "dl_helper/rl/", "dl_helper/tracker.py"):
            assert not any(bad in n for n in names), f"wheel 含旧文件: {bad}"
        # 不含旧重依赖
        for dep in ("torchmetrics", "stable-baselines3", "rllib", "autogluon"):
            assert dep.lower() not in content.lower(), f"wheel 含旧依赖: {dep}"
        # 无 alist/legacy extra
        assert "alist" not in content.lower() or "Provides-Extra: alist" not in content
        assert "legacy" not in content.lower() or "Provides-Extra: legacy" not in content


def test_wheel_dependencies(tmp_path):
    wheel = _build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as zf:
        metadata = [n for n in zf.namelist() if n.endswith("METADATA")][0]
        content = zf.read(metadata).decode("utf-8")
    for dep in ("torch", "accelerate", "numpy", "matplotlib", "scikit-learn",
                "safetensors", "PyYAML", "joblib", "requests"):
        assert f"Requires-Dist: {dep}" in content, f"缺核心依赖 {dep}"
