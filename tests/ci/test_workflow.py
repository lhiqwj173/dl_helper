"""任务 10.4：CI workflow 结构。"""
from __future__ import annotations

import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKFLOW = os.path.join(REPO, ".github", "workflows", "training-core.yml")


def _workflow():
    with open(WORKFLOW, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_matrix_windows_ubuntu_python310():
    wf = _workflow()
    strategy = wf["jobs"]["test"]["strategy"]
    os_list = strategy["matrix"]["os"]
    assert "windows-latest" in os_list
    assert "ubuntu-latest" in os_list
    assert "3.10" in strategy["matrix"]["python"]


def test_steps_include_gates():
    wf = _workflow()
    steps = [s.get("name", s.get("uses", "")) for s in wf["jobs"]["test"]["steps"]]
    for expected in ("Secret scan", "Full pytest", "Coverage gates", "Wheel audit", "Clean install probe"):
        assert expected in steps, f"缺少步骤 {expected}"


def test_no_gpu_or_network_steps():
    wf = _workflow()
    text = open(WORKFLOW, encoding="utf-8").read()
    assert "gpu" not in text.lower() and "nvidia" not in text.lower()
    assert "pip install torch" not in text  # 不联网重装 torch
    assert "pip install -e" in text  # 本地 editable 安装


def test_two_process_gloo_enforced():
    """OSR-001：CI 必须强制两进程 gloo 训练，且设置 DLH_ALLOW_MP。"""
    wf = _workflow()
    mp_steps = [s for s in wf["jobs"]["test"]["steps"] if "DLH_ALLOW_MP" in str(s)]
    assert mp_steps, "CI 缺少两进程 gloo 强制步骤"
    for step in mp_steps:
        assert step.get("env", {}).get("DLH_ALLOW_MP") == "1"
        run = step.get("run", "")
        assert "test_gloo_training.py" in run or "test_launcher.py" in run
