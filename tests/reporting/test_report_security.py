"""任务 4.5：报告 HTML escape 与安全。"""
from __future__ import annotations

import json
import os

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.reporting import generate_run_report, generate_sweep_report


def _fake_run(tmp_path, user_text=None):
    """构造含用户文本的 run Artifact。"""
    run_dir = str(tmp_path / "runs" / "sec")
    layout = RunLayout(run_dir)
    layout.ensure()
    layout.write_json("run-manifest.json", {"status": "succeeded", "run_id": "sec"})
    layout.write_json("metrics/summary.json", {
        "run_id": user_text or "sec", "backend": "torch", "status": "succeeded",
        "epoch": 1, "global_step": 8,
    })
    with open(layout.metrics_jsonl, "w", encoding="utf-8") as f:
        json.dump({"stage": "train", "epoch": 0, "global_step": 8,
                   "metrics": {"train/loss": 0.5}}, f, ensure_ascii=False)
        f.write("\n")
    return layout


def test_html_escape_user_content(tmp_path):
    payload = "<script>alert('x')</script>"
    layout = _fake_run(tmp_path, payload)
    index = generate_run_report(layout.run_dir)
    content = open(index, encoding="utf-8").read()
    # 原始 script 标签不存在
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_report_path_escape_rejected(tmp_path):
    from dl_helper.training.reporting import _safe_path
    run_dir = str(tmp_path / "runs" / "sec")
    layout = RunLayout(run_dir)
    layout.ensure()
    os.makedirs(layout.path("metrics"), exist_ok=True)
    with open(layout.metrics_jsonl, "w", encoding="utf-8") as f:
        f.write("\n")
    # 逃逸引用拒绝
    import pytest
    with pytest.raises(ValueError):
        _safe_path(run_dir, "../../../outside")


def test_sweep_report_escapes(tmp_path):
    sweep_dir = str(tmp_path / "sweeps" / "sec")
    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "sweep-manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "id": "sweep-sec",
            "ranking": [{"rank": 1, "trial": "<img src=x onerror=alert(1)>", "value": 0.9}],
            "best_trial": "<b>best</b>",
        }, f, ensure_ascii=False)
    index = generate_sweep_report(sweep_dir)
    content = open(index, encoding="utf-8").read()
    assert "<img src=x" not in content
    assert "&lt;img" in content
    assert "<b>best</b>" not in content  # 已转义


def test_sweep_report_idempotent(tmp_path):
    sweep_dir = str(tmp_path / "sweeps" / "idem")
    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "sweep-manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"id": "s", "ranking": [{"rank": 1, "trial": "a", "value": 0.5}]}, f, ensure_ascii=False)
    i1 = generate_sweep_report(sweep_dir)
    i2 = generate_sweep_report(sweep_dir)
    assert open(i1, encoding="utf-8").read() == open(i2, encoding="utf-8").read()
