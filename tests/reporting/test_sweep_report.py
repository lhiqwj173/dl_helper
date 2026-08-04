"""任务 6.5：sweep 聚合报告（成功/暂停/失败、HTML escape、幂等）。"""
from __future__ import annotations

import json
import os

from dl_helper.training.reporting import generate_sweep_report


def _sweep_with_manifest(tmp_path, payload):
    sweep_dir = tmp_path / "sweeps" / "s1"
    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "sweep-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return str(sweep_dir)


def test_success_report_shows_ranking_and_best(tmp_path):
    sweep_dir = _sweep_with_manifest(tmp_path, {
        "id": "s1",
        "ranking": [
            {"rank": 1, "trial": "lr-1e-3", "run_id": "s1--lr-1e-3", "value": 0.9},
            {"rank": 2, "trial": "lr-3e-4", "run_id": "s1--lr-3e-4", "value": 0.8},
        ],
        "best_trial": "lr-1e-3",
    })
    index = generate_sweep_report(sweep_dir)
    content = open(index, encoding="utf-8").read()
    assert "Sweep Report" in content
    assert "lr-1e-3" in content
    assert "0.9" in content


def test_paused_report_no_best(tmp_path):
    sweep_dir = tmp_path / "sweeps" / "p1"
    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "pause-manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"sweep_id": "p1", "current_run_id": "p1--b"}, f, ensure_ascii=False)
    index = generate_sweep_report(str(sweep_dir))
    content = open(index, encoding="utf-8").read()
    assert "Sweep Report" in content


def test_report_escapes_html(tmp_path):
    sweep_dir = _sweep_with_manifest(tmp_path, {
        "id": "s1",
        "ranking": [{"rank": 1, "trial": "<script>x</script>", "run_id": "r", "value": 1.0}],
        "best_trial": "<b>best</b>",
    })
    index = generate_sweep_report(sweep_dir)
    content = open(index, encoding="utf-8").read()
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_report_idempotent(tmp_path):
    sweep_dir = _sweep_with_manifest(tmp_path, {
        "id": "s1", "ranking": [{"rank": 1, "trial": "a", "value": 0.5}], "best_trial": "a",
    })
    i1 = generate_sweep_report(sweep_dir)
    i2 = generate_sweep_report(sweep_dir)
    assert open(i1, encoding="utf-8").read() == open(i2, encoding="utf-8").read()
