"""补充 reporting 分支覆盖（OSR-009 覆盖率门禁）。"""
from __future__ import annotations

import json
import os

from dl_helper.training.reporting import (
    _render_confusion_image,
    _render_context_table,
    _stage_metrics,
    generate_sweep_report,
)


def test_stage_metrics_skips_blank_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"stage": "train"}\n\n{"stage": "val"}\n', encoding="utf-8")
    out = _stage_metrics(str(path))
    assert set(out) == {"train", "val"}


def test_stage_metrics_missing_file(tmp_path):
    out = _stage_metrics(str(tmp_path / "nope.jsonl"))
    assert out == {}


def test_render_context_table_empty(tmp_path):
    assert _render_context_table({"unknown_key": 1}) == ""


def test_render_confusion_image_missing(tmp_path):
    assets = tmp_path / "assets"
    assert _render_confusion_image(str(tmp_path), str(assets), "val", {}) == ""


def test_sweep_report_ranking_not_list(tmp_path):
    s = tmp_path / "s"
    os.makedirs(s, exist_ok=True)
    json.dump({"ranking": "not-a-list", "best_trial": "a"},
              open(s / "sweep-manifest.json", "w", encoding="utf-8"))
    idx = generate_sweep_report(str(s))
    assert os.path.exists(idx)


def test_sweep_report_multiple_ranking(tmp_path):
    s = tmp_path / "s2"
    os.makedirs(s, exist_ok=True)
    json.dump({"ranking": [{"rank": 1, "trial": "a", "value": 0.5},
                           {"rank": 2, "trial": "b", "value": 0.4}],
               "best_trial": "a"},
              open(s / "sweep-manifest.json", "w", encoding="utf-8"))
    idx = generate_sweep_report(str(s))
    content = open(idx, encoding="utf-8").read()
    assert "best trial: a" in content
