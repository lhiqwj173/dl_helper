"""任务 6.5：未舍入排名、best 与并列稳定。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.sweep import SweepError, SweepManifest, TrialSpec, _compute_ranking


def _trial(name):
    return TrialSpec(name=name, variant=f"/tmp/{name}.yaml", resolved_config=f"/tmp/{name}.yaml")


def _manifest():
    return SweepManifest(
        schema_version=1, sweep_id="s1", experiment="e",
        base_config="/tmp/base.yaml", comparison_metric="val/f1_macro", mode="max",
        trials=[_trial("a"), _trial("b"), _trial("c")],
    )


def _write_run(output_root, run_id, metric, value):
    run_dir = os.path.join(output_root, "runs", run_id)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    # 最终 val 指标写入 summary（排名读取 summary 而非 metrics.jsonl）
    metric_name = metric.removeprefix("val/")
    with open(os.path.join(run_dir, "metrics", "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "stage_metrics": {"val": {metric: value}},
            "metric_definitions": {
                metric_name: {"formula_id": "f", "formula_version": 1,
                              "direction": "max" if "f1" in metric else "min",
                              "exact": True, "evaluation_scope": "full",
                              "averaging": "macro", "sample_weight_policy": "supported"},
            },
        }, f, ensure_ascii=False)


def test_ranking_max_mode(tmp_path):
    m = _manifest()
    configs = [(t, parse_config(default_schema())) for t in m.trials]
    _write_run(str(tmp_path), "s1--a", "val/f1_macro", 0.5)
    _write_run(str(tmp_path), "s1--b", "val/f1_macro", 0.9)
    _write_run(str(tmp_path), "s1--c", "val/f1_macro", 0.7)
    ranking = _compute_ranking(m, configs, str(tmp_path))
    assert [r["trial"] for r in ranking] == ["b", "c", "a"]
    assert ranking[0]["rank"] == 1


def test_ranking_min_mode(tmp_path):
    m = SweepManifest(schema_version=1, sweep_id="s1", experiment="e", base_config="/tmp/b",
                      comparison_metric="val/loss", mode="min",
                      trials=[_trial("a"), _trial("b")])
    configs = [(t, parse_config(default_schema())) for t in m.trials]
    _write_run(str(tmp_path), "s1--a", "val/loss", 1.5)
    _write_run(str(tmp_path), "s1--b", "val/loss", 0.5)
    ranking = _compute_ranking(m, configs, str(tmp_path))
    assert ranking[0]["trial"] == "b"


def test_tie_broken_by_yaml_order(tmp_path):
    m = _manifest()
    configs = [(t, parse_config(default_schema())) for t in m.trials]
    for t in m.trials:
        _write_run(str(tmp_path), f"s1--{t.name}", "val/f1_macro", 0.5)
    ranking = _compute_ranking(m, configs, str(tmp_path))
    assert [r["trial"] for r in ranking] == ["a", "b", "c"]  # YAML 顺序稳定


def test_uses_unrounded_raw_value(tmp_path):
    """显示舍入并列但原始值不同 → 排名用原始值。"""
    m = SweepManifest(schema_version=1, sweep_id="s1", experiment="e", base_config="/tmp/b",
                      comparison_metric="val/f1_macro", mode="max",
                      trials=[_trial("a"), _trial("b")])
    configs = [(t, parse_config(default_schema())) for t in m.trials]
    _write_run(str(tmp_path), "s1--a", "val/f1_macro", 0.951)
    _write_run(str(tmp_path), "s1--b", "val/f1_macro", 0.950)
    ranking = _compute_ranking(m, configs, str(tmp_path))
    assert ranking[0]["trial"] == "a"  # 0.951 > 0.950 即使都显示 0.95


def test_missing_value_fails(tmp_path):
    m = _manifest()
    configs = [(t, parse_config(default_schema())) for t in m.trials]
    _write_run(str(tmp_path), "s1--a", "val/f1_macro", 0.5)
    _write_run(str(tmp_path), "s1--b", "val/f1_macro", 0.9)
    # c 缺失
    with pytest.raises(SweepError):
        _compute_ranking(m, configs, str(tmp_path))


def test_non_finite_value_fails(tmp_path):
    m = SweepManifest(schema_version=1, sweep_id="s1", experiment="e", base_config="/tmp/b",
                      comparison_metric="val/f1_macro", mode="max",
                      trials=[_trial("a"), _trial("b")])
    configs = [(t, parse_config(default_schema())) for t in m.trials]
    _write_run(str(tmp_path), "s1--a", "val/f1_macro", 0.5)
    _write_run(str(tmp_path), "s1--b", "val/f1_macro", float("nan"))
    with pytest.raises(SweepError):
        _compute_ranking(m, configs, str(tmp_path))
