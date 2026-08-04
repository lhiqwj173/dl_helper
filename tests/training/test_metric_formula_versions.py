"""任务 2.4：公式版本不可变门禁 —— fixture 与实现版本一致且稳定。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.task import (
    MulticlassClassificationTask,
    MultilabelClassificationTask,
    RegressionTask,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "metric_goldens_v1.json"
)

# 不可变：任何影响数值的语义变更必须提高版本并新增 fixture
EXPECTED_FORMULA_VERSIONS = {
    "weighted_mean_loss": 1,
    "multiclass_accuracy": 1,
    "multiclass_balanced_accuracy": 1,
    "multiclass_precision_macro": 1,
    "multiclass_recall_macro": 1,
    "multiclass_f1_macro": 1,
    "multiclass_f1_weighted": 1,
    "multilabel_precision_macro": 1,
    "multilabel_recall_macro": 1,
    "multilabel_f1_macro": 1,
    "multilabel_f1_weighted": 1,
    "multilabel_precision_micro": 1,
    "multilabel_recall_micro": 1,
    "multilabel_f1_micro": 1,
    "multilabel_subset_accuracy": 1,
    "multilabel_hamming_loss": 1,
    "regression_mae": 1,
    "regression_mse": 1,
    "regression_r2_uniform": 1,
    "regression_r2_variance_weighted": 1,
}


def _all_definitions():
    tasks = [
        MulticlassClassificationTask(num_classes=3),
        MultilabelClassificationTask(num_labels=2),
        RegressionTask(num_targets=2),
    ]
    defs = {}
    for task in tasks:
        for name, d in task.metric_definitions.items():
            defs.setdefault(d.formula_id, d.formula_version)
            assert defs[d.formula_id] == d.formula_version, "同一 formula_id 版本漂移"
    return defs


def test_formula_versions_match_immutable_contract():
    defs = _all_definitions()
    assert set(defs) == set(EXPECTED_FORMULA_VERSIONS), (
        f"公式集合漂移: 实现 {sorted(set(defs))} vs 期望 {sorted(set(EXPECTED_FORMULA_VERSIONS))}"
    )
    for fid, ver in defs.items():
        assert ver == EXPECTED_FORMULA_VERSIONS[fid], f"{fid} 版本漂移 {ver}"


def test_fixture_formula_versions_match_implementation():
    defs = _all_definitions()
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    for case in fixture["cases"]:
        assert case["formula_version"] == 1
        for fid in case["formula_id"].values():
            assert defs[fid] == case["formula_version"], f"{fid} 与 fixture 版本不一致"


def test_formula_ids_in_fixture_are_implemented():
    defs = _all_definitions()
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    for case in fixture["cases"]:
        for fid in case["formula_id"].values():
            assert fid in defs, f"fixture 引用未实现 formula_id {fid}"


def test_fixture_generated_by_documented_script():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    assert fixture["generated_by"] == "generate_metric_goldens.py"
    assert fixture["scikit_learn_version"] == "1.6.1"
