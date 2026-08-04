"""任务 2.1/2.4：内置指标公式元数据一致性（formula_id/version、方向、sample-weight）。"""
from __future__ import annotations

import pytest

from dl_helper.training.task import (
    MulticlassClassificationTask,
    MultilabelClassificationTask,
    RegressionTask,
    SklearnMulticlassTask,
    SklearnMultilabelTask,
    SklearnRegressionTask,
)

# 每个公式版本必须唯一且稳定
FORMULA_VERSIONS: dict[str, int] = {}


def _collect(task):
    for name, d in task.metric_definitions.items():
        assert d.name == name
        assert d.implementation == "builtin_verified"
        assert d.evaluation_scope == "full"
        assert d.exact is True
        key = (d.formula_id, d.formula_version)
        if key in FORMULA_VERSIONS:
            # 同一 (formula_id, version) 只能对应一个指标名
            assert FORMULA_VERSIONS[key] == d.name, "formula_id/version 冲突"
        else:
            FORMULA_VERSIONS[key] = d.name
        # sample-weight 语义
        assert d.sample_weight_policy == "supported"


def test_torch_task_definitions_stable():
    _collect(MulticlassClassificationTask(num_classes=3))
    _collect(MultilabelClassificationTask(num_labels=2))
    _collect(RegressionTask(num_targets=2))
    _collect(SklearnMulticlassTask(classes=[0, 1, 2]))
    _collect(SklearnMultilabelTask(num_labels=2))
    _collect(SklearnRegressionTask(num_targets=1))


def test_metric_directions():
    mc = MulticlassClassificationTask(num_classes=3).metric_definitions
    assert mc["accuracy"].direction == "max"
    assert mc["loss"].direction == "min"
    assert mc["f1_weighted"].direction == "max"
    ml = MultilabelClassificationTask(num_labels=2).metric_definitions
    assert ml["hamming_loss"].direction == "min"
    assert ml["subset_accuracy"].direction == "max"
    reg = RegressionTask(num_targets=1).metric_definitions
    assert reg["mae"].direction == "min"
    assert reg["r2"].direction == "max"


def test_loss_definition_shared_across_tasks():
    for task in (MulticlassClassificationTask(num_classes=3),
                 MultilabelClassificationTask(num_labels=2),
                 RegressionTask(num_targets=1)):
        loss = task.metric_definitions["loss"]
        assert loss.formula_id == "weighted_mean_loss"
        assert loss.formula_version == 1


def test_sampled_scope_forbidden_from_full_defs():
    for task in (MulticlassClassificationTask(num_classes=3),
                 MultilabelClassificationTask(num_labels=2),
                 RegressionTask(num_targets=1)):
        for d in task.metric_definitions.values():
            assert d.evaluation_scope == "full"
