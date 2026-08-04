"""任务 1.4：双后端公共合同与导入边界测试。"""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch

from dl_helper.training.contracts import (
    DataIdentity,
    EstimatorBatch,
    LoaderDataModule,
    LossResult,
    MetricDefinition,
    PredictedBatch,
    PreparedBatch,
    ResumableMapDataModule,
    SklearnExperiment,
    SchedulerBinding,
    TorchExperiment,
    contract_splits,
    validate_backend_match,
    validate_experiment,
    validate_json_value,
)


def test_json_value_accepts_scalars():
    for v in [None, True, 1, 1.5, "x"]:
        validate_json_value(v)


def test_json_value_rejects_nan_inf():
    with pytest.raises(ValueError):
        validate_json_value(float("nan"))
    with pytest.raises(ValueError):
        validate_json_value({"a": [1.0, float("inf")]})


def test_json_value_rejects_non_string_keys():
    with pytest.raises(ValueError):
        validate_json_value({1: "a"})


def test_data_identity_non_empty():
    with pytest.raises(ValueError):
        DataIdentity("", "v1", "fp1")
    with pytest.raises(ValueError):
        DataIdentity("name", "", "fp1")
    with pytest.raises(ValueError):
        DataIdentity("name", "v1", "")


def test_incremental_train_fingerprint_restores_source_state():
    class Source:
        def __init__(self):
            self.cursor = 7

        def iter_epoch(self, epoch):
            assert epoch == 0
            self.cursor = 99
            yield EstimatorBatch(
                features=np.array([[1.0, 2.0]]),
                targets=np.array([1]),
                sample_count=1,
            )

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = int(state["cursor"])

    class DataModule:
        def __init__(self):
            self.source = Source()

        def identity(self):
            return DataIdentity("incremental", "1", "identity-fp")

        def incremental_train_data(self):
            return self.source

    dm = DataModule()
    splits = contract_splits(dm, dm.identity())
    assert splits["train"]["fingerprint"]
    assert dm.source.cursor == 7


def test_loader_datamodule():
    ds = torch.utils.data.TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    dm = LoaderDataModule(DataIdentity("d", "v1", "fp"), loader)
    assert dm.supports_mid_epoch_resume is False
    assert dm.state_dict() == {}
    with pytest.raises(ValueError):
        dm.load_state_dict({"epoch": 1})


def test_resumable_datamodule_state_roundtrip():
    ds = torch.utils.data.TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,)))

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.tensor(ys)

    dm = ResumableMapDataModule(
        DataIdentity("d", "v1", "fp"), lambda: ds, collate, batch_size=4, shuffle=True
    )
    state = dm.state_dict()
    dm2 = ResumableMapDataModule(
        DataIdentity("d", "v1", "fp"), lambda: ds, collate, batch_size=4, shuffle=True
    )
    dm2.load_state_dict(state)
    assert dm2.state_dict() == state


def test_prepared_batch_validation():
    with pytest.raises(ValueError):
        PreparedBatch(inputs=torch.randn(2), targets=torch.tensor([0, 1]), sample_count=0)
    with pytest.raises(ValueError):
        PreparedBatch(
            inputs=torch.randn(2), targets=torch.tensor([0, 1]), sample_count=2,
            sample_weight=torch.tensor([1.0]),  # 长度不一致
        )
    with pytest.raises(ValueError):
        PreparedBatch(
            inputs=torch.randn(2), targets=torch.tensor([0, 1]), sample_count=2,
            sample_weight=torch.tensor([1.0, float("nan")]),
        )
    with pytest.raises(ValueError):
        PreparedBatch(
            inputs=torch.randn(2), targets=torch.tensor([0, 1]), sample_count=2,
            sample_weight=torch.tensor([-1.0, 1.0]),
        )
    with pytest.raises(ValueError):
        PreparedBatch(
            inputs=torch.randn(2), targets=torch.tensor([0, 1]), sample_count=2,
            sample_weight=torch.tensor([0.0, 0.0]),
        )


def test_loss_result_validation():
    num = torch.tensor(1.0, requires_grad=True)
    ok = LossResult(num, 2.0)
    assert ok is not None
    with pytest.raises(ValueError):
        LossResult(torch.tensor(1.0), 2.0)  # numerator 无梯度
    with pytest.raises(ValueError):
        LossResult(torch.tensor(float("nan"), requires_grad=True), 2.0)
    with pytest.raises(ValueError):
        LossResult(torch.tensor(1.0, requires_grad=True), 0.0)
    with pytest.raises(ValueError):
        LossResult(torch.tensor(1.0, requires_grad=True), -1.0)


def test_predicted_batch_validation():
    tgt = np.array([0, 1], dtype=np.int64)
    pred = np.array([1, 1], dtype=np.int64)
    with pytest.raises(ValueError):
        PredictedBatch(targets=tgt, predictions=np.array([1], dtype=np.int64), sample_count=2)
    with pytest.raises(ValueError):
        PredictedBatch(targets=tgt, predictions=pred, sample_count=2,
                       scores=np.array([0.5, 0.5, 0.5]))  # 样本维不一致
    with pytest.raises(ValueError):
        PredictedBatch(targets=tgt, predictions=pred, sample_count=2,
                       scores=np.array([0.5, float("nan")]))
    with pytest.raises(ValueError):
        PredictedBatch(targets=tgt, predictions=pred, sample_count=2,
                       sample_weight=np.array([0.0, 0.0]))  # 权重和必须为正
    with pytest.raises(TypeError):
        PredictedBatch(targets=tgt, predictions=pred, sample_count=2,
                       sample_weight=np.array([1, 0], dtype=np.int64))  # 非浮点权重


def test_metric_definition_validation():
    def make(**kw):
        base = dict(
            name="acc", direction="max", formula_id="x", formula_version=1,
            averaging="macro", sample_weight_policy="supported", zero_division="zero",
            exact=True, evaluation_scope="full", parameters={}, implementation="builtin_verified",
        )
        base.update(kw)
        return MetricDefinition(**base)

    make()  # 合法
    with pytest.raises(ValueError):
        make(formula_version=0)
    with pytest.raises(ValueError):
        make(direction="up")
    with pytest.raises(ValueError):
        make(averaging="bad")
    with pytest.raises(ValueError):
        make(parameters={"x": float("nan")})
    with pytest.raises(ValueError):
        make(evaluation_scope="sampled", exact=True)


def test_scheduler_binding():
    ok = SchedulerBinding(scheduler="s", interval="epoch", monitor=None)
    assert ok is not None
    with pytest.raises(ValueError):
        SchedulerBinding(scheduler="s", interval="validation_metric", monitor=None)
    with pytest.raises(ValueError):
        SchedulerBinding(scheduler="s", interval="epoch", monitor="val/x")


def test_experiment_validation():
    exp = TorchExperiment(
        name="e", backend="torch",
        model_factory=lambda: torch.nn.Linear(3, 2),
        datamodule_factory=lambda: None,
        task_factory=lambda: None,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.01),
        scheduler_factory=lambda opt: None,
        model_config={},
    )
    validate_experiment(exp)
    validate_backend_match(exp, "torch")
    with pytest.raises(ValueError):
        validate_backend_match(exp, "sklearn")
    with pytest.raises(ValueError):
        validate_experiment(
            TorchExperiment(name="", backend="torch", model_factory=1, datamodule_factory=1,
                            task_factory=1, optimizer_factory=1, scheduler_factory=1, model_config={})
        )
    with pytest.raises(TypeError):
        validate_experiment(object())


def test_sklearn_experiment_validation():
    exp = SklearnExperiment(
        name="s", backend="sklearn",
        estimator_factory=lambda: None,
        datamodule_factory=lambda: None,
        task_factory=lambda: None,
        model_config={},
    )
    validate_experiment(exp)
    validate_backend_match(exp, "sklearn")


def test_import_boundaries_no_heavy_imports():
    """导入 dl_helper.training 不得触发 transformers/torchmetrics/RL/GUI/网络。"""
    heavy = {"transformers", "torchmetrics", "stable_baselines3", "imitation", "autogluon"}
    for mod in heavy:
        assert mod not in sys.modules, f"import dl_helper.training 触发了 {mod}"
