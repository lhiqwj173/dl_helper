"""分支补充：torch scheduler 分支、launcher、contracts 校验分支。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dl_helper.training.backends.torch_backend import (
    TorchBackendError,
    build_torch_components,
    validate_fresh_components,
)
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.contracts import (
    DataIdentity,
    DataModule,
    LoaderDataModule,
    SchedulerBinding,
    TorchExperiment,
    validate_data_identity,
    validate_torch_task,
)
from dl_helper.training.task import MulticlassClassificationTask


def _cfg(**patch):
    schema = default_schema()
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    for k, v in patch.items():
        schema[k] = v
    return parse_config(schema)


def _exp(sched_factory=None):
    return TorchExperiment(
        name="e", backend="torch",
        model_factory=lambda: torch.nn.Linear(4, 3),
        datamodule_factory=lambda: LoaderDataModule(
            DataIdentity("d", "v1", "fp"), torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,))), batch_size=8),
            val_dataloader=torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.randn(4, 4), torch.randint(0, 3, (4,))), batch_size=4),
            nominal_train_batch_size=8),
        task_factory=lambda: MulticlassClassificationTask(num_classes=3),
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler_factory=sched_factory or (lambda o: None),
        model_config={},
    )


def test_validation_metric_scheduler_monitor_missing():
    """validation_metric scheduler 的 monitor 必须由 Task 产生。"""
    sched = lambda o: SchedulerBinding(scheduler=torch.optim.lr_scheduler.StepLR(o, 1),
                                       interval="validation_metric", monitor="val/nonexistent")
    exp = _exp(sched_factory=sched)
    model, dm, task, opt, sched_b = build_torch_components(exp, _cfg())
    with pytest.raises(TorchBackendError):
        validate_fresh_components(model, dm, task, opt, sched_b, _cfg())


def test_validation_metric_scheduler_valid_monitor():
    sched = lambda o: SchedulerBinding(scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(o),
                                       interval="validation_metric", monitor="val/loss")
    exp = _exp(sched_factory=sched)
    model, dm, task, opt, sched_b = build_torch_components(exp, _cfg())
    validate_fresh_components(model, dm, task, opt, sched_b, _cfg())  # 不抛


def test_every_optimizer_steps_requires_resumable_dm():
    cfg = _cfg(checkpoint={"every_epochs": None, "every_optimizer_steps": 5,
                           "keep_last": 2})
    model, dm, task, opt, sched_b = build_torch_components(_exp(), cfg)
    with pytest.raises(TorchBackendError):
        validate_fresh_components(model, dm, task, opt, sched_b, cfg)


class TestContractBranches2:
    def test_validate_data_identity_type(self):
        with pytest.raises(TypeError):
            validate_data_identity("not-an-identity")

    def test_validate_torch_task_missing_member(self):
        class Bad:
            pass
        with pytest.raises(TypeError):
            validate_torch_task(Bad())

    def test_loader_dm_rejects_state(self):
        dm = LoaderDataModule(DataIdentity("d", "v1", "f"),
                              torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(4), torch.randn(4))))
        with pytest.raises(ValueError):
            dm.load_state_dict({"epoch": 1})

    def test_torch_task_metric_defs_mismatch(self):
        task = MulticlassClassificationTask(num_classes=3)
        # 复制但改名 → 校验失败
        bad = type("BadTask", (), dict(
            name="x", metric_definitions={"wrong": next(iter(task.metric_definitions.values()))},
            metric_state=lambda s: None, update_metrics=lambda s, p: None,
            prediction_arrays=lambda p: {}, report_kind=lambda: "x",
            prepare_batch=lambda b, s: None, forward=lambda m, p: None,
            loss=lambda o, p: None, to_predicted_batch=lambda o, p: None))()
        with pytest.raises(ValueError):
            validate_torch_task(bad)
