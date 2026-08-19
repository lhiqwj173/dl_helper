"""任务 3.2：Torch 组件预检负向用例。"""
from __future__ import annotations

import pickle

import pytest
import torch

from dl_helper.training.backends.torch_backend import (
    TorchBackendError,
    build_torch_components,
    validate_fresh_components,
)
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.contracts import DataIdentity, LoaderDataModule, SchedulerBinding, TorchExperiment
from dl_helper.training.task import MulticlassClassificationTask


def _cfg(**patch):
    schema = default_schema()
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(schema.get(k), dict):
            schema[k] = {**schema[k], **v}
        else:
            schema[k] = v
    return parse_config(schema)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def _dm(mid_epoch=False):
    ds = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,)))
    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    val_ds = torch.utils.data.TensorDataset(torch.randn(4, 4), torch.randint(0, 3, (4,)))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4)
    return LoaderDataModule(DataIdentity("d", "v1", "fp"), loader,
                            val_dataloader=val_loader, nominal_train_batch_size=8)


def _experiment(model_factory=None, dm_factory=None, task_factory=None, sched_factory=None):
    return TorchExperiment(
        name="e", backend="torch",
        model_factory=model_factory or MLP,
        datamodule_factory=dm_factory or _dm,
        task_factory=task_factory or (lambda: MulticlassClassificationTask(num_classes=3)),
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler_factory=sched_factory or (lambda o: None),
        model_config={},
    )


def test_valid_components_pass():
    exp = _experiment()
    model, dm, task, opt, sched = build_torch_components(exp, _cfg())
    validate_fresh_components(model, dm, task, opt, sched, _cfg())


def test_model_on_cuda_rejected():
    model = MLP()
    model.fc.weight.data = model.fc.weight.data.cuda() if torch.cuda.is_available() else model.fc.weight.data
    if not torch.cuda.is_available():
        pytest.skip("无 CUDA")
    exp = _experiment(model_factory=lambda: model)
    model, dm, task, opt, sched = build_torch_components(exp, _cfg())
    with pytest.raises(TorchBackendError):
        validate_fresh_components(model, dm, task, opt, sched, _cfg())


def test_ddp_wrapped_model_rejected():
    from torch.nn.parallel import DistributedDataParallel
    model = MLP()
    try:
        ddp = DistributedDataParallel(model)
    except Exception:
        pytest.skip("DDP 需要初始化")
    exp = _experiment(model_factory=lambda: ddp)
    model, dm, task, opt, sched = build_torch_components(exp, _cfg())
    with pytest.raises(TorchBackendError):
        validate_fresh_components(model, dm, task, opt, sched, _cfg())


def test_scheduler_not_serializable_rejected():
    class NotPicklable:
        def __reduce__(self):
            raise TypeError("no pickle")

    sched = lambda o: SchedulerBinding(scheduler=NotPicklable(), interval="epoch", monitor=None)  # noqa: E731
    exp = _experiment(sched_factory=sched)
    model, dm, task, opt, sched_b = build_torch_components(exp, _cfg())
    with pytest.raises(TorchBackendError):
        validate_fresh_components(model, dm, task, opt, sched_b, _cfg())


def test_datamodule_missing_member_rejected():
    class BadDM:
        pass

    exp = _experiment(dm_factory=lambda: BadDM())
    model, dm, task, opt, sched = build_torch_components(exp, _cfg())
    with pytest.raises(TorchBackendError):
        validate_fresh_components(model, dm, task, opt, sched, _cfg())


def test_every_optimizer_steps_requires_mid_resume():
    cfg = _cfg(checkpoint={"every_epochs": 1, "every_optimizer_steps": 5, "keep_last": 2})
    # LoaderDataModule 不支持中途恢复
    model, dm, task, opt, sched = build_torch_components(_experiment(), cfg)
    with pytest.raises(TorchBackendError):
        validate_fresh_components(model, dm, task, opt, sched, cfg)


def test_budget_policy_requires_mid_epoch_resume():
    from dl_helper.training.platform import ExecutionPolicy
    cfg = _cfg()
    model, dm, task, opt, sched = build_torch_components(_experiment(), cfg)
    # D-003：运行预算来自 ExecutionPolicy；有预算时 DataModule 必须支持中途恢复
    with pytest.raises(TorchBackendError):
        validate_fresh_components(
            model, dm, task, opt, sched, cfg,
            execution_policy=ExecutionPolicy(platform="local", max_minutes=30.0,
                                             shutdown_grace_minutes=5.0))


def test_selection_missing_with_val_rejected():
    cfg = _cfg(selection=None)
    model, dm, task, opt, sched = build_torch_components(_experiment(), cfg)
    with pytest.raises(Exception):
        validate_fresh_components(model, dm, task, opt, sched, cfg)
