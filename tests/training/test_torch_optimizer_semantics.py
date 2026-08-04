"""任务 3.3：精确梯度归一化 —— 不等尾批/sample weight 下与全局加权金标一致。"""
from __future__ import annotations

import torch

from dl_helper.training.backends.torch_backend import _normalize_gradients
from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
from dl_helper.training.task import MulticlassClassificationTask


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def _golden_gradient(model, task, batches, weights):
    """全局加权均值梯度金标：sum_m grad(numerator_m) / sum_m den_m。"""
    model.zero_grad()
    denom_total = 0.0
    for (inputs, targets), w in zip(batches, weights):
        prepared = task.prepare_batch((inputs, targets, w), "train")
        outputs = task.forward(model, prepared)
        loss = task.loss(outputs, prepared)
        loss.numerator.backward()
        denom_total += float(loss.denominator.detach())
    golden = {name: p.grad.clone() / denom_total
              for name, p in model.named_parameters() if p.grad is not None}
    return golden, denom_total


def test_accumulation_denominator_normalization():
    torch.manual_seed(0)
    model = _MLP()
    task = MulticlassClassificationTask(num_classes=3)
    # 两个 micro-batch，batch 大小不同（不等尾批），权重不同
    b1 = (torch.randn(4, 4), torch.randint(0, 3, (4,)))
    b2 = (torch.randn(2, 4), torch.randint(0, 3, (2,)))  # 不等尾批
    w1 = torch.tensor([1.0, 2.0, 0.5, 1.5])
    w2 = torch.tensor([2.0, 1.0])
    batches = [b1, b2]
    weights = [w1, w2]

    golden, denom_total = _golden_gradient(model, task, batches, weights)

    # 模拟 backend 步骤：Accelerate 的 1/acc_steps 被 backward(numerator * acc_steps) 抵消，
    # 得到 sum_m grad(numerator_m)，再归一化 scale = world/global_denom
    model.zero_grad()
    acc_steps = 2
    window_denom = 0.0
    for (inputs, targets), w in zip(batches, weights):
        prepared = task.prepare_batch((inputs, targets, w), "train")
        outputs = task.forward(model, prepared)
        loss = task.loss(outputs, prepared)
        # *acc_steps 抵消 Accelerate 的 /acc_steps → 净 backward(numerator)
        loss.numerator.backward()
        window_denom += float(loss.denominator.detach())

    class _Acc:
        num_processes = 1

    _normalize_gradients(model, _Acc(), window_denom)
    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, golden[name], atol=1e-6), (
            f"{name}: backend {p.grad} vs golden {golden[name]}"
        )
    assert abs(window_denom - denom_total) < 1e-9


def test_unequal_micro_batch_single_step():
    """最后不足窗口：单个 micro-batch 也执行同一公式。"""
    torch.manual_seed(1)
    model = _MLP()
    task = MulticlassClassificationTask(num_classes=3)
    b1 = (torch.randn(5, 4), torch.randint(0, 3, (5,)))
    w1 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    golden, denom_total = _golden_gradient(model, task, [b1], [w1])

    model.zero_grad()
    prepared = task.prepare_batch((b1[0], b1[1], w1), "train")
    outputs = task.forward(model, prepared)
    loss = task.loss(outputs, prepared)
    loss.numerator.backward()

    class _Acc:
        num_processes = 1

    _normalize_gradients(model, _Acc(), float(loss.denominator.detach()))
    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, golden[name], atol=1e-6), name
