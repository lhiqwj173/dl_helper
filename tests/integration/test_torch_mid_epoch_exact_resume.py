"""任务 OSR-004：Torch 中途精确恢复 —— 不重复 step、auto 损坏必须失败。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.checkpoint import CheckpointError, read_latest
from dl_helper.training.config import default_schema, parse_config


def _cfg(run_id, max_epochs, resume="none", every_steps=None, max_minutes=None):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 30, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["checkpoint"]["every_optimizer_steps"] = every_steps
    schema["checkpoint"]["keep_last"] = 10
    schema["checkpoint"]["resume"] = resume
    schema["runtime"]["max_minutes"] = max_minutes
    schema["runtime"]["shutdown_grace_minutes"] = 2
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def _metrics(layout):
    return [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]


class _AdvancingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def test_mid_epoch_resume_does_not_repeat_steps(tmp_path):
    """batch 检查点记录真实位置；resume 不重复已完成优化 step。"""
    run_dir = str(tmp_path / "runs" / "mid-epoch")
    cfg1 = _cfg("mid-epoch", max_epochs=1, resume="auto", every_steps=4, max_minutes=10)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    r1 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg1, layout1, 0, 1, "auto",
                    budget_monotonic=_AdvancingClock())
    assert r1.status == "preempted"
    # 检查点记录了 batch_in_epoch 位置
    latest = read_latest(layout1.path("checkpoints"))
    ckpt_dir = os.path.join(layout1.path("checkpoints"), latest["path"])
    manifest = json.load(open(os.path.join(ckpt_dir, "checkpoint-manifest.json"), encoding="utf-8"))
    assert manifest["batch_in_epoch"] > 0

    # resume：从 epoch 1 位置继续，不重复 epoch 0 的 8 个 step
    cfg2 = _cfg("mid-epoch", max_epochs=2, resume="auto", every_steps=4)
    layout2 = RunLayout(run_dir)
    layout2.ensure()
    r2 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg2, layout2, 0, 1, "auto")
    assert r2.status == "succeeded"
    # 两 epoch 各 8 batch → 16 step；若重复 epoch0 会 >16
    assert r2.global_step == 16
    # 指标不重复 epoch 0 train 记录
    metrics = _metrics(layout2)
    train_records = [m for m in metrics if m["stage"] == "train"]
    epochs = [m["epoch"] for m in train_records]
    assert len(set(epochs)) == 2  # 只有 epoch 0,1 各一次


def test_auto_resume_corrupt_latest_fails(tmp_path):
    """auto 恢复遇到损坏 latest 必须失败，不静默从零重训。"""
    run_dir = str(tmp_path / "runs" / "corrupt-auto")
    cfg1 = _cfg("corrupt-auto", max_epochs=1, resume="auto", every_steps=4, max_minutes=10)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    # 预算预占：产生 checkpoint + pause 终态，无 success 终态
    run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg1, layout1, 0, 1, "auto",
               budget_monotonic=_AdvancingClock())
    # 篡改 latest 指向的 checkpoint
    latest = read_latest(layout1.path("checkpoints"))
    ckpt_dir = os.path.join(layout1.path("checkpoints"), latest["path"])
    with open(os.path.join(ckpt_dir, "engine-state.json"), "w", encoding="utf-8") as f:
        f.write("{corrupt json")
    with pytest.raises((CheckpointError, Exception)):
        run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg1, layout1, 0, 1, "auto")


def test_auto_resume_no_checkpoint_starts_fresh(tmp_path):
    """auto 且确实无 latest → 从零开始。"""
    run_dir = str(tmp_path / "runs" / "auto-fresh")
    cfg = _cfg("auto-fresh", max_epochs=1, resume="auto", every_steps=4)
    layout = RunLayout(run_dir)
    layout.ensure()
    r = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, layout, 0, 1, "auto")
    assert r.epoch == 1 and r.global_step == 8


def _load_last_state(run_dir):
    import safetensors.torch
    path = os.path.join(run_dir, "models", "last", "model.safetensors")
    return safetensors.torch.load_file(path)


def test_resumed_trajectory_matches_continuous(tmp_path):
    """OSR-004：中途暂停恢复后的模型张量、step 与指标必须与连续训练一致。"""
    # 连续：2 epoch 不中断
    cont_dir = str(tmp_path / "runs" / "cont")
    layout_c = RunLayout(cont_dir)
    layout_c.ensure()
    rc = run_worker("experiments.toy_multiclass_resumable:build_experiment",
                    _cfg("cont", max_epochs=2, resume="none"), layout_c, 0, 1, "none")
    assert rc.global_step == 16

    # 恢复：epoch 0 中途预算预占 → checkpoint，再恢复至完成 2 epoch
    res_dir = str(tmp_path / "runs" / "res")
    layout_r = RunLayout(res_dir)
    layout_r.ensure()
    r1 = run_worker("experiments.toy_multiclass_resumable:build_experiment",
                    _cfg("res", max_epochs=1, resume="auto", every_steps=4, max_minutes=10),
                    layout_r, 0, 1, "auto", budget_monotonic=_AdvancingClock())
    assert r1.status == "preempted"
    latest = read_latest(layout_r.path("checkpoints"))
    ckpt_dir = os.path.join(layout_r.path("checkpoints"), latest["path"])
    ckpt_manifest = json.load(open(os.path.join(ckpt_dir, "checkpoint-manifest.json"), encoding="utf-8"))
    assert 0 < ckpt_manifest["batch_in_epoch"] < 8  # 真中途

    r2 = run_worker("experiments.toy_multiclass_resumable:build_experiment",
                    _cfg("res", max_epochs=2, resume="auto", every_steps=4),
                    layout_r, 0, 1, "auto")
    assert r2.status == "succeeded"
    assert r2.global_step == 16

    # 模型张量逐项一致
    state_c = _load_last_state(cont_dir)
    state_r = _load_last_state(res_dir)
    assert set(state_c) == set(state_r)
    for k in state_c:
        assert state_c[k].shape == state_r[k].shape, k
        assert bool((state_c[k] == state_r[k]).all()), f"张量 {k} 不一致"

    # 指标一致：连续与恢复的每轮 train/val 记录相等
    def _train_val(layout):
        recs = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
        return {stage: {rec["epoch"]: rec["metrics"] for rec in recs if rec["stage"] == stage}
                for stage in ("train", "val")}

    mc = _train_val(layout_c)
    mr = _train_val(layout_r)
    for stage in ("train", "val"):
        for ep in (0, 1):
            for k in mc[stage][ep]:
                assert mr[stage][ep][k] == pytest.approx(mc[stage][ep][k], rel=1e-6), \
                    f"{stage} epoch {ep} {k} 不一致"


def _shuffle_dm():
    import torch
    from dl_helper.training.contracts import DataIdentity, ResumableMapDataModule
    from torch.utils.data import TensorDataset

    def _collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.tensor(ys)

    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 4, generator=g)
    y = torch.randint(0, 2, (64,), generator=g)
    ds = TensorDataset(x, y)
    identity = DataIdentity("shuf-dm", "1.0", "fp-shuf")
    return ResumableMapDataModule(identity, lambda: ds, _collate, batch_size=8, shuffle=True)


def test_resumable_datamodule_epoch_advances_fresh_shuffle():
    """OSR-004：DataModule 每 epoch 推进并产生新 shuffle（不重复上一轮顺序）。"""
    dm = _shuffle_dm()
    loader0 = dm.train_dataloader()
    b0 = next(iter(loader0))[0].tolist()
    dm.advance_epoch()
    loader1 = dm.train_dataloader()
    b1 = next(iter(loader1))[0].tolist()
    assert b0 != b1  # epoch 1 顺序与 epoch 0 不同


def test_resumable_datamodule_resume_reproduces_batches():
    """OSR-004：同一 epoch 中途恢复后，剩余 batch 数据与连续运行逐项一致（RNG 不漂移）。"""
    import torch

    dm = _shuffle_dm()
    it = iter(dm.train_dataloader())
    for _ in range(3):
        next(it)
    b4_cont = next(it)  # 第 4 个 batch（consumed=3 之后）

    state = dm.state_dict()
    state["consumed_batches"] = 3
    dm2 = _shuffle_dm()
    dm2.load_state_dict(state)
    it2 = iter(dm2.train_dataloader())
    for _ in range(3):
        next(it2)
    b4_res = next(it2)

    for t_c, t_r in zip(b4_cont, b4_res):
        assert bool((t_c == t_r).all()), "恢复后 batch 数据不一致（RNG 漂移）"


def test_resumable_datamodule_boundary_resume_matches_continuous():
    """OSR-004：连续第二 epoch 与边界恢复（epoch 推进）产生相同 shuffle，轨迹不分叉。"""
    import torch

    dm = _shuffle_dm()
    list(dm.train_dataloader())  # 消费 epoch 0（seed 0）
    dm.advance_epoch()
    ep1_cont = [b[0].clone() for b in dm.train_dataloader()]  # 连续 epoch 1（seed 1）

    # 边界恢复：DataModule 从 epoch 1、consumed=0 开始
    dm2 = _shuffle_dm()
    dm2.load_state_dict({"epoch": 1, "consumed_batches": 0})
    ep1_res = [b[0].clone() for b in dm2.train_dataloader()]

    assert len(ep1_cont) == len(ep1_res)
    for t_c, t_r in zip(ep1_cont, ep1_res):
        assert bool((t_c == t_r).all()), "连续 epoch 1 与边界恢复 shuffle 不一致"
