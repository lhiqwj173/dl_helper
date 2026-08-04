"""任务 OSR-003：训练失败写脱敏 failure.json，primary/secondary，不吞 traceback。"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from dl_helper.training.config import default_schema


def _base_cfg(tmp_path, run_id):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["run"]["output_root"] = str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_failure_writes_redacted_failure_json(tmp_path):
    from dl_helper.training.cli import main

    cfg = _base_cfg(tmp_path, "err-artifact")
    with pytest.raises(Exception):
        main(["train", "--config", cfg, "--experiment", "nonexistent_module:build_experiment"])
    failure_path = os.path.join(str(tmp_path), "runs", "err-artifact", "failure.json")
    assert os.path.exists(failure_path), "失败必须写 failure.json"
    failure = json.load(open(failure_path, encoding="utf-8"))
    assert failure["exception_type"]
    assert "ModuleNotFoundError" in failure["exception_type"]
    assert "primary_exception" in failure
    assert "Traceback" in failure["traceback"]
    assert "ModuleNotFoundError" in failure["traceback"]
    # 无 success/pause 终态
    assert not os.path.exists(os.path.join(os.path.dirname(failure_path), "run-manifest.json"))


def test_failure_traceback_not_swallowed(tmp_path, capsys):
    from dl_helper.training.cli import main

    cfg = _base_cfg(tmp_path, "err-tb")
    with pytest.raises(ModuleNotFoundError):
        main(["train", "--config", cfg, "--experiment", "nonexistent_module:build_experiment"])
    failure_path = os.path.join(str(tmp_path), "runs", "err-tb", "failure.json")
    failure = json.load(open(failure_path, encoding="utf-8"))
    # traceback 保留原始异常链
    assert "ModuleNotFoundError" in failure["traceback"]


def test_pause_to_failed_transition_keeps_pause_when_replace_fails(tmp_path, monkeypatch):
    from dl_helper.training import artifacts

    run_dir = str(tmp_path / "runs" / "atomic-transition")
    os.makedirs(run_dir, exist_ok=True)
    artifacts.write_json(os.path.join(run_dir, "pause-manifest.json"),
                         {"status": "preempted", "resume_checkpoint": "ckpt-1"})
    original_replace = artifacts.os.replace

    def fail_replace(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError):
        artifacts.publish_terminal(run_dir, "failed", {"exception_type": "RuntimeError"})
    assert os.path.exists(os.path.join(run_dir, "pause-manifest.json"))
    assert not os.path.exists(os.path.join(run_dir, "failure.json"))
    monkeypatch.setattr(artifacts.os, "replace", original_replace)


def test_failure_records_train_position(tmp_path):
    """OSR-003：训练中途失败时 failure.json 记录阶段与训练位置。"""
    import sys as _sys

    from dl_helper.training.cli import main

    # 临时实验：loss 在首个 batch 抛错，触发训练循环失败
    exp = tmp_path / "fail_exp.py"
    exp.write_text('''
def build_experiment(config):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
    from dl_helper.training.task import MulticlassClassificationTask

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            return self.fc(x)

    def model_factory():
        return M()

    def datamodule_factory():
        g = torch.Generator().manual_seed(7)
        x = torch.randn(160, 8, generator=g)
        y = torch.randint(0, 3, (160,), generator=g)
        train_ds = TensorDataset(x[:128], y[:128])
        val_ds = TensorDataset(x[128:], y[128:])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)
        return LoaderDataModule(DataIdentity("fail-exp", "1", "fp"), train_loader,
                                val_dataloader=val_loader)

    def task_factory():
        class _Raising(MulticlassClassificationTask):
            def __init__(self):
                super().__init__(num_classes=3)

            def loss(self, outputs, prepared):
                raise RuntimeError("boom-mid-training")

        return _Raising()

    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=0.05)

    return TorchExperiment(name="fail-exp", backend="torch",
                           model_factory=model_factory,
                           datamodule_factory=datamodule_factory,
                           task_factory=task_factory,
                           optimizer_factory=optimizer_factory,
                           scheduler_factory=lambda o: None,
                           model_config=dict(config))
''', encoding="utf-8")

    cfg = _base_cfg(tmp_path, "err-pos")
    _sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="boom-mid-training"):
            main(["train", "--config", cfg, "--experiment", "fail_exp:build_experiment"])
    finally:
        _sys.path.remove(str(tmp_path))

    failure_path = os.path.join(str(tmp_path), "runs", "err-pos", "failure.json")
    failure = json.load(open(failure_path, encoding="utf-8"))
    assert failure["stage"] == "train"
    assert failure["epoch"] == 0
    assert failure["global_step"] == 0
    assert "Traceback" in failure["traceback"]


def test_failure_evidence_redacts_configured_secret(tmp_path, monkeypatch):
    """OSR-003：_secret_keys 已解析 → failure.json 全链路脱敏，不泄漏 Secret 值。"""
    from dl_helper.training.artifacts import RunLayout
    from dl_helper.training.cli import _write_failure_evidence
    from dl_helper.training.platform import Platform, SecretResolver

    monkeypatch.setenv("WECOM_TEST_SECRET", "s3cr3t-token-xyz")
    layout = RunLayout(str(tmp_path / "runs" / "sec"))
    layout.ensure()
    resolver = SecretResolver(Platform("local"))

    class Args:
        _run_dir = layout.run_dir
        _secret_resolver = resolver
        _secret_keys = ("WECOM_TEST_SECRET",)
        _secondary_errors = [{"service": "finalize", "event": "RUN_FAILED",
                              "error_type": "X", "message": "secondary with s3cr3t-token-xyz"}]

    try:
        raise RuntimeError("primary failed with s3cr3t-token-xyz")
    except RuntimeError as exc:
        _write_failure_evidence(Args(), exc)
    failure = json.load(open(os.path.join(layout.run_dir, "failure.json"), encoding="utf-8"))
    blob = json.dumps(failure, ensure_ascii=False)
    assert "s3cr3t-token-xyz" not in blob
    assert "[REDACTED]" in failure["message"]
    assert "[REDACTED]" in failure["secondary_errors"][0]["message"]
    assert "Traceback" in failure["traceback"]


def test_failure_written_despite_old_pause(tmp_path):
    """OSR-003：旧 pause 终态不丢弃失败证据，failure.json 仍写入。"""
    from dl_helper.training.artifacts import RunLayout, write_json
    from dl_helper.training.cli import _write_failure_evidence

    layout = RunLayout(str(tmp_path / "runs" / "pause-fail"))
    layout.ensure()
    write_json(os.path.join(layout.run_dir, "pause-manifest.json"), {"status": "preempted"})

    class Args:
        _run_dir = layout.run_dir
        _secret_resolver = None
        _secret_keys = ()
        _secondary_errors = []

    try:
        raise RuntimeError("resume failed")
    except RuntimeError as exc:
        _write_failure_evidence(Args(), exc)
    failure_path = os.path.join(layout.run_dir, "failure.json")
    assert os.path.exists(failure_path), "旧 pause 不得丢弃失败证据"
    failure = json.load(open(failure_path, encoding="utf-8"))
    assert "resume failed" in failure["message"]
    # OSR-003：旧 pause 被原子替换为唯一 FAILED 终态（不并存多终态）
    from dl_helper.training.artifacts import existing_terminal
    assert existing_terminal(layout.run_dir) == "failure.json"
    assert not os.path.exists(os.path.join(layout.run_dir, "pause-manifest.json"))
