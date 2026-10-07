"""跨会话恢复必须保留检查点位置的指标与预测历史。"""
from __future__ import annotations

import json

import pytest

from dl_helper.training.artifacts import sha256_manifest
from dl_helper.training.checkpoint import (
    HISTORY_DIR, HISTORY_MANIFEST, CheckpointError, HistoryRecoveryError,
    _snapshot_run_history, restore_run_history,
)


def test_history_restores_into_new_session_and_rolls_back_later_files(tmp_path):
    source = tmp_path / "source"
    metrics = source / "metrics" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"stage":"train","epoch":0}\n', encoding="utf-8")
    prediction = source / "predictions" / "val" / "part.npz"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"prediction-at-checkpoint")
    auxiliary = source / "models" / "diagnostic" / "model.safetensors"
    auxiliary.parent.mkdir(parents=True)
    auxiliary.write_bytes(b"auxiliary-model-at-checkpoint")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _snapshot_run_history(str(checkpoint), str(source), position_epoch=1, include_models=True)
    saved = json.loads((checkpoint / HISTORY_MANIFEST).read_text(encoding="utf-8"))
    assert saved["files"] == sha256_manifest(str(checkpoint / HISTORY_DIR))

    destination = tmp_path / "new-session"
    destination.mkdir()
    newer = destination / "predictions" / "test" / "later.npz"
    newer.parent.mkdir(parents=True)
    newer.write_bytes(b"not-in-checkpoint")
    restore_run_history(str(checkpoint), str(destination), position_epoch=1)
    assert (destination / "metrics" / "metrics.jsonl").read_text(encoding="utf-8") == metrics.read_text(encoding="utf-8")
    assert (destination / "predictions" / "val" / "part.npz").read_bytes() == prediction.read_bytes()
    assert (destination / "models" / "diagnostic" / "model.safetensors").read_bytes() == auxiliary.read_bytes()
    assert not newer.exists()


def test_history_rejects_missing_or_tampered_metrics(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(CheckpointError, match="metrics.jsonl"):
        _snapshot_run_history(str(checkpoint), str(source), position_epoch=1)
    with pytest.raises(HistoryRecoveryError, match="旧检查点未包含运行历史"):
        restore_run_history(str(checkpoint), str(source), position_epoch=1)

    metrics = source / "metrics" / "metrics.jsonl"
    metrics.parent.mkdir()
    metrics.write_text('{"stage":"train","epoch":0}\n', encoding="utf-8")
    (checkpoint / HISTORY_DIR).rmdir()
    _snapshot_run_history(str(checkpoint), str(source), position_epoch=1)
    (checkpoint / HISTORY_DIR / "metrics" / "metrics.jsonl").write_text("bad\n", encoding="utf-8")
    with pytest.raises(HistoryRecoveryError, match="清单不一致"):
        restore_run_history(str(checkpoint), str(tmp_path / "new-session"), position_epoch=1)


def test_sklearn_checkpoint_can_have_no_metric_history(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _snapshot_run_history(str(checkpoint), str(source), position_epoch=1,
                          require_metrics=False)
    destination = tmp_path / "new-session"
    destination.mkdir()
    restore_run_history(str(checkpoint), str(destination), position_epoch=1,
                        require_metrics=False)
