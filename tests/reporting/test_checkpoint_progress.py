"""checkpoint 进度报告：snapshot 合同、固定轴趋势图、任务副图与 checkpoint 集成。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.sklearn_backend import probe_sklearn_learning_rate
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.checkpoint import CheckpointError, load_torch_checkpoint
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.reporting import (
    PROGRESS_REPORT_SCHEMA_VERSION,
    generate_checkpoint_progress_report,
    validate_progress_snapshot,
)


def _snapshot(**overrides):
    base = {
        "schema_version": PROGRESS_REPORT_SCHEMA_VERSION,
        "backend": "torch",
        "run_id": "run-progress",
        "checkpoint_id": "epoch-000001-step-00000008",
        "created_utc": "2026-08-22T00:00:00Z",
        "position": {"epoch": 1, "batch_in_epoch": 0, "global_step": 8},
        "max_epochs": 5,
        "partial_epoch": None,
        "learning_rate": {"source": "optimizer"},
    }
    base.update(overrides)
    return base


def _write_history(run_dir, records):
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    with open(os.path.join(run_dir, "metrics", "metrics.jsonl"), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# snapshot 合同（fail-fast）
# --------------------------------------------------------------------------

def test_validate_snapshot_rejects_invalid():
    with pytest.raises(ValueError):
        validate_progress_snapshot({"schema_version": 999})
    with pytest.raises(ValueError):
        validate_progress_snapshot(_snapshot(backend="mxnet"))
    with pytest.raises(ValueError):
        validate_progress_snapshot(_snapshot(run_id=""))
    bad = _snapshot()
    bad["position"] = {"epoch": -1, "batch_in_epoch": 0, "global_step": 0}
    with pytest.raises(ValueError):
        validate_progress_snapshot(bad)
    with pytest.raises(ValueError):
        validate_progress_snapshot(_snapshot(max_epochs=0))
    with pytest.raises(ValueError):
        validate_progress_snapshot(_snapshot(learning_rate={"source": "magic"}))
    with pytest.raises(ValueError):
        validate_progress_snapshot(_snapshot(partial_epoch={"epoch": 2, "train_loss": 0.1,
                                                            "learning_rate": 0.01}))
    with pytest.raises(ValueError):
        validate_progress_snapshot(_snapshot(partial_epoch={"epoch": 1, "train_loss": "x",
                                                            "learning_rate": 0.01}))
    assert validate_progress_snapshot(_snapshot())["backend"] == "torch"


def test_generate_report_allows_empty_history(tmp_path):
    """首个 checkpoint 前无任何 metrics 记录时仍可生成报告（空曲线 + N/A lr）。"""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    out = os.path.join(run_dir, "progress")
    path = generate_checkpoint_progress_report(run_dir, _snapshot(), out)
    html = open(path, encoding="utf-8").read()
    assert "data:image/png;base64," in html
    assert "Task metrics" in html  # 无活跃组时显示占位说明


# --------------------------------------------------------------------------
# 报告生成：固定轴、副图、学习率来源、幂等
# --------------------------------------------------------------------------

def _multiclass_history():
    return [
        {"stage": "train", "epoch": 0, "global_step": 8,
         "metrics": {"train/loss": 1.2, "train/accuracy": 0.5, "train/f1_macro": 0.48},
         "extended": {"train/confusion_weighted": [[1, 0], [0, 1]]},
         "learning_rate": 0.05},
        {"stage": "val", "epoch": 0, "global_step": 8,
         "metrics": {"val/loss": 0.9, "val/accuracy": 0.7, "val/f1_macro": 0.66},
         "extended": {}},
        {"stage": "train", "epoch": 1, "global_step": 16,
         "metrics": {"train/loss": 0.8, "train/accuracy": 0.6, "train/f1_macro": 0.58},
         "extended": {"train/confusion_weighted": [[2, 0], [0, 2]]},
         "learning_rate": 0.04},
        {"stage": "val", "epoch": 1, "global_step": 16,
         "metrics": {"val/loss": 0.7, "val/accuracy": 0.75, "val/f1_macro": 0.71},
         "extended": {}},
    ]


def test_report_contains_fixed_axis_and_group(tmp_path):
    from dl_helper.training.reporting import _active_progress_groups

    run_dir = str(tmp_path / "run")
    _write_history(run_dir, _multiclass_history())
    out = os.path.join(run_dir, "ckpt", "progress")
    path = generate_checkpoint_progress_report(run_dir, _snapshot(), out)
    html = open(path, encoding="utf-8").read()
    assert "epoch 1 batch 0" in html.replace("\n", " ")
    assert "data:image/png;base64," in html
    # multiclass 副图组渲染，且组内每个出现的标量键都有对应子图（含 f1_macro）；
    # 共享聚合键 f1_macro 由 extended 后缀判别归 multiclass，不得溢出到 multilabel 组
    assert "multiclass subplots: accuracy, f1_macro" in html
    assert "metrics multilabel" not in html
    # 组键覆盖 Task 实际产出的全部标量键（task.py _multiclass_definitions）
    groups = dict(_active_progress_groups({
        "scalars": {name: {} for name in (
            "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_weighted")},
        "extended_keys": {"train/confusion_weighted"}}))
    assert set(groups["multiclass"]) == {
        "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted"}
    # snapshot 落盘且与输入一致
    saved = json.load(open(os.path.join(out, "progress-snapshot.json"), encoding="utf-8"))
    assert saved["max_epochs"] == 5
    # 幂等
    html2 = open(generate_checkpoint_progress_report(run_dir, _snapshot(), out),
                 encoding="utf-8").read()
    assert html2 == html


def test_report_multilabel_full_scalar_keys(tmp_path):
    """multilabel 组必须覆盖全部实际标量键（macro/micro/weighted 变体），无死键。"""
    from dl_helper.training.reporting import _TASK_METRIC_GROUPS, _active_progress_groups

    run_dir = str(tmp_path / "run")
    records = [
        {"stage": "train", "epoch": 0, "global_step": 8,
         "metrics": {"train/loss": 0.5, "train/subset_accuracy": 0.7,
                     "train/hamming_loss": 0.1, "train/precision_macro": 0.6,
                     "train/recall_micro": 0.65, "train/f1_micro": 0.62},
         "extended": {"train/per_label": {"precision": [0.6]}},
         "learning_rate": 0.01},
        {"stage": "val", "epoch": 0, "global_step": 8,
         "metrics": {"val/loss": 0.45, "val/subset_accuracy": 0.72,
                     "val/hamming_loss": 0.09, "val/precision_macro": 0.63,
                     "val/recall_micro": 0.68, "val/f1_micro": 0.65},
         "extended": {}},
    ]
    _write_history(run_dir, records)
    out = os.path.join(run_dir, "progress")
    html = open(generate_checkpoint_progress_report(run_dir, _snapshot(), out),
                encoding="utf-8").read()
    assert "multilabel subplots: subset_accuracy, hamming_loss, precision_macro, recall_micro, f1_micro" in html
    # 组定义与 task.py _multilabel_definitions 精确对齐，无死键
    multilabel_names = next(names for group, names in _TASK_METRIC_GROUPS if group == "multilabel")
    assert set(multilabel_names) == {
        "subset_accuracy", "hamming_loss", "precision_macro", "precision_micro",
        "recall_macro", "recall_micro", "f1_macro", "f1_micro", "f1_weighted"}
    # per_label extended 判别归属 multilabel，共享聚合键不溢出到 multiclass
    groups = dict(_active_progress_groups({
        "scalars": {name: {} for name in multilabel_names},
        "extended_keys": {"train/per_label"}}))
    assert set(groups["multilabel"]) == set(multilabel_names)
    assert "multiclass" not in groups


def test_report_lr_sources(tmp_path):
    run_dir = str(tmp_path / "run")
    _write_history(run_dir, _multiclass_history())
    out = os.path.join(run_dir, "progress")
    # estimator_config
    snap = _snapshot(backend="sklearn",
                     learning_rate={"source": "estimator_config", "param_name": "learning_rate_init",
                                    "config_value": 0.001, "history": None})
    html = generate_checkpoint_progress_report(run_dir, snap, out) and open(
        os.path.join(out, "index.html"), encoding="utf-8").read()
    assert "learning_rate_init" in html
    # estimator_history
    snap = _snapshot(backend="sklearn",
                     learning_rate={"source": "estimator_history", "param_name": "lr_history_",
                                    "config_value": None, "history": [[0, 0.01], [1, 0.008]]})
    html = generate_checkpoint_progress_report(run_dir, snap, out) and open(
        os.path.join(out, "index.html"), encoding="utf-8").read()
    assert "history_points" in html
    # unavailable → N/A 标注
    snap = _snapshot(backend="sklearn",
                     learning_rate={"source": "unavailable", "param_name": None,
                                    "config_value": None, "history": None})
    generate_checkpoint_progress_report(run_dir, snap, out)
    html = open(os.path.join(out, "index.html"), encoding="utf-8").read()
    assert "N/A" in html


def test_report_mid_epoch_partial(tmp_path):
    run_dir = str(tmp_path / "run")
    _write_history(run_dir, _multiclass_history())
    snap = _snapshot(position={"epoch": 2, "batch_in_epoch": 3, "global_step": 19},
                     partial_epoch={"epoch": 2, "train_loss": 0.55, "learning_rate": 0.039})
    out = os.path.join(run_dir, "progress")
    html = open(generate_checkpoint_progress_report(run_dir, snap, out), encoding="utf-8").read()
    assert "partial.train_loss" in html
    assert "partial.learning_rate" in html


def test_report_regression_group(tmp_path):
    run_dir = str(tmp_path / "run")
    _write_history(run_dir, [
        {"stage": "train", "epoch": 0, "global_step": 8,
         "metrics": {"train/loss": 1.2, "train/mae": 0.4, "train/mse": 0.2, "train/r2": 0.1},
         "extended": {"train/per_target": {"mae": [0.4, 0.5]}, "learning_rate": 0.05}},
        {"stage": "val", "epoch": 0, "global_step": 8,
         "metrics": {"val/loss": 0.9, "val/mae": 0.35, "val/mse": 0.18, "val/r2": 0.15},
         "extended": {"val/per_target": {"mae": [0.35, 0.45]}}},
    ])
    out = os.path.join(run_dir, "progress")
    html = open(generate_checkpoint_progress_report(run_dir, _snapshot(), out),
                encoding="utf-8").read()
    assert "metrics regression" in html
    assert "metrics multiclass" not in html  # 只渲染当前任务支持的组


def test_corrupt_metrics_jsonl_aborts_report(tmp_path):
    """metrics.jsonl 含不可解析行时必须中止报告生成，不得静默跳过。"""
    run_dir = str(tmp_path / "run")
    _write_history(run_dir, _multiclass_history())
    with open(os.path.join(run_dir, "metrics", "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"stage": "train", "epoch": 2, "global_step": 24, "met')  # 崩溃残留截断行
    with pytest.raises(ValueError, match="无法解析"):
        generate_checkpoint_progress_report(
            run_dir, _snapshot(), os.path.join(run_dir, "progress"))


def test_missing_identity_fields_aborts_report(tmp_path):
    """合法 JSON 但缺少 stage/epoch 身份字段的记录必须中止，不得静默丢弃曲线点。"""
    run_dir = str(tmp_path / "run")
    _write_history(run_dir, _multiclass_history())
    with open(os.path.join(run_dir, "metrics", "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"global_step": 24, "metrics": {"train/loss": 0.5}},
                           ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="身份字段"):
        generate_checkpoint_progress_report(
            run_dir, _snapshot(), os.path.join(run_dir, "progress"))


# --------------------------------------------------------------------------
# sklearn 学习率探测
# --------------------------------------------------------------------------

def test_probe_sklearn_learning_rate_sources():
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mlp = MLPRegressor(hidden_layer_sizes=(4,), max_iter=1)
    mlp.partial_fit([[0.0], [1.0]], [0.0, 1.0])
    info = probe_sklearn_learning_rate(mlp)
    assert info["source"] == "estimator_config"
    assert info["param_name"] == "learning_rate_init"
    assert info["config_value"] > 0

    pipe = Pipeline([("scale", StandardScaler()),
                     ("mlp", MLPRegressor(hidden_layer_sizes=(4,), max_iter=1))])
    pipe.fit([[0.0], [1.0]], [0.0, 1.0])
    info = probe_sklearn_learning_rate(pipe)
    assert info["source"] == "estimator_config"
    assert info["param_name"] == "learning_rate_init"

    sgd = SGDRegressor(eta0=0.01, learning_rate="invscaling", max_iter=1, tol=None)
    sgd.fit([[0.0], [1.0]], [0.0, 1.0])
    info = probe_sklearn_learning_rate(sgd)
    assert info["source"] == "estimator_config"
    assert info["param_name"] == "eta0"

    lin = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    assert probe_sklearn_learning_rate(lin)["source"] == "unavailable"

    mlp2 = MLPRegressor(hidden_layer_sizes=(4,), max_iter=1)
    mlp2.partial_fit([[0.0], [1.0]], [0.0, 1.0])
    mlp2.learning_rate_history_ = [0.01, 0.009, 0.008]
    info = probe_sklearn_learning_rate(mlp2)
    assert info["source"] == "estimator_history"
    assert info["history"] == [[0, 0.01], [1, 0.009], [2, 0.008]]


# --------------------------------------------------------------------------
# Torch checkpoint 集成
# --------------------------------------------------------------------------

def _torch_cfg(run_id, max_epochs):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 30, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = 1
    schema["checkpoint"]["keep_last"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def _run_torch(tmp_path, run_id, max_epochs=2):
    layout = RunLayout(str(tmp_path / "runs" / run_id))
    layout.ensure()
    cfg = _torch_cfg(run_id, max_epochs)
    result = run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    return layout, cfg, result


def test_torch_checkpoint_contains_verified_report(tmp_path):
    layout, _cfg, result = _run_torch(tmp_path, "progress-torch")
    assert result.status == "succeeded"
    ckpts = sorted(d for d in os.listdir(layout.path("checkpoints")) if d.startswith("epoch-"))
    assert ckpts
    for name in ckpts:
        cdir = layout.path("checkpoints", name)
        assert os.path.exists(os.path.join(cdir, "progress", "index.html"))
        assert os.path.exists(os.path.join(cdir, "progress", "progress-snapshot.json"))
        manifest = json.load(open(os.path.join(cdir, "checkpoint-manifest.json"), encoding="utf-8"))
        assert os.path.join("progress", "index.html") in manifest["files"]
        assert os.path.join("progress", "progress-snapshot.json") in manifest["files"]
    # metrics.jsonl train 记录携带学习率（报告 lr 数据源）
    records = [json.loads(line) for line in open(layout.metrics_jsonl, encoding="utf-8")
               if line.strip()]
    train_records = [r for r in records if r["stage"] == "train"]
    assert train_records
    assert all("learning_rate" in r and r["learning_rate"] > 0 for r in train_records)
    # run report 同样含固定轴趋势图与 multiclass 副图
    from dl_helper.training.reporting import generate_run_report
    index = generate_run_report(layout.run_dir)
    html = open(index, encoding="utf-8").read()
    assert "Trend (fixed epoch axis)" in html
    assert "data:image/png;base64," in html


def test_corrupt_progress_report_blocks_resume(tmp_path):
    from dl_helper.training.backends.torch_backend import (
        config_fingerprint_resume,
    )
    from dl_helper.training.engine import EngineState

    layout, cfg, _result = _run_torch(tmp_path, "progress-corrupt")
    latest = json.load(open(layout.path("checkpoints", "latest.json"), encoding="utf-8"))
    cdir = layout.path("checkpoints", latest["path"])
    manifest = json.load(open(os.path.join(cdir, "checkpoint-manifest.json"), encoding="utf-8"))
    snap_path = os.path.join(cdir, "progress", "progress-snapshot.json")
    with open(snap_path, "r", encoding="utf-8") as f:
        original = f.read()
    # 篡改字节 → 与 manifest checksum 不一致
    with open(snap_path, "w", encoding="utf-8") as f:
        f.write(original + " ")
    engine_state = EngineState(
        backend="torch", run_id="progress-corrupt",
        config_fingerprint=config_fingerprint_resume(cfg),
        metric_name="val/loss", mode="min", patience=30, min_delta=0.0,
    )
    # 恢复前校验必须失败，不尝试反序列化
    with pytest.raises(CheckpointError):
        load_torch_checkpoint(
            _StubAccelerator(), layout.path("checkpoints"), engine_state,
            _StubDataModule(), lambda: {}, config_fingerprint_resume(cfg),
            manifest["data_fingerprint"], manifest["model_signature"],
        )
    # 缺失文件同样拒绝
    os.remove(snap_path)
    with pytest.raises(CheckpointError):
        load_torch_checkpoint(
            _StubAccelerator(), layout.path("checkpoints"), engine_state,
            _StubDataModule(), lambda: {}, config_fingerprint_resume(cfg),
            manifest["data_fingerprint"], manifest["model_signature"],
        )


class _StubAccelerator:
    process_index = 0

    def wait_for_everyone(self):
        pass


class _StubDataModule:
    pass


# --------------------------------------------------------------------------
# 中途 checkpoint 快照（budget PREEMPTED → partial loss/lr）
# --------------------------------------------------------------------------

class _AdvancingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def test_mid_epoch_checkpoint_snapshot_has_partial(tmp_path):
    from dl_helper.training.platform import ExecutionPolicy

    budget = ExecutionPolicy(platform="local", max_minutes=10.0, shutdown_grace_minutes=2.0)
    run_dir = str(tmp_path / "runs" / "progress-partial")
    layout = RunLayout(run_dir)
    layout.ensure()
    cfg = _torch_cfg("progress-partial", max_epochs=3)
    result = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, layout,
                        0, 1, "auto", budget_monotonic=_AdvancingClock(),
                        execution_policy=budget)
    assert result.status == "preempted"
    latest = json.load(open(layout.path("checkpoints", "latest.json"), encoding="utf-8"))
    snap = json.load(open(
        os.path.join(layout.path("checkpoints"), latest["path"],
                     "progress", "progress-snapshot.json"),
        encoding="utf-8",
    ))
    # budget 在 optimizer step 边界中断：当前 epoch 未完成，快照必须携带 partial loss/lr
    assert snap["position"]["batch_in_epoch"] > 0
    assert snap["partial_epoch"] is not None
    partial = snap["partial_epoch"]
    assert partial["epoch"] == snap["position"]["epoch"]
    assert partial["train_loss"] > 0
    assert partial["learning_rate"] > 0
    html_path = os.path.join(layout.path("checkpoints"), latest["path"],
                             "progress", "index.html")
    html = open(html_path, encoding="utf-8").read()
    assert "partial.train_loss" in html


# --------------------------------------------------------------------------
# sklearn incremental checkpoint 集成
# --------------------------------------------------------------------------

def _incr_cfg(run_id, max_epochs):
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "incremental", "evaluation_batch_size": 4096, "n_jobs": None,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": max_epochs, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 20, "min_delta": 0.0}
    schema["checkpoint"] = {"every_epochs": 1, "every_optimizer_steps": None, "keep_last": None}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    return parse_config(schema)


def test_sklearn_checkpoint_contains_report_with_lr_availability(tmp_path):
    from dl_helper.training.backends.sklearn_backend import (
        build_sklearn_experiment, run_sklearn_worker_experiment,
    )
    from dl_helper.training.platform import Platform

    run_dir = str(tmp_path / "runs" / "progress-skl")
    cfg = _incr_cfg("progress-skl", max_epochs=1)
    layout = RunLayout(run_dir)
    layout.ensure()
    experiment = build_sklearn_experiment(
        "experiments.sklearn_incremental:build_experiment", cfg.experiment
    )
    result = run_sklearn_worker_experiment(experiment, cfg, Platform(), layout)
    assert result.status == "succeeded"
    ckpts = sorted(d for d in os.listdir(layout.path("checkpoints")) if d.startswith("epoch-"))
    assert ckpts
    for name in ckpts:
        cdir = layout.path("checkpoints", name)
        assert os.path.exists(os.path.join(cdir, "progress", "index.html"))
        assert os.path.exists(os.path.join(cdir, "progress", "progress-snapshot.json"))
        manifest = json.load(open(os.path.join(cdir, "checkpoint-manifest.json"), encoding="utf-8"))
        assert os.path.join("progress", "index.html") in manifest["files"]
        assert os.path.join("progress", "progress-snapshot.json") in manifest["files"]
        snap = json.load(open(os.path.join(cdir, "progress", "progress-snapshot.json"),
                              encoding="utf-8"))
        # SGDClassifier(learning_rate='optimal', eta0=0.0)：无数值学习率 → 明确 N/A，不伪造
        assert snap["learning_rate"]["source"] == "unavailable"
        assert snap["partial_epoch"] is None
        assert snap["max_epochs"] == 1
    html = open(os.path.join(layout.path("checkpoints"), ckpts[-1], "progress", "index.html"),
                encoding="utf-8").read()
    assert "N/A" in html
    assert "data:image/png;base64," in html
