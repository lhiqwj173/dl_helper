"""任务 OSR-005：run manifest 完整性 —— checksum、MetricDefinition、模型/报告/服务引用。"""
from __future__ import annotations

import json
import os

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.sklearn_backend import (
    build_sklearn_experiment,
    run_sklearn_worker_experiment,
)
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import Platform


def _torch_cfg(run_id, max_epochs=1):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 20, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def test_torch_manifest_schema_complete(tmp_path):
    cfg = _torch_cfg("schema-torch")
    layout = RunLayout(str(tmp_path / "runs" / "schema-torch"))
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    manifest = json.load(open(layout.path("run-manifest.json"), encoding="utf-8"))
    # checksum 清单
    assert "artifacts" in manifest and manifest["artifacts"]
    for rel, meta in manifest["artifacts"].items():
        assert "size" in meta and "sha256" in meta
    # MetricDefinition
    assert "metric_definitions" in manifest
    assert "accuracy" in manifest["metric_definitions"]
    # 模型引用
    assert manifest["model_artifact"]["best_path"] == "models/best/model.safetensors"
    # 报告路径
    assert manifest["report"] == os.path.join("report", "index.html").replace("\\", "/")
    # 完整 schema：时间/source/环境/真实服务结果（OSR-005）
    assert manifest["created_utc"]
    assert manifest["source_revision"]
    assert "environment" in manifest and "resources" in manifest["environment"]
    assert manifest["environment"]["seed"] == cfg.run.seed
    # 服务禁用时 degraded 为真实空列表，audit 为 None（OSR-005：不伪造）
    assert manifest["services"]["degraded"] == []
    assert manifest["services"]["audit"] is None
    # 独立文件
    assert os.path.exists(layout.evaluation_contract_json)
    assert os.path.exists(layout.environment_json)
    assert os.path.exists(layout.path("report", "index.html"))
    # eval contract 含 DataIdentity、per-split fingerprint、label schema、model signature（OSR-005）
    contract = json.load(open(layout.evaluation_contract_json, encoding="utf-8"))
    assert "data_identity" in contract and contract["data_identity"]["name"]
    assert "splits" in contract and "val" in contract["splits"]
    assert contract["splits"]["val"]["fingerprint"]  # per-split fingerprint 非空
    assert contract["label_schema"]["kind"] == "classification"
    assert "num_classes" in contract["label_schema"]
    assert "model_signature" in contract and "num_parameters" in contract["model_signature"]
    assert "accuracy" in contract["metric_definitions"]
    # summary 状态一致
    summary = json.load(open(layout.summary_json, encoding="utf-8"))
    assert summary["status"] == manifest["status"] == "succeeded"
    # 终态唯一
    assert not os.path.exists(layout.path("pause-manifest.json"))
    assert not os.path.exists(layout.path("failure.json"))


def test_sklearn_manifest_schema_complete(tmp_path):
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096,
                                     "n_jobs": None, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 5, "min_delta": 0.0}
    schema["run"]["id"] = "schema-skl"
    schema["checkpoint"] = {"every_epochs": None, "every_optimizer_steps": None,
                            "keep_last": 1}
    cfg = parse_config(schema)
    layout = RunLayout(str(tmp_path / "runs" / "schema-skl"))
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_batch:build_experiment", cfg.experiment)
    run_sklearn_worker_experiment(exp, cfg, Platform(), layout)
    manifest = json.load(open(layout.path("run-manifest.json"), encoding="utf-8"))
    assert "artifacts" in manifest and manifest["artifacts"]
    assert manifest["backend"] == "sklearn"
    assert manifest["model_artifact"]["format"] == "joblib"
    assert os.path.exists(layout.evaluation_contract_json)
    # OSR-005：sklearn 持久合同含 splits/label schema/model signature
    contract = json.load(open(layout.evaluation_contract_json, encoding="utf-8"))
    assert "data_identity" in contract
    assert "splits" in contract and "val" in contract["splits"]
    assert "label_schema" in contract
    assert "model_signature" in contract
    assert "metric_definitions" in contract
