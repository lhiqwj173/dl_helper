"""补充 cli.py / platform.py / metrics.py 覆盖率。"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from dl_helper.training.config import default_schema, parse_config


def _write_cfg(tmp_path, run_id, output_root=None):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["run"]["output_root"] = output_root or str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_cli_report_command(tmp_path):
    from dl_helper.training.backends.torch_backend import run_worker
    from dl_helper.training.artifacts import RunLayout
    cfg_path = _write_cfg(tmp_path, "cli-rpt")
    cfg = parse_config(yaml.safe_load(open(cfg_path, encoding="utf-8")))
    layout = RunLayout(str(tmp_path / "runs" / "cli-rpt"))
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    from dl_helper.training.cli import main
    code = main(["report", "--run", layout.run_dir])
    assert code == 0
    assert os.path.exists(layout.path("report", "index.html"))


def test_cli_train_preflight_variant(tmp_path):
    from dl_helper.training.cli import main
    base = _write_cfg(tmp_path, "cli-doc")
    variant = tmp_path / "v.yaml"
    variant.write_text(yaml.safe_dump({"training": {"max_epochs": 1}}, allow_unicode=True), encoding="utf-8")
    code = main(["train", "--config", base, "--variant", str(variant),
                 "--experiment", "experiments.toy_multiclass:build_experiment",
                 "--preflight-only"])
    assert code == 0


def test_cli_sweep_report_command(tmp_path):
    from dl_helper.training.cli import main
    sweep_dir = tmp_path / "sweeps" / "s"
    os.makedirs(sweep_dir, exist_ok=True)
    json.dump({"id": "s", "ranking": [{"rank": 1, "trial": "a", "value": 0.5}], "best_trial": "a"},
              open(os.path.join(sweep_dir, "sweep-manifest.json"), "w", encoding="utf-8"))
    code = main(["sweep-report", "--sweep-dir", str(sweep_dir)])
    assert code == 0


# ---------- platform ----------
def test_platform_environment_manifest():
    from dl_helper.training.platform import Platform
    p = Platform("local")
    env = p.environment_manifest()
    assert env["platform"] == "local"
    assert env["logical_cpus"] >= 1
    assert "python" in env


def test_platform_kaggle_output_root_default(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    from dl_helper.training.platform import Platform
    p = Platform("kaggle")
    cfg = parse_config(default_schema())
    out = p.resolve_output_root(cfg)
    assert out == "/kaggle/working/dl-helper-runs"


# ---------- metrics extended ----------
def test_metrics_extended_compute():
    import numpy as np
    from dl_helper.training.metrics import MulticlassState, MultilabelState, RegressionState
    from dl_helper.training.contracts import PredictedBatch

    mc = MulticlassState("val", [0, 1, 2])
    mc.update(PredictedBatch(targets=np.array([0, 1, 2]), predictions=np.array([0, 1, 2]),
                             sample_count=3))
    ext = mc.extended_compute()
    assert "val/per_class" in ext
    assert "val/confusion_weighted" in ext

    ml = MultilabelState("val", 2)
    ml.update(PredictedBatch(targets=np.array([[0, 1], [1, 0]]), predictions=np.array([[0, 1], [1, 0]]),
                             scores=np.array([[0.1, 0.9], [0.9, 0.1]]), sample_count=2))
    ext = ml.extended_compute()
    assert "val/per_label" in ext

    reg = RegressionState("val", 2)
    reg.update(PredictedBatch(targets=np.array([[1.0, 2.0], [3.0, 4.0]]),
                              predictions=np.array([[1.1, 2.1], [3.1, 4.1]]), sample_count=2))
    ext = reg.extended_compute()
    assert "val/per_target" in ext


# ---------- cli._build_services 分支 ----------
def _alist_cfg(policy="record"):
    schema = default_schema()
    schema["remote"] = {"type": "alist", "host": "https://alist.example.invalid",
                        "base_path": "/x", "user_secret_key": "ALIST_USER",
                        "password_secret_key": "ALIST_PWD", "connect_timeout_seconds": 1,
                        "read_timeout_seconds": 1, "max_attempts": 2, "async_upload": False,
                        "failure_policy": policy}
    return parse_config(schema)


def _wecom_cfg(policy="record"):
    schema = default_schema()
    schema["notifications"] = {"type": "wecom", "corp_id_secret_key": "WECOM_CORP_ID",
                               "corp_secret_key": "WECOM_CORP_SECRET",
                               "agent_id_secret_key": "WECOM_AGENT_ID",
                               "to_user": "u", "connect_timeout_seconds": 1,
                               "read_timeout_seconds": 1, "max_attempts": 2,
                               "failure_policy": policy}
    return parse_config(schema)


def _svc_layout(tmp_path):
    from dl_helper.training.artifacts import RunLayout
    layout = RunLayout(str(tmp_path / "runs" / "svc"))
    layout.ensure()
    return layout


def test_build_services_disabled_returns_none(tmp_path):
    import dl_helper.training.cli as cli
    from dl_helper.training.platform import Platform
    svc = cli._build_services(parse_config(default_schema()), Platform("local"),
                              _svc_layout(tmp_path))
    assert svc is None


def test_build_services_alist_record(tmp_path):
    import dl_helper.training.cli as cli
    from dl_helper.training.platform import Platform
    svc = cli._build_services(_alist_cfg("record"), Platform("local"), _svc_layout(tmp_path))
    assert svc is not None
    assert svc._wecom is None
    assert not svc._policy.is_required


def test_build_services_alist_required(tmp_path):
    import dl_helper.training.cli as cli
    from dl_helper.training.platform import Platform
    svc = cli._build_services(_alist_cfg("required"), Platform("local"), _svc_layout(tmp_path))
    assert svc._policy.is_required


def test_build_services_wecom_record(tmp_path):
    import dl_helper.training.cli as cli
    from dl_helper.training.platform import Platform
    svc = cli._build_services(_wecom_cfg("record"), Platform("local"), _svc_layout(tmp_path))
    assert svc is not None
    assert svc._wecom is not None
    assert not svc._wecom_policy.is_required
    assert not svc._policy.is_required  # store 默认 record


def test_build_services_wecom_required(tmp_path):
    import dl_helper.training.cli as cli
    from dl_helper.training.platform import Platform
    svc = cli._build_services(_wecom_cfg("required"), Platform("local"), _svc_layout(tmp_path))
    assert svc._wecom_policy.is_required  # OSR-002：wecom 独立策略
    assert not svc._policy.is_required  # store 不被折叠为最严
