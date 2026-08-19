"""补充 sweep.py / cli.py 覆盖率。"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from dl_helper.training.config import default_schema, parse_config


def _base_yaml(output_root):
    return {
        "schema_version": 1,
        "run": {"name": "t", "id": None, "output_root": output_root,
                "source_revision": None, "seed": 42, "tags": {}},
        "experiment": {"lr": 0.01},
        "training": {"max_epochs": 1, "log_every_steps": 1},
        "backend": {"type": "torch", "torch": {
            "gradient_accumulation_steps": 1, "mixed_precision": "no", "compile": False,
            "clip_grad_norm": 1.0, "deterministic": "off", "matmul_precision": "high",
            "find_unused_parameters": False}, "sklearn": None},
        "distributed": {"num_processes": 1},
        "selection": {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0},
        "checkpoint": {"every_epochs": None, "every_optimizer_steps": None,
                       "keep_last": 1},

        "report": {"enabled": True, "curve_sample_limit": 100000,
                   "prediction_sample_limit": 10000, "prediction_splits": ["val"]},
        "remote": {"type": "none"},
        "notifications": {"type": "none"},
    }


def _write_sweep(tmp_path, output_root, id_="cov-sweep"):
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "base.yaml").write_text(yaml.safe_dump(_base_yaml(output_root), allow_unicode=True),
                                         encoding="utf-8")
    for name, lr in [("a", 0.01), ("b", 0.05)]:
        (sweep_dir / f"{name}.yaml").write_text(
            yaml.safe_dump({"experiment": {"lr": lr}}, allow_unicode=True), encoding="utf-8")
    man = {
        "schema_version": 1,
        "sweep": {"id": id_, "experiment": "experiments.toy_multiclass:build_experiment",
                  "base_config": "./base.yaml", "comparison_metric": "val/loss", "mode": "min",
                  "trials": [{"name": "a", "variant": "./a.yaml"}, {"name": "b", "variant": "./b.yaml"}]},
    }
    path = sweep_dir / "sweep.yaml"
    path.write_text(yaml.safe_dump(man, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_sweep_resolve_output_root_from_config(tmp_path):
    from dl_helper.training.sweep import _resolve_output_root, parse_sweep_manifest
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    resolved = _resolve_output_root(m)
    assert resolved == output_root


def test_sweep_checksum_stable(tmp_path):
    from dl_helper.training.sweep import parse_sweep_manifest, sweep_checksum
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m1 = parse_sweep_manifest(manifest)
    m2 = parse_sweep_manifest(manifest)
    assert sweep_checksum(m1) == sweep_checksum(m2)
    assert len(sweep_checksum(m1)) == 64


def test_sweep_resolve_trial_configs(tmp_path):
    from dl_helper.training.sweep import parse_sweep_manifest, resolve_trial_configs
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    m = parse_sweep_manifest(manifest)
    configs = resolve_trial_configs(m)
    assert len(configs) == 2
    for trial, cfg in configs:
        assert cfg.run.id is None  # variant 不提供 run.id


def test_sweep_generate_report_failure_path(tmp_path):
    """无 manifest 的 sweep 目录生成进度报告。"""
    from dl_helper.training.reporting import generate_sweep_report
    sweep_dir = tmp_path / "sweeps" / "s"
    os.makedirs(sweep_dir, exist_ok=True)
    index = generate_sweep_report(str(sweep_dir))
    assert os.path.exists(index)


def test_cli_train_variant(tmp_path):
    from dl_helper.training.cli import main
    base = str(tmp_path / "base.yaml")
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = "cli-v"
    schema["run"]["output_root"] = str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    open(base, "w", encoding="utf-8").write(yaml.safe_dump(schema, allow_unicode=True))
    variant = tmp_path / "v.yaml"
    variant.write_text(yaml.safe_dump({"training": {"max_epochs": 1}}, allow_unicode=True), encoding="utf-8")
    code = main(["train", "--config", base, "--variant", str(variant),
                 "--experiment", "experiments.toy_multiclass:build_experiment"])
    assert code == 0


def test_cli_train_preflight_emit_contract(tmp_path, capsys):
    from dl_helper.training.cli import main
    base = str(tmp_path / "base.yaml")
    schema = default_schema()
    schema["run"]["output_root"] = str(tmp_path)
    schema["run"]["id"] = "cli-doc-emit"
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    open(base, "w", encoding="utf-8").write(yaml.safe_dump(schema, allow_unicode=True))
    code = main(["train", "--config", base, "--preflight-only",
                 "--experiment", "experiments.toy_multiclass:build_experiment"])
    assert code == 0
    out = capsys.readouterr().out
    assert '"backend": "torch"' in out or '"backend"' in out
