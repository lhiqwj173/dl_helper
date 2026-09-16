"""任务 6.3：隔离顺序 trial coordinator。"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from dl_helper.training.sweep import SweepError, run_sweep


def _base_yaml(output_root):
    return {
        "schema_version": 1,
        "run": {"name": "toy-lr", "id": None, "output_root": output_root,
                "source_revision": None, "seed": 42, "tags": {}},
        "experiment": {"lr": 0.01},
        "training": {"max_epochs": 1, "log_every_steps": 1},
        "backend": {"type": "torch", "torch": {
            "gradient_accumulation_steps": 1, "mixed_precision": "no", "compile": False,
            "clip_grad_norm": 1.0, "deterministic": "off", "matmul_precision": "high",
            "find_unused_parameters": False}, "sklearn": None},
        "distributed": {"num_processes": 1},
        "selection": {"metric": "val/loss", "mode": "min", "patience": 20, "min_delta": 0.0},
        "checkpoint": {"every_epochs": None, "every_optimizer_steps": None,
                       "keep_last": 1},

        "report": {"enabled": True, "curve_sample_limit": 100000,
                   "prediction_sample_limit": 10000, "prediction_splits": ["val"]},
        "remote": {"type": "none"},
        "notifications": {"type": "none"},
    }


def _write_sweep(tmp_path, output_root, comparison="val/loss", mode="min",
                 trials=None, experiment="experiments.toy_multiclass:build_experiment"):
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    base = sweep_dir / "base.yaml"
    base.write_text(yaml.safe_dump(_base_yaml(output_root), allow_unicode=True), encoding="utf-8")
    trials = trials or [("lr-1e-2", 0.01), ("lr-5e-2", 0.05)]
    variants = []
    for name, lr in trials:
        v = sweep_dir / f"{name}.yaml"
        v.write_text(yaml.safe_dump({"experiment": {"lr": lr}}, allow_unicode=True), encoding="utf-8")
        variants.append({"name": name, "variant": f"./{name}.yaml"})
    manifest = {
        "schema_version": 1,
        "sweep": {
            "id": "toy-lr-sweep",
            "experiment": experiment,
            "base_config": "./base.yaml",
            "comparison_metric": comparison,
            "mode": mode,
            "trials": variants,
        },
    }
    path = sweep_dir / "sweep.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_sweep_runs_sequentially_and_ranks(tmp_path):
    output_root = str(tmp_path / "out")
    manifest = _write_sweep(tmp_path, output_root)
    code = run_sweep(manifest)
    assert code == 0
    sweep_dir = os.path.join(output_root, "sweeps", "toy-lr-sweep")
    assert os.path.exists(os.path.join(sweep_dir, "sweep-manifest.json"))
    assert os.path.exists(os.path.join(sweep_dir, "best-trial.json"))
    # 两个 trial run 产物
    for name in ("lr-1e-2", "lr-5e-2"):
        assert os.path.exists(os.path.join(output_root, "runs", f"toy-lr-sweep--{name}",
                                           "metrics", "summary.json"))
    # trials.jsonl 记录
    lines = [json.loads(l) for l in open(os.path.join(sweep_dir, "trials.jsonl"), encoding="utf-8")]
    assert len(lines) >= 2
    # 排名
    m = json.load(open(os.path.join(sweep_dir, "sweep-manifest.json"), encoding="utf-8"))
    assert len(m["ranking"]) == 2
    assert m["best_trial"] in ("lr-1e-2", "lr-5e-2")


def test_sweep_trial_failure_stops_and_no_best(tmp_path):
    output_root = str(tmp_path / "out2")
    manifest = _write_sweep(tmp_path, output_root, experiment="nonexistent_module:build")
    code = run_sweep(manifest)
    assert code != 0
    sweep_dir = os.path.join(output_root, "sweeps", "toy-lr-sweep")
    assert os.path.exists(os.path.join(sweep_dir, "failure.json"))
    assert not os.path.exists(os.path.join(sweep_dir, "sweep-manifest.json"))
    assert not os.path.exists(os.path.join(sweep_dir, "best-trial.json"))


def test_concurrent_lock_fails(tmp_path):
    from dl_helper.training.sweep import _acquire_lock
    sweep_dir = os.path.join(str(tmp_path), "sweeps", "locked")
    os.makedirs(sweep_dir, exist_ok=True)
    fd1 = _acquire_lock(sweep_dir)
    try:
        with pytest.raises(SweepError):
            _acquire_lock(sweep_dir)
    finally:
        os.close(fd1)


def _write_existing_pause(output_root):
    from dl_helper.training.artifacts import write_json

    sweep_dir = os.path.join(output_root, "sweeps", "toy-lr-sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    write_json(os.path.join(sweep_dir, "pause-manifest.json"), {"status": "preempted"})
    return sweep_dir


def test_resume_preflight_failure_replaces_pause(tmp_path, monkeypatch):
    from dl_helper.training import sweep as sweep_module

    output_root = str(tmp_path / "out-preflight")
    manifest = _write_sweep(tmp_path, output_root)
    sweep_dir = _write_existing_pause(output_root)
    monkeypatch.setattr(sweep_module, "_validate_resume", lambda *_args: None)
    monkeypatch.setattr(
        sweep_module,
        "_emit_evaluation_contract",
        lambda *_args: {"valid": False, "errors": ["contract drift"]},
    )

    assert run_sweep(manifest, resume=True) == 1
    assert os.path.exists(os.path.join(sweep_dir, "failure.json"))
    assert not os.path.exists(os.path.join(sweep_dir, "pause-manifest.json"))


def test_resume_trial_failure_replaces_pause(tmp_path, monkeypatch):
    from dl_helper.training import sweep as sweep_module

    output_root = str(tmp_path / "out-trial")
    manifest = _write_sweep(tmp_path, output_root)
    sweep_dir = _write_existing_pause(output_root)
    monkeypatch.setattr(sweep_module, "_validate_resume", lambda *_args: None)
    monkeypatch.setattr(sweep_module, "_emit_evaluation_contract", lambda *_args: {"valid": True})
    monkeypatch.setattr(sweep_module, "_compare_contracts", lambda *_args: None)
    monkeypatch.setattr(sweep_module, "_run_trial_subprocess", lambda *_args: 7)

    assert run_sweep(manifest, resume=True) == 7
    assert os.path.exists(os.path.join(sweep_dir, "failure.json"))
    assert not os.path.exists(os.path.join(sweep_dir, "pause-manifest.json"))


def test_resume_required_service_failure_replaces_pause(tmp_path, monkeypatch):
    from dl_helper.training import sweep as sweep_module
    from dl_helper.training.services import ServiceDeliveryError

    class RequiredServices:
        def start_sweep(self, sweep_id, **fields):
            return None

        def trial_event(self, sweep_id, trial, status, **fields):
            return None

        def finalize_sweep(self, sweep_id, status, **fields):
            raise ServiceDeliveryError("required service failed")

    output_root = str(tmp_path / "out-service")
    manifest = _write_sweep(tmp_path, output_root)
    sweep_dir = _write_existing_pause(output_root)
    monkeypatch.setattr(sweep_module, "_build_sweep_services", lambda *_args: RequiredServices())
    monkeypatch.setattr(sweep_module, "_validate_resume", lambda *_args: None)
    monkeypatch.setattr(sweep_module, "_emit_evaluation_contract", lambda *_args: {"valid": True})
    monkeypatch.setattr(sweep_module, "_compare_contracts", lambda *_args: None)
    monkeypatch.setattr(sweep_module, "_run_trial_subprocess", lambda *_args: 7)

    with pytest.raises(ServiceDeliveryError):
        run_sweep(manifest, resume=True)
    assert os.path.exists(os.path.join(sweep_dir, "failure.json"))
    assert not os.path.exists(os.path.join(sweep_dir, "pause-manifest.json"))


class _RecordingServices:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def start_sweep(self, sweep_id, **fields):
        self.events.append((f"sweep_started:{sweep_id}", fields))

    def trial_event(self, sweep_id, trial, status, **fields):
        self.events.append((f"trial_{status}:{trial}", fields))

    def finalize_sweep(self, sweep_id, status, **fields):
        self.events.append((f"sweep_{status}", fields))


def _patch_sweep_services(monkeypatch, services, handler):
    from dl_helper.training import sweep as sweep_module

    monkeypatch.setattr(sweep_module, "_build_sweep_services", lambda *_args: services)
    monkeypatch.setattr(sweep_module, "_emit_evaluation_contract", lambda *_args: {"valid": True})
    monkeypatch.setattr(sweep_module, "_compare_contracts", lambda *_args: None)
    monkeypatch.setattr(sweep_module, "_run_trial_subprocess", handler)


def test_sweep_notification_fields_on_success(tmp_path, monkeypatch):
    """sweep/trial 成功通知：进度、比较指标副本与 best 值。"""
    from dl_helper.training.artifacts import write_json

    output_root = str(tmp_path / "out-notify-ok")
    manifest = _write_sweep(tmp_path, output_root)
    services = _RecordingServices()

    def fake_trial(_manifest, trial, _config, run_id, _sweep_dir, _project_dir=None):
        write_json(os.path.join(output_root, "runs", run_id, "metrics", "summary.json"), {
            "schema_version": 1,
            "metric_definitions": {
                "loss": {"exact": True, "evaluation_scope": "full", "direction": "min"},
            },
            "stage_metrics": {"val": {"val/loss": 0.25 if trial.name == "lr-1e-2" else 0.1}},
        })
        return 0

    _patch_sweep_services(monkeypatch, services, fake_trial)

    assert run_sweep(manifest) == 0
    assert services.events[0] == ("sweep_started:toy-lr-sweep",
                                  {"trials": 2, "comparison": "val/loss", "mode": "min"})
    assert ("trial_started:lr-1e-2", {"progress": "1/2"}) in services.events
    assert ("trial_succeeded:lr-5e-2", {"progress": "2/2", "metrics": "val/loss=0.1"}) in services.events
    assert ("sweep_succeeded", {"best": "lr-5e-2", "metrics": "val/loss=0.1"}) in services.events


def test_sweep_notification_fields_on_trial_failure(tmp_path, monkeypatch):
    """Trial 失败通知：进度、退出码与 run failure.json 的失败信息。"""
    from dl_helper.training.artifacts import write_json

    output_root = str(tmp_path / "out-notify-fail")
    manifest = _write_sweep(tmp_path, output_root)
    services = _RecordingServices()

    def fake_trial(_manifest, trial, _config, run_id, _sweep_dir, _project_dir=None):
        write_json(os.path.join(output_root, "runs", run_id, "failure.json"), {
            "schema_version": 1,
            "exception_type": "RuntimeError",
            "message": "CUDA out of memory",
            "epoch": 3,
            "global_step": 120,
        })
        return 7

    _patch_sweep_services(monkeypatch, services, fake_trial)

    assert run_sweep(manifest) == 7
    failure_fields = {"error_type": "RuntimeError", "message": "CUDA out of memory",
                      "epoch": 3, "step": 120}
    assert ("trial_failed:lr-1e-2", {"progress": "1/2", **failure_fields}) in services.events
    assert ("sweep_failed", {"trial": "lr-1e-2", "exit_code": 7, "progress": "1/2",
                             **failure_fields}) in services.events


def test_trial_checkpoint_field_reads_run_pause_manifest(tmp_path):
    """暂停通知的恢复检查点取自 run pause manifest；缺失/损坏不阻断通知。"""
    from dl_helper.training.artifacts import write_json
    from dl_helper.training.sweep import _trial_checkpoint_field

    output_root = str(tmp_path)
    write_json(os.path.join(output_root, "runs", "r1", "pause-manifest.json"),
               {"schema_version": 1, "resume_checkpoint": "epoch-000003-step-00000120"})
    assert _trial_checkpoint_field(output_root, "r1") == "epoch-000003-step-00000120"
    assert _trial_checkpoint_field(output_root, "missing") == ""
    with open(os.path.join(output_root, "runs", "r1", "pause-manifest.json"), "w",
              encoding="utf-8") as f:
        f.write("{not json")
    assert _trial_checkpoint_field(output_root, "r1") == ""
