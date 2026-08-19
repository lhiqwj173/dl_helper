"""任务 3.7：CLI 参数矩阵与命令分派。"""
from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from dl_helper.training.cli import build_parser, main
from dl_helper.training.config import default_schema


def test_parser_exposes_four_commands_without_doctor():
    parser = build_parser()
    sub = next(a for a in parser._actions if getattr(a, "dest", None) == "command")
    for cmd in ("train", "report", "sweep", "sweep-report"):
        assert cmd in sub.choices
    assert "doctor" not in sub.choices


def test_train_success(tmp_path):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 10, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = "cli-train"
    schema["run"]["output_root"] = str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")

    code = main([
        "train",
        "--config", str(cfg_path),
        "--experiment", "experiments.toy_multiclass:build_experiment",
    ])
    assert code == 0


def test_train_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["nonexistent"])


def test_train_missing_config_raises(tmp_path):
    """缺失配置文件：main 原样 raise，由入口以非零退出。"""
    with pytest.raises(Exception):
        main([
            "train",
            "--config", str(tmp_path / "missing.yaml"),
            "--experiment", "experiments.toy_multiclass:build_experiment",
        ])


def test_train_preflight_only_success(tmp_path):
    schema = default_schema()
    schema["run"]["id"] = "cli-doctor"
    schema["run"]["output_root"] = str(tmp_path)
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    cfg_path = tmp_path / "doctor.yaml"
    cfg_path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    code = main([
        "train",
        "--config", str(cfg_path),
        "--experiment", "experiments.toy_multiclass:build_experiment",
        "--preflight-only",
    ])
    assert code == 0


# ---------- D-001：库模块边界校验 ----------
def _boundary_cfg(tmp_path, run_id="boundary", output_root=None):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["run"]["output_root"] = output_root or str(tmp_path)
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    path = tmp_path / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_boundary_rejects_dl_helper_experiment(tmp_path):
    from dl_helper.training.cli import CliError
    cfg_path = _boundary_cfg(tmp_path)
    with pytest.raises(CliError, match="库包 dl_helper 内"):
        main(["train", "--config", cfg_path,
              "--experiment", "dl_helper.training.backends.torch_backend:build_torch_components"])


def test_boundary_rejects_config_inside_package(tmp_path, monkeypatch):
    import dl_helper.training.cli as cli
    from dl_helper.training.cli import CliError
    monkeypatch.setattr(cli, "_DL_HELPER_PACKAGE_REALPATH", str(tmp_path / "pkg"))
    cfg_path = tmp_path / "pkg" / "in.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(yaml.safe_dump(default_schema(), allow_unicode=True), encoding="utf-8")
    with pytest.raises(CliError, match="库包 dl_helper 目录内"):
        main(["train", "--config", str(cfg_path),
              "--experiment", "experiments.toy_multiclass:build_experiment"])


def test_boundary_rejects_output_root_inside_package(tmp_path, monkeypatch):
    import dl_helper.training.cli as cli
    from dl_helper.training.cli import CliError
    monkeypatch.setattr(cli, "_DL_HELPER_PACKAGE_REALPATH", str(tmp_path / "pkg"))
    # 配置在包外，output root 在包内
    cfg_path = _boundary_cfg(tmp_path, output_root=str(tmp_path / "pkg" / "runs"))
    with pytest.raises(CliError, match="库包 dl_helper 目录内"):
        main(["train", "--config", cfg_path,
              "--experiment", "experiments.toy_multiclass:build_experiment"])


def test_boundary_external_project_passes(tmp_path):
    # 配置与 output root 均在包外、experiment 为外部项目 → 允许（preflight-only 证明通过边界）
    from dl_helper.training.cli import main
    cfg_path = _boundary_cfg(tmp_path)
    code = main(["train", "--config", cfg_path, "--preflight-only",
                 "--experiment", "experiments.toy_multiclass:build_experiment"])
    assert code == 0


# ---------- D-002：resume 只留 none/required，省略为内部 auto ----------
def test_parser_rejects_explicit_resume_auto():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--resume", "auto", "--config", "x.yaml",
                           "--experiment", "y:build"])


def test_parser_resume_choices_are_none_and_required():
    parser = build_parser()
    sub = next(a for a in parser._actions if getattr(a, "dest", None) == "command")
    train_parser = sub.choices["train"]
    res = next(a for a in train_parser._actions if a.dest == "resume")
    assert set(res.choices) == {"none", "required"}
    assert res.default is None


def test_train_omitted_resume_resolves_to_internal_auto(tmp_path, monkeypatch):
    """省略 --resume 时内部使用 auto；有最新本地 checkpoint 则恢复。"""
    import dl_helper.training.backends.torch_backend as tb
    from dl_helper.training.backends.base import BackendResult

    cfg_path = _boundary_cfg(tmp_path, run_id="auto-default")
    seen = {}

    def fake_worker(experiment_ref, config, layout, rank, world, resume, **kw):
        seen["resume"] = resume
        return BackendResult(status="succeeded", epoch=1, batch_in_epoch=0, global_step=1)

    monkeypatch.setattr(tb, "run_worker", fake_worker)
    code = main(["train", "--config", cfg_path,
                 "--experiment", "experiments.toy_multiclass:build_experiment"])
    assert code == 0
    assert seen["resume"] == "auto"
