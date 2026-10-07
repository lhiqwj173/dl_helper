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


def test_train_success(tmp_path, monkeypatch):
    from dataclasses import replace
    from dl_helper.training.platform import Platform
    resolve = Platform.resolve_torch_resources
    monkeypatch.setattr(Platform, "resolve_torch_resources",
                        lambda self, config, batch: replace(resolve(self, config, batch), num_workers=0,
                                                            persistent_workers=False, prefetch_factor=None))
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
    import json
    from zipfile import ZipFile
    with ZipFile(tmp_path / "runs/cli-train/checkpoint-archives/last-checkpoint.zip") as archive:
        manifest = json.loads(archive.read("resume-manifest.json").decode("utf-8"))
    assert manifest["phase"] == "final" and manifest["position"]["epoch"] == 1


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


def test_checkpoint_paths_parser_and_preflight(tmp_path):
    from dl_helper.training.cli import CliError
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "x.yaml", "--experiment", "x:build",
                              "--checkpoint-input", "first", "--checkpoint-input", "second"])
    assert args.checkpoint_input == ["first", "second"]
    cfg_path = _boundary_cfg(tmp_path)
    with pytest.raises(CliError, match="output-root"):
        main(["train", "--config", cfg_path, "--experiment", "experiments.toy_multiclass:build_experiment",
              "--checkpoint-export-dir", str(tmp_path.parent / "outside"), "--preflight-only"])


@pytest.mark.parametrize("dataset_available", [False, True])
def test_dataset_resume_keeps_alist_fallback(tmp_path, monkeypatch, dataset_available):
    """有 Dataset 用 Dataset；无 Dataset 仍查询原 AList 恢复入口。"""
    import dl_helper.training.cli as cli
    import dl_helper.training.doctor as doctor
    import dl_helper.training.backends.torch_backend as backend
    from dl_helper.training.backends.base import BackendResult
    from dl_helper.training.config import parse_config
    from dl_helper.training.checkpoint_archive import export_checkpoint_archive
    from test_checkpoint_archive import make_checkpoint

    schema = default_schema()
    schema["run"].update(id="archive-test", source_revision="source-v1", output_root=str(tmp_path))
    schema["distributed"]["num_processes"] = 1
    schema["remote"] = {"type": "alist", "host": "https://alist.example", "base_path": "/runs",
                        "user_secret_key": "USER", "password_secret_key": "PASSWORD",
                        "connect_timeout_seconds": 1, "read_timeout_seconds": 1, "max_attempts": 1,
                        "async_upload": False, "failure_policy": "required"}
    cfg = parse_config(schema)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    inputs = tmp_path / "dataset"
    if dataset_available:
        source = tmp_path / "source"
        checkpoint = make_checkpoint(source, cfg)
        export_checkpoint_archive(source, checkpoint, inputs, cfg)

    calls = []

    class Services:
        def restore_latest_checkpoint(self, run_id):
            calls.append(run_id)
            # 服务本身的 TAR/GZIP 恢复由 test_alist_store 覆盖。
            return None

    monkeypatch.setattr(doctor, "validate_training_start", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_build_services", lambda *a: Services())
    seen = []

    def worker(experiment, config, layout, *a, **k):
        from dl_helper.training.checkpoint import read_latest
        seen.append(read_latest(layout.path("checkpoints")) is not None)
        return BackendResult(status="succeeded", epoch=1, batch_in_epoch=0, global_step=1)

    monkeypatch.setattr(backend, "run_worker", worker)
    assert cli.main(["train", "--config", str(cfg_path), "--experiment", "experiments.toy_multiclass:build_experiment",
                     "--checkpoint-input", str(inputs)]) == 0
    assert calls == ([] if dataset_available else ["archive-test"])
    assert seen == [dataset_available]


def test_project_checkpoint_validator_rejects_before_worker(tmp_path, monkeypatch):
    import dl_helper.training.cli as cli
    import dl_helper.training.backends.torch_backend as backend
    from dl_helper.training.config import parse_config
    from test_checkpoint_archive import make_checkpoint

    schema = default_schema()
    schema["run"].update(id="archive-test", source_revision="source-v1", output_root=str(tmp_path))
    schema["distributed"]["num_processes"] = 1
    config = parse_config(schema)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    source = tmp_path / "source"
    make_checkpoint(source, config)
    (tmp_path / "fixture_checkpoint_validator.py").write_text(
        'def validate(run_dir, checkpoint_dir):\n    raise ValueError("辅助权重与历史指标不一致")\n', encoding="utf-8")
    monkeypatch.setattr(backend, "run_worker", lambda *a, **k: pytest.fail("校验失败后不得进入训练"))
    with pytest.raises(ValueError, match="辅助权重与历史指标不一致"):
        cli.main(["train", "--config", str(cfg_path), "--project-dir", str(tmp_path),
                  "--experiment", "experiments.toy_multiclass:build_experiment",
                  "--checkpoint-input", str(source),
                  "--checkpoint-validator", "fixture_checkpoint_validator:validate"])
