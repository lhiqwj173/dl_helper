"""OSR-004/OSR-005：省略 sweep --project-dir 时按调用者当前目录解析外部项目。

验证分三层，全部不执行完整训练，保证在合理时间内稳定完成：
1. CLI 层：`_cmd_sweep` 省略 --project-dir 时解析为调用者 cwd 并传给 run_sweep。
2. 子进程 argv 层：预检(contract)与 trial 子进程省略 project_dir 时，argv 都携带
   `--project-dir <调用者 cwd>`，从而子进程能在该 cwd 导入外部 Experiment。
3. 轻量端到端：仅 preflight-only 子进程（零拟合），确认外部 cwd 下能导入 Experiment。

现有测试被 tests/conftest.py 全局注入 examples/ 到 PYTHONPATH 掩盖；本测试显式清空
PYTHONPATH。
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

from dl_helper.training.sweep import parse_sweep_manifest, resolve_trial_configs

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXAMPLES = os.path.join(_REPO, "examples")


def test_cmd_sweep_resolves_caller_cwd(monkeypatch, tmp_path):
    """OSR-004：_cmd_sweep 省略 --project-dir 时把调用者 cwd（绝对路径）传给 run_sweep。"""
    import dl_helper.training.cli as cli
    import dl_helper.training.sweep as sweep_mod

    args = types.SimpleNamespace(command="sweep", sweep="x.yaml",
                                 project_dir=None, resume=False)
    captured = {}

    def fake_run_sweep(manifest, resume=False, project_dir=None):
        captured["manifest"] = manifest
        captured["resume"] = resume
        captured["project_dir"] = project_dir
        return 0

    monkeypatch.setattr(sweep_mod, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(cli, "_prepare_project_dir",
                        lambda p: os.path.realpath(str(tmp_path)))
    monkeypatch.chdir(tmp_path)
    assert cli._cmd_sweep(args) == 0
    assert captured["manifest"] == "x.yaml"
    assert captured["resume"] is False
    assert captured["project_dir"] == os.path.realpath(str(tmp_path)), \
        "省略 --project-dir 必须解析为调用者 cwd 的实路径"


def _external_manifest_and_trial(tmp_path):
    """复制示例 sweep 配置到临时目录并改写 base output_root，返回 (manifest_path, trial)。"""
    import shutil

    import yaml

    sweep_src = os.path.join(_EXAMPLES, "configs", "sweeps", "toy-learning-rate")
    sweep_tmp = tmp_path / "sweep"
    shutil.copytree(sweep_src, sweep_tmp)
    base_path = sweep_tmp / "base.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base["run"]["output_root"] = str(tmp_path / "out")
    base_path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    manifest_path = str(sweep_tmp / "sweep.yaml")
    manifest = parse_sweep_manifest(manifest_path)
    trial = manifest.trials[0]
    return manifest_path, manifest, trial


def test_trial_subprocess_passes_default_cwd(tmp_path, monkeypatch):
    """OSR-004：trial 子进程接收 run_sweep 解析后的 cwd，argv 携带 --project-dir <cwd>。"""
    from dl_helper.training.sweep import _run_trial_subprocess

    _, manifest, trial = _external_manifest_and_trial(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    cwd = os.path.realpath(_EXAMPLES)

    calls = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["cwd"] = kwargs.get("cwd")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    build_experiment_cfg = resolve_trial_configs(manifest)[0][1]

    # project_dir 由上层 run_sweep 解析为调用者 cwd 后传入
    code = _run_trial_subprocess(manifest, trial, build_experiment_cfg,
                                 run_id="toy-lr-sweep-v1--lr-1e-2",
                                 sweep_dir=str(tmp_path / "sweep"), project_dir=cwd)
    assert code == 0
    assert "--project-dir" in calls["argv"]
    assert calls["argv"][calls["argv"].index("--project-dir") + 1] == cwd
    # --experiment 指向 manifest 声明的外部项目实验
    assert "--experiment" in calls["argv"]
    assert calls["argv"][calls["argv"].index("--experiment") + 1] == manifest.experiment


def test_contract_subprocess_passes_default_cwd(tmp_path, monkeypatch):
    """OSR-004：预检(contract)子进程接收解析后的 cwd，argv 携带 --project-dir <cwd>。"""
    from dl_helper.training.sweep import _emit_evaluation_contract

    _, manifest, trial = _external_manifest_and_trial(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    cwd = os.path.realpath(_EXAMPLES)

    captured = {"args": None, "cwd": None}

    def fake_run(argv, **kwargs):
        captured["args"] = argv
        captured["cwd"] = kwargs.get("cwd")
        contract = ('{"schema_version": 1, "backend": "torch", '
                    '"experiment_ref": "experiments.toy_multiclass:build_experiment", '
                    '"valid": true, "errors": [], "splits": {"val": {"fingerprint": "f"}}}')
        return types.SimpleNamespace(returncode=0, stdout=contract + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    build_experiment_cfg = resolve_trial_configs(manifest)[0][1]

    result = _emit_evaluation_contract(trial, build_experiment_cfg, manifest, project_dir=cwd)
    assert result is not None and result.get("valid") is True
    assert "--project-dir" in captured["args"]
    assert captured["args"][captured["args"].index("--project-dir") + 1] == cwd
    assert "--preflight-only" in captured["args"]


def test_run_sweep_resolves_cwd_for_subprocesses(tmp_path, monkeypatch):
    """OSR-005(组合)：run_sweep(project_dir=None) 把调用者 cwd 传给预检与 trial 的公共入口。"""
    import dl_helper.training.sweep as sweep

    _, manifest, _ = _external_manifest_and_trial(tmp_path)
    manifest_path = os.path.join(str(tmp_path / "sweep"), "sweep.yaml")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(_EXAMPLES)

    captured = {}

    def fake_locked(manifest, sweep_dir, layout, resume, output_root, services, project_dir):
        captured["project_dir"] = project_dir
        return 0

    monkeypatch.setattr(sweep, "_run_sweep_locked", fake_locked)
    code = sweep.run_sweep(manifest_path, project_dir=None)
    assert code == 0
    assert captured["project_dir"] == os.path.realpath(_EXAMPLES), \
        "run_sweep 省略 project_dir 必须解析为调用者 cwd 实路径"


def test_external_project_preflight_imports_from_cwd(tmp_path, monkeypatch):
    """OSR-005(轻量)：从外部 cwd 启动 preflight（零拟合）能导入该目录 Experiment。"""
    _, manifest, _ = _external_manifest_and_trial(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(_EXAMPLES)

    proc = subprocess.run(
        [sys.executable, "-m", "dl_helper.training.cli", "train",
         "--config", os.path.join(str(tmp_path / "sweep"), "base.yaml"),
         "--variant", os.path.join(str(tmp_path / "sweep"), "variants", "lr-1e-2.yaml"),
         "--experiment", manifest.experiment, "--preflight-only",
         "--project-dir", os.path.realpath(_EXAMPLES)],
        cwd=os.path.realpath(_EXAMPLES),
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    assert proc.returncode == 0, f"preflight 失败(rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    # contract 输出应有效且指向该外部项目实验；未导入成功会以非零 rc/错误 stderr 失败
    assert "experiments.toy_multiclass" in proc.stdout