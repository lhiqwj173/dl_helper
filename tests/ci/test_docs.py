"""任务 4.1/4.2：文档与 Notebook 合同。

Kaggle 文档/Notebook 的命令必须可直接执行（subprocess + sys.executable、无 PowerShell/
反引号续行/本机解释器路径）；不得出现已删除的 `--resume auto`、`runtime`/`checkpoint.resume`
默认项；所有仓库示例引用必须解析到 `examples/`。
"""
from __future__ import annotations

import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KAGGLE_DOC = os.path.join(ROOT, "docs", "training", "kaggle.md")
NOTEBOOKS = [
    os.path.join(ROOT, "notebook", "kaggle_train_stage1_epoch5.ipynb"),
    os.path.join(ROOT, "notebook", "kaggle_train_stage2_resume_to_epoch15.ipynb"),
]
DOC_FILES = glob.glob(os.path.join(ROOT, "docs", "training", "*.md")) + [os.path.join(ROOT, "README.md")]


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _notebook_code(path: str) -> str:
    data = json.loads(_read(path))
    return "\n".join("".join(c.get("source", [])) for c in data["cells"] if c["cell_type"] == "code")


def test_notebooks_are_valid_json():
    for path in NOTEBOOKS:
        data = json.loads(_read(path))
        assert data["nbformat"] == 4


def _no_powershell_continuation(content: str, path: str) -> None:
    """拒绝反引号续行（行尾单个反引号）；markdown 代码围栏 ``` 不受影响。"""
    for lineno, line in enumerate(content.splitlines(), start=1):
        if line.count("`") == 1 and line.rstrip().endswith("`"):
            raise AssertionError(f"{path}:{lineno} 疑似 PowerShell 反引号续行: {line!r}")


def test_kaggle_docs_use_python_not_powershell():
    for path in [KAGGLE_DOC] + NOTEBOOKS:
        content = _read(path)
        assert "powershell" not in content.lower(), path
        assert "D:/programs/miniconda3" not in content, path
        _no_powershell_continuation(content, path)


def test_kaggle_docs_have_no_removed_defaults():
    """--resume auto 与 root runtime/checkpoint.resume 默认项必须消失。"""
    for path in [KAGGLE_DOC] + NOTEBOOKS:
        content = _read(path)
        assert "--resume auto" not in content, path
        assert '"--resume", "auto"' not in content, path
    # 删除说明允许提到旧字段名，但不得作为可配置 YAML 键出现
    kaggle = _read(KAGGLE_DOC)
    assert "runtime.max_minutes" not in kaggle
    assert "max_minutes:" not in kaggle
    assert "resume:" not in kaggle
    for path in NOTEBOOKS:
        code = _notebook_code(path)
        assert "max_minutes" not in code, path
        assert "checkpoint['resume']" not in code and "resume':" not in code, path


def test_notebooks_subprocess_handle_exit_codes():
    for path in NOTEBOOKS:
        code = _notebook_code(path)
        assert "sys.executable" in code, path
        assert "subprocess" in code, path
        assert "returncode" in code, path


def test_repo_example_references_resolve_to_examples():
    """所有仓库示例引用必须指向 examples/ 下的真实文件。"""
    references = set()
    for path in [KAGGLE_DOC] + DOC_FILES:
        content = _read(path)
        for m in re.finditer(r"examples/(experiments|configs)/[A-Za-z0-9_./\-]+", content):
            references.add(os.path.normpath(m.group(0)))
    assert references, "未找到任何 examples/ 引用，路径测试失效"
    for rel in sorted(references):
        target = os.path.join(ROOT, rel)
        assert os.path.exists(target), f"文档示例引用不存在: {rel}"


def test_kaggle_doc_makes_execution_contracts_explicit():
    content = _read(KAGGLE_DOC)
    assert "0" in content and "75" in content
    # 文档必须说明 75 不是失败
    assert "75" in content and "失败" in content
    assert "execution-policy.json" in content


def test_kaggle_doc_source_revision_is_explicit():
    """OSR-002：指南配置示例的 source_revision 为非空无空白版本标识，不得为 None。"""
    content = _read(KAGGLE_DOC)
    block = content.split("config = {", 1)[1].split("config_path.write_text", 1)[0]
    assert '"source_revision": None' not in block
    assert re.search(r'"source_revision"\s*:\s*"[^"\s]+"', block), \
        "source_revision 必须是非空无空白版本标识"
    assert "必须显式提供" in content, "文档必须说明非 Git 项目应显式提供 source_revision"


def test_kaggle_doc_sweep_accepts_preempt():
    """OSR-003：sweep 单元把 0 和 75 都作为受控结果，不把预算暂停当失败。"""
    content = _read(KAGGLE_DOC)
    assert "not in (0, 75)" in content, "sweep 单元必须接受 0 与 75"
    assert "预算保护暂停" in content


_EXTERNAL_EXPERIMENT_SRC = '''\
"""OSR-002：非 Git 外部项目的最小 sklearn 实验（供文档指南 preflight 验证）。"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dl_helper.training.contracts import (
    DataIdentity,
    EstimatorBatch,
    SklearnExperiment,
)
from dl_helper.training.task import SklearnMulticlassTask


class _ToyBatchDM:
    def __init__(self, seed=0, n_train=80, n_val=20, n_features=6, n_classes=3):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n_train + n_val, n_features))
        y = rng.integers(0, n_classes, n_train + n_val)
        self._x = x
        self._y = y
        self._n_train = n_train
        self._identity = DataIdentity("doc-guide-batch", "1.0", "fp-doc-guide-batch")

    def setup(self, stage):
        return None

    def identity(self):
        return self._identity

    def full_train_data(self):
        return EstimatorBatch(features=self._x[: self._n_train],
                              targets=self._y[: self._n_train], sample_count=self._n_train)

    def evaluation_batches(self, stage):
        if stage == "val":
            yield EstimatorBatch(features=self._x[self._n_train:], targets=self._y[self._n_train:],
                                 sample_count=self._x.shape[0] - self._n_train)


def build_experiment(config: dict) -> SklearnExperiment:
    def estimator_factory():
        return make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True, random_state=42))

    def task_factory():
        return SklearnMulticlassTask(classes=[0, 1, 2])

    return SklearnExperiment(
        name="doc-guide-batch",
        backend="sklearn",
        estimator_factory=estimator_factory,
        datamodule_factory=lambda: _ToyBatchDM(),
        task_factory=task_factory,
        model_config=dict(config),
    )
'''


def test_kaggle_guide_config_preflights_without_git(tmp_path):
    """OSR-002：按指南（显式非空 source_revision）在非 Git 项目目录 preflight 不因 revision 失败。"""
    import subprocess
    import sys as _sys

    import yaml

    from dl_helper.training.config import default_schema

    project = tmp_path / "my-project"
    configs = project / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (project / "my_proj.py").write_text(_EXTERNAL_EXPERIMENT_SRC, encoding="utf-8")

    schema = default_schema()
    schema["run"].update({"name": "my-project", "id": None, "output_root": None,
                          "source_revision": "my-project-v1"})
    schema["experiment"] = {"n_classes": 3}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096,
                                     "n_jobs": None, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    cfg_path = configs / "kaggle.yaml"
    cfg_path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")

    proc = subprocess.run(
        [_sys.executable, "-m", "dl_helper.training.cli", "train",
         "--config", str(cfg_path), "--project-dir", str(project),
         "--experiment", "my_proj:build_experiment", "--preflight-only"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, f"preflight 失败(rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    assert "无法获取 Git revision" not in proc.stderr
