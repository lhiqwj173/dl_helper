"""任务 1.4：公共导入边界 —— 无副作用导入、不触发重库/网络/Secret。"""
from __future__ import annotations

import subprocess
import sys

import pytest


HEAVY_MODULES = ("transformers", "torchmetrics", "stable_baselines3", "imitation",
                 "autogluon", "pandas")


def test_import_training_no_side_effects():
    """子进程导入 dl_helper.training，断言不触发重库、Secret 或网络。"""
    code = (
        "import sys\n"
        "import dl_helper.training\n"
        "heavy = " + repr(HEAVY_MODULES) + "\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, f'import 触发了重库: {loaded}'\n"
        "import dl_helper\n"
        "assert dl_helper.__version__ == '1.0.0'\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_public_api_exports():
    from dl_helper.training import (
        DataIdentity, EstimatorBatch, Experiment, LoaderDataModule, LossResult,
        MetricDefinition, MetricState, PredictedBatch, PreparedBatch,
        ResumableMapDataModule, SchedulerBinding, SklearnExperiment, TorchExperiment,
    )
    for symbol in (
        "DataIdentity", "EstimatorBatch", "Experiment", "LoaderDataModule", "LossResult",
        "MetricDefinition", "MetricState", "PredictedBatch", "PreparedBatch",
        "ResumableMapDataModule", "SchedulerBinding", "SklearnExperiment", "TorchExperiment",
    ):
        assert symbol in dir(dl_helper_module()), symbol


def dl_helper_module():
    import dl_helper.training as m
    return m
