"""dl_helper.training —— Kaggle 通用深度学习训练平台公共包。

本包导入无副作用：不连接网络、不解析 Secret、不构造实验、不导入 transformers 等重库。
"""
from .contracts import (
    DataIdentity,
    EstimatorBatch,
    Experiment,
    LoaderDataModule,
    LossResult,
    MetricDefinition,
    MetricState,
    PredictedBatch,
    PreparedBatch,
    ResumableMapDataModule,
    SchedulerBinding,
    SklearnExperiment,
    TorchExperiment,
)

__all__ = [
    "DataIdentity",
    "EstimatorBatch",
    "Experiment",
    "LoaderDataModule",
    "LossResult",
    "MetricDefinition",
    "MetricState",
    "PredictedBatch",
    "PreparedBatch",
    "ResumableMapDataModule",
    "SchedulerBinding",
    "SklearnExperiment",
    "TorchExperiment",
]

__version__ = "1.0.0"
