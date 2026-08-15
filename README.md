# dl-helper 1.0.0

Kaggle 通用深度学习训练平台：显式双后端（PyTorch / scikit-learn）、严格配置、流式指标、可审计恢复与生命周期服务。

> **破坏性版本**：本版本完整替换旧训练体系。旧 `dl_helper.trainer/tester/tracker` 等模块不再存在；最后一个旧版 commit 为 `62ee1e4bbf42065ec07dbd0fc8d5b4f9b642f7fd`。不提供兼容 shim 或迁移代理。

## 快速开始

```bash
# 本地训练（torch）
D:/programs/miniconda3/python.exe -m dl_helper.training.cli doctor \
    --config configs/sweeps/toy-learning-rate/base.yaml \
    --experiment experiments.toy_multiclass:build_experiment
D:/programs/miniconda3/python.exe -m dl_helper.training.cli train \
    --config configs/sweeps/toy-learning-rate/base.yaml \
    --experiment experiments.toy_multiclass:build_experiment
D:/programs/miniconda3/python.exe -m dl_helper.training.cli report --run runs/<run-id>
```

## 五个命令

| 命令 | 作用 |
|---|---|
| `doctor` | 不训练的后端感知预检；`--emit-evaluation-contract` 输出 sweep 可比性合同 |
| `train` | 运行一次训练；`--variant` 合并严格 patch；`--resume auto/required` |
| `report` | 只读 Artifact 生成离线 HTML 报告（幂等） |
| `sweep` | 顺序运行多 variant；`--resume` 恢复暂停 sweep |
| `sweep-report` | 生成 sweep 聚合报告 |

退出码：`0` 成功、`75` PREEMPTED（可恢复暂停）、其他非零为失败。

## 核心文档

- [训练指南](docs/training/guide.md) — 从零到出报告的全流程教程（新手上路先读这里）
- [配置](docs/training/configuration.md) — schema v1、base/variant、跨字段约束
- [指标](docs/training/metrics.md) — 流式指标、sklearn 金标、公式版本
- [自定义任务](docs/training/custom-task.md) — Task 协议与 PredictedBatch
- [sklearn](docs/training/sklearn.md) — batch/incremental、Pipeline、sample-weight
- [服务](docs/training/services.md) — AList、企业微信、required/record 策略
- [sweep](docs/training/sweeps.md) — manifest、可比性、排名
- [Kaggle](docs/training/kaggle.md) — 固定 revision、预算、恢复
- [Artifacts](docs/training/artifacts.md) — run/sweep schema、检查点

## 依赖

Python `>=3.10,<3.13`。核心依赖：torch、accelerate、numpy、matplotlib、scikit-learn、safetensors、PyYAML、joblib、requests。`dev` extra：pytest、pytest-cov、build。
