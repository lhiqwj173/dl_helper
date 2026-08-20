# 训练指南

> 面向第一次使用本平台的人。**从头到尾读完这一篇，你就能把「一个训练项目」从零训练到「出模型、出报告」。**
> 其余参考文档都是在某一步需要深挖时再看（文末有速查表）。所有命令默认在**仓库根目录**下运行。

---

## 0. 一分钟理解这个平台

把平台想象成一个**训练工厂**，输入两样东西，输出一套完整产物：

```
你的训练项目目录
  ├── Experiment 模块   <项目>/my_project.py       →  定义「模型长什么样、数据从哪来、用什么指标」
  └── 配置 YAML          <项目>/configs/train.yaml →  定义「怎么训：训几轮、用什么后端、多久存一次」
                        │
                        ▼
  4 个命令：train（自动预检）→ report 报告（→ 可选 sweep 调参）
                        │
                        ▼
  产物：runs/<run-id>/  模型权重、流式指标、预测、HTML 报告、检查点
```

三条最重要的约定：

1. **命令统一入口**：`D:/programs/miniconda3/python.exe -m dl_helper.training.cli <命令>`
2. **退出码**：`0` 成功 · `75` 可恢复暂停（PREEMPTED，预算到点，可继续训）· 其他非零 = 失败
3. **产物目录**：`<run.output_root>/runs/<run-id>/`。本地不写 `output_root` 时默认落在**当前目录**。

下面用一个真实可运行的例子带你走完整条路：`examples/experiments/toy_multiclass.py`（多分类 MLP）+ `examples/configs/sweeps/toy-learning-rate/base.yaml`，以仓库 `examples/` 作为 `--project-dir` 运行。

---

## 1. 第一步：写 Experiment 模块

每个训练项目对应一个 Python 模块，放在**你自己的项目目录**里（仓库自带示例见 `examples/experiments/`）。它唯一的任务：**导出一个 `build_experiment(config)` 函数**，返回一个 `TorchExperiment`（PyTorch）或 `SklearnExperiment`（scikit-learn）。

> `examples/experiments/` 下已有 10 个现成示例，先抄后改是最快的入门方式：
> `toy_multiclass.py` / `toy_regression.py` / `toy_multilabel.py` / `mnist.py` / `sklearn_batch.py` / `sklearn_incremental.py` / `sklearn_pipeline.py` / `toy_custom_task.py` / `toy_multi_input.py` / `toy_multiclass_resumable.py`

### Torch 实验：6 个工厂

```python
# my_project.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dl_helper.training.contracts import DataIdentity, LoaderDataModule, TorchExperiment
from dl_helper.training.task import MulticlassClassificationTask

def build_experiment(config: dict) -> TorchExperiment:
    in_dim   = int(config.get("in_dim", 8))
    num_classes = int(config.get("num_classes", 3))
    batch_size  = int(config.get("batch_size", 16))

    # 1) 数据：TrainLoader 与（可选的）ValLoader
    #    config 里的一切都来自配置文件的 experiment 段
    def datamodule_factory():
        g = torch.Generator().manual_seed(42)
        x = torch.randn(160, in_dim, generator=g)
        y = torch.randint(0, num_classes, (160,), generator=g)
        return LoaderDataModule(
            DataIdentity("my-project", "1.0", "fp-my-project-42"),  # 数据身份，用于恢复校验
            DataLoader(TensorDataset(x[:128], y[:128]), batch_size=batch_size),
            val_dataloader=DataLoader(TensorDataset(x[128:], y[128:]), batch_size=batch_size),
            nominal_train_batch_size=batch_size,
        )

    # 2) 模型：每次调用都返回一个全新 nn.Module
    def model_factory():
        return nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, num_classes))

    # 3) 任务：决定损失函数 + 自动生成该任务的指标（见下）
    def task_factory():
        return MulticlassClassificationTask(num_classes=num_classes)

    # 4) 优化器（接收模型参数）
    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=float(config.get("lr", 0.05)))

    # 5) 调度器（不需要就返回 None）
    def scheduler_factory(optimizer):
        return None

    return TorchExperiment(
        name="my-project",          # 用于默认 run-id，如 my-project-20260814-...
        backend="torch",
        model_factory=model_factory,
        datamodule_factory=datamodule_factory,
        task_factory=task_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        model_config=dict(config),  # 原样保存配置，报告/审计里可见
    )
```

**关于 Task（第 3 点），大多数情况不用写指标**。平台内置三个开箱即用的任务类，各自的指标已自动生成：

| 任务类 | 适用 | 自动指标 |
|---|---|---|
| `MulticlassClassificationTask` | 多分类 | accuracy、balanced_accuracy、macro/weighted P/R/F1、per-class |
| `MultilabelClassificationTask` | 多标签 | micro/macro/weighted F1、subset_accuracy、hamming_loss |
| `RegressionTask` | 回归 | MAE、MSE、R2 |

只有当你要跑自己的损失、自定义评估逻辑时才需要继承 Task（见 [custom-task.md](custom-task.md)）。

### 数据模块怎么选

- **`LoaderDataModule`**：一次性把训练/验证集准备好（如 `toy_multiclass.py`）。适合短训练、不需要中途恢复。
- **`ResumableMapDataModule`**：数据集由「工厂函数 + 状态」构成，训练被打断后能**从断点精确恢复**。长训练、Kaggle 预算训练**必须用它**（见 `examples/experiments/toy_multiclass_resumable.py`、`mnist.py`）。Kaggle 执行策略开启运行预算或省略 `--resume`（自动恢复）时，需要可恢复的数据模块。

### sklearn 实验

换 `SklearnExperiment`，工厂变成 4 个：`estimator_factory` / `datamodule_factory` / `task_factory` / `model_config`，任务类换成 `Sklearn*Task`（`SklearnMulticlassTask` 等）。sklearn 有两种 fit 模式（见 [sklearn.md](sklearn.md)）：

- **`batch`**：一次 `fit(X, y)`。限制严格：`max_epochs=1`；不支持恢复（显式 `--resume required` 在预检阶段失败）。
- **`incremental`**：`partial_fit` 逐 batch 训练，可预算暂停恢复。数据源需要实现 `iter_epoch` / `state_dict` / `load_state_dict`（照抄 `examples/experiments/sklearn_incremental.py`）。

---

## 2. 第二步：写配置 YAML

配置是**唯一**的输入格式，严格解析（schema v1）：不认模板、不认环境变量插值、不认 YAML 合并/别名，重复 key 直接报错。照下面的最小可用配置改即可：

```yaml
# <project-dir>/configs/my_project.yaml
schema_version: 1
run:
  name: my-project        # 用于默认 run-id 前缀
  id: null                # 留空则自动生成 <name>-<UTC时间戳>；要恢复/对齐就手写一个固定 id
  output_root: null       # 输出根目录，null = 当前目录（Kaggle 默认 /kaggle/working/dl-helper-runs）
  source_revision: null   # tag/分支/短 SHA 均可；留空时从 project-dir 的 Git 读取
  seed: 42
  tags: {}
experiment:               # ← 这个 dict 原样传给 build_experiment(config)
  in_dim: 8
  num_classes: 3
  lr: 0.05
training:
  max_epochs: 20
  log_every_steps: 20
backend:
  type: torch             # 或 sklearn
  torch:                  # type=torch 时 sklearn 必须为 null（反之亦然）
    gradient_accumulation_steps: 1
    mixed_precision: "no" # auto / no / fp16 / bf16（no/off 要加引号，否则 YAML 解析成布尔）
    compile: false
    clip_grad_norm: 1.0
    deterministic: strict # strict / warn / off
    matmul_precision: high
    find_unused_parameters: false
  sklearn: null
distributed:
  num_processes: 1        # 或 auto（Kaggle 自动用满可见 GPU）
selection:                # 早停 + 最优模型选择；没有验证集就必须整段写成 null
  metric: val/loss
  mode: min
  patience: 5
  min_delta: 0.0
checkpoint:
  every_epochs: 1         # 每几轮存一次检查点
  every_optimizer_steps: null
  keep_last: 2            # 最多保留几个
report:
  enabled: true
  curve_sample_limit: 100000
  prediction_sample_limit: 10000
  prediction_splits: [val]
remote:                   # 默认无远端；进阶见第 8 节
  type: none
notifications:            # 默认无通知；进阶见第 8 节
  type: none
```

**最容易踩的坑**（完整约束见 [configuration.md](configuration.md)）：

- `backend.type` 必须和 `build_experiment` 返回的类型一致；未选的后端分支必须写 `null`。
- 有验证集时 `selection` 必须存在；`selection.mode` 必须和指标方向一致（loss→min，accuracy→max）。
- Kaggle 执行策略开启运行预算时，数据模块必须支持中途恢复。
- sklearn `fit_mode=batch`：`max_epochs=1`，且不支持恢复。

---

## 3. 第三步：train（自动预检）

训练会先检查配置和后端是否合法，预检失败会聚合错误并立即终止：

```bash
D:/programs/miniconda3/python.exe -m dl_helper.training.cli train \
    --project-dir D:/work/my-project \
    --config D:/work/my-project/configs/my_project.yaml \
    --experiment my_project:build_experiment
```

它会检查：配置合法性、Experiment 与后端是否匹配、数据身份、选择指标、资源、磁盘，以及 Kaggle 的执行策略和必需服务。sweep 的可比性合同由内部预检自动生成。

> `--experiment` 的写法是 `模块路径:函数名`。必须把项目目录通过 `--project-dir` 传入，示例则用仓库的 `examples/` 作为 project dir。

---

## 4. 第四步：train 训练

```bash
D:/programs/miniconda3/python.exe -m dl_helper.training.cli train \
    --project-dir D:/work/my-project \
    --config D:/work/my-project/configs/my_project.yaml \
    --experiment my_project:build_experiment
```

跑完你的项目目录下会出现 `runs/<run-id>/`，里面是完整产物：

```
runs/<run-id>/
  run-manifest.json           # 成功终态清单（含全部产物 SHA256）
  config.resolved.yaml        # 本次实际生效的完整配置（还原现场用）
  environment.json            # 运行环境
  metrics/metrics.jsonl       # 流式指标（每步追加）
  metrics/summary.json        # 汇总指标
  checkpoints/latest.json     # 最新检查点指针
  models/best/  models/last/  # 最优 / 最后模型（.safetensors 或 .joblib）
  predictions/val/            # 验证集预测
  report/index.html           # 训练报告
```

**常用选项**：

| 选项 | 作用 |
|---|---|
| `--variant <file>` | 叠加一个严格 patch YAML（只覆盖 `experiment` 里的值），不改 base 文件 |
| `--run-id <id>` | 显式指定 run-id（恢复、对齐、发报告时用） |
| `--resume none/required` | 显式覆盖恢复策略；省略时内部自动恢复 |

**退出码含义**：`0` 成功；`75` 预算到点暂停（PREEMPTED，产物保留，可以续训）；其他非零失败——先看 `runs/<run-id>/failure.json`（已脱敏的异常与调用栈），再修。

> 想先小跑一把验证流程，可以直接用仓库自带的 toy 示例：
> `--project-dir <repo>/examples --config <repo>/examples/configs/sweeps/toy-learning-rate/base.yaml --experiment experiments.toy_multiclass:build_experiment`

---

## 5. 第五步：report 生成 HTML 报告

训练完成后（或恢复前），从产物生成离线报告：

```bash
D:/programs/miniconda3/python.exe -m dl_helper.training.cli report --run runs/<run-id>
```

纯只读、幂等，可重复生成。加 `--out <目录>` 可以指定输出位置。报告包含运行上下文、各 stage 指标表，以及分类任务的混淆矩阵 / 回归任务的预测散点图。

---

## 6. 进阶一：sweep 超参对比

要对比多个超参组合，不用手动跑多次——写一个 **variant 覆盖** + 一份 **sweep manifest**。

variant 是只含覆盖字段的严格 YAML（递归合并）：

```yaml
# <project-dir>/configs/sweeps/my-sweep/variants/lr-1e-3.yaml
experiment:
  lr: 0.001
```

sweep manifest 把 base + 有序 trials + 比较指标钉在一起：

```yaml
# <project-dir>/configs/sweeps/my-sweep/sweep.yaml
schema_version: 1
sweep:
  id: my-sweep-v1
  experiment: experiments.my_project:build_experiment
  base_config: ./base.yaml
  comparison_metric: val/loss   # 必须 val/ 前缀、可参与比较的指标
  mode: min
  trials:
    - {name: lr-1e-2, variant: ./variants/lr-1e-2.yaml}
    - {name: lr-3e-3, variant: ./variants/lr-3e-3.yaml}
    - {name: lr-1e-3, variant: ./variants/lr-1e-3.yaml}
```

运行：

```bash
# 跑完整 sweep（顺序子进程；任一失败立即停；预算暂停可续）
python -m dl_helper.training.cli sweep --sweep <project-dir>/configs/sweeps/my-sweep/sweep.yaml \
  --project-dir <project-dir>
# 恢复被暂停的 sweep
python -m dl_helper.training.cli sweep --sweep <project-dir>/configs/sweeps/my-sweep/sweep.yaml \
  --project-dir <project-dir> --resume
# 生成聚合对比报告
python -m dl_helper.training.cli sweep-report --sweep-dir <sweep 输出目录>
```

完整规则见 [sweeps.md](sweeps.md)。

---

## 7. 进阶二：接入服务（AList 远端归档 / 企业微信通知）

想让**远端自动归档产物**或**训练结束推企业微信**，把配置里对应的段从 `none` 打开即可。**密码等敏感值不写进 YAML**——配置里放的是环境变量名（Secret key），运行时从环境变量读取：

```yaml
remote:                    # 远端归档（需要 AList 服务）
  type: alist
  host: https://your-alist.example.com
  base_path: /dl-helper/my-project
  user_secret_key: ALIST_USER        # 环境变量名，运行时读 ALIST_USER
  password_secret_key: ALIST_PWD
  connect_timeout_seconds: 10
  read_timeout_seconds: 60
  max_attempts: 3
  async_upload: false
  failure_policy: required            # required=归档失败则训练算失败；record=只记 degraded
notifications:             # 企业微信通知
  type: wecom
  corp_id_secret_key: WECOM_CORP_ID
  corp_secret_key: WECOM_CORP_SECRET
  agent_id_secret_key: WECOM_AGENT_ID
  to_user: "@all"
  connect_timeout_seconds: 10
  read_timeout_seconds: 30
  max_attempts: 3
  failure_policy: required
```

运行前先在环境里设好对应变量（如 `ALIST_USER=...`），训练自动预检会校验 Secret。上传顺序是「先归档不可变文件 → 回读 SHA 校验 → 再写 manifest」；服务审计记录在 `runs/<run-id>/services/service-audit.jsonl`。细节见 [services.md](services.md)。

---

## 8. 进阶三：上 Kaggle

平台为 Kaggle 做了专门适配：

- 任意环境变量以 `KAGGLE` 开头即自动识别为 Kaggle 环境。
- 数据**必须**显式挂在 `/kaggle/input/...`；输出默认 `/kaggle/working/dl-helper-runs`。
- `distributed.num_processes: auto` 会用满所有可见 GPU（CPU 环境 = 1）。
- **预算训练**：Kaggle 由平台执行策略固定为 **660 分钟训练 + 10 分钟收尾**。系统在成功 batch/optimizer step 后做硬截止保护，并在每个完整 epoch 结束时按完整 epoch 平均耗时预测下一轮；若下一轮无法在 650 分钟截止前完成，则当前边界执行“存检查点 → 推送 AList → 刷新通知/服务 → 写 pause manifest”，退出码为 `75`。新 session 里用同样的命令 + **同一个 `--run-id`**（省略 `--resume` 即自动恢复）继续训。预测只是估算，硬截止仍可能提前触发。预算值记录在 run 目录 `execution-policy.json`，用户配置不包含 `runtime`。
- Kaggle 强制启用 AList 和企业微信，且两者 `failure_policy=required`；训练预检会列出缺失 Secret 并终止。

仓库中的 `notebook/kaggle_train_stage1_epoch5.ipynb` 与 `notebook/kaggle_train_stage2_resume_to_epoch15.ipynb` 是当前 15 epoch 两阶段训练示例：第一本在第 5 轮保存并推送可恢复检查点，第二本使用同一 run ID 恢复并完成到第 15 轮。普通训练任务也可以按本页第 1~5 步在自己的 Kaggle Notebook 中运行。

---

## 9. 收尾：产物与恢复语义

- **所有产物原子写入**，文本 UTF-8；`run-manifest.json` 是成功终态，`pause-manifest.json` 是暂停终态，`failure.json` 是失败终态，三者互斥，只存在一个。
- **恢复是严格校验的**：config / backend / data / model 的指纹 + 运行时版本精确匹配才允许加载，防住"换了配置还续训"的坑。
- 产物与检查点格式见 [artifacts.md](artifacts.md)。

---

## 10. 速查表：我想做什么 → 看哪篇

| 场景 | 文档 |
|---|---|
| **新手上路 / 完整流程** | 本指南 |
| 配置全部字段与跨字段约束 | [configuration.md](configuration.md) |
| 写自定义 Task / 自定义损失与评估 | [custom-task.md](custom-task.md) |
| 指标定义、公式版本、流式语义 | [metrics.md](metrics.md) |
| sklearn batch / incremental | [sklearn.md](sklearn.md) |
| 服务（AList / 企业微信 / 失败策略） | [services.md](services.md) |
| sweep 超参对比 | [sweeps.md](sweeps.md) |
| Kaggle 训练 | [kaggle.md](kaggle.md) |
| 产物目录 / 检查点 / 恢复 | [artifacts.md](artifacts.md) |
| 破坏性迁移 / 已删除字段 | [breaking-removal.md](breaking-removal.md) |
