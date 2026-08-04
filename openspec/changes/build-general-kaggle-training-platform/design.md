# Design: Kaggle 通用深度学习训练平台

## Context

当前主调用链为 `test_base -> trainer.run -> notebook_launcher -> run_fn_gpu -> train_fn/val_fn/test_fn -> Tracker -> save_predictions/package_root`。模型和 DataLoader 本身可替换，但 Trainer、Tracker、金融预测、远程存储和平台终止逻辑共享全局状态和同一文件，导致通用样例可以开始训练，却不能可靠完成、恢复或输出结果。

本设计采用仓库级替换：先建立 `dl_helper.training` 的完整新内核与测试，再在同一 change 中删除旧训练器、模型、实验和独立训练栈。最终发布物只有一套共享训练生命周期和 PyTorch/sklearn 两个明确 backend，AList/企业微信由配置启用；不保留旧 import、legacy extra、领域模型或兼容代理。

## Goals

- 以显式协议支持任意 PyTorch 模型、嵌套输入、DataLoader、自定义任务，以及 sklearn estimator/Pipeline 的 batch/incremental 训练。
- 在本地与 Kaggle 共享同一训练语义，并自动使用 Kaggle 可见 CUDA 资源。
- 以版本化公式、sample weight、稳定矩和显式金标精确、流式、分布式地计算核心指标，不让内存随样本数增长。
- 通过不可变检查点、兼容指纹和原子发布实现可审计恢复。
- 产出固定 schema 的训练 Artifact 和任务化 HTML 报告。
- 全链路 Fail Fast，不隐藏配置、网络、指标或文件错误。

## Non-Goals

- 不统一或迁移 RL、模仿学习、AutoGluon、GUI 与旧 C++ 扩展；这些旧实现直接移出本仓库。
- 不推断数据文件格式、标签或任务类型。
- 不支持多节点和 TPU/XLA。
- 不实现自动超参数搜索、自动 OOM 降级或自动调批量；只顺序执行用户显式列出的 variant。
- 不迁移任何旧历史实验或模型；需要继续使用的实验由用户基于新合同重新声明。

## Repository Replacement Boundary

实施完成后，`dl_helper/` 受跟踪内容只能包含新的 `__init__.py` 和 `training/`；通用示例只能位于 `experiments/`，自动测试只能位于根级 `tests/`。以下内容必须从 Git 索引和工作树删除：

- 旧包目录：`dl_helper/acc/`、`models/`、`other_tests/`、`rl/`、`tests/`、`transforms/`。
- 旧包根模块：除 `dl_helper/__init__.py` 外，当前 HEAD 已有的全部 `.py`，包括 trainer/tester/tracker/data/scheduler/tool、AutoGluon、压缩、索引和传输模块。
- 旧运行资产：`cpp/`、`参考/`、现有 `envs/`、现有 `notebook/`、`setup.py`、`requirements.txt`、受跟踪 wheel/tarball/checkpoint；随后只新建 `envs/kaggle_bootstrap.py` 和 `notebook/kaggle_training_template.ipynb`。

`.git/`、`.gitignore`、OpenSpec、Agent/编辑器配置不属于训练实现，不因本 change 删除。删除门禁由 `git ls-files` 驱动，禁止通过 `.gitignore` 隐藏仍受跟踪的旧文件。用户目录、Kaggle Dataset、远程 AList 和未被 Git 跟踪的外部训练产物不在删除范围。

## Target Module Boundaries

| 文件 | 稳定职责 |
|---|---|
| `dl_helper/training/contracts.py` | `Experiment`、`DataModule`、`Task`、`PreparedBatch`、`LossResult`、`SchedulerBinding`、`DataIdentity` 协议和 dataclass |
| `dl_helper/training/config.py` | YAML schema v1、严格解析、跨字段验证、规范化序列化和指纹 |
| `dl_helper/training/task.py` | 多分类、多标签、回归 Task 与默认模型调用规则 |
| `dl_helper/training/metrics.py` | 流式状态、分布式 sum 归约和精确指标计算 |
| `dl_helper/training/engine.py` | backend 无关的 run 状态机、指标/Artifact/服务编排，不包含模型库细节 |
| `dl_helper/training/backends/base.py` | backend 结果、能力和生命周期协议 |
| `dl_helper/training/backends/torch_backend.py` | PyTorch worker、DDP/AMP、优化器和 scheduler 语义 |
| `dl_helper/training/backends/sklearn_backend.py` | sklearn clone、batch fit、incremental partial_fit、预测和 joblib 状态 |
| `dl_helper/training/launcher.py` | 单进程和 `notebook_launcher` 多进程启动；仅传递实验引用和纯配置 |
| `dl_helper/training/checkpoint.py` | 不可变检查点、manifest、latest 指针、保留策略、恢复校验 |
| `dl_helper/training/artifacts.py` | 运行目录、JSON/JSONL、预测分片、原子文件写入和 Artifact schema |
| `dl_helper/training/platform.py` | Local/Kaggle 检测、资源发现、路径、Secret resolver、doctor 和运行预算 |
| `dl_helper/training/remote.py` | `ArtifactStore`、`LocalArtifactStore`、`AListArtifactStore` 和异步同步器 |
| `dl_helper/training/notifications.py` | 企业微信客户端、生命周期事件模板、重试、Secret 脱敏和投递审计 |
| `dl_helper/training/sweep.py` | base/variant 解析、顺序 trial 子进程、恢复、比较和 sweep manifest |
| `dl_helper/training/reporting.py` | 从已落盘 Artifact 生成分类、多标签、回归或通用 HTML 报告 |
| `dl_helper/training/cli.py` | `doctor`、`train`、`report`、`sweep`、`sweep-report` 命令和非零退出行为 |
| `dl_helper/training/__init__.py` | 只导出稳定公共类型和入口，不触发 torchmetrics、transformers 或网络导入 |
| `dl_helper/__init__.py` | 只暴露新平台版本；不重导出或代理任何旧符号 |

## Public Contracts

### Experiment

```python
@dataclass(frozen=True)
class TorchExperiment:
    name: str
    backend: Literal["torch"]
    model_factory: Callable[[], torch.nn.Module]
    datamodule_factory: Callable[[], DataModule]
    task_factory: Callable[[], TorchTask]
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]
    scheduler_factory: Callable[[torch.optim.Optimizer], SchedulerBinding | None]
    model_config: Mapping[str, JSONValue]

@dataclass(frozen=True)
class SklearnExperiment:
    name: str
    backend: Literal["sklearn"]
    estimator_factory: Callable[[], BaseEstimator]
    datamodule_factory: Callable[[], SklearnDataModule]
    task_factory: Callable[[], SklearnTask]
    model_config: Mapping[str, JSONValue]

Experiment = TorchExperiment | SklearnExperiment
```

- CLI 参数使用 `--experiment module.path:build_experiment`；该函数签名必须是 `build_experiment(experiment_config: Mapping[str, JSONValue]) -> Experiment`，返回值只能是上述两个 frozen dataclass。
- launcher 在父进程只校验引用格式，不导入实验模块；Torch worker 独立导入以避免启动前初始化 CUDA，sklearn trial 也在独立子进程构造以隔离 native thread pool 和全局状态。
- Torch 工厂不得返回已经移动到设备、已由 DDP 包装或已执行优化步的对象；sklearn 工厂返回值必须可由 `sklearn.base.clone(..., safe=True)` 克隆并经 `check_is_fitted` 确认为未拟合。
- sklearn Pipeline、ColumnTransformer 和满足 clone/get_params/set_params/fit/predict 的第三方 wrapper 可直接使用；Task 声明与 `is_classifier/is_regressor` 不一致、缺 predict 或配置要求的 predict_proba/decision_function/partial_fit 时预检失败。
- `model_config` 必须可规范化为 JSON，用于兼容指纹；Torch 额外计算参数 name/shape/dtype，sklearn 记录类的 module/qualname、`get_params(deep=True)` 规范 JSON 和库版本作为 `model_signature`。

### DataModule

```python
class DataModule(Protocol):
    supports_mid_epoch_resume: bool
    nominal_train_batch_size: int | None
    def setup(self, stage: Literal["fit", "test", "predict"]) -> None: ...
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader | None: ...
    def test_dataloader(self) -> DataLoader | None: ...
    def predict_dataloader(self) -> DataLoader | None: ...
    def identity(self) -> DataIdentity: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

`DataIdentity` 固定包含 `name: str`、`version: str`、`fingerprint: str`。三者均不能为空；引擎不自动把路径或文件时间当成可信数据版本。`nominal_train_batch_size` 为正整数时用于计算全局有效批量；动态 batch 必须设为 `None`，报告将有效批量标为 `dynamic` 并记录实际 `sample_count` 的最小值和最大值。正常的最后一个不足批次不视为动态 batch，也不得因此失败。无中途恢复能力的 DataModule 令 `supports_mid_epoch_resume=False`，其 `state_dict` 返回空 mapping，并且只能创建 epoch 边界检查点。

提供两个便利实现：

- `LoaderDataModule`：直接包装用户 DataLoader，支持任意 batch，但不声明中途恢复。
- `ResumableMapDataModule`：由 dataset/collate 工厂和 DataLoader 参数构造，保存 epoch、已消费批次数和 sampler 状态；严格确定性模式下要求随机增强由样本键、epoch 和 seed 决定，不依赖未保存的 worker 全局 RNG。

### Sklearn Data Contract

```python
@dataclass(frozen=True)
class EstimatorBatch:
    features: Any
    targets: Any
    sample_count: int
    sample_weight: np.ndarray | None = None
    sample_ids: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

class IncrementalBatchSource(Protocol):
    classes: np.ndarray | None
    nominal_batch_size: int | None
    supports_mid_fit_resume: bool
    def iter_epoch(self, epoch: int) -> Iterable[EstimatorBatch]: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

class SklearnEvaluationDataModule(Protocol):
    def setup(self, stage: Literal["fit", "test", "predict"]) -> None: ...
    def evaluation_batches(self, stage: Literal["train", "val", "test", "predict"]) -> Iterable[EstimatorBatch]: ...
    def identity(self) -> DataIdentity: ...

class SklearnBatchDataModule(SklearnEvaluationDataModule, Protocol):
    def full_train_data(self) -> EstimatorBatch: ...

class SklearnIncrementalDataModule(SklearnEvaluationDataModule, Protocol):
    def incremental_train_data(self) -> IncrementalBatchSource: ...

SklearnDataModule = SklearnBatchDataModule | SklearnIncrementalDataModule
```

`fit_mode=batch` 要求 `SklearnBatchDataModule` 并恰好执行一次 estimator.fit；`fit_mode=incremental` 要求 `SklearnIncrementalDataModule` 并逐批执行 partial_fit。engine 通过显式 experiment mode 验证所需协议，不捕获 `AttributeError` 猜测模式。features 保持 estimator 接受的 ndarray/scipy sparse/array-like 类型，不自动 densify 或转 Tensor；sample_count、targets、weight 和 sample_ids 第一维必须一致，weight 必须有限且非负且总和大于零。

batch estimator 不支持 mid-fit checkpoint；incremental classifier 首批必须由 Task 提供完整、顺序固定的 classes，后续类别集合变化立即失败。预处理必须包含在 sklearn Pipeline 内并只对 train fit；engine 不对 val/test 单独拟合 scaler、encoder 或 imputer，防止数据泄漏。

当训练数据带 `sample_weight` 时，`backend.sklearn.sample_weight_parameter` 必须显式给出传给 `fit/partial_fit` 的参数名（普通 estimator 通常为 `sample_weight`，Pipeline 可为 `step__sample_weight`）；未配置时必须失败，不得丢弃权重。引擎不探测或改写第三方 metadata routing；参数不被 estimator 接受时保留其原始 `TypeError`。评价阶段的 `sample_weight` 始终经 `PredictedBatch` 进入共享指标合同，与训练参数转发彼此独立。

### PreparedBatch And Model Invocation

```python
@dataclass(frozen=True)
class PreparedBatch:
    inputs: Any
    targets: Any
    sample_count: int
    sample_weight: torch.Tensor | None = None
    metadata: Mapping[str, Any] | None = None

@dataclass(frozen=True)
class LossResult:
    numerator: torch.Tensor
    denominator: torch.Tensor | int | float
```

默认模型调用规则固定为：

```python
if isinstance(inputs, Mapping):
    outputs = model(**inputs)
elif isinstance(inputs, tuple):
    outputs = model(*inputs)
else:
    outputs = model(inputs)
```

`list` 被视为单个输入，避免无法区分“模型位置参数”和“序列样本”；需要位置参数的 Task 必须转换为 tuple。`sample_count` 必须为正整数且等于本批评价样本数。`sample_weight` 缺省表示每个样本权重为 1；提供时必须是一维、长度等于 sample_count、有限且非负的浮点 Tensor，单批总权重必须大于零。metadata 只允许 Task/PredictionWriter 消费，训练引擎不得解释字段含义。

### Task

```python
@dataclass(frozen=True)
class PredictedBatch:
    targets: Any
    predictions: Any
    sample_count: int
    scores: Any | None = None
    sample_weight: Any | None = None
    sample_ids: Any | None = None
    metadata: Mapping[str, Any] | None = None

class EvaluationTask(Protocol):
    name: str
    metric_definitions: Mapping[str, MetricDefinition]
    def metric_state(self, stage: str) -> MetricState: ...
    def update_metrics(self, state: MetricState, predicted: PredictedBatch) -> None: ...
    def prediction_arrays(self, predicted: PredictedBatch) -> Mapping[str, np.ndarray]: ...
    def report_kind(self) -> str: ...

class TorchTask(EvaluationTask, Protocol):
    def prepare_batch(self, batch: Any, stage: str) -> PreparedBatch: ...
    def forward(self, model: nn.Module, prepared: PreparedBatch) -> Any: ...
    def loss(self, outputs: Any, prepared: PreparedBatch) -> LossResult: ...
    def to_predicted_batch(self, outputs: Any, prepared: PreparedBatch) -> PredictedBatch: ...

class SklearnTask(EvaluationTask, Protocol):
    estimator_kind: Literal["classifier", "regressor"]
    classes: np.ndarray | None
    required_prediction: Literal["predict", "decision_function", "predict_proba"]
    def predict_batch(self, estimator: BaseEstimator, batch: EstimatorBatch) -> PredictedBatch: ...

@dataclass(frozen=True)
class MetricDefinition:
    name: str
    direction: Literal["min", "max"]
    formula_id: str
    formula_version: int
    averaging: Literal["none", "micro", "macro", "weighted", "uniform_average", "variance_weighted"]
    sample_weight_policy: Literal["supported", "required", "forbidden"]
    zero_division: Literal["zero", "one", "error", "not_applicable"]
    exact: bool
    evaluation_scope: Literal["full", "sampled"]
    parameters: Mapping[str, JSONValue]
    implementation: Literal["builtin_verified", "custom"]

class MetricState(Protocol):
    def reset(self) -> None: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
    def reduction_state(self) -> Mapping[str, tuple[torch.Tensor, Literal["sum", "min", "max", "merge_weighted_moments"]]]: ...
    def load_reduced_state(self, state: Mapping[str, torch.Tensor]) -> None: ...
    def compute(self) -> Mapping[str, float]: ...
```

- `LossResult.numerator` 必须是有限、可微的标量 Tensor，表示本 micro-batch 的损失加权和；`denominator` 是对应有限正权重和且不得带梯度。引擎以 `sum(numerator)/sum(denominator)` 记录 loss，不猜测 criterion reduction，也不接受已经做未知 mean reduction 的标量。
- `MetricState.reduction_state` 只能返回固定 shape Tensor 和显式 `sum/min/max/merge_weighted_moments` 操作；最后一种操作只接受末维为 `(weight, mean, M2)` 的 float64 状态，并按固定 rank 顺序使用 Chan 合并公式。各 rank 的键、shape、dtype 与操作必须完全一致，不允许包含样本维数组。空 split、非有限结果或不一致状态立即失败。
- `PredictedBatch` 是两后端共用的唯一评价入口；targets/predictions/scores/weight/sample_ids 的样本维和有限性统一校验。sklearn Task 必须显式声明 prediction capability，缺失时失败，不从 `hasattr` 顺序选择替代输出。
- `MetricDefinition` 是 manifest、summary 和报告的强制评价元数据；同名指标只有全部字段完全一致才可跨 split、rank 或 trial 比较。内置指标固定 `builtin_verified`、`evaluation_scope=full` 和 design 指定的 formula_id/version；抽样 ROC/PR 只能作为 `evaluation_scope=sampled` 的可视化统计，不得注册成 selection 或 sweep comparison metric。
- 内置 `MulticlassClassificationTask` 使用 logits `[N,C]` 与 long target `[N]`。
- 内置 `MultilabelClassificationTask` 使用 logits/target `[N,L]`、BCE-with-logits 和显式阈值，默认阈值 `0.5`。
- 内置 `RegressionTask` 接受 prediction/target `[N]` 或 `[N,D]`，内部统一为 `[N,D]`。
- 自定义 TorchTask 可以覆盖 `forward/to_predicted_batch`；自定义 SklearnTask 可以固定 estimator 输出到 PredictedBatch 的映射。共享 engine 只消费 PredictedBatch 和 MetricState，不解释模型输出或业务字段。

### SchedulerBinding

```python
@dataclass(frozen=True)
class SchedulerBinding:
    scheduler: Any
    interval: Literal["optimizer_step", "epoch", "validation_metric"]
    monitor: str | None
```

- `validation_metric` 必须配置存在的 `val/...` monitor；其他 interval 的 monitor 必须为 `None`。
- 引擎不通过类名或 `hasattr` 猜测调用策略。
- 调度器注册到 Accelerator 检查点；状态不可序列化时启动即失败。

### EngineState

`EngineState` 是 backend-neutral 的版本化可序列化对象，固定保存 backend、当前 `epoch`、`batch_in_epoch`、`global_step`、best 指标值及其 epoch/step、early-stopping patience 计数、当前阶段和各阶段尚未完成的部分 `MetricState.state_dict()`。Torch 将它注册到 Accelerate checkpoint；sklearn incremental 将它以严格结构化状态保存并由 checkpoint manifest 校验。恢复时必须先验证 schema、backend 和指标定义/状态兼容性，再继续数据源；中途检查点不得丢弃本 epoch 已累计但尚未汇总的指标。

## Strict Configuration Schema V1

唯一输入格式为 UTF-8 YAML。解析使用 `yaml.safe_load`，再映射到 frozen dataclass；所有文本读写显式指定 `encoding='utf-8'`。未知字段、重复 YAML key、非有限数值、隐式字符串布尔值和跨字段冲突都抛出 `ConfigError`。

```yaml
schema_version: 1
run:
  name: mnist
  id: mnist-baseline-v1
  output_root: null
  source_revision: null
  seed: 42
  tags: {}
experiment: {}
training:
  max_epochs: 20
  log_every_steps: 20
backend:
  type: torch
  torch:
    gradient_accumulation_steps: 1
    mixed_precision: auto
    compile: false
    clip_grad_norm: 1.0
    deterministic: strict
    matmul_precision: high
    find_unused_parameters: false
  sklearn: null
distributed:
  num_processes: auto
selection:
  metric: val/loss
  mode: min
  patience: 5
  min_delta: 0.0
checkpoint:
  every_epochs: 1
  every_optimizer_steps: null
  keep_last: 2
  resume: none
runtime:
  max_minutes: null
  shutdown_grace_minutes: 10
report:
  enabled: true
  curve_sample_limit: 100000
  prediction_sample_limit: 10000
  prediction_splits: [val, test]
remote:
  type: none
notifications:
  type: none
```

sklearn backend 的对应分支固定为：

```yaml
backend:
  type: sklearn
  torch: null
  sklearn:
    fit_mode: batch
    evaluation_batch_size: 4096
    n_jobs: auto
    random_state: run_seed
    sample_weight_parameter: null
```

Torch 分支字段为：

```yaml
backend:
  type: torch
  torch:
    gradient_accumulation_steps: 1
    mixed_precision: auto
    compile: false
    clip_grad_norm: 1.0
    deterministic: strict
    matmul_precision: high
    find_unused_parameters: false
```

枚举与跨字段行为：

| 字段 | 允许值与行为 |
|---|---|
| `backend.type` | `torch/sklearn`，必须与 Experiment dataclass 一致；未选 backend 分支必须为 null |
| `backend.torch.mixed_precision` | `auto/no/fp16/bf16`；auto 在 CUDA bf16 可用时选 bf16，否则 CUDA 选 fp16，CPU 选 no；显式不支持值立即失败 |
| `backend.torch.deterministic` | `strict/warn/off`；strict 调用 deterministic algorithms 且 `warn_only=False`；warn 为 True；off 不修改 |
| `backend.torch.matmul_precision` | `highest/high/medium`，直接传给 PyTorch |
| `backend.sklearn.fit_mode` | `batch/incremental`；batch 只调用一次 fit、training.max_epochs 必须为 1、checkpoint.resume 必须为 none 且不声明中途恢复；incremental 要求 partial_fit 和可恢复 batch source |
| `backend.sklearn.n_jobs` | null、正整数或 `auto`；非 null 时 estimator 必须暴露顶层 n_jobs，auto 解析为逻辑 CPU 数，写入 resolved estimator params |
| `backend.sklearn.random_state` | `run_seed/require_explicit`；run_seed 将所有值为 null 的递归 `*random_state` 参数设为 run.seed；其他非整数值失败 |
| `backend.sklearn.sample_weight_parameter` | null 或非空参数路径；训练数据存在 sample_weight 时必须非 null，并原样传给 fit/partial_fit；不存在权重时必须为 null，防止配置与真实数据语义脱节 |
| `distributed.num_processes` | Torch 为正整数或 `auto`；auto 为可见 CUDA 数量，CPU 为 1；sklearn 必须为 1 |
| `selection` | 有验证集时必须存在；无验证集时必须为 null；metric 必须由 Task 产生，mode 必须等于 MetricDefinition.direction |
| `checkpoint.resume` | `none/auto/required`；auto 仅在同一 run ID 找到兼容 latest 时恢复；required 找不到即失败 |
| `checkpoint.every_optimizer_steps` | 仅 Torch 可设置，且 DataModule 必须支持中途恢复；sklearn incremental 使用 batch 边界 checkpoint |
| `runtime.max_minutes` | Kaggle profile 必须是正数；Torch 或 sklearn incremental 有值时数据源必须支持中途恢复；sklearn batch 仅允许 Local 且必须为 null，因为 fit 不能提供可审计暂停点 |
| `run.id` | Kaggle profile 必填并匹配 `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`；本地缺省时生成 UTC 时间和指纹后缀 |
| `run.output_root` | Kaggle 缺省解析为 `/kaggle/working/dl-helper-runs`；本地缺省为 `<cwd>/runs`；Kaggle 不允许写 `/kaggle/input` |
| `run.source_revision` | 可为 null 或非空字符串；Kaggle 必须解析为 40 位 Git SHA 且与 clean HEAD 完全一致；本地无法从 Git 获得 revision 时必须显式提供 |
| `remote.type` | `none/alist`；alist 需要下述完整配置和 Secret，缺任一项失败 |
| `notifications.type` | `none/wecom`；wecom 需要下述 Secret key、投递策略与目标用户 |

AList 配置固定为：

```yaml
remote:
  type: alist
  host: https://alist.example.invalid
  base_path: /dl-helper-runs
  user_secret_key: ALIST_USER
  password_secret_key: ALIST_PWD
  connect_timeout_seconds: 10
  read_timeout_seconds: 600
  max_attempts: 3
  async_upload: true
  failure_policy: required
```

host 不提供默认 IP；配置只保存 Secret 名，不保存值。Kaggle 使用 `kaggle_secrets.UserSecretsClient.get_secret`，Local 使用同名环境变量。Secret 缺失时不得回退匿名连接。`failure_policy` 为 `required/record`：required 在安全边界重新抛出并阻止成功终态，record 将失败写入 service audit 后继续，但绝不静默忽略。

企业微信配置固定为：

```yaml
notifications:
  type: wecom
  corp_id_secret_key: WECOM_CORP_ID
  corp_secret_key: WECOM_CORP_SECRET
  agent_id_secret_key: WECOM_AGENT_ID
  to_user: "@all"
  connect_timeout_seconds: 10
  read_timeout_seconds: 30
  max_attempts: 3
  failure_policy: record
```

企业微信 API host 固定为 `https://qyapi.weixin.qq.com`，不接受用户覆盖；只发送 UTF-8 text 应用消息。Secret resolver 和脱敏规则与 AList 共用。`to_user` 必须为非空字符串且不得来自 Secret 值；按官方应用消息合同，最终 `content.encode("utf-8")` 不得超过 2048 字节。平台只使用固定内置模板，并在 Unicode code point 边界按 UTF-8 字节数裁剪可变的异常消息和路径；event、status、run/sweep/trial ID 与异常类型不得被裁掉。

## Base, Variant And Sweep Configuration

普通 `train` 使用一个完整 base config，并可额外接收一个 `--variant <patch.yaml>`。variant 是不含 `schema_version` 的严格 UTF-8 YAML mapping，按以下唯一规则合并：mapping 递归合并，scalar/list/null 整体替换；禁止重复 key、YAML merge key/alias、环境变量插值、模板表达式、文件 include 和网络 URL。base/variant 路径先 resolve，并拒绝符号链接逃逸。variant 只允许覆盖 `run.name/tags`、`experiment`、`training`、已选择的 `backend` 子树、`selection`、`checkpoint` 的频率/保留参数、`runtime`、`report`、`remote` 超时/重试/策略和 `notifications` 超时/重试/策略；不得覆盖 schema、run.id、backend type、output root、source revision、Secret key、host/base path 或 distributed process count。最终 merged config 必须重新通过完整 schema 和全部跨字段校验，不允许直接修改已构造 dataclass；base/variant SHA256、规范化 resolved config、resume fingerprint 和 tuning fingerprint 全部落盘。

`tuning_fingerprint` 只覆盖 `experiment/training/backend/selection` 中会改变拟合结果或停止位置的字段。sweep 中不同 trial 的 tuning fingerprint 必须唯一，服务超时、tag 或报告采样上限的差异不能把相同训练伪装成不同 trial。variant 禁止改变 `run.seed`，因此同一 sweep 的随机性基线一致；如需多 seed 稳健性评估，必须显式创建不同 sweep，而不是把 seed 当成可排名超参。

sweep manifest 是独立 schema：

```yaml
schema_version: 1
sweep:
  id: mnist-learning-rate-v1
  experiment: experiments.mnist:build_experiment
  base_config: ./mnist/base.yaml
  comparison_metric: val/f1_macro
  mode: max
  trials:
    - name: lr-1e-3
      variant: ./mnist/variants/lr-1e-3.yaml
    - name: lr-3e-4
      variant: ./mnist/variants/lr-3e-4.yaml
```

路径必须是相对 manifest 的本地相对路径，resolve 后仍位于 manifest 父目录树内；trial name 匹配 run ID 字符集且唯一，至少两个 trial，按 YAML 顺序执行。run ID 固定派生为 `<sweep.id>--<trial.name>`，variant 不得提供 run.id。comparison metric 必须是 `val/` 前缀、`evaluation_scope=full`、`exact=true` 的 MetricDefinition，mode 必须等于其 direction；test 指标可以展示但严禁参与选择和排名。

sweep 在任何 trial 拟合前，依次以独立 `doctor --emit-evaluation-contract` 子进程解析全部 trial。所有合同必须具有相同的 Experiment 引用、backend、DataIdentity、train/val/test split fingerprint、Task name、标签/目标 schema、MetricDefinition 全集、comparison metric 和 direction；模型 signature 与允许调参值可以不同。任一合同构造失败或不可比，整个 sweep 在零优化步时失败。coordinator 本身不导入实验、sklearn、torch 或初始化 CUDA。

sweep coordinator 使用 `sys.executable -m dl_helper.training.cli train` 和 argv 在全新子进程顺序运行，并以原子独占 lock 防止相同 sweep ID 并发。任何 trial 返回非零且非 75 时写 sweep failure 并停止，FAILED sweep 不允许原 ID 续跑；返回 75 时写 sweep pause manifest 并返回 75。`sweep --resume` 只接受完整 PREEMPTED manifest：复核 manifest/base/variant/已完成 run 的 checksum 与 fingerprint，跳过已成功 trial并继续同一 paused trial；发现任何漂移立即失败。

全部 trial 成功后才读取每个 `metrics/summary.json` 的未舍入全量 comparison metric，NaN/Inf/缺失或定义漂移立即失败；并列按 YAML trial 顺序稳定排序。随后原子写 `best-trial.json`、生成 `sweep-report/index.html`，完成 required AList/企业微信服务，再最后发布唯一 `sweep-manifest.json`。排名不得读取报告显示的舍入值或抽样曲线。

## Target Training State Machine

共享状态按以下顺序单向推进：

```text
CREATED -> PREFLIGHTED -> SERVICES_STARTED -> PREPARED -> [RESUMED] -> FITTING
FITTING -> EVALUATING -> CHECKPOINTING -> FITTING
FITTING -> TESTING -> FINALIZING -> SERVICES_FINALIZED -> SUCCEEDED
any non-terminal state -> FAILED
runtime budget -> CHECKPOINTING -> FINALIZING -> PREEMPTED(exit 75)
```

`PREEMPTED` 是有完整可恢复检查点的预期非成功终态，使用退出码 `75`；Kaggle 编排器可据此启动下一会话。未产生完整检查点的预算耗尽属于 `FAILED`。共享 engine 只编排状态、Artifact、MetricDefinition 和服务，不包含 torch/sklearn 特有对象；backend worker 返回统一 `BackendResult`（阶段、位置、PredictedBatch 流、模型 Artifact、可恢复状态和环境统计）。

共享编排伪代码：

```python
config = load_and_validate_config()
experiment = import_ref_and_build(config.experiment)
platform = detect_platform()
platform.preflight(config, experiment)
services = build_services(config, platform.secret_resolver)
audit(services.start_run_or_trial())
worker = backend_registry.create(config.backend.type)
result = worker.run(experiment, config, platform, artifact_writer)
validate_result_against_evaluation_contract(result)
generate_report_from_artifacts()
publish_artifacts_to_store()
audit(services.finalize(result.status))
write_exactly_one_terminal_manifest_last()
```

PyTorch worker 伪代码：

```python
accelerator = Accelerator(mixed_precision=resolved_precision,
                          gradient_accumulation_steps=accumulation_steps)
model, datamodule, task, optimizer, scheduler = build_torch_components()
validate_fresh_components_and_register_checkpoint_state()
model, optimizer, loaders = accelerator.prepare(model, optimizer, loaders)
resume_accelerate_state_if_requested()
for epoch in remaining_epochs:
    window_denominator = 0.0
    for raw_batch in train_loader:
        with accelerator.accumulate(model):
            prepared = task.prepare_batch(raw_batch, "train")
            outputs = task.forward(model, prepared)
            loss = task.loss(outputs, prepared)
            validate_loss_result(loss)
            accelerator.backward(loss.numerator * accumulation_steps)
            window_denominator += detached(loss.denominator)
            update_full_metrics(task.to_predicted_batch(outputs, prepared))
            if accelerator.sync_gradients:
                global_denominator = all_reduce_sum(window_denominator)
                accelerator.unscale_gradients(optimizer)
                multiply_each_gradient(world_size / global_denominator)
                clip_if_configured_and_validate_finite_gradients()
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                step_optimizer_scheduler(); global_step += 1
                window_denominator = 0.0
                checkpoint_or_budget_boundary_if_needed()
    reduce_persist_and_validate_train_metrics()
    evaluate_val_and_update_selection_if_present()
    step_epoch_or_metric_scheduler()
    save_epoch_checkpoint_if_due()
evaluate_test_if_present(); export_safetensors_best_and_last()
```

PyTorch 的 backward 使用损失加权和而非局部均值。Accelerate 对每次 backward 的固定 accumulation 除数由乘回 `accumulation_steps` 抵消；DDP 已对各 rank 梯度取平均，因此在 unscale 后、clip 前乘 `world_size/global_denominator`，得到当前 accumulation window 全部 rank、全部样本的精确加权均值梯度。最后不足 accumulation 的窗口执行同一公式。`global_denominator<=0`、非有限梯度或 optimizer step 失败立即终止。global step 只在 optimizer 成功 step 后增加，batch count 和评价样本数独立记录。

sklearn batch worker 固定执行：clone 未拟合 estimator -> 解析 n_jobs/random_state -> 一次 `fit(X,y[,sample_weight])` -> `check_is_fitted` -> 按 train/val/test evaluation batch 生成 PredictedBatch -> 选择 best=last -> 导出 joblib。它不创建 mid-fit/latest checkpoint，`training.max_epochs=1`、`checkpoint.resume=none`、Local profile、`runtime.max_minutes=null` 是前置条件；fit 异常直接失败，不伪造恢复。

sklearn incremental worker 固定执行：clone 未拟合 estimator -> 恢复或构造 IncrementalBatchSource -> 对 epoch/batch 顺序调用 `partial_fit` -> 每个 batch 成功后增加 batch/global step -> 在首个 classifier batch 传入 Task 声明的完整 classes，之后不再改变 classes -> 在配置边界评价 val/update selection -> 在 batch/epoch 边界以可信 joblib + source state 创建检查点。预算触发只发生在成功 partial_fit 之后；完整 joblib/source/EngineState/manifest 发布后才可 PREEMPTED/75。两种 sklearn 模式均为单进程，不调用 Accelerate、DDP、AMP、optimizer 或 scheduler。

任一 backend 产生的 PredictedBatch 都通过同一 MetricState 更新、同一 prediction writer 落盘。非有限 loss、prediction、score、weight、metric 或状态错误立即失败；`failure.json` 脱敏记录后重新抛出原异常，不由服务异常覆盖。

## Metric Semantics And Distributed Reduction

所有内置评价先验证 `N>0`、shape/schema 固定、target/prediction/scores 有限。`sample_weight` 缺省时令 `w_i=1`；提供时必须是一维 `[N]` float、有限、非负且每批 `sum(w)>0`。状态同时保存原始 `sample_count:int64` 和 `weight_sum:float64`，报告不得用权重和冒充样本数。每个 stage 的 summary 必须携带完整 MetricDefinition、有效样本数、权重和及 computed UTC；selection、early stopping 和 sweep 只能使用 `exact=true/evaluation_scope=full` 的有限标量。

### Common Loss

每批累计 `loss_numerator += LossResult.numerator.detach()` 和 `loss_denominator += denominator`，跨 rank 均用 float64 sum reduce；最终 `loss=loss_numerator/loss_denominator`。该值是全量评价单元的加权均值，与用于 backward 的局部/分布式缩放分开计算。denominator 非正、numerator 非标量或任何值非有限均报错。内置 Torch Task 的单位固定为：多分类每样本交叉熵；多标签每样本先对全部 label 取 BCE 均值；回归每样本先对全部 target 取平方误差均值。对应 sample weight 只在样本维应用一次。

### Multiclass

- 类别顺序由 Task 显式声明为 `C>=2` 个唯一值。Torch logits 必须为 `[N,C]` 并以 argmax 得到预测；sklearn 预测标签必须可按声明顺序无歧义映射。未知 target/predicted class 立即失败。
- 状态包含加权混淆矩阵 `M[i,j]=sum_k(w_k * 1[y_k=i and pred_k=j])`（float64）以及原始混淆计数（int64）；未提供权重时两者数值一致。分布式仅 component-wise sum。
- `accuracy=trace(M)/sum(M)`；class k 的 `precision=M[k,k]/sum_i M[i,k]`、`recall=M[k,k]/sum_j M[k,j]`、`f1=2PR/(P+R)`，任一零分母按 0 处理并在 MetricDefinition 标记 `zero_division=zero`。
- macro precision/recall/F1 对全部声明类别等权平均，包括 weighted support 为零的类别；weighted precision/recall/F1 按真实 weighted support 加权。`balanced_accuracy` 只对真实 weighted support 大于零的类别 recall 求均值，并显式列出缺失类别，行为与 sklearn 1.6.1 金标一致。
- 内置全量标量至少包含 accuracy、balanced_accuracy、precision_macro、recall_macro、f1_macro 和 f1_weighted；per-class precision/recall/F1/support 作为同一精确状态的向量输出。单标签 multiclass 不重复输出与 accuracy 数学等价的 micro F1，报告应解释为“未提供冗余指标”，不得填充伪值。

### Multilabel

- label 顺序和阈值向量由 Task 显式声明；target 只能是 bool 或 0/1 `[N,L]`，score 必须为同 shape 的概率或经 sigmoid 后概率。默认每个 label 阈值为 0.5，实际阈值写入 MetricDefinition。
- sample weight 沿 label 维广播一次；每个 label 累计 float64 TP/FP/FN/TN 和 int64 原始计数。per-label precision/recall/F1 零分母为 0；micro 指标先跨 label 求和 TP/FP/FN，macro 对全部声明 label 等权平均，weighted F1 按正类 weighted support 加权。
- `subset_accuracy=sum_i(w_i * 1[all_l pred_il=target_il])/sum_i(w_i)`；`hamming_loss=sum_i,l(w_i * 1[pred_il!=target_il])/(L*sum_i w_i)`。二者分别为 max/min 方向，避免用单一 micro F1 隐藏每样本完全匹配质量。

### Regression

- prediction/target 统一为 float64 `[N,D]`，sample weight 沿 target 维广播。每个 target 累计 weighted absolute error、weighted squared error和合并式 `(weight, mean_y, M2_y)`；M2 使用 weighted Welford 更新，并按固定 rank 顺序用 Chan 公式合并，避免大偏置目标上 `sum(y^2)-sum(y)^2/n` 的灾难性消减。
- 每目标 `MAE=sum(w*abs_error)/sum(w)`、`MSE=sum(w*squared_error)/sum(w)`、`R2=1-SSE/M2_y`。当 `M2_y=0` 时遵循 sklearn `force_finite=True`：SSE=0 得 1，否则得 0；不能用 epsilon 将近似常量目标改判为常量。
- 总体 MAE/MSE 使用 `uniform_average`；R2 同时输出 `uniform_average` 与 `variance_weighted`。后者按每目标 M2 加权；全部 M2 为零时按 sklearn 语义退化为 uniform average，并在 MetricDefinition parameters 中记录该规则。
- 负 M2 只允许在 `8 * float64 epsilon * max(weight, abs(mean), 1)` 的舍入界内归零，超过界限立即视为状态损坏；最终标量必须有限。

### Scientific Verification Gate

- 金标固定为 scikit-learn 1.6.1：`confusion_matrix/accuracy_score/balanced_accuracy_score/precision_recall_fscore_support/multilabel_confusion_matrix/accuracy_score/hamming_loss/mean_absolute_error/mean_squared_error/r2_score(force_finite=True)`，所有调用显式传 labels、average、zero_division、sample_weight 和 multioutput，不依赖库默认值。
- 测试矩阵必须覆盖：无权重/整数权重/非整数权重、随机 batch 分块、单/双 rank、缺失真实类别、从未预测类别、全负多标签、极端类别不平衡、最后不足批次、常量与近常量目标、多输出不同方差、大偏置目标、状态保存恢复。内置值对金标绝对误差 `<=1e-6`；大偏置 R2 另要求与直接 float64 两遍计算 `<=1e-10`。
- 每个内置 formula_id/version 均有不可变 golden fixture。修改公式、默认阈值、zero-division、sample-weight 或聚合行为必须增加 formula_version，旧 run 报告仍按其记录版本解释，不得静默套用新定义。

内置状态只归约固定大小状态，不 gather 全量预测。float64 用于权重、loss、混淆与回归状态，int64 用于原始计数；最终 JSON 转为有限 Python float/int。模型选择不预设“最好指标”：用户必须选择与业务目标一致且由 Task 声明方向的验证指标，系统只保证计算合同和可比性，不声称 accuracy、F1 或 R2 在所有问题上具有普适优越性。

## Prediction And Visualization Sampling

- 摘要指标永远来自全量流式状态。
- 曲线数据使用确定性优先级抽样：对 `(run_seed, split, stable_sample_id)` 计算 64 位 hash，保留 hash 最小的 `curve_sample_limit` 个样本；没有 stable ID 时使用全局递增样本位置。多 rank 在主进程合并各自候选后再次取最小值。
- 报告必须显示 `sampled/total`，不得把抽样 ROC/PR/AUC 标记为全量。
- Built-in Task 的 prediction shard 是无 pickle 的压缩 NPZ，字段只能是数值、bool 或固定宽度 Unicode ndarray，禁止 object dtype。文件名为 `part-rank{rank:05d}-{index:06d}.npz`。
- 自定义 Task 返回的 arrays 第一维必须等于 sample_count；字段名匹配 `[A-Za-z][A-Za-z0-9_.-]*`。

## Artifact Schema

```text
runs/<run-id>/
  run-manifest.json
  pause-manifest.json
  config.resolved.yaml
  environment.json
  evaluation-contract.json
  services/service-audit.jsonl
  logs/train.log
  metrics/metrics.jsonl
  metrics/summary.json
  checkpoints/
    latest.json
    epoch-000001-step-00000050/
      torch-state/ | estimator.joblib + engine-state.json + source-state/
      checkpoint-manifest.json
  models/best/model-manifest.json + model.safetensors|model.joblib
  models/last/model-manifest.json + model.safetensors|model.joblib
  predictions/<split>/prediction-manifest.json
  predictions/<split>/part-rank00000-000000.npz
  report/index.html
  report/assets/*.png
  failure.json

sweeps/<sweep-id>/
  sweep-manifest.json | pause-manifest.json | failure.json
  sweep.resolved.yaml
  contracts/<trial-name>.json
  trials.jsonl
  services/service-audit.jsonl
  best-trial.json
  sweep-report/index.html
  sweep-report/assets/*.png
```

- `run-manifest.json` 是最后写入的成功标志，包含 schema、run ID、backend、状态、开始/结束 UTC、source revision、base/variant/config/tuning/data/model 指纹、环境摘要、MetricDefinition、best/last、服务结果、完整 artifact 相对路径与 SHA256。
- `pause-manifest.json` 是完成可恢复持久化后最后写入的 PREEMPTED 标志，包含同一复现上下文、恢复 checkpoint ID、暂停位置、远程 flush 结果、报告路径和 checksum；恢复启动后先原子移除旧 pause manifest，最终发布新的唯一终态。
- `run-manifest.json`、`pause-manifest.json` 与 `failure.json` 三者互斥。失败时 `failure.json` 包含异常类型、消息、阶段、epoch、step 和 traceback，所有 Secret 值先从文本替换为 `[REDACTED]`。
- `model-manifest.json` 固定记录 backend、format、format_version、model_signature、origin_run_id、created_utc、文件 size/SHA256 和产生它的 Python/numpy/torch 或 sklearn/scipy/joblib 精确版本。Torch 只导出 CPU contiguous state dict 的 safetensors；sklearn 只导出 joblib，不生成扩展名伪装的 pickle。
- sweep 三个终态文件同样互斥；成功 sweep manifest 最后写入并列出有序 trial run ID/status/config fingerprint/未舍入 comparison value、排名、best-trial checksum、聚合报告和服务审计 checksum。失败/暂停不得存在 best-trial 或成功 manifest。
- JSON、YAML、CSV、JSONL 和日志均 UTF-8；二进制格式不适用文本 encoding。
- 原子文本写入使用同目录临时文件、flush、`os.fsync`、`os.replace`。只有主进程写共享文本和报告。

## Checkpoint And Resume

检查点目录不可变。Torch backend 的 Accelerate 状态包含模型、优化器、调度器、scaler、RNG、DataModule 和 EngineState；sklearn incremental 状态包含 fitted estimator.joblib、结构化 EngineState、IncrementalBatchSource 状态、RNG 和评价状态。两者都先写入新 staging 目录，完成后生成 manifest，再将 staging 重命名为最终 checkpoint ID，最后原子替换 `latest.json`。manifest 字段固定为：

| 字段 | 含义 |
|---|---|
| `schema_version` | `1` |
| `run_id/checkpoint_id` | 运行与不可变检查点 ID |
| `created_utc/epoch/batch_in_epoch/global_step` | 恢复位置 |
| `config_fingerprint` | 排除允许变化字段后的 SHA256 |
| `backend/data_fingerprint/model_signature` | 后端、数据与模型兼容标识 |
| `runtime_versions` | Python 及 torch/accelerate 或 sklearn/numpy/scipy/joblib 的精确版本 |
| `files` | 每个相对文件的 size 与 SHA256 |
| `complete` | 只有全部文件 fsync 后才为 true |

允许恢复时变化的配置键只有：`training.max_epochs`（只能增大）、`runtime.*`、`checkpoint.every_*`、`checkpoint.keep_last`、`report.*`、`remote/notifications` 的超时与重试策略。其他变化导致 `CheckpointCompatibilityError`，错误中列出差异键但不含 Secret。sweep trial 恢复还必须匹配 sweep/manifest/base/variant checksum 和 trial 派生 run ID。

中途恢复要求数据源声明能力并保存当前位置，同时 EngineState 保存本 epoch 部分 MetricState；引擎恢复后不得重复已完成 optimizer/partial_fit step 或重复累计已消费 batch。Torch epoch 边界恢复适用于所有 DataModule；sklearn batch 不支持任何恢复；sklearn incremental 在 batch 边界恢复。保留策略只删除当前 run 的、manifest 完整且不是 latest/best 引用的旧检查点。

joblib/pickle 具有代码执行能力。平台只能加载由当前 run ID 先前写出的 regular file：resolved path 必须位于当前 run/checkpoint 根内且不是符号链接，origin_run_id、backend、model signature、全部文件 SHA256 和 runtime_versions 必须与当前预检完全一致。用户给定路径、其他 run、缺 manifest、版本范围仅兼容但不精确相同的 joblib 一律拒绝；失败时不得尝试 `joblib.load` 获取更多信息。AList 下载也必须先校验外层 archive/manifest，再落到隔离 staging 并复核全部文件，之后才允许反序列化。

## Kaggle Platform Behavior

### Detection And Paths

- 任一环境键以 `KAGGLE` 开头时识别 Kaggle，否则为 Local。
- Kaggle 输入必须由 Experiment config 显式给出绝对 `/kaggle/input/...` 路径；禁止选择“第一个输入目录”。
- 所有新 Artifact 写入 `/kaggle/working/dl-helper-runs/<run-id>`，除非显式给出仍位于 `/kaggle/working` 内的 output root。
- 对 `/kaggle/input` 的写尝试和越界 output root 在训练前失败。

### Resource Resolution

- Torch 的 `num_processes=auto` 使用 `torch.cuda.device_count()`；没有 CUDA 时为 1。sklearn 固定单进程 coordinator，estimator 内部并行只由解析后的顶层 `n_jobs` 控制，environment 记录逻辑 CPU 与 native thread 相关环境变量。
- 不通过 GPU 型号分支。environment manifest 记录每个设备名称、总显存、compute capability、CUDA、cuDNN 和可用混合精度。
- `nominal_train_batch_size` 为整数时，全局有效批量记录为 `nominal_train_batch_size * num_processes * gradient_accumulation_steps`；实际批次除最后一个不足批次外不得超过声明值。该字段为 `None` 时记录为 `dynamic`，并从 PreparedBatch.sample_count 统计每 rank 实际批量最小值/最大值，不伪造固定有效批量。
- `LoaderDataModule` 的 `num_workers=auto` 解析为 `min(8, max(1, os.cpu_count() // num_processes))`；CUDA 自动启用 pin memory，worker>0 自动启用 persistent workers 和 prefetch factor 2。解析值写入 resolved config。
- `torch.compile` 默认关闭；显式开启而后端不支持时失败，不退回 eager。

### Doctor

`doctor --profile <local|kaggle> --config <file> [--variant <file>] --experiment <ref>` 在不训练的情况下验证：依赖版本、实验与 backend 匹配、配置、数据/evaluation contract、输入存在且可读、输出可写、至少 5 GiB 可用空间、run ID、运行时预算、恢复能力、Secret 和服务 host。Torch 额外检查设备、AMP、DDP、DataLoader 和 scheduler；sklearn 额外检查 clone/未拟合状态、estimator kind、prediction capability、fit/partial_fit、n_jobs/random_state/sample_weight 参数及可信持久化版本。任何失败返回非零。

### Runtime Budget

- Kaggle 配置必须显式指定 `max_minutes` 和 `shutdown_grace_minutes`，且 grace 小于 max。
- sklearn batch 因没有可控制的 fit 中断/恢复点，不允许 Kaggle profile；doctor 必须建议改用支持 partial_fit 的 incremental estimator 或在 Local profile 运行，不能仅发 warning 后继续。
- worker 使用 monotonic clock；每个 batch 后按 `elapsed >= max_minutes - shutdown_grace_minutes` 判断，命中后停止新优化步，不动态估算保存耗时。整个 grace 窗口只用于 checkpoint、remote flush 和报告。
- 有中途恢复能力时立即保存完整检查点、flush 远程并生成当前报告，最后原子发布 `pause-manifest.json`，然后以 `PREEMPTED`/75 退出。
- 任一步骤在 deadline 前无法完成或 remote required 同步失败时为 FAILED，不输出伪成功。

### Pinned Bootstrap

Kaggle 模板要求用户提供 `DL_HELPER_GIT_REF`，只接受 40 位 commit SHA。模板 clone 该 revision 后比较 `git rev-parse HEAD`，使用 `pip install -e . --no-deps`，随后运行 doctor。不得执行 `git pull`、下载 `master` 单文件或静默安装/升级 torch、accelerate。

## Lifecycle Services

### ArtifactStore And AList

`ArtifactStore` 是 run/sweep 共享基础设施，不属于某个 Experiment 或 Kaggle 专用钩子。固定能力为 `publish_checkpoint`、`fetch_latest_checkpoint`、`publish_run_bundle`、`publish_sweep_bundle` 和 `flush`；LocalArtifactStore 始终启用，`remote.type=alist` 时额外启用 AListArtifactStore。Experiment/Task 不得接收客户端或主动上传。

- 新客户端直接使用核心依赖 `requests.Session`，显式 connect/read timeout。401/403 和 AList 非零认证/参数业务码不重试；连接错误、timeout、HTTP 5xx 和明确 transient server code 按 2、4、8 秒且不超过 `max_attempts=3` 重试，耗尽后以原异常为 cause 抛出 `ServiceDeliveryError`。
- archive 使用 tar.gz level 1；归档成员必须是 run/sweep 根内 regular file，拒绝符号链接、绝对路径和 `..`。本地先计算 size/SHA256。远程根固定为 `<base_path>/runs/<run-id>/` 与 `<base_path>/sweeps/<sweep-id>/`，run ID/sweep ID 只作为已校验的单个 path segment。
- checkpoint 发布顺序：上传不可变 archive -> 轮询 info/list 到 size 匹配 -> 通过 raw URL 流式回读并验证 SHA256 -> 上传 checkpoint manifest -> 回读校验 manifest -> 最后更新 `checkpoints/latest.json`。读取者忽略没有完整 manifest/latest 的对象；服务端 size 一致不构成成功。
- run/sweep bundle 排除已经独立发布的 checkpoint archive和所有 staging/tmp/lock 文件，包含 metrics、模型、预测、报告、resolved config、evaluation contract 和 service audit。发布顺序同样为 immutable archive 回读校验 -> bundle manifest 回读校验 -> terminal manifest 最后发布；pause 前必须先成功发布其引用 checkpoint。FAILED bundle 可以包含 failure.json，但绝不发布 success terminal。
- async 模式仅主进程启动一个 `daemon=False` worker 和容量 1 的待处理 checkpoint 队列。新 checkpoint 可以替换尚未开始的旧 pending 项，但不能取消 active 上传；run/sweep terminal bundle 不参与合并且必须等待所有 active/pending checkpoint。异常存入同步器并在下一安全边界或 final flush 重新抛出。
- `failure_policy=required` 时，服务启动/边界/final flush 失败阻止 SUCCEEDED 或 PREEMPTED；`record` 时运行继续，但每次失败必须先追加 service audit，并在终态 manifest 汇总 degraded service 状态。两种策略均禁止只 log 后遗忘。进程结束前必须 join，Secret 不进入 repr、日志、配置、manifest、URL query 的审计文本或异常正文。

### Enterprise WeChat Notification

`notifications.py` 内置独立企业微信客户端，不导入 `py_ext`，API host 固定为官方 `https://qyapi.weixin.qq.com`。生命周期事件固定为 `RUN_STARTED/RUN_SUCCEEDED/RUN_PREEMPTED/RUN_FAILED`、`SWEEP_STARTED/TRIAL_STARTED/TRIAL_SUCCEEDED/TRIAL_PREEMPTED/TRIAL_FAILED/SWEEP_SUCCEEDED/SWEEP_PREEMPTED/SWEEP_FAILED`；每个事件使用 `sha256(scope_id,event,attempt_scope)` 的稳定 event_id，恢复时根据 service audit 不重复发送已经成功的事件。

- token 仅通过 `GET /cgi-bin/gettoken` 的 `corpid/corpsecret` 获取；验证 HTTP、JSON、`errcode==0`、非空 access_token 和正 `expires_in`。每个客户端使用 monotonic clock 与锁缓存到 `expires_in-120s`（剩余不足 120 秒则按 80% 生命周期），不写磁盘、不使用进程级全局单例。
- 应用消息只通过 `POST /cgi-bin/message/send?access_token=...` 发送 `touser/agentid/msgtype=text/text.content`。agent_id Secret 必须解析为正十进制整数；`errcode==0` 才成功。已知 token 失效/过期码只允许清缓存、强制取新 token 后重放一次；网络/5xx/系统繁忙按同一有限重试器处理，其他 HTTP/业务错误不重试。
- 固定模板只包含平台、事件、run/sweep/trial ID、UTC 时间、耗时、未舍入 comparison/best metric 的格式化副本、状态、报告相对路径和失败异常类型/脱敏消息。最终 text content 必须不超过 2048 UTF-8 bytes；裁剪只作用于可变消息/路径，按 code point 边界执行，关键身份字段放不下时直接失败而不是发送含糊通知。
- `failure_policy` 与 AList 相同。required 的 STARTED 投递失败发生在首个拟合 step 前；required 的成功/暂停事件失败阻止对应终态 manifest。运行本身已经失败时，通知或远程失败只记为 secondary service error：`failure.json` 保留训练异常为 `primary_exception`，CLI 最终重新抛出原训练异常，绝不让通知异常替换其 traceback。

### Service Audit And Ordering

每次调用先/后各写一条 UTF-8 JSONL 审计记录，固定字段为 schema、event_id、scope、service、event、attempt、started/finished UTC、duration_ms、outcome、HTTP status、脱敏业务码和 error_type；禁止记录 token URL、请求认证参数或 response body。正常终态顺序固定为：核心 Artifact 完成 -> service audit flush -> AList terminal bundle（若启用）-> 企业微信终态事件（若启用）-> 再次发布包含最终 audit 的小型 service manifest -> 本地 terminal manifest 最后写入。跨两个外部服务无法提供原子事务，因此任何 required 失败都保留 FINALIZING/failure 证据；重跑 finalization 时以 checksum/event_id 幂等复核，不重做训练或重复成功消息。

## Reporting

`report` 命令只读取 Artifact，不导入用户模型或数据代码，因此可在训练后重复生成且幂等。相同输入 Artifact 产生同名覆盖的 report 文件，不改动指标、模型或检查点。

- 通用页：运行状态、backend、配置摘要、环境、数据 identity、模型参数量或 estimator 参数、有效批量、耗时、吞吐、peak CUDA allocated/reserved、学习率、loss 和用户指标曲线；sklearn 不适用项明确标为 N/A，不绘制伪造的 epoch/GPU 曲线。
- 指标定义页：逐项展示 formula_id/version、direction、average、sample-weight、zero-division、full/sampled、有效样本数/权重和与参数；summary 原始值保留双精度，UI 显示值可格式化但不得反向用于选择。
- 多分类页：原始/归一化混淆矩阵、per-class precision/recall/F1/support、抽样 ROC/PR；类别超过 20 时图表显示 support 最大 10 和 F1 最低 10，完整表仍保留。
- 多标签页：per-label 指标、micro/macro 指标、阈值和抽样 ROC/PR；同样使用 20 项图表规则。
- 回归页：每目标 MSE/MAE/R2、抽样 predicted-vs-actual hexbin、残差直方图和 residual-vs-prediction。
- 自定义页：始终提供通用曲线；Task 可通过受限 `ReportSection` 返回表格或 Matplotlib figure，不允许注入原始 HTML。
- 所有用户文本 HTML escape；图像使用相对 assets，报告可离线打开。

`sweep-report` 同样只读 sweep 与各 run Artifact，固定包含按未舍入 validation comparison metric 排名的总表、best trial、每个 trial 的 base/variant/tuning fingerprint、关键 resolved 参数差异、状态/耗时/资源、comparison 曲线和单 run 报告相对链接。失败或暂停 sweep 只能生成进度报告，不显示 best/ranking；test 指标可展示但必须标注“不参与调参选择”。

## Fail-Fast And Security

- CLI 顶层只负责记录脱敏 failure artifact 后 `raise`；不得 `os._exit(0)`、`pkill`、通用捕获后继续或返回安全默认值。
- 网络下载、上传、安装和子进程都检查返回码与响应；只有配置明确的重试可捕获异常，并在耗尽后重新抛出原异常链。
- 新代码不得包含 bare `except`、`except Exception: pass`、隐式 bool 字符串转换或未知配置忽略。
- 旧凭证消费模块与历史 Notebook 全部删除；新 AList/企业微信服务只从环境或 Kaggle Secret 获取认证信息，启用服务但缺失 Secret 时在首个拟合 step 前失败。发布前必须轮换历史代码暴露过的凭证，提案和新文档不得复述旧值。
- CI 扫描 Git 受跟踪的 `.py/.ipynb/.yaml/.yml/.toml/.md`：敏感变量名绑定非空字符串字面量、`set_token(<literal>)`、以及长度至少 20 且 Shannon entropy 大于 4.0 的字符串均失败。仅允许明确识别的 Git SHA、SHA256、URL、绝对路径、`.invalid` 域名和 `${SECRET_KEY}`/大写 Secret key 引用；豁免必须由解析规则判定，不允许按整文件跳过。
- Git 历史清理与外部凭证轮换不由 Apply Agent 自动执行，但发布检查表必须要求仓库所有者确认。

## Dependency And Packaging Decision

新增 `pyproject.toml`，项目版本固定为 `1.0.0`，支持 Python `>=3.10,<3.13`。核心依赖范围固定为：

- `torch>=2.4,<2.8`
- `accelerate>=1.6,<2`
- `numpy>=2,<3`
- `matplotlib>=3.8,<4`
- `scikit-learn>=1.5,<2`
- `safetensors>=0.5,<1`
- `PyYAML>=6,<7`
- `joblib>=1.4,<2`
- `requests>=2.32,<3`

`joblib` 与 `requests` 是平台代码直接导入且 AList/企业微信属于内置通用服务，因此均为直接核心依赖，不再提供 `alist` extra。`dev` extra 增加 `pytest>=8,<9`、`pytest-cov>=5,<7`、`build>=1,<2`。不提供 `legacy` extra；Pandas、RLlib、Stable-Baselines3、imitation、AutoGluon、GUI、TorchMetrics、py-ext 及旧 C++ 扩展依赖全部移除。删除 `setup.py` 和 `requirements.txt`，`pyproject.toml` 是唯一元数据与依赖来源。

Kaggle 不主动替换平台 torch；doctor 核验实际版本是否落在范围内。版本不兼容立即失败并要求使用兼容镜像或项目 revision，不在训练 Notebook 中动态升级大型框架。

## Decisions

### D-001 单一新平台完整替换旧体系

选择新建 `dl_helper.training`，并在同一 change 删除旧入口、模型、实验和额外训练栈。用户明确不要求兼容；保留双栈会继续携带重依赖、明文凭证、失配脚本和领域耦合。既不原地改造旧 Trainer，也不提供 shim；回滚单位是整个 Git revision。

### D-002 Accelerate 作为唯一分布式抽象

继续使用已安装并验证的 Accelerate 1.6，不直接手写 DDP/AMP/checkpoint。所有 worker 共享相同 engine；launcher 只传实验引用和配置。

### D-003 显式 Task 与 PreparedBatch

不从 batch 长度、Tensor 维度或模型输出猜测任务。默认调用覆盖 tensor、mapping 和 tuple，其他结构由 Task 处理，从而支持任意模型同时保持错误可定位。

### D-004 自研有限核心流式指标

内置指标使用加权混淆矩阵、TP/FP/FN/TN 和 mergeable weighted moments，不在核心导入 TorchMetrics。这样避免当前 `torchmetrics -> transformers` 的重导入链和全量 logits；公式、sample-weight、zero-division、聚合与版本由 MetricDefinition 和 sklearn 金标固定。复杂自定义指标仍通过 Task 插件实现。

### D-005 性能优化显式且可审计

AMP auto、DataLoader 建议值和多 GPU有确定规则；`torch.compile`、find-unused-parameters 和动态 batch 不自动启用。OOM 与 graph break 直接失败，不改变批量或训练语义。

### D-006 Kaggle 配额不硬编码

平台配额可能随账户和资源变化，因此不保留当前 9/12 小时猜测。Kaggle profile 强制用户给出 max/grace，运行时据此可恢复退出。

### D-007 不可变检查点与兼容指纹

检查点不覆盖写入，latest 最后发布。只允许列出的运行/报告参数变化，避免用不同模型或数据静默恢复。Accelerate 保存训练状态，DataModule 保存数据位置。

### D-008 epoch 恢复普适，中途恢复能力显式

任意 DataLoader 都可在 epoch 边界恢复；中途恢复只对实现状态协议的数据模块启用。相比重复本轮样本或假装精确恢复，该约束更符合通用训练的可审计性。

### D-009 安全 AList ArtifactStore 而非实验副作用

AList 是内置、配置可关闭的通用 ArtifactStore，host 和 Secret 必须显式配置；Experiment/Task 不导入或连接它。新客户端按 REF-006 协议实现 run/sweep/checkpoint 的超时、校验和与发布顺序，不复用无超时单例。

### D-010 固定 Artifact schema 与离线报告

JSONL/JSON/NPZ/HTML 让 Kaggle 输出可直接检查，也允许不重新加载模型地重建报告。不把 Notebook 单元输出或在线 dashboard 作为唯一成果。

### D-011 异步上传错误延迟到安全边界但最终失败

后台同步提高 GPU 利用率，但错误必须在下一边界或 final flush 重新抛出。只合并尚未开始的 pending checkpoint，避免无限上传队列。

### D-012 平台依赖范围与 doctor

使用版本范围而非锁死 Kaggle 大型框架补丁版本；每次运行记录实际环境并在训练前核验。Notebook 禁止静默升级框架，保证启动行为可复现。

### D-013 删除清单和负向门禁固定替换边界

以 Repository Replacement Boundary 为规范删除旧体系，不修复或迁移其中任何模块。测试同时断言旧 import 失败、禁止路径不再受 Git 跟踪、构建 wheel 不含旧包、依赖元数据不含旧重依赖；这避免仅移动文件、用 `.gitignore` 隐藏或保留不可见兼容层。

### D-014 backend-neutral engine 与 PyTorch/sklearn 双实现

共享 engine 只持有生命周期、评价和 Artifact 合同，PyTorch 与 sklearn worker 分别实现框架语义。sklearn batch 只做一次不可恢复 fit，incremental 只在成功 partial_fit batch 边界恢复；二者都通过 PredictedBatch 使用同一指标。相比把 estimator 包成伪 `nn.Module`，该边界保留 Pipeline/sparse/sample_weight/n_jobs/random_state 的原生行为，也不虚构 optimizer/epoch/AMP。

### D-015 版本化科学评价合同

每个指标必须声明公式版本、方向、average、sample-weight、zero-division、精确性和 full/sampled scope；selection/sweep 只接受全量精确验证指标。回归采用 weighted Welford/Chan moments，分类采用 float64 加权统计，内置公式以 sklearn 1.6.1 显式参数金标和极端数值矩阵锁定。这样既支持生产加权样本，也防止库默认值、抽样 AUC、显示舍入或公式升级静默改变结论。

### D-016 AList 与企业微信作为通用生命周期服务

两项服务由统一 Secret resolver、required/record policy 和 service audit 驱动，覆盖 run、trial、sweep 的开始/暂停/成功/失败；不再散落到训练脚本或依赖 `py_ext`。企业微信只调用官方 token/应用消息 API，AList 发布完整 run/sweep/checkpoint Artifact。外部服务无法跨系统事务，因此以稳定 event_id、不可变 bundle、terminal-last 和可重入 finalization 保证可审计幂等，且任何 secondary 服务异常不得覆盖原训练异常。

### D-017 base/variant 与显式顺序 sweep

一个完整 base 配置配合多个严格 patch 复用同一 Experiment；独立 sweep manifest 固定 trial 顺序、验证比较指标和派生 run ID。所有 trial 在零优化步时完成独立 evaluation contract 预检，顺序子进程隔离库全局状态，暂停可续、失败不可伪跳过，只有全部成功才排名和产生 best。拒绝自动搜索与并发 GPU 争抢，是为了在 Kaggle 有限会话中优先保证复现、恢复与公平比较。

## Alternatives Rejected

- PyTorch Lightning：会引入第二套训练生命周期和较大迁移成本，现有 Accelerate 已满足 DDP/AMP/checkpoint 基础需求。
- Hugging Face Trainer：对通用非 Transformer 模型、任意 batch 和任意结构输出约束过强。
- 继续扩展 `test_base`：命名与组织已经把实验、测试、平台和领域逻辑混合，无法提供严格公共合同。
- 并行保留旧 API 或 legacy extra：会让发布包继续携带两套生命周期、RL/GUI 重依赖和已知安全问题，与用户要求的完整重构冲突。
- 全量预测 gather 后计算 sklearn：指标精确但内存和通信随样本线性增长，不适合大规模 Kaggle 数据。
- 自动探测损失/任务/指标：会隐藏模型输出错误，违反 Fail Fast。
- 自动下载最新版依赖：提高短期便利但破坏 Kaggle 复现并可能消耗大部分会话启动时间。
- 将 sklearn estimator 包装成 PyTorch 模型：会丢失 Pipeline、sparse、sample_weight 和 estimator 参数语义，并伪造不存在的反向传播、AMP 与 checkpoint 能力。
- 直接调用 GridSearchCV/Optuna：会形成第二套并发、评价、恢复和报告生命周期；本 change 只对显式 variant 做可审计顺序比较。
- 继续复用 `py_ext.wechat`：旧实现硬编码身份并使用进程级单例，无法满足 Secret、超时、审计、run/sweep 事件和原异常保护合同。

## Replacement And Rollback Sequence

1. 建立新包装、配置、合同和 CPU 测试；新模块不得引用旧代码。
2. 实现版本化流式指标、backend-neutral engine、PyTorch/sklearn worker、checkpoint、Artifact 和报告。
3. 实现 Local/Kaggle platform、AList、企业微信、base/variant/sweep、CLI 与 Notebook 模板。
4. 创建完全独立的 PyTorch/sklearn 通用示例和 sweep 配置，不复制旧模型或实验代码。
5. 按固定删除清单移除旧包、实验、RL/AutoML/GUI、Notebook/env、C++ 和二进制资产，并运行 Git inventory、import graph、wheel contents 和 Secret 门禁。
6. 只发布新 CLI/配置/服务/调参文档，完成 Windows/Linux CI 与 Kaggle GPU 冒烟。

回滚必须恢复到发布记录中的最后一个旧版 commit，不在新 revision 内恢复单个旧文件或启用兼容开关。新运行目录与旧格式不共享 latest 指针；不得把新 manifest 目录转换或覆盖为旧 7z 缓存。

## Test Strategy By Decision

| 决策 | 必测内容 |
|---|---|
| D-001/D-003 | Tensor、mapping、tuple、多输入和自定义输出；旧模块导入全部失败 |
| D-002/D-005 | CPU、模拟单 CUDA配置、两进程 gloo；累积后 optimizer step 数与有效批量 |
| D-004 | sklearn 金标、缺失类别、零分母、常量回归、两进程归约、内存状态大小 |
| D-006/D-012 | Kaggle env/path/设备/版本/磁盘/预算 doctor 正负用例 |
| D-007/D-008 | epoch 与 mid-epoch 恢复、损坏 checksum、指纹差异、允许延长 epochs |
| D-009/D-011 | AList mock server 超时、5xx 重试、401 不重试、异步异常传播、发布顺序 |
| D-010 | 三种内置报告、类别>20、抽样标签、HTML escape、幂等重建 |
| D-013 | Git 禁止路径为零、wheel 无旧包、旧 import 失败、旧重依赖为零、Secret 扫描通过 |
| D-014 | sklearn batch/incremental、Pipeline/sparse/sample_weight、n_jobs/random_state、可信 joblib、与 Torch 共享 PredictedBatch |
| D-015 | sklearn 1.6.1 显式金标、随机分块/权重/极端不平衡/近常量/大偏置、多 rank 与公式版本漂移 |
| D-016 | AList 完整 run/sweep 发布、企业微信 token/消息/2048 UTF-8 bytes、required/record 审计、原异常保护 |
| D-017 | 严格 patch 合并、路径逃逸、trial contract 可比性、顺序/失败/暂停恢复、未舍入稳定排名 |
