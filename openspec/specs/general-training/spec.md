# general-training Specification

## Purpose
TBD - created by archiving change build-general-kaggle-training-platform. Update Purpose after archive.
## Requirements
### Requirement: 显式双后端实验工厂
系统 MUST 从 `--project-dir` 指定的外部训练项目通过 `module:function` 引用构造 frozen Experiment，并仅接受 backend 与配置一致的 TorchExperiment 或 SklearnExperiment。训练项目 MUST NOT 成为 `dl_helper` 模块；配置文件和 output root MUST 位于实际导入的 `dl_helper` 包目录之外。仓库自带训练项目和配置只能作为 `examples/` 下的示例，其导入和执行 MUST 与任意外部用户项目使用相同入口。wheel 是否携带示例不属于运行合同。

#### Scenario: 构造 PyTorch 自定义模型
- **WHEN** 外部工厂返回满足合同的任意全新 `torch.nn.Module`、DataModule、Task、optimizer 和 scheduler 工厂
- **THEN** Torch worker 独立构造组件并在首个优化步前完成合同预检

#### Scenario: 构造 sklearn Pipeline
- **WHEN** 外部工厂返回可由 `clone(safe=True)` 克隆的未拟合 estimator 或 Pipeline 及对应 DataModule/Task
- **THEN** sklearn worker 保留 ndarray/sparse/array-like 输入并按声明 fit mode 训练

#### Scenario: 使用兼容第三方 wrapper
- **WHEN** XGBoost、LightGBM、CatBoost 或其他可选库对象满足 sklearn estimator 协议和当前 Task 要求
- **THEN** 系统通过 sklearn backend 使用该对象，但项目不自动安装其第三方库

#### Scenario: 运行仓库示例
- **WHEN** 用户将仓库 `examples/` 作为 project dir 并引用其中 Experiment 与配置
- **THEN** 系统通过标准外部项目入口运行示例，不要求修改或导入 `dl_helper` 业务模块

#### Scenario: 训练内容进入库模块
- **WHEN** Experiment 引用为 `dl_helper`/`dl_helper.*`，或配置/output root 位于实际 `dl_helper` 包目录内
- **THEN** 系统在导入 Experiment 或创建产物前失败并指出训练内容必须属于库外用户项目

#### Scenario: 实验或 backend 不匹配
- **WHEN** 引用无效、返回对象不满足合同、TorchExperiment 配置为 sklearn，或 SklearnExperiment 配置为 torch
- **THEN** 系统在任何拟合 step 前抛出具体异常并以非零状态结束

### Requirement: 通用 PyTorch DataModule 合同
Torch backend MUST 接受产生任意 PyTorch DataLoader 的 DataModule，要求非空 DataIdentity、显式固定或 dynamic nominal train batch size，并明确区分 epoch 边界与中途恢复能力。系统 MUST NOT 根据目录首个文件或 batch shape 猜测数据身份。

#### Scenario: 包装任意 DataLoader
- **WHEN** LoaderDataModule 提供 train、可选 val/test/predict DataLoader 和完整 DataIdentity
- **THEN** 系统按存在的 split 执行并允许 epoch 边界恢复

#### Scenario: 请求不支持的中途恢复
- **WHEN** 配置 optimizer-step checkpoint 或运行预算，但 DataModule 声明不支持中途恢复
- **THEN** 系统在训练前失败，不通过重复本 epoch 样本隐式回退

#### Scenario: dynamic 与最后不足批次
- **WHEN** nominal batch 为 null，或固定 batch 仅最后一个 batch 较小
- **THEN** 前者记录 dynamic 与实际范围，后者仍按声明值计算有效批量且不误判违规

### Requirement: sklearn 数据与 estimator 合同
sklearn backend MUST 显式区分提供完整训练集的 batch DataModule 与提供可恢复 batch source 的 incremental DataModule。features MUST 保持 estimator 原生接受的类型，预处理 MUST 在训练 Pipeline 内只对 train fit，系统 MUST NOT 对 val/test 独立拟合预处理器。

#### Scenario: batch fit
- **WHEN** fit_mode=batch、max_epochs=1、resume=none、Local profile 且 DataModule 提供 full_train_data
- **THEN** 系统 clone estimator、恰好调用一次 fit、评价所需 split，并导出 best=last joblib 模型

#### Scenario: incremental partial_fit
- **WHEN** fit_mode=incremental 且 estimator、DataModule 和分类 classes 满足增量合同
- **THEN** 系统按 epoch/batch 顺序调用 partial_fit，并只在成功 batch 边界创建可恢复状态

#### Scenario: 训练 sample weight
- **WHEN** EstimatorBatch 提供有限非负 sample_weight
- **THEN** 配置必须显式提供 fit/partial_fit 参数路径并原样转发；缺失、被 estimator 拒绝或权重总和非正时立即失败

#### Scenario: 防止数据泄漏
- **WHEN** 用户将 scaler、encoder 或 imputer 放在 train Pipeline 之外并要求引擎对评价数据拟合
- **THEN** 系统拒绝该隐式行为，不使用 val/test 统计量训练预处理器

### Requirement: 统一评价 Task 与 PredictedBatch
两个 backend MUST 通过 EvaluationTask 将真实值、预测值、可选 score、sample weight、sample ID 和 metadata 转换为唯一 PredictedBatch 合同，并以同一 MetricState 评价。系统 MUST NOT 按 `hasattr` 优先级替换 Task 明确要求的 predict、predict_proba 或 decision_function。

#### Scenario: 两后端评价相同数据
- **WHEN** TorchTask 与 SklearnTask 产生语义相同的 PredictedBatch
- **THEN** 系统应用完全相同的 MetricDefinition、校验、流式状态和 Artifact schema

#### Scenario: estimator 缺少声明输出
- **WHEN** SklearnTask 要求 predict_proba，但 estimator 只实现 predict
- **THEN** train 自动预检在拟合前失败，不改用 decision_function 或硬标签伪造概率

#### Scenario: 预测 batch 合同错误
- **WHEN** sample_count 非正，或 targets/predictions/scores/weight/sample_ids 的样本维不一致
- **THEN** 系统立即抛出 Task 合同错误

### Requirement: 嵌套批次与任意 PyTorch 模型调用
TorchTask MUST 将任意 batch 转换为 PreparedBatch。默认调用 MUST 对 Mapping 使用关键字参数、对 tuple 使用位置参数、对其他对象使用单参数，并允许 Task 完全覆盖 forward。

#### Scenario: mapping 与 tuple 多输入
- **WHEN** inputs 分别为 Mapping 或 tuple
- **THEN** 系统分别执行 `model(**inputs)` 或 `model(*inputs)`，不修改键和位置

#### Scenario: list 输入
- **WHEN** inputs 是 list
- **THEN** 系统将其作为一个模型参数；需要位置展开的 Task 必须显式转换为 tuple

#### Scenario: PreparedBatch 权重错误
- **WHEN** sample_weight 不是长度 N 的有限非负一维浮点 Tensor或批内权重和不为正
- **THEN** 系统在 loss/metric 更新前失败

### Requirement: 内置与自定义任务
系统 MUST 提供多分类、多标签和单/多目标回归的 TorchTask 与 SklearnTask 适配，并允许自定义 Task 定义预测映射、LossResult、MetricState、prediction arrays 和报告类型。系统 MUST NOT 从输出维度自动选择任务、损失或标签顺序。

#### Scenario: 多分类、多标签与回归
- **WHEN** 输入分别满足显式声明的 logits/class、label/threshold 或连续 target schema
- **THEN** 对应 Task 产生版本化 MetricDefinition 和合法 PredictedBatch

#### Scenario: 自定义结构化输出
- **WHEN** 自定义 TorchTask 覆盖 forward/to_predicted_batch，或 SklearnTask 显式映射 estimator 输出
- **THEN** backend-neutral engine 只消费公开合同而不解释业务字段

#### Scenario: shape 或 dtype 错误
- **WHEN** 内置 Task 收到不符合声明的 rank、dtype、类别、label 或 target 数
- **THEN** 系统报告期望/实际值并失败，不 squeeze/reshape 到猜测形状

### Requirement: 精确的 PyTorch 优化循环
Torch backend MUST 在 CPU、单 CUDA 和多 CUDA 使用一致的 AMP、梯度累积、梯度裁剪和 SchedulerBinding 语义。LossResult MUST 提供损失加权和与正 denominator；系统 MUST 在每个 accumulation window 对全部 rank 的 denominator 归约并在 clip 前规范化梯度，使不等尾批和 sample weight 下的梯度等于全局加权均值。

#### Scenario: 梯度累积与不等尾批
- **WHEN** accumulation steps 为 4 且最后窗口不足 4 个 micro-batch或不同 rank denominator 不同
- **THEN** optimizer/global step 仅对实际窗口各增加一次，梯度与同数据单批全局加权金标一致

#### Scenario: scheduler interval
- **WHEN** interval 为 optimizer_step、epoch 或 validation_metric
- **THEN** 系统只在对应边界调用一次；validation_metric 必须指向存在的全量验证指标

#### Scenario: 非有限训练状态
- **WHEN** loss numerator/denominator、unscaled gradient、clip norm 或关键模型参数出现 NaN/Inf
- **THEN** 系统写脱敏失败 Artifact 并抛出异常，不继续 optimizer step

### Requirement: 验证选择与早停
有验证 split 时系统 MUST 要求 direction 与 MetricDefinition 一致的全量精确 selection metric，并按 min_delta/patience 选择 best 与早停；无验证 split 时 MUST 禁止 selection，只产生 last 模型。

#### Scenario: 保存最佳模型
- **WHEN** 当前未舍入验证值按方向改善超过 min_delta
- **THEN** 系统记录 epoch/global step、保存 backend 对应 best 模型并更新引用

#### Scenario: selection 指标不可用
- **WHEN** 指标缺失、非有限、sampled、非 exact 或方向不一致
- **THEN** 系统失败，不切换到 val/loss 或报告舍入值

#### Scenario: 早停
- **WHEN** 连续 patience 个完整验证边界未改善
- **THEN** 系统先安全保存当前状态，再停止拟合并进入 test/finalize

### Requirement: backend-aware 可审计恢复
系统 MUST 使用不可变 checkpoint、完整 manifest、SHA256、latest-last 和严格兼容指纹。Torch MUST 恢复模型/优化器/scaler/scheduler/RNG/DataModule/EngineState；sklearn incremental MUST 恢复可信 estimator joblib、batch source、RNG、EngineState 和部分指标；sklearn batch MUST 明确拒绝恢复。

#### Scenario: Torch 或 incremental 兼容恢复
- **WHEN** latest 完整且 config/backend/data/model/runtime version 全部兼容
- **THEN** 系统从下一未完成位置继续，不重复 optimizer/partial_fit step 或指标累计

#### Scenario: 不可信 joblib
- **WHEN** joblib 来自其他 run/用户路径、是 symlink、缺 manifest、checksum 不符或 sklearn/numpy/scipy/joblib/Python 版本非精确匹配
- **THEN** 系统在调用 joblib.load 前拒绝恢复

#### Scenario: 损坏或配置漂移
- **WHEN** 文件缺失、latest/manifest 不完整或非允许配置、数据、Task、指标定义、模型字段变化
- **THEN** 系统列出不含 Secret 的差异并失败，不尝试旧 checkpoint

### Requirement: 单一新平台与旧 API 移除
系统 MUST 只发布 `dl_helper.training` 新公共 API，并删除旧 trainer/tester/tracker/train_param/models/rl/tests 等模块。系统 MUST NOT 提供重导出、兼容 shim、弃用代理或 legacy extra。

#### Scenario: 新核心无副作用导入
- **WHEN** 用户执行 `import dl_helper.training`
- **THEN** 导入成功且不触发 transformers、GUI、网络、Secret 解析或实验构造

#### Scenario: 导入已删除模块
- **WHEN** 用户导入旧 trainer/tester/models/rl/实验路径
- **THEN** Python 报告模块不存在且不重定向

#### Scenario: 构建 wheel
- **WHEN** 检查候选 wheel
- **THEN** wheel 只包含新包代码，不包含旧 API、领域模型或兼容层

### Requirement: 默认自动恢复的训练入口
`train` 在用户省略 `--resume` 时 MUST 使用内部自动恢复策略：优先使用本地 latest checkpoint，本地不存在时查询已配置远程服务，两处都不存在时开始新训练。公开 CLI 只允许显式 `none|required`；配置 schema MUST NOT 包含恢复策略。发现任何候选 checkpoint 后仍 MUST 执行完整兼容校验，不得因自动模式静默忽略损坏或漂移。

#### Scenario: 首次运行省略 resume
- **WHEN** 用户未传 `--resume` 且本地和远程都不存在 checkpoint
- **THEN** 系统从头开始训练，不要求用户填写 `--resume auto`

#### Scenario: 跨会话自动恢复
- **WHEN** 用户未传 `--resume`、本地无 checkpoint 且远程存在兼容 latest
- **THEN** 系统下载并严格校验 checkpoint，从下一未完成位置继续

#### Scenario: 显式恢复策略
- **WHEN** 用户传入 `--resume none` 或 `--resume required`
- **THEN** 系统分别禁止恢复，或在无法找到兼容 checkpoint 时非零失败

#### Scenario: 已删除的恢复配置
- **WHEN** 用户显式传入 `--resume auto` 或 YAML 包含 `checkpoint.resume`
- **THEN** CLI 或严格配置解析立即失败，不兼容、不忽略也不转换旧输入
