## MODIFIED Requirements

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

## ADDED Requirements

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
