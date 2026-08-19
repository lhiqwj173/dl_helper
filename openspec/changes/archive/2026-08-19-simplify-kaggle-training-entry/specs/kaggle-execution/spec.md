## MODIFIED Requirements

### Requirement: 解耦训练项目的 Kaggle 启动
Kaggle bootstrap MUST 只安装 dl-helper，不运行或引用库仓库内的训练项目。Git ref MAY 是 tag、分支或短 SHA，也 MAY 省略；安装 MUST 使用 `pip install -e . --no-deps`。训练时 MUST 显式提供外部 `--project-dir`、`--config`、`--experiment`，并 MAY 使用 `--output-root` 覆盖输出目录。官方 Kaggle 文档中的可执行命令 MUST 是使用当前 kernel `sys.executable` 的 Python Notebook 单元，不得要求 PowerShell 或开发者本机解释器路径。

#### Scenario: 外部训练项目启动
- **WHEN** 用户安装 dl-helper 并提供外部项目目录、配置和 Experiment 引用
- **THEN** train 自动预检后启动训练，dl-helper 无需因训练项目变化而发布新版本

#### Scenario: Notebook 直接执行
- **WHEN** 用户按 Kaggle 指南依次运行安装、训练、恢复、sweep 或报告单元
- **THEN** 每个命令都通过 `subprocess` 和 `sys.executable` 在 Notebook 环境执行，并明确处理 0、75 与失败退出码

#### Scenario: 可选版本标识错误
- **WHEN** 显式 Git ref 无法 checkout、source revision 含空白或依赖超出范围
- **THEN** Notebook 单元非零停止且不拟合

### Requirement: 运行预算与可恢复退出
Kaggle profile MUST 通过独立于用户配置的执行策略自动使用 660 分钟总预算与 10 分钟 shutdown grace；用户配置 schema 和 CLI MUST NOT 接受预算参数，Local profile MUST 不启用运行预算。系统 MUST 在 `elapsed >= 650 minutes` 的成功 batch（Torch 为 optimizer step）后停止新拟合 step；同时 MUST 在每个完整 epoch 结束后，以已完成完整 epoch 的平均耗时预测下一 epoch，若平均耗时大于剩余训练时间，则在当前 epoch 边界停止。两种暂停路径都必须保存可恢复 checkpoint、刷新 required 服务、生成当前报告和 pause manifest。无法形成可恢复状态 MUST 为 FAILED，独立 execution-policy Artifact MUST 记录实际 660/10 值。

#### Scenario: Kaggle 自动预算
- **WHEN** Kaggle 用户启动训练
- **THEN** 自动预检、父进程和 worker 一致使用 660/10，并在 execution-policy Artifact 中记录该值

#### Scenario: 已删除的 runtime 配置
- **WHEN** 用户 YAML 包含根级 `runtime`
- **THEN** 严格配置解析按未知字段立即失败，不兼容、不忽略也不转换旧输入

#### Scenario: Local 默认不限时
- **WHEN** Local 用户启动训练
- **THEN** 系统不创建运行预算，不因 Kaggle 固定值暂停本地调试

#### Scenario: Torch 安全暂停
- **WHEN** budget 命中且 DataModule 支持中途恢复
- **THEN** 系统不再执行 optimizer step，保存完整状态、服务 flush、报告和 pause manifest，返回 75

#### Scenario: sklearn incremental 安全暂停
- **WHEN** budget 在一次成功 partial_fit 后命中
- **THEN** 系统保存 estimator/source/EngineState/metric 状态并完成同样 PREEMPTED/75 流程

#### Scenario: 完整 epoch 预测暂停
- **WHEN** 一个完整 epoch 结束且其后续 epoch 的平均耗时预计超过剩余训练时间
- **THEN** 系统在该 epoch 边界保存最新 checkpoint，提交 AList并刷新 required 服务，生成 pause manifest，返回 75；恢复的半个 epoch 不得计入平均值

#### Scenario: grace 或持久化失败
- **WHEN** 最终预算不一致、checkpoint 不完整或 required service 在 deadline 前失败
- **THEN** 运行 FAILED 且不存在 SUCCEEDED/PREEMPTED manifest
