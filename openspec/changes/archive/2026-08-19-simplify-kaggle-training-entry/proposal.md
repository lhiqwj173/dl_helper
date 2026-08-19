# Change: 简化 Kaggle 训练入口并隔离示例项目

## Why

当前仓库虽然已支持外部 `--project-dir`，但顶层 `experiments/`、`configs/` 仍容易被误解为库模块；训练命令和文档还要求用户重复填写 `--resume auto` 与 Kaggle 运行预算，并在 Notebook 指南中混入不可直接执行的 PowerShell 示例。这增加了 Kaggle 首次训练、跨 Session 恢复和 sweep 的认知负担。

## What Changes

- 将仓库自带实验和配置移动到明确的 `examples/` 项目树，只作为可运行示例和测试夹具。运行入口拒绝 `dl_helper.*` 实验引用以及位于库包目录内的配置和输出路径，确保真实训练代码、配置与产物只能属于用户项目。
- `train` 在未传 `--resume` 时内部使用自动恢复；只保留 `--resume none|required` 作为显式覆盖，删除公开的 `auto` 参数值和 YAML `checkpoint.resume` 字段。
- 从用户配置 schema 删除 `runtime`。Kaggle 通过独立执行策略自动使用固定的 660 分钟总预算与 10 分钟收尾窗口；Local 不启用运行预算。
- 重写 Kaggle 指南为逐单元 Notebook 工作流，所有命令使用 Python `subprocess` 与当前 kernel 的 `sys.executable`，完整覆盖安装、项目组织、Secrets、首次训练、退出码、跨 Session 恢复、sweep 和本地调试。
- 更新 README、示例配置、Notebook、测试和当前 OpenSpec，删除要求用户手工填写上述默认项的说明。

## Success Criteria

- Kaggle 用户的训练命令无需 `--resume auto`，业务 YAML 无需 `runtime`，仍能自动预算暂停、上传 AList 并在下一 Session 自动恢复。
- 文档中的 Kaggle 代码块可以直接粘贴到 Python Notebook 单元执行，不包含 PowerShell 续行符或 Windows 解释器路径。
- `examples/` 之外不存在仓库内置训练项目或业务配置；wheel 是否携带 examples 不构成运行合同。
- 旧 `--resume auto`、YAML `checkpoint.resume` 和根级 `runtime` 被严格拒绝，不保留兼容分支或旧路径代理。

## Scope

- Affected specs: `general-training`, `kaggle-execution`
- Affected code: `dl_helper/training/cli.py`, `config.py`, `doctor.py`, sweep 子进程参数构造与相关测试
- Affected data/config: `experiments/`, `configs/` 移入 `examples/`
- Affected docs: README、`docs/training/*`、Kaggle Notebook 模板

## Non-Goals

- 不改变模型、DataModule、Task、checkpoint 格式或 AList/企业微信协议。
- 不自动发现用户项目、配置文件、数据集或 Experiment 引用。
- 不移除高级的 `--resume none|required` 控制，也不让 sklearn batch 获得不可实现的中途恢复能力。

## Migration And Rollback

- 这是有意的破坏性清理：仓库内部引用一次性迁移到 `examples/`，外部项目必须删除 `checkpoint.resume` 和 `runtime`，并停止传入 `--resume auto`。
- 不提供旧路径 package、schema alias、弃用期或静默忽略。未知字段和已删除参数立即失败并指出替代行为。
- checkpoint 和 Artifact 格式不变；回滚代码不需要迁移已有 checkpoint 数据。
