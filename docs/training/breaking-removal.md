# 破坏性迁移（1.0.0 系列）

`dl.helper` 的版本演进会删除过期配置面与旧入口。本文档记录实际断点，便于升级与回滚判定。

## 已删除的公共配置面

- `checkpoint.resume`：恢复策略不再由 YAML 配置。省略 `--resume` 为内部自动恢复；只能显式 `--resume none|required`。
- 根级 `runtime`（`runtime.max_minutes` / `runtime.shutdown_grace_minutes`）：运行预算改由平台执行策略提供，Kaggle 恒为 660 分钟训练 + 10 分钟收尾（run 目录 `execution-policy.json` 审计），Local 不启用预算。
- 显式 `--resume auto`：`auto` 不再是可选 CLI 值；省略参数即自动恢复。
- `doctor` 命令：预检并入 `train` 启动路径，不再有独立命令；sweep 可比性预检使用隐藏的 `--preflight-only`。

## 已删除的仓库目录

- 顶层 `experiments/` 移至 `examples/experiments/`，`configs/` 移至 `examples/configs/`。仓库自带实验与配置只作为示例与测试夹具，不再是库模块；通过 `--project-dir <repo>/examples` 使用。
- 新公共 API 只位于 `dl_helper.training`；旧 `dl_helper.trainer/tester/tracker/train_param/...` 等模块不受支持。

## 兼容性承诺

- 不提供 shim、重导出、`__getattr__`、契约参数代理或静默忽略：未知字段和已删除参数立即失败并指出替代行为。
- checkpoint 与 Artifact 格式不变，回滚代码不需要迁移已有 checkpoint 数据。
- 外部项目必须删除 `checkpoint.resume` 与 `runtime`，并停止传入 `--resume auto`。