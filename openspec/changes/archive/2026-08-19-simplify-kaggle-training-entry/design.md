## Context

`dl_helper.training.cli._cmd_train` 当前以 CLI 和 YAML 两处共同决定恢复策略；`RuntimeConfig` 混在业务 `Config` 中，并借助配置序列化传给 Torch worker。顶层 `experiments/` 与 `configs/` 也没有表达“仅示例”。这些边界让平台策略、用户业务配置和仓库演示相互混杂。Kaggle 文档的 sweep 示例还使用 PowerShell，无法直接作为 Notebook Python 单元运行。

## Goals

- Kaggle 常规训练只要求项目目录、配置、Experiment 引用和稳定 run ID。
- 平台安全默认值由库控制并在独立 execution-policy Artifact 中可审计。
- 示例可以运行，但在目录、导入路径和文档上都与工具库模块隔离。
- 删除错误的旧参数和 schema 字段，不保留兼容代码或静默恢复降级。

## Decisions

### D-001 示例统一放入外部项目形态的 `examples/`

将 `experiments/` 移为 `examples/experiments/`，将 `configs/` 移为 `examples/configs/`。示例命令使用 `--project-dir <repo>/examples` 与 `experiments.<name>:build_experiment`。测试需要示例时显式使用该 project dir。wheel 是否携带示例不影响本合同；关键边界是示例和真实训练代码都不得成为 `dl_helper` Python 模块。

CLI 在导入前执行三项边界校验：Experiment 模块名不得为 `dl_helper` 或以 `dl_helper.` 开头；配置文件不得位于实际导入的 `dl_helper` 包目录内；output root 不得位于该包目录内。违反任一条件立即抛出 `CliError`。这直接阻止用户通过修改库模块承载训练业务。

不采用继续保留顶层目录并只加 README 注释，因为目录结构仍会暗示它们属于库的正式运行面。

### D-002 train 默认自动恢复

`build_parser()` 的 `--resume` choices 只保留 `none|required`，default 使用内部常量 `auto`；因此用户无法再显式输入多余的 `--resume auto`。`_cmd_train` 直接使用解析后的策略。用户省略参数时：本地 latest 存在则恢复；本地不存在且服务启用则查询 AList；两处均无 checkpoint 时从头开始。`required` 仍要求存在兼容 checkpoint，`none` 明确禁止恢复。

从 `CheckpointConfig`、严格 YAML schema、variant allowlist、指纹和示例中删除 `resume`。sklearn batch 在内部 auto 下不查询或加载 checkpoint；显式 `required` 对 batch 在预检阶段失败，显式 `none` 正常运行。

### D-003 Kaggle 固定预算，Local 不自动限时

从公共 `Config` 和严格 YAML schema 完全删除 `RuntimeConfig/runtime`。在平台模块新增不可由 YAML 构造的 frozen `ExecutionPolicy`，并定义唯一 Kaggle 策略：

| 字段 | Kaggle 策略值 | Local 策略值 |
|---|---:|---:|
| `max_minutes` | `660` | `null` |
| `shutdown_grace_minutes` | `10` | `10` |

CLI 检测平台后构造 `ExecutionPolicy`，将其显式传给自动预检、单进程 backend 和 launcher。launcher 在 spawn 参数中单独传递纯 dict，并在子进程严格重建策略，不再借用业务配置序列化。`config.resolved.yaml` 只保存用户配置；新增 `execution-policy.json` 记录平台、resume policy、max minutes 和 grace。650 分钟为训练截止，10 分钟用于 checkpoint、AList、通知、报告和终态刷新，平台 720 分钟上限之外另留约 60 分钟缓冲。

严格 schema 遇到根级 `runtime` 或 `checkpoint.resume` 必须按未知字段失败。自动预检校验 `ExecutionPolicy`，Kaggle policy 缺失或值不一致立即失败。

不采用从 Notebook 启动时间推导剩余配额，因为 Kaggle 没有稳定、已被本项目验证的会话剩余时间 API；也不使用 720 分钟满额，避免平台终止早于 required 服务完成。

### D-004 Notebook 命令只使用当前 Python kernel

Kaggle 指南中的可执行命令统一写为 Python 单元：

```python
import subprocess
import sys

proc = subprocess.run(
    [sys.executable, "-m", "dl_helper.training.cli", "train", ...],
    cwd=project_dir,
    check=False,
)
if proc.returncode not in (0, 75):
    raise RuntimeError(f"训练失败，退出码: {proc.returncode}")
```

安装步骤也使用 `subprocess.run(..., check=True)`。首次训练必须展示 `75` 是成功保存后的暂停；恢复步骤再次执行同一训练单元和同一 run ID，不再添加 `--resume auto`。只有需要“必须恢复、没有 checkpoint 就失败”的审计场景才展示 `--resume required`。sweep 与 sweep-report 同样使用 `sys.executable`，不出现 PowerShell、反引号续行或本机绝对 Python 路径。

## Data Flow

1. CLI 解析参数，缺省 resume 得到内部 `auto`；显式只接受 none/required。
2. 加载外部 YAML；严格拒绝已删除的 resume/runtime 字段。
3. 检测平台并构造独立 `ExecutionPolicy`：Kaggle 为 660/10，Local 无预算。
4. 自动预检验证用户 config、execution policy、Secrets、AList 和企业微信。
5. `auto` 先查本地 latest，再查 AList；不存在则新训，存在则严格验证后恢复。
6. worker 使用 resolved budget 训练，并按硬截止与 epoch 预测安全暂停。

## Failure Behavior And Invariants

- 自动恢复不等于宽松恢复：发现 checkpoint 后任何 checksum、数据、模型、版本或配置不兼容都必须失败，不能静默从头开始。
- `required` 在本地和 AList 都没有 checkpoint 时必须失败。
- Kaggle 执行策略必须始终为 660/10，预检、父进程和 worker 不得看到不同值。
- 示例移动后不得通过兼容 Python package 或重导出保留旧 `experiments.*` 路径。
- `dl_helper.*` 实验引用以及包目录内配置/输出路径必须在导入或训练前失败。
- `--resume auto`、`checkpoint.resume` 和 `runtime` 必须被拒绝，不得接受后忽略。
- 文档代码必须显式处理退出码 0、75 和其他非零值。

## Test Strategy

- CLI parser：省略 `--resume` 得到内部 auto；显式 none/required 保持；显式 auto 解析失败。
- CLI integration：auto 无 checkpoint 新训、local latest 恢复、AList latest 恢复、损坏 checkpoint fail-fast。
- Config/platform：runtime 和 checkpoint.resume 被拒绝；Local policy 无预算；Kaggle policy 为 660/10；execution-policy Artifact 记录实际值。
- 示例路径：所有引用更新，示例 import/build/preflight 通过，旧顶层路径不存在，库模块引用和包内路径被拒绝。
- 文档/Notebook：JSON 可解析；扫描禁止 Kaggle 可执行块出现 `powershell`、反引号续行、`D:/programs/miniconda3` 或 `--resume auto`；subprocess 示例包含 0/75 处理。
