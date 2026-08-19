# dl-helper

`dl-helper` 是面向 Kaggle 的训练工具库。训练项目不在本库包内：你的模型、数据读取和任务定义放在自己的项目目录，通过 `--project-dir` 加入运行时导入路径；配置和产物目录也由命令或 YAML 显式指定。仓库自带的实验与配置只作为 `examples/` 下的可运行示例和测试夹具，不是库模块。

## 最小运行

```bash
python -m dl_helper.training.cli train \
  --project-dir D:/work/my-project \
  --config D:/work/my-project/configs/train.yaml \
  --output-root D:/work/my-project/runs \
  --experiment my_experiment:build_experiment
```

`train` 启动前会自动执行配置、数据路径、ExecutionPolicy、后端、磁盘、版本和服务预检；预检失败会列出全部缺失/错误字段并立即终止。库不再暴露 `doctor` 命令。

退出码：`0` 成功，`75` 预算保护暂停且已保存检查点（可继续训），其他非零表示失败。省略 `--resume` 时按内部自动恢复策略：本地 latest 优先，无则查 AList，两处都无则从头开始；`--resume none` 禁止恢复，`--resume required` 无兼容检查点即失败。

## Kaggle

Kaggle 训练必须在配置中启用 AList 和企业微信，且两者 `failure_policy` 都必须是 `required`。AList 用户名/密码和企业微信凭证从 Kaggle Secrets（本地调试时也可用同名环境变量）读取；任何 Secret 缺失或配置字段错误都会在训练前终止。Kaggle 的运行预算由库的平台执行策略自动固定为 **660 分钟训练 + 10 分钟收尾窗口**（run 目录下 `execution-policy.json` 可审计），用户配置不再包含 `runtime`。训练会在成功 batch/optimizer step 后做硬截止保护，并在每个完整 epoch 结束时按完整 epoch 平均耗时预测下一轮；预测无法在截止前完成时，于当前 epoch 边界保存并推送 checkpoint，再以 `75` 暂停。

```bash
python -m dl_helper.training.cli train \
  --project-dir /kaggle/working/my-project \
  --config /kaggle/working/my-project/configs/kaggle.yaml \
  --experiment my_experiment:build_experiment \
  --run-id my-run-001
```

第一个 Session 因预算返回 `75` 后，在新的 Session 使用同一 `--run-id` 再次运行同一条命令即可自动恢复（无需再填任何恢复参数）。完整的 Kaggle 安装、Secrets、恢复和 sweep 流程见 [Kaggle 指南](docs/training/kaggle.md)。

## 文档

- [训练指南](docs/training/guide.md)
- [配置](docs/training/configuration.md)
- [Kaggle 训练、恢复与调参](docs/training/kaggle.md)
- [sweep](docs/training/sweeps.md)
- [服务](docs/training/services.md)
- [产物与检查点](docs/training/artifacts.md)