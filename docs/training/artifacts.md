# Artifacts

```
runs/<run-id>/run-manifest.json | pause-manifest.json | failure.json  # 三者互斥
runs/<run-id>/config.resolved.yaml, environment.json, evaluation-contract.json
runs/<run-id>/metrics/metrics.jsonl, metrics/summary.json
runs/<run-id>/checkpoints/latest.json, checkpoints/<epoch-step>/
runs/<run-id>/models/best|last/{model-manifest.json, model.safetensors|model.joblib}
runs/<run-id>/predictions/<split>/{prediction-manifest.json, part-rank*.npz}
runs/<run-id>/report/index.html
```

- 所有文本 UTF-8；原子写入同目录 tmp+flush+fsync+replace。
- 检查点不可变：staging → manifest（全部文件 SHA256）→ 原子 rename → latest 最后发布。
- 预算预测暂停发生在完整 epoch 边界：先确保该边界存在最新 checkpoint，再提交配置的 AList 服务；终态服务刷新完成后才写入 `pause-manifest.json`。硬截止触发时同样保存可恢复的 batch/optimizer 位置。
- 恢复校验归档完整性、RUN_ID、backend、模型参数结构及状态格式；不比较当前配置、数据指纹或 source_revision。运行时版本漂移会提示，实际状态加载失败直接终止。
- Torch 从早停终点检查点恢复时，保留已完成的 train/val 指标和 selection 状态，直接进入 test、模型导出和报告收尾；不再执行本轮验证或提交同名检查点。已有检查点无需迁移。
- 终态最后发布且互斥；resume 先移除旧 pause manifest。

## 本地成果包

成功运行会生成 `runs/<run-id>.run-bundle.zip` 和同名 `.sha256`，
与 AList 是否启用无关；本地打包失败会直接抛出异常。
成果包不含 `checkpoints/`、`checkpoint-archives/` 和远端发布标记。
恢复训练继续使用 `checkpoint-archives/last-checkpoint.zip`，成果包不是续训包。

项目如需在训练后导出 TorchScript、业务评价或其他文件，应先将它们写入 run 目录，
完成项目自己的完整性校验，再重新调用公共接口：

```python
from dl_helper.training.bundles import export_run_bundle

output = export_run_bundle(run_dir, "/kaggle/working/run-bundle.zip")
```

也可调用 `dl_helper.training.cli bundle --run RUN_DIR --out OUTPUT.zip`。
输出必须位于 run 目录之外，防止将旧成果包再次打入新包。
该接口只接受成功终态，核验原清单已登记的文件，并自动纳入后处理新增的普通文件。
ZIP 保留原始 `run-manifest.json`，独立的 `bundle-manifest.json` 登记实际包内文件、
大小、SHA-256 及排除规则；原清单中的检查点引用不表示这些文件属于成果包。
文件顺序、时间戳及权限固定，逐文件流式写入并从 ZIP 回读验证后原子替换目标。
本地与 AList 发布共用同一归档实现，AList 只增加上传、回读和远端校验。
