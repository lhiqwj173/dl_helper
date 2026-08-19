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
- 恢复先校验 config/backend/data/model fingerprint 与 runtime 精确版本，再加载。
- 终态最后发布且互斥；resume 先移除旧 pause manifest。
