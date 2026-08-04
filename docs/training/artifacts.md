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
- 恢复先校验 config/backend/data/model fingerprint 与 runtime 精确版本，再加载。
- 终态最后发布且互斥；resume 先移除旧 pause manifest。
