# 配置

唯一输入格式为 UTF-8 YAML（schema_version=1）。解析使用严格 loader：拒绝重复 key、YAML merge/alias、模板、环境插值、include、URL 与未知字段。

## base 示例

```yaml
schema_version: 1
run: {name: mnist, id: null, output_root: null, source_revision: null, seed: 42, tags: {}}
experiment: {}
training: {max_epochs: 20, log_every_steps: 20}
backend:
  type: torch
  torch: {gradient_accumulation_steps: 1, mixed_precision: "auto", compile: false, clip_grad_norm: 1.0, deterministic: strict, matmul_precision: high, find_unused_parameters: false}
  sklearn: null
distributed: {num_processes: auto}
selection: {metric: val/loss, mode: min, patience: 5, min_delta: 0.0}
checkpoint: {every_epochs: 1, every_optimizer_steps: null, keep_last: 2, resume: none}
runtime: {max_minutes: null, shutdown_grace_minutes: 10}
report: {enabled: true, curve_sample_limit: 100000, prediction_sample_limit: 10000, prediction_splits: [val]}
remote: {type: none}
notifications: {type: none}
```

## 关键约束

- `backend.type` 为 `torch/sklearn`；未选分支必须为 `null`。
- `mixed_precision` 取值 `auto/no/fp16/bf16`（`no`/`off` 等 YAML 歧义词需加引号）。
- sklearn `fit_mode=batch` 要求 `max_epochs=1`、`resume=none`、`max_minutes=null`。
- `selection` 有验证集时必须存在，无验证集时必须为 `null`；`mode` 必须等于 `MetricDefinition.direction`。
- `runtime.max_minutes` 要求 DataModule 支持中途恢复；sklearn batch 禁止。
- Kaggle：`run.id` 必填、`source_revision` 为 40 位 SHA、输出位于 `/kaggle/working`。

## variant

variant 是不含 `schema_version` 的严格 YAML，mapping 递归合并、scalar/list/null 整体替换；禁止覆盖 run.id/seed、backend type、output root、source revision、Secret key、host 或 distributed process count。
