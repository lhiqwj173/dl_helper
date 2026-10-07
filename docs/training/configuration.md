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
checkpoint: {every_epochs: 1, every_optimizer_steps: null, keep_last: 2}
report: {enabled: true, curve_sample_limit: 100000, prediction_sample_limit: 10000, prediction_splits: [val]}
remote: {type: none}
notifications: {type: none}
```

> 配置**不含** `runtime` 与 `checkpoint.resume`（D-002/D-003）。恢复策略由 CLI 内部自动决定或显式
> `--resume none|required` 覆盖；运行预算由平台执行策略提供（Kaggle 恒为 660 分钟训练 + 10 分钟收尾，
> 记录在 run 目录 `execution-policy.json`，Local 不启用预算）。

## 关键约束

`train --use-alist` 默认开启，使用 YAML 的 `remote` 配置。`train --no-use-alist` 仅覆盖 `remote` 为 `none`，不读取 AList Secret、不下载远程检查点、不创建 AList 客户端或上传任务。本地恢复按 `--resume` 独立执行：省略时自动查找本地检查点，`none` 从头训练，`required` 要求有效本地检查点。原后端、单机多卡资源配置、平台预算及本地产物保留。企业微信独立按 `notifications` 配置执行，启用时仍需提供有效凭证。YAML 本身仍须符合 schema v1。

- `backend.type` 为 `torch/sklearn`；未选分支必须为 `null`。
- `mixed_precision` 取值 `auto/no/fp16/bf16`（`no`/`off` 等 YAML 歧义词需加引号）。
- sklearn `fit_mode=batch` 要求 `max_epochs=1`，且不支持恢复（显式 `--resume required` 在预检阶段失败）。
- `selection` 有验证集时必须存在，无验证集时必须为 `null`；`mode` 必须等于 `MetricDefinition.direction`。
- Kaggle：输出必须位于 `/kaggle/working`；`source_revision` 只要是无空白版本标识即可（tag、分支、短 SHA 均可）。
- Kaggle 默认要求启用 AList 且其 `failure_policy=required`；`--no-use-alist` 关闭 AList。企业微信独立按配置启用或关闭，使用配置中的 `required`/`record` 策略。平台执行策略固定 660 分钟训练 + 10 分钟收尾（截止 650 分钟）：系统在每个成功 batch/optimizer step 后做硬截止保护，并在每个完整 epoch 后按平均 epoch 耗时预测下一轮，预测不足时在边界保存 checkpoint 后以 `75` 暂停，AList 开启时上传。要求开启预算时 DataModule 支持中途恢复。`runtime`/`checkpoint.resume` 写入即按未知字段失败。

## 单机检查点路径

单进程 Torch 每次检查点保存还会同步生成 ZIP（默认 `<run_dir>/checkpoint-archives`）。
新增 CLI 参数 `--checkpoint-input PATH`（可重复）支持只读 Dataset、ZIP、展开恢复包及旧 run；
`--checkpoint-export-dir PATH` 指定 output-root 下下载目录。两者仅属于执行策略，
不增加 YAML 字段，不改变恢复指纹。省略 `--resume` 时比较本地与 Dataset 进度，
均无检查点再查询原 AList；AList 上传和远程恢复协议保持原样。
ZIP 与原始目录遵循 `checkpoint.keep_last`，null 保留全部，有限值保留最近对应数量。

## variant

variant 是不含 `schema_version` 的严格 YAML，mapping 递归合并、scalar/list/null 整体替换；禁止覆盖 run.id/seed、backend type、output root、source revision、Secret key、host 或 distributed process count。
