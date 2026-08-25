# Proposal: add checkpoint progress reports

## Why

当前 checkpoint 只保存恢复状态，用户必须回到 run 根目录并手工查看日志或等待终态报告，才能了解训练进展。同时报告缺少按固定最大 epoch 轴绘制的 loss、学习率和任务指标趋势。

## What Changes

- 每个完整 Torch 与 sklearn incremental checkpoint 都内嵌只读进度 HTML 报告和进度快照数据。
- 进度报告复用现有 `metrics.jsonl` 历史；Torch 中途 checkpoint 额外记录当前部分 epoch 的 loss/学习率摘要。
- 主图使用 `1..training.max_epochs` 固定 x 轴，绘制 train/val loss 与学习率。sklearn estimator 暴露数值型 learning-rate 参数或训练历史时必须可视化；确实不存在或只有隐式调度时明确显示 N/A。
- 副图按多分类、多标签、回归三类任务指标分组；仅渲染当前 Task 支持的指标组。
- 报告文件进入 checkpoint SHA256 manifest，损坏时恢复前校验失败。
- 不改变 checkpoint 兼容指纹与状态恢复语义；旧 checkpoint 仍可加载，但不含新增内嵌报告。

## Impact

- **Affected specs:** `general-training`, `training-reporting`
- **Affected code:** `dl_helper/training/reporting.py`, `dl_helper/training/checkpoint.py`, Torch/sklearn backend checkpoint helpers, integration tests
