# 指标

内置指标使用固定大小流式状态（内存不随样本数增长），跨 rank 以 sum 或 weighted-moment merge 归约。每个指标携带 `MetricDefinition`（formula_id/version、direction、averaging、sample-weight、zero-division、exact、scope）。

- 多分类：float64 加权混淆矩阵 → accuracy、balanced_accuracy、macro/weighted P/R/F1、per-class 向量。
- 多标签：per-label TP/FP/FN/TN → micro/macro/weighted F1、subset_accuracy、hamming_loss。
- 回归：weighted Welford moments → per-target 与 uniform/variance-weighted MAE/MSE/R2；常量 target 按 `force_finite` 语义。

金标固定为 scikit-learn 1.6.1 显式参数。修改任何影响数值的语义必须提升 `formula_version` 并新增 golden fixture（`tests/fixtures/metric_goldens_v1.json`）。

selection / early-stop / sweep comparison 只接受 `exact=true`、`evaluation_scope=full` 的有限标量；报告显示的舍入值或抽样曲线不参与选择。
