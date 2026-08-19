# sklearn

两个 fit mode：

- `batch`：`clone(safe=True)` 未拟合 estimator → 一次 `fit(X, y[, sample_weight])` → 评价 → best=last joblib。仅 `max_epochs=1`，且不支持恢复（显式 `--resume required` 在预检阶段失败，内部自动恢复不查询检查点）。
- `incremental`：`partial_fit` 逐 batch，classifier 首批传入完整 classes，batch 边界可信 joblib + source state 检查点，支持 Kaggle 预算暂停恢复。每个完整 epoch 结束后用完整 epoch 平均耗时预测下一轮；预测不足时在已推进的 epoch 边界保存 joblib/source checkpoint 并提交 AList，再返回 `75`。中途恢复的半个 epoch 不纳入平均值。

Pipeline 预处理只在 train fit（防止 val/test 泄漏）。训练数据带 sample_weight 时 `sample_weight_parameter` 必须显式给出参数路径（如 `sample_weight` 或 `step__sample_weight`）。

joblib 是代码执行边界：只加载当前 run 自产、manifest/checksum/runtime 精确匹配的可信模型；任何外部/其他 run/版本漂移在 `joblib.load` 前拒绝。
