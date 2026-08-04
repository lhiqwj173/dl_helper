# sklearn

两个 fit mode：

- `batch`：`clone(safe=True)` 未拟合 estimator → 一次 `fit(X, y[, sample_weight])` → 评价 → best=last joblib。仅 Local、`max_epochs=1`、`resume=none`、`max_minutes=null`。
- `incremental`：`partial_fit` 逐 batch，classifier 首批传入完整 classes，batch 边界可信 joblib + source state 检查点，支持 Kaggle 预算暂停恢复。

Pipeline 预处理只在 train fit（防止 val/test 泄漏）。训练数据带 sample_weight 时 `sample_weight_parameter` 必须显式给出参数路径（如 `sample_weight` 或 `step__sample_weight`）。

joblib 是代码执行边界：只加载当前 run 自产、manifest/checksum/runtime 精确匹配的可信模型；任何外部/其他 run/版本漂移在 `joblib.load` 前拒绝。
