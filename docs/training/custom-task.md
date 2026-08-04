# 自定义任务

Task 通过统一合同接入：`PreparedBatch` → `forward` → `LossResult` → `to_predicted_batch` → `PredictedBatch` → `MetricState`。

```python
class CustomTask(RegressionTask):
    def forward(self, model, prepared):
        return torch.abs(model(prepared.inputs))  # 结构化输出

    def loss(self, outputs, prepared):
        return LossResult(numerator=(outputs - prepared.targets).pow(2).sum(),
                          denominator=float(prepared.sample_count))

    def to_predicted_batch(self, outputs, prepared):
        return PredictedBatch(targets=..., predictions=..., sample_count=...)
```

默认模型调用规则：Mapping→`model(**inputs)`、tuple→`model(*inputs)`、其他→单参数。list 作为单参数（需要位置展开必须转 tuple）。

engine 只消费 `PredictedBatch` 与 `MetricState`，不解释业务字段。prediction arrays 必须数值/bool/固定宽度 Unicode，禁止 object dtype。
