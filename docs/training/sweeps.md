# Sweep

sweep manifest 固定 Experiment 引用、完整 base、有序 variant、`val/` 前缀 comparison metric 与 direction。

```yaml
schema_version: 1
sweep:
  id: toy-lr-v1
  experiment: experiments.toy_multiclass:build_experiment
  base_config: ./base.yaml
  comparison_metric: val/loss
  mode: min
  trials:
    - {name: lr-1e-2, variant: ./variants/lr-1e-2.yaml}
    - {name: lr-3e-3, variant: ./variants/lr-3e-3.yaml}
```

- 路径相对 manifest 且位于其目录树内；trial name 唯一且匹配 run ID 字符集。
- run ID 派生为 `<sweep-id>--<trial-name>`；tuning fingerprint 必须唯一。
- 所有 trial 在零拟合 step 完成独立自动预检并比较 evaluation contract（Experiment/backend/DataIdentity/Task/MetricDefinition 一致）。
- 顺序子进程运行；任一失败立即停止且不产生 best；75 写 pause manifest。
- 排名读取各 run summary 的未舍入 comparison 值；并列按 YAML 顺序稳定。
