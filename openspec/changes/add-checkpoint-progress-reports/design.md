# Design: add checkpoint progress reports

## Approach

在 checkpoint staging 完成状态序列化后、manifest 计算前生成 `progress/index.html` 与 `progress/progress-snapshot.json`。这样报告天然进入不可变目录和完整 manifest 校验。报告生成器继续只读取已落盘 Artifact：run 级历史来自 `metrics/metrics.jsonl`，checkpoint 当前值由调用方传入显式 snapshot，不从内存对象隐式导出。

Torch 主图的数据源是已完成 epoch 的 train/val 记录和每次 optimizer step 后的标量学习率；中途 checkpoint 将当前 partial train loss 与当前 learning rate 写入 snapshot 并以最后一个 epoch 点展示。sklearn 会从 fitted estimator（含 Pipeline 嵌套步骤）探测 `learning_rate`、`learning_rate_init` 和 `eta0` 等数值参数；若 estimator 另外暴露迭代学习率历史则优先使用历史。存在可用值时与 loss 同图可视化，否则只绘制阶段 loss 并把学习率标注为 N/A，不用静态配置值伪造实际优化轨迹。

副图根据 metric key 和 extended 结构识别任务组：`accuracy/balanced_accuracy` 属于 multiclass；`subset_accuracy/per_label` 属于 multilabel；`mae/mse/r2/per_target` 属于 regression。每组一个子图，同一 metric 的 train/val 曲线共用坐标轴，x 轴统一为 `1..max_epochs`。

## Risks

- 分布式保存必须只在 main process 写报告，并在 manifest 提交前后保留现有 barrier。
- Matplotlib 渲染失败不得静默产出空报告；应抛出异常并中止 checkpoint，符合 fail-fast 合同。
- 新增 PNG 会增加 checkpoint 体积；报告使用固定尺寸、关闭网格细节并将图片限制为必要趋势图。
