## ADDED Requirements

### Requirement: 训练过程进度可视化
训练报告 MUST 使用 `training.max_epochs` 作为固定 x 轴上限，MUST 绘制各阶段可用 loss；Torch optimizer 学习率以及 sklearn estimator 暴露的数值型 learning-rate 配置或训练历史 MUST 在主图叠加，完全不存在或只有不可观测调度时 MUST 明确标注 N/A。报告 MUST 按多分类、多标签和回归任务类型组织指标副图，且只渲染当前任务实际支持的指标组。checkpoint 进度报告 MUST 展示 checkpoint position、partial state 和截至当前位置的全部历史曲线。

#### Scenario: Torch 进度主图
- **WHEN** Torch run 生成 run report 或 checkpoint progress report
- **THEN** 主图 x 轴范围固定为 `1..max_epochs`，分别呈现 train/val loss 与学习率趋势

#### Scenario: sklearn 可用学习率
- **WHEN** sklearn fitted estimator 或其 Pipeline 步骤暴露数值型 learning-rate 参数或迭代历史
- **THEN** 报告在主图中可视化该学习率，并区分配置初值与实际历史

#### Scenario: sklearn 学习率缺失
- **WHEN** sklearn estimator 不暴露学习率参数或只使用不可观测的内部调度
- **THEN** 报告将学习率显示为 N/A，loss 仍使用同一固定 epoch 轴且不伪造数值

#### Scenario: 任务指标副图
- **WHEN** metrics history 中出现 multiclass、multilabel 或 regression 指标
- **THEN** 对应任务组的每个标量指标都有独立子图，并区分 train/val 曲线

#### Scenario: 中途 checkpoint 快照
- **WHEN** Torch 在 epoch 未完成时保存 checkpoint
- **THEN** 报告包含 current epoch、batch/global step、partial loss 和当前 learning rate
