## ADDED Requirements

### Requirement: 版本化科学指标定义
每个 Task 输出的指标 MUST 携带名称、direction、formula_id/version、averaging、sample-weight policy、zero-division、exact、full/sampled scope、参数和实现来源。selection、early stopping 和 sweep comparison MUST 只接受有限的 `exact=true`、`evaluation_scope=full` 指标；系统 MUST NOT 用报告舍入值或抽样曲线替代原始全量值。

#### Scenario: 指标定义完整
- **WHEN** 任一 stage 完成评价
- **THEN** summary 和报告逐项保存完整 MetricDefinition、原始 sample_count、weight_sum 和未舍入 float 值

#### Scenario: 同名定义漂移
- **WHEN** rank、split、resume 或 sweep trial 的同名指标在公式版本、average、zero-division、阈值或 sample-weight policy 上不同
- **THEN** 系统在比较或归约前失败

#### Scenario: 公式升级
- **WHEN** 实现改变会影响指标数值的任何语义
- **THEN** 实现必须增加 formula_version，并按旧 Artifact 记录的版本解释旧 run，不静默覆盖

### Requirement: 全量流式与分布式指标
系统 MUST 使用与样本数无关的固定大小状态计算 loss、多分类、多标签和回归摘要。MetricState MUST 支持 reset、状态保存恢复、固定 shape 的 sum/min/max 或 weighted-moment merge 归约和 compute；内置摘要 MUST NOT gather 全量预测。

#### Scenario: 多分类显式金标
- **WHEN** 样例包含缺失真实类别、从未预测类别、极端不平衡和声明顺序
- **THEN** accuracy、balanced accuracy、macro precision/recall/F1、weighted F1 和 per-class 指标与 sklearn 1.6.1 显式 labels/average/zero_division 金标绝对误差不超过 `1e-6`

#### Scenario: 多标签显式金标
- **WHEN** 样例包含全负 label、不同阈值和完全匹配/部分匹配样本
- **THEN** per-label、micro/macro/weighted 指标、subset accuracy 和 hamming loss 与 sklearn 金标在 `1e-6` 内一致

#### Scenario: 稳定回归金标
- **WHEN** target 包含常量、近常量、多输出不同方差或大偏置小波动数据
- **THEN** MAE/MSE、raw/uniform/variance-weighted R2 符合 sklearn force_finite 语义；大偏置 R2 与直接 float64 两遍计算误差不超过 `1e-10`

#### Scenario: 随机分块与两进程
- **WHEN** 同一加权数据采用不同 batch 分块或两个 rank 分片
- **THEN** 归约结果与单进程全量金标在 `1e-6` 内一致，且状态大小不随 N 增长

#### Scenario: 状态错误或空 split
- **WHEN** reduction key/shape/dtype/op 不一致、weighted moment 非法、状态含样本维数组，或声明 split 无样本
- **THEN** 系统失败，不输出零指标或空成功报告

### Requirement: 统一 sample-weight 语义
内置 Torch/sklearn 指标 MUST 将缺省权重视为 1，并对一维有限非负 sample weight 按样本维应用一次。状态 MUST 分开记录原始样本数和权重和；权重 shape 错误、负值、非有限或批权重和非正 MUST 立即失败。

#### Scenario: 加权多分类
- **WHEN** 样本使用非整数权重
- **THEN** 系统以 float64 加权混淆矩阵计算指标，并与 sklearn sample_weight 金标一致

#### Scenario: 加权多标签
- **WHEN** 每个样本有一个权重和多个 label
- **THEN** 权重沿 label 维只广播一次，micro/macro/subset/hamming 指标与金标一致

#### Scenario: 加权多输出回归
- **WHEN** 每个样本有一个权重和多个连续 target
- **THEN** 权重沿 target 维应用，MAE/MSE/R2 与 sklearn sample_weight/multioutput 金标一致

#### Scenario: 样本数与权重和展示
- **WHEN** weight_sum 与 sample_count 不相等
- **THEN** summary/report 同时展示两者，不用其中一个冒充另一个

### Requirement: 版本化 backend-aware 训练 Artifact
每个 run MUST 生成固定目录 schema、UTF-8 配置/日志/JSON/JSONL、evaluation contract、服务审计、环境、指标、backend 对应模型和唯一终态。SUCCEEDED 或 PREEMPTED manifest MUST 最后原子写入并列出 Artifact checksum；run/pause/failure MUST 三者互斥。

#### Scenario: Torch 成功产物
- **WHEN** Torch run 完成
- **THEN** manifest 引用 best/last safetensors、指标定义、配置/数据/模型指纹、服务状态、报告和有效 checksum

#### Scenario: sklearn 成功产物
- **WHEN** sklearn run 完成
- **THEN** manifest 引用带可信来源与精确 runtime version 的 best/last joblib model manifest，Torch 不适用字段标为 N/A

#### Scenario: 失败或可恢复暂停
- **WHEN** 运行失败，或预算暂停且 checkpoint/服务/report 完成
- **THEN** 分别只产生脱敏 failure.json，或只产生含恢复 checkpoint 的 pause-manifest.json

#### Scenario: UTF-8 文本
- **WHEN** 配置、标签、服务审计或错误含中文
- **THEN** 所有文本 Artifact 以显式 UTF-8 无损读取

### Requirement: 通用无 pickle 预测分片
系统 MUST 允许 Task 将 val/test/predict 结果保存为无 pickle NPZ 分片和 manifest，并校验字段名、dtype、第一维和 checksum。系统 MUST NOT 将领域字段或 joblib/pickle 混入预测分片。

#### Scenario: 保存预测
- **WHEN** report.prediction_splits 包含当前 split
- **THEN** 每个 rank 写唯一数值/bool/固定宽度 Unicode 分片，manifest 列出 schema、样本数和 checksum

#### Scenario: 不安全数组
- **WHEN** Task 返回 object dtype、非法字段名、shape 不符或路径逃逸
- **THEN** 系统拒绝写入并失败

#### Scenario: 自定义领域数组
- **WHEN** 自定义 Task 返回符合通用合同的任意字段
- **THEN** 系统按原字段保存且不解释业务 metadata

### Requirement: 可审计抽样可视化
系统 MUST 使用确定性 hash 优先级抽样限制预测与 ROC/PR/散点数据，报告 MUST 显示 sampled/total，并将抽样图与全量摘要清楚区分。

#### Scenario: 重复生成
- **WHEN** 对同一 Artifact 重复 report
- **THEN** 选择相同样本并产生相同图表数据

#### Scenario: 多 rank 候选
- **WHEN** 样本分布于多个 rank
- **THEN** 主进程按稳定 sample ID 合并候选后保留全局最小 hash；没有稳定 ID 时记录位置抽样及其可复现范围

#### Scenario: 抽样指标不可用于选择
- **WHEN** ROC/PR/AUC 仅由抽样数据计算
- **THEN** 页面标注 sampled/total，且配置验证拒绝将其作为 selection/sweep metric

### Requirement: 直观且忠实的离线 HTML 报告
系统 MUST 只读取已落盘 Artifact 生成可离线打开、HTML escape、幂等的通用和任务报告；MUST 展示指标定义与适用边界，不导入用户模型或数据代码。

#### Scenario: 分类和多标签报告
- **WHEN** classification run 成功或暂停
- **THEN** 报告包含全量 summary、加权/原始混淆统计、per-class/per-label 表和明确抽样的 ROC/PR

#### Scenario: 回归报告
- **WHEN** regression run 成功或暂停
- **THEN** 报告包含每目标和聚合 MAE/MSE/R2、抽样 predicted-vs-actual/残差图及常量目标规则

#### Scenario: sklearn 报告
- **WHEN** backend 为 sklearn
- **THEN** 报告展示 estimator/Pipeline 参数与 CPU/thread 上下文，并将 optimizer、学习率、CUDA/AMP 等不存在项标为 N/A 而非伪造曲线

#### Scenario: 大类别和 HTML 注入
- **WHEN** 类别超过 20 或用户文本含 HTML/script
- **THEN** 图表按固定 support/F1 规则裁剪但完整表保留，所有用户内容被转义

### Requirement: 报告展示资源与复现上下文
系统 MUST 展示 source revision、base/variant/config/tuning checksum、DataIdentity、split fingerprint、model signature/runtime version、seed、确定性模式、资源、批量、耗时、吞吐、服务状态和报告生成版本。

#### Scenario: Kaggle 多 GPU
- **WHEN** 两 GPU Torch run 完成
- **THEN** 报告列出两设备、每设备批量、累积、正确全局有效批量、吞吐与 peak memory

#### Scenario: CPU 或 sklearn
- **WHEN** CPU Torch 或 sklearn run 完成
- **THEN** 不适用的 CUDA/optimizer 字段显示 N/A，逻辑 CPU、n_jobs 与其他复现字段仍完整
