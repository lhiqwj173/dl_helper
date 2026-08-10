# experiment-sweeps Specification

## Purpose
TBD - created by archiving change build-general-kaggle-training-platform. Update Purpose after archive.
## Requirements
### Requirement: 严格 sweep manifest 与 trial 派生
系统 MUST 接受一个 UTF-8 sweep manifest，固定 sweep ID、Experiment 引用、完整 base config、本地有序 variant 列表、validation comparison metric 和 direction。路径 MUST 相对 manifest、解析后位于其目录树内；至少两个唯一 trial，run ID MUST 固定为 `<sweep-id>--<trial-name>`。

#### Scenario: 合法多 variant sweep
- **WHEN** manifest 列出两个以上合法 variant
- **THEN** 系统按 YAML 顺序生成唯一 resolved config、tuning fingerprint、evaluation contract 和派生 run ID

#### Scenario: 路径或名称错误
- **WHEN** base/variant 是 URL、绝对路径、symlink 逃逸，或 trial/sweep 名称非法/重复
- **THEN** coordinator 在任何 trial 拟合前失败

#### Scenario: 相同训练伪装不同 trial
- **WHEN** 两个 trial 只有 tag、报告或服务参数不同而 tuning fingerprint 相同
- **THEN** sweep 拒绝运行，不生成重复排名项

### Requirement: 零优化步可比性预检
coordinator MUST 在训练前通过独立 doctor 子进程构造每个 trial 的 evaluation contract，并要求 Experiment 引用、backend、DataIdentity、split fingerprint、Task、标签/目标 schema、MetricDefinition、comparison metric/direction 相同。comparison MUST 是 `val/` 前缀的全量精确指标；test 或 sampled 指标 MUST NOT 排名。

#### Scenario: 可比较 trial
- **WHEN** 只有模型 signature 或允许调参值不同，评价合同其余字段相同
- **THEN** 预检通过并可进入第一个 trial

#### Scenario: 数据或指标漂移
- **WHEN** variant 改变数据 split、标签顺序、Task、阈值、sample-weight、formula version 或 comparison direction
- **THEN** 整个 sweep 在零 fit/partial_fit/optimizer step 时失败并列出差异

#### Scenario: test leakage 防护
- **WHEN** manifest 将 test 指标指定为 comparison
- **THEN** 配置验证失败；test 结果只能展示且标记不参与选择

### Requirement: 隔离顺序 trial 编排
sweep coordinator MUST 以 `sys.executable -m dl_helper.training.cli train` 在全新子进程按 manifest 顺序运行 trial，MUST 持有原子独占 sweep lock，且自身 MUST NOT 导入实验、torch/sklearn 或初始化 CUDA。系统 MUST NOT 并发争抢同一 Kaggle GPU，也不自动生成网格/随机/贝叶斯参数。

#### Scenario: 顺序成功
- **WHEN** 前一 trial 返回 0 且 manifest/checksum/fingerprint 有效
- **THEN** coordinator 记录 trials.jsonl 后才启动下一 trial

#### Scenario: trial 失败
- **WHEN** 任一 trial 返回非零且非 75
- **THEN** coordinator 立即停止、写 sweep failure，后续 trial 不启动且不生成 best/ranking

#### Scenario: 并发相同 sweep
- **WHEN** 第二 coordinator 尝试取得同一 sweep ID lock
- **THEN** 第二进程非零失败，不等待后写或共享 run 目录

### Requirement: sweep 暂停与严格恢复
trial 返回 75 时 coordinator MUST 发布完整 sweep pause manifest 并返回 75。`sweep --resume` MUST 只接受 PREEMPTED sweep，复核 sweep/base/variant/contract/已完成 run 的 checksum 与 fingerprint，跳过完整成功 trial并恢复原 paused trial。FAILED sweep MUST NOT 使用原 ID 续跑。

#### Scenario: 当前 trial 暂停
- **WHEN** 第 k 个 trial 安全 PREEMPTED
- **THEN** pause manifest 记录已完成 trial、当前 run/checkpoint、剩余顺序和全部 checksum，不启动第 k+1 个 trial

#### Scenario: 合法恢复
- **WHEN** 下一 Kaggle 会话以未漂移 manifest 执行 sweep resume
- **THEN** 前 k-1 个 trial 仅校验不重跑，第 k 个 trial 从自己的 checkpoint 恢复

#### Scenario: 恢复输入漂移
- **WHEN** manifest/base/variant/contract/成功 run 任一内容变化或缺失
- **THEN** 恢复失败，不重算 fingerprint 后继续

#### Scenario: FAILED sweep 续跑
- **WHEN** 用户对 FAILED sweep 使用 resume
- **THEN** 系统拒绝，并要求修正后使用新 sweep ID 保留失败审计

### Requirement: 科学排名与 best trial
只有全部 trial SUCCEEDED 后系统 MUST 从各 run `metrics/summary.json` 读取未舍入的全量 validation comparison value，按固定 direction 排名，NaN/Inf/缺失/定义漂移 MUST 失败；相同值 MUST 按 YAML trial 顺序稳定排序。best-trial、聚合报告、required 服务和成功 manifest MUST 按 terminal-last 顺序完成。

#### Scenario: 全部成功排名
- **WHEN** 所有 comparison value 合法
- **THEN** 系统写有序排名和 best-trial checksum，生成聚合报告并最后发布 sweep success manifest

#### Scenario: 显示舍入并列但原值不同
- **WHEN** 两个 UI 显示值相同而原始 float 不同
- **THEN** 排名使用原始值；仅原始值相等才按 trial 顺序打破并列

#### Scenario: 指标无效
- **WHEN** 任一 summary 缺指标、值非有限、definition/direction 不同或只存在 sampled 值
- **THEN** sweep FAILED 且不存在 best-trial/success manifest

### Requirement: 离线 sweep 报告
系统 MUST 只读取 sweep 与 run Artifact 生成离线、HTML escape、幂等的聚合报告，展示 trial 状态、resolved 调参差异、tuning fingerprint、资源/耗时、validation comparison、排名和单 run 报告链接。失败/暂停报告 MUST 只展示进度，不伪造 best。

#### Scenario: 成功报告
- **WHEN** sweep SUCCEEDED
- **THEN** 报告显示稳定排名、best、未舍入来源、格式化显示值、关键参数差异和所有 trial 链接

#### Scenario: 暂停或失败报告
- **WHEN** sweep PREEMPTED 或 FAILED
- **THEN** 报告显示已完成/当前/未开始及错误/恢复位置，不显示完整排名或 best

#### Scenario: test 指标展示
- **WHEN** trial run 包含 test 指标
- **THEN** 报告可展示但明确标注“不参与调参选择”
