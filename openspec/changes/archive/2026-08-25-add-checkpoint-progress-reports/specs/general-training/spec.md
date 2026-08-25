## MODIFIED Requirements

### Requirement: backend-aware 可审计恢复
系统 MUST 使用不可变 checkpoint、完整 manifest、SHA256、latest-last 和严格兼容指纹。每个 complete checkpoint MUST 包含当前进度的离线 HTML report 与机器可读 progress snapshot，且两者 MUST 进入 checksum manifest。Torch MUST 恢复模型/优化器/scaler/scheduler/RNG/DataModule/EngineState；sklearn incremental MUST 恢复可信 estimator joblib、batch source、RNG、EngineState 和部分指标；sklearn batch MUST 明确拒绝恢复。

#### Scenario: Torch 或 incremental 兼容恢复
- **WHEN** latest 完整且 config/backend/data/model/runtime version 全部兼容
- **THEN** 系统从下一未完成位置继续，不重复 optimizer/partial_fit step 或指标累计

#### Scenario: checkpoint 含当前进度报告
- **WHEN** Torch 或 sklearn incremental 成功提交任一 complete checkpoint
- **THEN** checkpoint 目录包含通过 checksum 校验的 `progress/index.html` 和 `progress/progress-snapshot.json`

#### Scenario: 进度报告损坏
- **WHEN** 内嵌 report 文件缺失或字节与 manifest 不一致
- **THEN** 系统在恢复前抛出 checkpoint 校验错误，不尝试反序列化或部分恢复

#### Scenario: 不可信 joblib
- **WHEN** joblib 来自其他 run/用户路径、是 symlink、缺 manifest、checksum 不符或 sklearn/numpy/scipy/joblib/Python 版本非精确匹配
- **THEN** 系统在调用 joblib.load 前拒绝恢复

#### Scenario: 损坏或配置漂移
- **WHEN** 文件缺失、latest/manifest 不完整或非允许配置、数据、Task、指标定义、模型字段变化
- **THEN** 系统列出不含 Secret 的差异并失败，不尝试旧 checkpoint
