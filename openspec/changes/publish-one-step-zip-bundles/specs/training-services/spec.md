## MODIFIED Requirements

### Requirement: 通用 ArtifactStore 生命周期
系统 MUST 将 LocalArtifactStore 始终用于本地产物，并可配置 AListArtifactStore 统一发布 run、checkpoint 和 sweep；Experiment/Task MUST NOT 接收服务客户端或自行上传。AList MUST 使用显式 HTTP(S) host/base path（允许 `http://IP`）、Kaggle Secret或同名环境变量、有限 timeout/retry、回读 checksum 与 terminal-last 发布。新发布的 run 和 sweep 成果 MUST 分别为可直接解压的 `run-bundle.zip` 与 `sweep-bundle.zip`；checkpoint MUST 保持可恢复的 `archive.tar.gz`。

#### Scenario: 发布可恢复 checkpoint
- **WHEN** checkpoint archive 上传且远程 size 可见
- **THEN** 系统仍须 raw 回读验证 SHA256，再发布并回读 manifest，最后更新 latest；读取者忽略任何不完整对象

#### Scenario: 发布完整 run 或 sweep
- **WHEN** run/sweep 核心产物完成
- **THEN** 系统发布排除 staging/lock 和已独立 checkpoint 的不可变 ZIP bundle，回读校验后最后发布对应 terminal manifest

#### Scenario: 历史成果与检查点兼容
- **WHEN** 新版本发布成果或恢复旧运行的 checkpoint
- **THEN** 新成果使用 ZIP，旧远端 `.tar.gz` 保持不变，checkpoint 继续按原有 TAR/GZIP 路径恢复

#### Scenario: 路径与 archive 安全
- **WHEN** ID/path 含非法 segment，或 archive 成员是绝对路径、`..`、symlink 或逃逸根目录
- **THEN** 系统在上传/解压前失败

#### Scenario: 认证和临时错误
- **WHEN** AList 返回 401/403/认证业务码，或网络 timeout/5xx
- **THEN** 前者不重试；后者只按配置的 2/4/8 秒有限重试，耗尽后保留原异常链
