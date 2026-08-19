# training-services Specification

## Purpose
TBD - created by archiving change build-general-kaggle-training-platform. Update Purpose after archive.
## Requirements
### Requirement: 通用 ArtifactStore 生命周期
系统 MUST 将 LocalArtifactStore 始终用于本地产物，并可配置 AListArtifactStore 统一发布 run、checkpoint 和 sweep；Experiment/Task MUST NOT 接收服务客户端或自行上传。AList MUST 使用显式 HTTPS host/base path、Kaggle Secret或同名环境变量、有限 timeout/retry、回读 checksum 与 terminal-last 发布。

#### Scenario: 发布可恢复 checkpoint
- **WHEN** checkpoint archive 上传且远程 size 可见
- **THEN** 系统仍须 raw 回读验证 SHA256，再发布并回读 manifest，最后更新 latest；读取者忽略任何不完整对象

#### Scenario: 发布完整 run 或 sweep
- **WHEN** run/sweep 核心产物完成
- **THEN** 系统发布排除 staging/lock 和已独立 checkpoint 的不可变 bundle，回读校验后最后发布对应 terminal manifest

#### Scenario: 路径与 archive 安全
- **WHEN** ID/path 含非法 segment，或 archive 成员是绝对路径、`..`、symlink 或逃逸根目录
- **THEN** 系统在上传/解压前失败

#### Scenario: 认证和临时错误
- **WHEN** AList 返回 401/403/认证业务码，或网络 timeout/5xx
- **THEN** 前者不重试；后者只按配置的 2/4/8 秒有限重试，耗尽后保留原异常链

### Requirement: 有界异步 AList 同步
仅主进程 MUST 可创建一个非 daemon AList worker 和容量 1 的 pending checkpoint 队列；新 checkpoint MAY 替换尚未开始的 pending 项，但 MUST NOT 取消 active 项。terminal bundle MUST 等待所有 checkpoint，并在进程退出前 join。

#### Scenario: pending 合并
- **WHEN** active checkpoint 上传期间连续产生多个新 checkpoint
- **THEN** 仅最新未开始项保留在 pending，被替换 checkpoint 仍完整保留本地

#### Scenario: 异步异常
- **WHEN** worker 上传失败
- **THEN** 异常在下一安全边界或 final flush 按 failure policy 处理，不只写日志

#### Scenario: terminal flush
- **WHEN** run/sweep 准备 SUCCEEDED 或 PREEMPTED
- **THEN** worker 已完全 join、引用 checkpoint 已发布、terminal bundle 已校验后才能继续终态

### Requirement: 固定企业微信生命周期通知
系统 MUST 内置不依赖 py-ext 的企业微信客户端，host 固定为 `https://qyapi.weixin.qq.com`，只用官方 gettoken 与 application message API 发送 UTF-8 text。通知 MUST 覆盖 run、trial、sweep 的 STARTED/SUCCEEDED/PREEMPTED/FAILED 事件，并以稳定 event_id 去重。

#### Scenario: token 获取与缓存
- **WHEN** 首次投递或缓存 token 临近到期
- **THEN** 客户端校验 HTTP/JSON/errcode/token/expires_in，使用 monotonic 安全提前量缓存；token 不落盘且无全局单例

#### Scenario: 发送应用消息
- **WHEN** token 有效且固定事件模板字段合法
- **THEN** 客户端发送 touser、正整数 agentid、msgtype=text 和 text.content，只有 errcode=0 记为成功

#### Scenario: token 失效
- **WHEN** message API 返回明确 token 失效或过期码
- **THEN** 客户端只允许清缓存、刷新 token 并重放一次；其他业务错误不重试

#### Scenario: UTF-8 消息上限
- **WHEN** 固定模板超过 2048 UTF-8 bytes
- **THEN** 系统只按 code point 边界裁剪异常消息/路径并保留 event/status/scope ID/异常类型；关键字段仍放不下则失败

### Requirement: 服务失败策略与审计
AList 和企业微信 MUST 分别配置 `required` 或 `record`，每次调用 MUST 写结构化 UTF-8 service audit。Kaggle 运行 MUST 同时启用 AList 和企业微信且两者均为 `required`，并在训练前聚合报告所有缺失 Secret key；本地调试 MAY 关闭服务。required 失败 MUST 阻止成功/暂停终态；record 失败 MAY 继续但 MUST 在 audit 与 terminal manifest 标为 degraded。任何 secondary 服务异常 MUST NOT 覆盖原训练异常。

#### Scenario: required STARTED 失败
- **WHEN** required 企业微信或启动阶段 AList 预检失败
- **THEN** 系统在首个 optimizer/fit/partial_fit step 前终止

#### Scenario: record 终态失败
- **WHEN** 核心训练成功但 record 服务投递失败
- **THEN** run/sweep 可成功，manifest 明确记录 service=degraded、错误类型和对应 audit checksum

#### Scenario: 训练与通知同时失败
- **WHEN** backend 先抛出训练异常且 FAILED 通知也失败
- **THEN** failure.json 将训练错误作为 primary、服务错误作为 secondary，CLI 重新抛出原训练异常 traceback

#### Scenario: 审计脱敏
- **WHEN** 服务发生任意成功、重试或失败
- **THEN** audit 记录 event_id、scope、attempt、UTC、duration、outcome、脱敏状态码/error_type，不含 token URL、认证参数、Secret 或 response body

### Requirement: 可重入服务终结
系统 MUST 以 immutable bundle checksum、稳定 event_id 和 terminal-last 顺序支持 FINALIZING 重入；外部服务之间不声称原子事务。重入 MUST 复核已成功动作并只补做缺失步骤，不重新训练或重复成功消息。

#### Scenario: AList 已成功但通知中断
- **WHEN** finalization 在 bundle 发布后、终态通知前中断
- **THEN** 重入复用相同 bundle checksum，只补投未成功 event 并最后发布 terminal manifest

#### Scenario: 本地终态已存在
- **WHEN** 相同 run/sweep 的完整 terminal manifest 已存在且 checksum 匹配
- **THEN** 重复 finalization 幂等返回，不覆盖或重复服务动作

#### Scenario: 终结状态漂移
- **WHEN** audit、bundle、event 或 candidate terminal checksum 与已有记录不一致
- **THEN** 系统失败并保留诊断，不选择任一版本继续
