## ADDED Requirements

### Requirement: 严格 base 与 variant 配置
系统 MUST 使用 schema_version=1 的完整 UTF-8 YAML base，并可合并至多一个不含 schema_version 的严格 variant。mapping MUST 递归合并，scalar/list/null MUST 整体替换；合并后 MUST 重新执行完整 schema/跨字段校验。系统 MUST 拒绝未知/重复 key、错误类型、非有限值、字符串 bool、YAML merge/alias、模板、环境插值、include、URL、路径逃逸和 variant 禁止字段。

#### Scenario: 合法 base 与 variant
- **WHEN** variant 只覆盖允许的 run tag、experiment/training、选中 backend 参数、selection、checkpoint/runtime/report 或服务策略
- **THEN** 系统生成 frozen resolved config，并保存 base/variant SHA256、config/resume/tuning fingerprint

#### Scenario: merge 后跨字段冲突
- **WHEN** patch 单独合法但合并后造成 backend 分支、resume、预算、selection 或 sample-weight 冲突
- **THEN** 系统指出最终字段路径并失败，不绕过完整校验

#### Scenario: 禁止基础设施漂移
- **WHEN** variant 修改 schema、run.id/seed、backend type、output root、source revision、Secret key、host/base path 或 distributed process count
- **THEN** 系统拒绝 patch

#### Scenario: YAML 隐式能力
- **WHEN** 配置使用重复 key、merge key、alias、`compile: "false"`、模板或 include
- **THEN** 解析失败，不接受 PyYAML 默认覆盖或字符串 truthiness

### Requirement: Fail Fast 进程与子进程语义
系统 MUST 让未恢复异常产生非零退出码和完整 traceback，禁止通用捕获后继续、`os._exit(0)`、`pkill` 或安全默认值。仅完整 PREEMPTED 使用 75；sweep MUST 原样区分 0、75 和其他非零。

#### Scenario: backend 异常
- **WHEN** forward、fit、partial_fit、predict 或 metric 抛出异常
- **THEN** CLI 保存脱敏 failure Artifact、重新抛出原异常并非零返回

#### Scenario: bootstrap 或 trial 子进程失败
- **WHEN** Git、pip、report 或 trial 子进程非零
- **THEN** 上层立即停止，不执行后续训练或生成伪排名

#### Scenario: 预期暂停
- **WHEN** 完整 checkpoint、required service flush 和报告成功
- **THEN** CLI 原子发布 pause manifest 并返回 75，不与 FAILED/SUCCEEDED 混淆

### Requirement: 复现实验 manifest
系统 MUST 记录 source revision、base/variant/resolved config、DataIdentity/split fingerprint、Task/MetricDefinition、model signature、backend runtime versions、OS/设备、seed、确定性和线程配置。Kaggle MUST 要求固定 clean source revision。

#### Scenario: Git Kaggle 运行
- **WHEN** 工作树来自 Git checkout
- **THEN** Kaggle source revision 必须与 clean HEAD 的 40 位 SHA 完全一致

#### Scenario: 非 Git Local
- **WHEN** 无法获取 Git SHA
- **THEN** 用户必须显式提供 source revision，否则复现预检失败

#### Scenario: sklearn 随机性
- **WHEN** random_state=run_seed
- **THEN** 所有递归 null `*random_state` 参数设为 run.seed，已有整数保持；非整数值失败并在 manifest 列出 resolved params

#### Scenario: PyTorch strict 确定性
- **WHEN** deterministic=strict 遇到非确定性算子
- **THEN** PyTorch 异常向上传播，不切换 warn/off

### Requirement: 可信 joblib 反序列化
系统 MUST 将 joblib/pickle 视为代码执行边界，只允许加载当前 run 自身生成且经完整 manifest 验证的 sklearn model/checkpoint。校验 MUST 在 `joblib.load` 前完成，并要求 regular non-symlink path、根目录边界、origin run/backend/model signature、SHA256 和 Python/sklearn/numpy/scipy/joblib 精确版本一致。

#### Scenario: 当前 run 合法恢复
- **WHEN** 本地或 AList 下载的 joblib 满足全部前置校验
- **THEN** 系统才调用 joblib.load，并在加载后再次验证 fitted estimator kind/signature

#### Scenario: 外部或漂移 joblib
- **WHEN** 路径由用户提供、来自其他 run、是 symlink、checksum 错误、缺 manifest 或 runtime 仅版本范围兼容
- **THEN** 系统在反序列化前失败且不尝试探测文件内容

#### Scenario: AList staging
- **WHEN** 从远程恢复 sklearn checkpoint
- **THEN** 系统先校验 archive/manifest、解压到隔离 staging、拒绝危险成员并复核每个文件，之后才加载

### Requirement: Secret 与凭证安全
受跟踪源码、Notebook、配置、fixture、日志和 Artifact MUST NOT 包含真实 AList、企业微信或第三方凭证。Secret MUST 只在启用服务的预检阶段从 Kaggle Secrets 或同名环境变量解析并统一脱敏；历史暴露凭证 MUST 在发布前由仓库所有者确认轮换。

#### Scenario: 启用服务但 Secret 缺失
- **WHEN** AList 或企业微信配置存在但任一 Secret 不存在
- **THEN** 系统在首个拟合 step 前报告缺失的 key，不显示值或回退匿名/硬编码身份

#### Scenario: Secret 扫描
- **WHEN** CI 扫描受跟踪 Python/YAML/TOML/Markdown/Notebook
- **THEN** 敏感变量非空字面量、固定 set_token 和高熵长字符串导致失败，仅精确规则识别的 SHA/URL/路径/.invalid/${SECRET_KEY} 可豁免

#### Scenario: 下游正文泄露
- **WHEN** HTTP 或 estimator 异常文本含已解析 Secret
- **THEN** 日志、audit、failure 和通知内容写入前全部替换为 `[REDACTED]`

### Requirement: 依赖与导入边界
项目 MUST 仅使用 `pyproject.toml` 声明 Python、核心依赖和 dev extra；joblib/requests MUST 是直接核心依赖，MUST NOT 存在 alist 或 legacy extra。Kaggle MUST 核验而不是静默升级框架。导入 core MUST 不触发用户实验、TorchMetrics、transformers、RL、GUI、网络或 Secret。

#### Scenario: 安装默认项目
- **WHEN** 安装候选 wheel
- **THEN** 元数据包含 torch/accelerate/numpy/matplotlib/sklearn/safetensors/PyYAML/joblib/requests 的 design 版本范围，不含旧重依赖

#### Scenario: 请求已删除 extra
- **WHEN** 用户请求 alist/legacy extra 或审计 wheel metadata
- **THEN** extra 不存在，wheel 版本为 1.0.0，且不含 RL/AutoML/GUI/TorchMetrics/py-ext/C++ 依赖

#### Scenario: Kaggle 依赖不兼容
- **WHEN** doctor 发现 backend runtime 超出支持范围
- **THEN** 非零返回实际/支持版本，不执行 pip upgrade

### Requirement: 自动化科学正确性门禁
仓库 MUST 以根级 tests 为唯一 pytest 目录，覆盖合同、两 backend、科学指标、两进程、恢复、Artifact、Kaggle、服务、sweep、报告、安全和删除面。Windows/Linux CPU CI MUST 通过，Kaggle GPU 冒烟 MUST 留证；新 training package line/branch coverage MUST 分别不低于 85%/75%。

#### Scenario: 指标金标矩阵
- **WHEN** CI 运行 sklearn 1.6.1 golden fixtures 与随机分块/权重/极端数值测试
- **THEN** 内置指标满足 design 的 `1e-6` 与大偏置 R2 `1e-10` 门槛

#### Scenario: backend 端到端矩阵
- **WHEN** 集成测试运行 Torch 多输入/多标签/回归/恢复和 sklearn batch/Pipeline/sparse/incremental/恢复
- **THEN** 每例断言 step、模型、指标定义、checksum、服务审计和报告，而非只断言无异常

#### Scenario: 跨平台与覆盖率
- **WHEN** Windows/Ubuntu Python 3.10 CI 执行
- **THEN** 测试不依赖 GPU/网络/真实 Secret，且 line/branch 两阈值独立通过

#### Scenario: Kaggle 发布证据
- **WHEN** 候选 revision 发布
- **THEN** 真实 Kaggle CUDA 完成 doctor、Torch 多 GPU、一次暂停恢复、AList/企业微信审计、报告和 manifest，并另以 sklearn incremental CPU smoke 核验 backend contract

### Requirement: 旧训练体系完整移除
候选 revision MUST 从 Git 跟踪内容和 wheel 删除旧训练器、模型、变换、历史实验、RL/AutoML/GUI、旧 Notebook/env、C++、vendored 包和 checkpoint。删除 MUST NOT 通过 shim、改名或 gitignore 隐藏。

#### Scenario: Git inventory
- **WHEN** CI 检查 git ls-files
- **THEN** dl_helper 除新 `__init__.py` 和 `training/` 外无受跟踪文件，固定禁止资产不存在

#### Scenario: 隔离 wheel import
- **WHEN** 从仓库外 cwd 安装并探测候选 wheel
- **THEN** 新 API 成功且所有旧 import 失败，不受源码目录或 editable 安装污染

#### Scenario: 删除边界
- **WHEN** 实施仓库清理
- **THEN** 只删除固定的受跟踪项目资产，不删除用户目录、Kaggle Dataset、AList 或外部 runs
