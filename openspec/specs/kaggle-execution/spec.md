# kaggle-execution Specification

## Purpose
TBD - created by archiving change build-general-kaggle-training-platform. Update Purpose after archive.
## Requirements
### Requirement: Kaggle 平台与路径合同
系统 MUST 通过环境识别 Kaggle，要求显式 `/kaggle/input/...` 数据路径，并将新产物限制在 `/kaggle/working`。系统 MUST NOT 自动选择首个输入目录或写只读输入目录。

#### Scenario: 合法路径
- **WHEN** 输入存在于显式 `/kaggle/input/<dataset>/...` 且 output root 位于 `/kaggle/working`
- **THEN** doctor 通过并为 run/sweep 创建隔离目录

#### Scenario: 输入含糊或路径逃逸
- **WHEN** 输入只给 `/kaggle/input`、不存在、经 symlink 逃逸，或输出解析到 working 外
- **THEN** doctor 在训练前非零失败

### Requirement: Kaggle PyTorch 资源自动利用
Torch backend MUST 通过可见设备发现 CPU/CUDA，`num_processes=auto` MUST 使用全部可见 CUDA 且 CPU 使用一个进程。系统 MUST 记录解析资源与有效批量，不按 GPU 型号硬编码。

#### Scenario: 两张可见 GPU
- **WHEN** 暴露两张 CUDA 且 num_processes=auto
- **THEN** launcher 启动两个 worker、各绑定一个设备，并记录正确全局有效批量

#### Scenario: dynamic batch
- **WHEN** nominal batch 为 null
- **THEN** 报告显示 dynamic 和每 rank 实际 sample_count 范围，不从 DataLoader 属性猜测

#### Scenario: 请求资源不可满足
- **WHEN** 进程数超过可见 GPU，或显式 AMP/compile 不受支持
- **THEN** doctor 失败，不回退单卡、其他精度或 eager

#### Scenario: AMP auto
- **WHEN** mixed_precision=auto
- **THEN** 支持 bf16 的 CUDA 选 bf16、其他 CUDA 选 fp16、CPU 选 no，并写入 resolved config

### Requirement: Kaggle sklearn 增量执行
Kaggle profile MUST 只接受单进程、可恢复的 sklearn incremental mode，并按逻辑 CPU 解析 estimator 顶层 n_jobs。系统 MUST 拒绝无法提供受控暂停点的 batch fit，而不是暗示其能跨 Kaggle 会话恢复。

#### Scenario: incremental estimator
- **WHEN** estimator 支持 partial_fit、数据源支持 batch resume、distributed.num_processes=1 且预算有效
- **THEN** doctor 通过，并在 batch 边界检查预算与发布可信 joblib checkpoint

#### Scenario: batch estimator
- **WHEN** Kaggle profile 配置 fit_mode=batch
- **THEN** doctor 在拟合前失败，并指出必须改用 incremental mode 或 Local profile

#### Scenario: n_jobs auto
- **WHEN** sklearn n_jobs=auto 且 estimator 暴露顶层 n_jobs
- **THEN** 系统解析为逻辑 CPU 数、写入 estimator 参数与 environment；未暴露时失败

### Requirement: 固定 revision 的 Kaggle 启动
Kaggle 模板 MUST 要求 40 位 Git commit SHA，检出后核对 HEAD，使用 `pip install -e . --no-deps` 并先执行 doctor。模板 MUST NOT 下载浮动 master、git pull 或静默升级框架。

#### Scenario: 固定提交启动
- **WHEN** 用户提供合法可获取 commit SHA
- **THEN** 模板检出相同 SHA、安装并在 doctor 通过后启动 train 或 sweep

#### Scenario: revision 或依赖错误
- **WHEN** SHA 缺失/错误、HEAD 不一致或依赖超出范围
- **THEN** Notebook 单元非零停止且不拟合

### Requirement: backend-aware Kaggle Doctor
系统 MUST 提供不执行训练的 doctor，检查共享配置、实验、依赖、数据/evaluation contract、路径、至少 5 GiB 磁盘、预算、恢复、Secret 与服务配置，并根据 backend 检查 PyTorch 或 sklearn 专属能力。doctor MUST 支持输出 sweep 可比性所需的结构化 evaluation contract。

#### Scenario: Torch 完整预检
- **WHEN** Torch 配置、设备、AMP、DDP、DataModule 和服务均满足合同
- **THEN** doctor 返回 0 和不含 Secret 的解析摘要

#### Scenario: sklearn 完整预检
- **WHEN** incremental estimator 的 clone、kind、partial_fit、prediction、classes、n_jobs/random_state/sample-weight 和 source resume 均满足合同
- **THEN** doctor 返回 0 并输出 evaluation contract，不创建 fitted estimator

#### Scenario: 多项静态失败
- **WHEN** 同时存在多个可独立检查的错误
- **THEN** doctor 一次列出全部错误后非零返回

#### Scenario: 无训练副作用
- **WHEN** 运行 doctor 或 emit evaluation contract
- **THEN** 不调用 fit/partial_fit/optimizer step，不创建 checkpoint、远程目录或通知

### Requirement: 运行预算与可恢复退出
Kaggle profile MUST 要求正 max/grace 且 grace 小于 max，并在 `elapsed >= max-grace` 的成功 batch 后停止新拟合 step，使用完整 grace 保存 checkpoint、刷新 required 服务、生成当前报告和 pause manifest。无法形成可恢复状态 MUST 为 FAILED。

#### Scenario: Torch 安全暂停
- **WHEN** budget 命中且 DataModule 支持中途恢复
- **THEN** 系统不再执行 optimizer step，保存完整状态、服务 flush、报告和 pause manifest，返回 75

#### Scenario: sklearn incremental 安全暂停
- **WHEN** budget 在一次成功 partial_fit 后命中
- **THEN** 系统保存 estimator/source/EngineState/metric 状态并完成同样 PREEMPTED/75 流程

#### Scenario: grace 或持久化失败
- **WHEN** grace 配置错误、checkpoint 不完整或 required service 在 deadline 前失败
- **THEN** 运行 FAILED 且不存在 SUCCEEDED/PREEMPTED manifest

### Requirement: Kaggle CUDA 性能配置
Torch backend MUST 记录并使用解析后的 DataLoader worker、pin memory、persistent worker、prefetch、AMP 和有效批量。torch.compile MUST 默认关闭，显式失败不得回退 eager；OOM MUST NOT 自动降低 batch。

#### Scenario: 自动 DataLoader 参数
- **WHEN** 使用 LoaderDataModule、CUDA 和 num_workers=auto
- **THEN** worker 按 CPU/进程规则解析，启用合法 pin/persistent/prefetch 并记录结果

#### Scenario: compile 或 OOM 失败
- **WHEN** 显式 compile 不兼容或出现 CUDA OOM
- **THEN** 系统记录当前批量/累积/显存上下文并保留 traceback，不修改训练语义继续
