# Change: 构建 Kaggle 通用深度学习训练平台

## Why

当前仓库已经包含基于 PyTorch、Accelerate 的训练循环，以及混合精度、多 GPU、检查点和静态图表能力，但这些能力与 LOB 数据字段、AList、微信通知、Kaggle 固定路径和历史实验脚本紧密耦合。当前 HEAD 还存在分类指标实参顺序错误、回归路径执行分类后处理、异常以退出码 0 结束、配置字段静默丢弃、明文凭证和大量失配实验入口，因此不能作为可验证、可复现的通用训练平台。

本变更以新的通用训练平台完整替换现有训练体系，使用户可以显式提供任意 PyTorch 数据加载器、模型和任务逻辑，在 Kaggle CPU/单 GPU/多 GPU 环境中可靠训练、恢复、评估并生成直观成果报告。旧训练入口、模型、历史实验和独立 RL/AutoML 训练栈不再兼容或保留，避免新旧两套生命周期、依赖和安全边界长期并存。

## Current Behavior

- `dl_helper.tester.test_base` 暴露模型、数据、损失、优化器和变换钩子，但 `dl_helper.trainer.run_fn_gpu` 在训练结束后无条件解析金融元数据并上传 AList。
- `dl_helper.tracker.Tracker` 将整轮预测和标签反复拼接到主进程，分类 F1/Recall 调用顺序错误，回归与分类共享了不兼容的置信度输出路径。
- `dl_helper/tests/` 包含 260 个 Python 文件和 181 个 `test_base` 实验类，但不是稳定自动化测试集；其中 124 个脚本仍调用已删除的 `cache_data` 模式。
- Kaggle Notebook 从 `master` 实时下载环境和单个日期实验脚本，运行版本不可复现，并在源码和 Notebook 中保存凭证。
- `setup.py` 没有声明运行依赖，`requirements.txt` 未锁定关键版本，也没有 CI、严格配置 schema 或稳定公共 API。
- 当前 Git 还跟踪 370 个 `dl_helper` 文件，其中包含 271 个历史实验文件、15 个旧模型文件、59 个 RL 文件，以及约 40 MiB 的训练 checkpoint；这些资产不属于可发布的通用训练库。

## Goals And Observable Success Criteria

- 用户通过 `module:function` 实验工厂和严格 YAML 配置启动训练；内置 PyTorch 与 scikit-learn 两个后端。PyTorch 后端接受任意 `torch.nn.Module`，sklearn 后端接受可 clone、fit、predict 的 estimator/Pipeline 及兼容第三方 wrapper。
- 内置多分类、多标签和单/多目标回归 Task；其他任务通过同一 Task 协议接入，不依赖输入形状或字段名称推断。
- 单机 CPU、单 CUDA 和多 CUDA 使用同一训练语义；Kaggle 默认使用全部可见 CUDA 设备，并准确记录每设备批量、梯度累积和全局有效批量。
- 内置指标与固定 sklearn 金标样例在 `1e-6` 绝对误差内一致；跨两进程归约与单进程结果在 `1e-6` 内一致。
- 每个内置指标随 Artifact 输出定义、方向、平均方式、sample-weight 语义、精确/抽样标记和有效样本数；随机分块、跨 rank、极端类别、常量目标和加权样例均与 scikit-learn 1.6.1 金标一致。
- 训练过程不在 GPU 上保留整轮 logits；内置指标内存随类别数或目标数增长，而不是随样本数增长。
- 每个成功运行都生成版本化 manifest、解析后配置、环境信息、JSONL 指标、摘要、best/last 模型、可选预测分片和可直接打开的 HTML 报告。
- Kaggle 运行使用显式输入路径、`/kaggle/working` 输出、固定 Git revision、运行时预算和安全 Secret；支持带校验和的本地/AList 检查点恢复。
- AList ArtifactStore 与企业微信通知是实验无关的内置基础设施；启用后覆盖 run/sweep 的开始、成功、暂停和失败生命周期，并以显式 required/record 失败策略运行。
- 同一 Experiment 可由一个完整 base YAML 和多个严格 variant patch 复用；sweep 按清单顺序在独立子进程运行、比较同一全量指标并生成离线对比报告，不复制训练脚本。
- 任一输入、状态、指标、输出、远程同步或配置错误均产生非零退出码和完整 traceback；不得记录并继续、返回伪成功或回退到不同训练语义。
- 仓库扫描不再发现已知 AList、索引服务或第三方数据明文凭证；历史已泄露凭证的外部轮换是发布前置条件。
- 安装后的 `dl_helper` 只公开新训练平台；`dl_helper.trainer/tester/tracker/train_param/models/rl/tests` 等旧模块导入必须失败，仓库不再跟踪旧实验、旧模型、训练 checkpoint、vendored wheel/tarball 或旧启动 Notebook。

## What Changes

- 新建 `dl_helper.training` 公共包，包含严格配置、协议、训练引擎、启动器、流式指标、检查点、平台适配、报告和远程 ArtifactStore。
- 新建 backend 边界与 PyTorch/sklearn 两个实现；sklearn 支持普通 batch estimator、Pipeline，以及提供 `partial_fit` 的可恢复增量 estimator。
- 新建 `dl_helper.training.cli`，提供 `doctor`、`train`、`report`、`sweep`、`sweep-report` 五个非交互命令；所有本地命令通过 `D:/programs/miniconda3/python.exe -m dl_helper.training.cli ...` 验证。
- 增加 Kaggle Notebook 模板和从零编写的 MNIST、双输入分类、多标签、回归、自定义 Task 示例；模板要求固定 Git revision，不再下载浮动 `master` 文件。
- 增加标准运行目录 schema、不可变检查点 manifest、配置/数据/模型兼容指纹和安全 AList 同步协议。
- 增加固定企业微信应用消息客户端、生命周期事件模板、Secret 注入、有限重试和通知审计 Artifact，不复用 `py_ext` 单例或硬编码身份信息。
- 增加精确流式分类/多标签/回归指标、分布式归约、预测分片和任务化 HTML 报告。
- 增加 base/variant 配置合并、sweep manifest、trial 隔离恢复、可比性预检、best-trial 指针和聚合 HTML 报告。
- 增加 `pyproject.toml`、最小训练依赖与可选 extras、根级真实测试目录、Windows/Linux CPU CI 和 Kaggle 合同测试。
- **BREAKING** 删除旧监督训练器、数据/指标/调度器、模型、变换、历史实验、RL/AutoML/GUI 训练栈、旧 Notebook/env 脚本、C++ 扩展和仓库内二进制依赖/检查点，不提供兼容 shim 或弃用期。
- 删除 `setup.py`、`requirements.txt` 和 legacy extra；`pyproject.toml` 是唯一包元数据与依赖来源。

## Scope

### In Scope

- PyTorch 监督训练、scikit-learn batch/incremental estimator 训练及通过 Task 插件实现的自定义训练任务。
- torchvision/timm/transformers 等返回 `nn.Module` 的模型工厂，以及 XGBoost/LightGBM/CatBoost 等满足 sklearn estimator 协议的可选 wrapper；项目不自动安装这些第三方库。
- Map-style、Iterable-style 或用户自定义 DataLoader；中途精确恢复仅对声明并实现状态协议的 DataModule 开放。
- CPU、单 CUDA、多 CUDA DDP、AMP、梯度累积、梯度裁剪、显式调度器策略和可选 `torch.compile`。
- Kaggle Notebook 的环境预检、资源发现、路径约束、运行时预算、产物持久化和 AList 跨会话恢复。
- 企业微信 run/sweep 生命周期通知，以及同一实验的确定性多 variant 顺序训练与对比报告。
- 分类、多标签、回归和通用指标曲线的静态报告。
- 对旧训练体系及其受跟踪实验资产的完整移除和新仓库边界门禁。

### Non-Goals

- 不重写、迁移或继续发布 Stable-Baselines3、RLlib、imitation、AutoGluon、GUI 和旧 C++ 扩展；对应代码和依赖从本仓库删除。未来如需这些能力，必须以独立 change 和新插件合同重新引入。
- 不自动猜测原始文件格式、标签列、模型结构、损失函数或指标方向；用户必须通过 DataModule、Task 和 Experiment 显式声明。
- 不提供模型搜索、NAS、自动批量回退、自动 OOM 降级、自动超参数搜索或分布式多节点训练。
- `sweep` 只执行用户显式列出的 variant，不生成参数组合、不做贝叶斯/网格/随机搜索，也不并发争抢同一 Kaggle GPU。
- 不支持 Kaggle TPU/XLA；本变更的 Kaggle 加速范围是 CPU 和 CUDA。TPU 需要后续独立规格和真实 Kaggle 验证。
- 不承诺不同 PyTorch/CUDA/硬件版本之间逐位一致；同一受支持环境和严格确定性配置下必须保持状态与优化步一致。
- 不自动重写 Git 历史或调用外部服务轮换凭证；仓库清理属于实施，历史清理与凭证轮换由仓库所有者在发布前完成。

## Affected Capabilities

- `general-training`：新增通用实验、数据、任务、优化和恢复合同。
- `kaggle-execution`：新增 Kaggle 资源、路径、预算、Secret 和远程恢复合同。
- `training-reporting`：新增流式指标、预测产物和 HTML 报告合同。
- `training-reliability`：新增 Fail Fast、配置验证、复现、安全和测试门禁。
- `training-services`：新增通用 AList ArtifactStore、企业微信通知和生命周期投递合同。
- `experiment-sweeps`：新增 base/variant 配置复用、顺序 trial 编排、恢复和对比报告合同。

## Affected Code And Artifacts

- 新建：`dl_helper/training/`（含 `backends/`、`notifications.py`、`sweep.py`）、`tests/training/`、`tests/integration/`、`tests/kaggle/`、`tests/services/`、`tests/sweeps/`、`experiments/`、`configs/`、`notebook/kaggle_training_template.ipynb`、`pyproject.toml`。
- 替换：`dl_helper/__init__.py` 只导出新平台版本和公共入口；`envs/` 只保留新 `kaggle_bootstrap.py`；`notebook/` 只保留新 Kaggle 模板。
- 删除：现有 `dl_helper/` 中除被替换 `__init__.py` 外的全部旧文件，包括 `trainer.py`、`tester.py`、`tracker.py`、`train_param.py`、`data.py`、`scheduler.py`、`tool.py`、`models/`、`transforms/`、`tests/`、`other_tests/`、`rl/`、AutoGluon、压缩/索引/传输辅助模块。
- 删除：旧 `envs/` 与 `notebook/` 内容、`cpp/`、`参考/`、`setup.py`、`requirements.txt`、仓库内 wheel/tarball、checkpoint 和其他生成二进制资产；编辑器/Agent 配置与 OpenSpec 文件不属于删除范围。
- 运行产物：`runs/<run-id>/` 下的 manifest、配置、日志、指标、模型、检查点、预测和报告。

## Breaking Removal And Migration

- 新 API 只使用 `dl_helper.training`；旧 `dl_helper.trainer.run`、`test_base`、旧模型类、旧 RL/AutoML 入口和旧 import path 立即移除，不提供重导出、警告代理或自动转换器。
- 旧实验配置、`Params` 字典、7z/AList 缓存和旧 checkpoint 不受支持；用户必须以新 Experiment/DataModule/Task 和 schema v1 配置重新表达仍需保留的实验。
- 普通 sklearn batch estimator 不声明 mid-fit 恢复；Kaggle 的可恢复预算模式只接受提供 `partial_fit` 且 DataModule 可恢复的 incremental estimator。batch estimator 可在本地无运行预算配置下训练，系统不得伪造恢复能力。
- variant patch 只覆盖 base config 中允许变化的训练/模型参数；sweep 强制所有 trial 使用同一 Experiment 引用、backend、DataIdentity、Task/标签 schema 和比较指标，避免把不同数据或任务错误排名。
- 本变更只提供从零编写的通用示例，不复制旧 LOB 模型、超参数或金融输出器；任意领域预测通过公开 Task 的 prediction arrays 合同由用户重新实现。
- 删除仅作用于 Git 受跟踪源码和仓库资产，不扫描或删除用户机器、Kaggle Dataset、AList 服务器及既有 `runs/` 外部产物。
- 发布版本固定为 `1.0.0`，以明确表示旧 Python API 和训练资产不兼容；最后一个旧版源码提交固定记录为 `62ee1e4bbf42065ec07dbd0fc8d5b4f9b642f7fd`。

## Risks, Release And Rollback

- 分布式与恢复语义错误可能造成重复优化步或错误 best 模型；以单/双进程金标集、状态恢复对照和不可变 manifest 降低风险。
- 删除旧模块会使所有旧 import 立即失败；这是显式接受的破坏性变更，通过主版本升级、删除清单和负向 import 测试固定边界，不保留双栈。
- 删除历史实验和二进制 checkpoint 会缩减仓库但不可从新版本工作树直接恢复；Git 历史仍保留原内容，发布说明必须记录最后一个旧版 commit。
- AList 上传中断可能留下不完整对象；只有 archive 校验完成后才发布 manifest，只有 manifest 校验完成后才更新 `latest.json`。
- sklearn joblib 模型只能从本次运行产生且 checksum/版本完全匹配的可信 Artifact 恢复；不得加载用户提供或来源不可信的 pickle/joblib。
- 企业微信通知失败是否影响训练由显式 `required/record` 策略决定；无论哪种策略都必须写入投递审计，失败通知不得覆盖原训练异常。
- sweep 中任一 trial 失败即停止，任一 trial 预算暂停则整个 sweep 可恢复暂停；不允许跳过失败 trial 后生成伪完整排名。
- 报告抽样曲线不是全量指标；报告必须明确标记抽样数，摘要指标始终使用全量流式统计。
- 发布前必须完成凭证轮换确认、Linux/Windows CI、Kaggle GPU 冒烟运行和 Secret 扫描。
- 回滚必须整体恢复到发布说明记录的最后一个旧版 commit；同一 revision 内不提供旧入口开关。新运行目录与旧格式隔离，回滚不得转换、覆盖或删除新平台已生成的 Artifact。
