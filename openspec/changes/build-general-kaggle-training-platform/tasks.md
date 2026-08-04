## 1. Packaging, Configuration And Contracts

- [x] 1.1 建立唯一包元数据与测试收集边界
  - 依据：`依赖与导入边界`、`单一新平台与旧 API 移除`、D-001、D-012、D-016、REF-002
  - 修改：新建 `pyproject.toml`，固定版本 `1.0.0`、Python `>=3.10,<3.13`、design 中 torch/accelerate/numpy/matplotlib/sklearn/safetensors/PyYAML/joblib/requests 核心范围与 dev extra；配置 pytest `testpaths=["tests"]` 和 package discovery。
  - 约束：不得提供 alist/legacy extra；不得声明 RL、AutoML、GUI、TorchMetrics、py-ext 或旧 C++ 依赖；旧元数据在第 9 阶段删除。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pip install -e '.[dev]' --no-deps`；预期 editable metadata 成功且依赖/extra 快照符合 design。

- [x] 1.2 实现严格 base 配置 schema v1
  - 依据：`严格 base 与 variant 配置`、`复现实验 manifest`、D-007、D-012
  - 修改：新建 `dl_helper/training/config.py`，实现重复 key 检测、安全 YAML loader、frozen typed config、backend 分支、AList/企业微信配置、规范化序列化、跨字段校验和完整/resume fingerprint。
  - 约束：全部文本 I/O 显式 `encoding='utf-8'`；未知字段、字符串 bool、NaN/Inf、未选 backend 非 null、sklearn batch/Kaggle、Secret 值进入配置均必须失败。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_config_base.py`；预期合法快照与每类负向字段/跨字段用例通过。

- [x] 1.3 实现严格 variant resolver 与 tuning fingerprint
  - 依据：`严格 base 与 variant 配置`、`严格 sweep manifest 与 trial 派生`、D-017
  - 修改：在 `config.py` 实现 mapping 递归、scalar/list/null 替换、允许路径白名单、base/variant SHA256、resolved/tuning fingerprint 和 symlink/root 边界；在 `cli.py train` 接入 `--variant`。
  - 约束：拒绝 YAML merge/alias、模板、环境插值、include、URL、路径逃逸以及 run.id/seed、backend type、revision、host/Secret/process 等禁止覆盖；合并后必须重跑完整 schema。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_variant_config.py`；预期 merge 表、禁止路径、相同 tuning 与跨字段失败全部通过。

- [x] 1.4 实现稳定双后端公共合同
  - 依据：`显式双后端实验工厂`、`通用 PyTorch DataModule 合同`、`sklearn 数据与 estimator 合同`、D-003、D-014
  - 修改：新建 `contracts.py`、`backends/base.py` 与无副作用 `training/__init__.py`，实现 design 全部 frozen dataclass/Protocol、JSONValue、DataIdentity、EvaluationContract、BackendResult 和运行时前置校验。
  - 约束：公共导入不得连接网络、解析 Secret、构造实验或导入 transformers；Torch/sklearn 合同不得以 `Any` 隐藏关键 backend 状态或静默协议探测。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_contracts.py tests/training/test_import_boundaries.py`；预期双后端正负合同和 import graph 通过。

- [x] 1.5 实现统一 Task、PredictedBatch 与内置任务适配
  - 依据：`统一评价 Task 与 PredictedBatch`、`嵌套批次与任意 PyTorch 模型调用`、`内置与自定义任务`、D-003、D-014
  - 修改：新建 `task.py`，实现 PreparedBatch/LossResult/PredictedBatch 验证、Torch 默认 mapping/tuple/single 调用、多分类/多标签/回归 TorchTask 与 SklearnTask 适配及 prediction capability 声明。
  - 约束：不推断任务、shape、label 顺序或 estimator 输出；sample_count/weight/schema 必须精确；缺 predict_proba 等要求时不允许回退。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_builtin_tasks.py tests/training/test_model_invocation.py tests/training/test_predicted_batch.py`；预期两 backend、三任务与自定义输出正负用例通过。

## 2. Scientific Metrics

- [x] 2.1 实现 MetricDefinition 与 MetricState 归约合同
  - 依据：`版本化科学指标定义`、`全量流式与分布式指标`、D-004、D-015、REF-007
  - 修改：新建 `metrics.py`，实现公式元数据验证、state reset/save/load、sum/min/max/merge_weighted_moments 固定 shape 归约、sample_count/weight_sum 和 finite compute 门禁。
  - 约束：同名定义漂移、样本维 state、rank key/shape/dtype/op 不一致、空 split 或非有限输出立即失败；selection 仅接受 exact/full。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_metric_definition.py tests/training/test_metric_state_contract.py`；预期公式版本和全部状态负向用例通过。

- [x] 2.2 实现加权多分类与多标签流式指标
  - 依据：`全量流式与分布式指标`、`统一 sample-weight 语义`、D-004、D-015、REF-007
  - 修改：在 `metrics.py` 实现 float64 weighted/raw confusion、per-class/label、macro/weighted/micro、balanced accuracy、subset accuracy、hamming loss、阈值与 zero_division=0 语义。
  - 约束：sample weight 只按样本应用一次；macro 包含全部声明类别/label；balanced accuracy 只含正真实 weighted support；不得输出冗余 micro multiclass 伪指标。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_metrics_classification_goldens.py`；预期无权重/整数/浮点权重、缺失类、全负 label 与 sklearn 1.6.1 误差不超过 `1e-6`。

- [x] 2.3 实现稳定加权回归与 loss 统计
  - 依据：`全量流式与分布式指标`、`统一 sample-weight 语义`、`精确的 PyTorch 优化循环`、D-015、REF-007
  - 修改：在 `metrics.py` 实现 LossResult numerator/denominator float64 状态、weighted Welford update、固定 rank Chan merge、per-target MAE/MSE/R2 与 uniform/variance-weighted 聚合。
  - 约束：常量目标严格采用 sklearn force_finite；近常量不得 epsilon 改判；M2 负值只按 design 舍入界处理，非法状态失败。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_metrics_regression_goldens.py`；预期常量/近常量/大偏置/多输出/加权金标通过，大偏置 R2 误差不超过 `1e-10`。

- [x] 2.4 建立公式版本不可变 golden 与随机分块门禁
  - 依据：`版本化科学指标定义`、`自动化科学正确性门禁`、D-015、REF-007
  - 修改：新建 `tests/fixtures/metric_goldens_v1.json`、`tests/training/test_metric_chunking.py` 和 fixture 生成说明；固定输入、显式 sklearn 参数、期望定义和值，并以随机 batch 切分/状态恢复重复验证。
  - 约束：测试运行不得现场用待测实现生成期望；fixture 变更必须伴随 formula_version 变化；所有 fixture JSON 以 UTF-8 读取。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_metric_chunking.py tests/training/test_metric_formula_versions.py`；预期 100 组固定 seed 分块与不可变版本断言通过。

## 3. Backend Workers And Engine

- [x] 3.1 实现 backend-neutral 生命周期与 EngineState
  - 依据：`统一评价 Task 与 PredictedBatch`、`验证选择与早停`、`Fail Fast 进程与子进程语义`、D-014
  - 修改：新建 `engine.py`，实现共享状态机、BackendResult 消费、stage 指标、selection/early stop、Artifact/服务边界和 backend-neutral EngineState 序列化。
  - 约束：engine 不导入领域代码或包含 torch/sklearn 训练细节；所有状态单向，终态互斥，primary 异常向上传播。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_engine_state_machine.py tests/training/test_selection.py`；预期状态转换、best/patience、无 val、方向/定义错误通过。

- [x] 3.2 实现 Torch 组件预检与 launcher
  - 依据：`显式双后端实验工厂`、`通用 PyTorch DataModule 合同`、`Kaggle PyTorch 资源自动利用`、D-002、D-003、REF-003、REF-004
  - 修改：新建 `backends/torch_backend.py` 和 `launcher.py`，实现 worker 内延迟实验导入、全新组件验证、Accelerator/loader prepare、SchedulerBinding 注册及 1/多进程启动。
  - 约束：父进程不初始化 CUDA；已上设备/DDP/训练过模型失败；scheduler 不可序列化失败；spawn 入口兼容 Windows/Linux。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_torch_preflight.py tests/training/test_launcher.py`；预期延迟导入、1/2 进程和组件负向用例通过。

- [x] 3.3 实现精确 PyTorch 优化、评价与导出
  - 依据：`精确的 PyTorch 优化循环`、`验证选择与早停`、D-002、D-005、D-015
  - 修改：在 `torch_backend.py` 实现 numerator backward、抵消固定 accumulation 除数、global denominator all-reduce、unscale 后 world-size 规范化、clip/step/scheduler、PredictedBatch 指标、best/last safetensors。
  - 约束：最后不足窗口和不等 rank/sample weight 必须精确；global step 仅成功 optimizer step 增加；NaN/Inf/OOM/compile 不兼容不回退。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_torch_optimizer_semantics.py tests/integration/test_torch_training.py`；预期梯度与单批金标一致，step/scheduler/模型 Artifact 正确。

- [x] 3.4 实现 sklearn estimator 预检与参数解析
  - 依据：`显式双后端实验工厂`、`sklearn 数据与 estimator 合同`、`复现实验 manifest`、D-014、REF-008
  - 修改：新建 `backends/sklearn_backend.py`，实现 clone/check_is_fitted、classifier/regressor/required prediction/fit mode 验证，递归 random_state、顶层 n_jobs 和 sample_weight 参数解析。
  - 约束：不通过 AttributeError 猜测模式；已有 fitted 状态、非整数 random_state、缺顶层 n_jobs、Task kind/capability 不符全部在 fit 前失败。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_sklearn_preflight.py`；预期 Pipeline/RandomForest/SGD 与第三方 fake wrapper 正负矩阵通过。

- [x] 3.5 实现 sklearn batch worker
  - 依据：`sklearn 数据与 estimator 合同`、`统一评价 Task 与 PredictedBatch`、D-014、REF-008
  - 修改：在 `sklearn_backend.py` 实现一次 fit、显式 sample_weight 参数、train/val/test 分批 prediction、shared metrics、best=last 与 joblib model result。
  - 约束：仅 Local、max_epochs=1、resume=none、runtime.max_minutes=null；features 不 densify；Pipeline 预处理只 fit train；fit 次数必须精确为 1。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_sklearn_batch.py`；预期 ndarray/sparse/Pipeline/sample-weight、数据泄漏和非法配置用例通过。

- [x] 3.6 实现 sklearn incremental worker
  - 依据：`sklearn 数据与 estimator 合同`、`Kaggle sklearn 增量执行`、`backend-aware 可审计恢复`、D-014、REF-008
  - 修改：在 `sklearn_backend.py` 实现 epoch/batch partial_fit、首批完整 classes、global step、评价/selection、source state、batch checkpoint 与预算事件。
  - 约束：classes 必须有序固定；step 仅成功 partial_fit 增加；数据源不可恢复或后续类别漂移失败；不调用 Accelerate/AMP/optimizer/scheduler。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_sklearn_incremental.py`；预期 SGD 分类/回归、sample weight、暂停恢复和 classes 负向用例通过。

- [x] 3.7 实现五个 CLI 命令与退出码
  - 依据：`Fail Fast 进程与子进程语义`、`backend-aware Kaggle Doctor`、`离线 sweep 报告`、D-014、D-017
  - 修改：新建 `cli.py`、`training/__main__.py`，实现 doctor/train/report/sweep/sweep-report argparse、`--variant`、`--resume`、evaluation-contract 输出和 0/75/其他非零传播。
  - 约束：顶层捕获只允许写脱敏失败证据后原样 raise；不得吞 traceback、返回伪 0 或在 report/sweep coordinator 导入用户模型。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_cli.py tests/training/test_cli_exit_codes.py`；预期参数矩阵与 0/1/75 传播通过。

## 4. Artifacts, Checkpoints And Reporting

- [x] 4.1 实现固定 run/sweep schema 与原子写入
  - 依据：`版本化 backend-aware 训练 Artifact`、`科学排名与 best trial`、D-010、D-017
  - 修改：新建 `artifacts.py`，实现 design 的 run/sweep 目录、UTF-8 JSON/YAML/JSONL/log、同目录 tmp+flush+fsync+replace、SHA256 清单、路径边界和 terminal 互斥。
  - 约束：仅主进程写共享文件；success/pause/failure 最后发布且互斥；终态不引用缺失/未校验文件。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_artifacts.py tests/sweeps/test_sweep_artifacts.py`；预期中文、checksum、原子中断、路径逃逸和终态快照通过。

- [x] 4.2 实现预测分片与确定性抽样
  - 依据：`通用无 pickle 预测分片`、`可审计抽样可视化`、D-010
  - 修改：在 `artifacts.py` 实现 numeric/bool/fixed-Unicode NPZ shard、prediction manifest 和基于 seed/split/stable ID 的 64-bit hash top-k、跨 rank 候选合并。
  - 约束：拒绝 object dtype/非法字段/shape/checksum；summary 不使用抽样；无稳定 ID 时必须记录位置抽样限制。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_prediction_shards.py tests/training/test_priority_sampling.py`；预期安全 schema、确定性、跨 rank 与 sampled/total 通过。

- [x] 4.3 实现 Torch 不可变 checkpoint 与 safetensors manifest
  - 依据：`backend-aware 可审计恢复`、`版本化 backend-aware 训练 Artifact`、D-007、D-008、REF-003
  - 修改：新建 `checkpoint.py`，实现 Torch staging->Accelerate state->manifest->immutable dir->latest、EngineState/DataModule/MetricState、保留策略和 best/last safetensors manifest。
  - 约束：不得覆盖完整 checkpoint或删除 latest/best；损坏 latest 不尝试旧项；mid-epoch 必须恢复精确位置与部分指标。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_torch_checkpoint_resume.py tests/training/test_checkpoint_corruption.py`；预期连续/恢复 step、权重、指标一致且全部损坏用例失败。

- [x] 4.4 实现可信 sklearn joblib checkpoint 与模型 manifest
  - 依据：`可信 joblib 反序列化`、`backend-aware 可审计恢复`、D-014、REF-008
  - 修改：在 `checkpoint.py` 实现 incremental estimator/source/EngineState staging、joblib/model manifest、regular-file/root/symlink/origin/signature/SHA/runtime exact pre-load 校验和安全 AList staging。
  - 约束：任何校验必须先于 joblib.load；外部/其他 run/版本范围兼容均拒绝；batch 不创建 latest；加载后复核 fitted kind/signature。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/training/test_sklearn_persistence_security.py tests/integration/test_sklearn_checkpoint_resume.py`；预期合法恢复及恶意路径/篡改/版本漂移全部通过。

- [x] 4.5 实现 backend-aware run 离线报告
  - 依据：`直观且忠实的离线 HTML 报告`、`报告展示资源与复现上下文`、D-010、D-015
  - 修改：新建 `reporting.py`，只读 Artifact 生成通用/指标定义/多分类/多标签/回归页面、escaped HTML、相对 PNG、类别>20 规则和 Torch/sklearn N/A 上下文。
  - 约束：不导入实验；不接受 raw HTML；原始指标与抽样图分开；关闭 Matplotlib figure；重复生成幂等。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/reporting/test_run_reports.py tests/reporting/test_report_security.py`；预期两 backend、三任务、暂停、抽样标签、HTML escape 和非空图片通过。

## 5. AList And Enterprise WeChat Services

- [x] 5.1 实现统一 Secret resolver、脱敏与 service audit
  - 依据：`服务失败策略与审计`、`Secret 与凭证安全`、D-016、REF-001
  - 修改：在 `platform.py`/新建 `services.py` 实现 Kaggle Secrets/Local env 同名解析、值注册脱敏、required/record policy、event audit JSONL、primary/secondary error model。
  - 约束：启用服务的 Secret 在首个拟合 step 前解析；值不进入 repr/log/config/audit/error；服务错误不得覆盖训练 traceback。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/services/test_secret_resolver.py tests/services/test_service_audit.py tests/services/test_error_precedence.py`；预期缺 Secret、中文 UTF-8、required/record 和原异常保护通过。

- [x] 5.2 实现完整 AListArtifactStore
  - 依据：`通用 ArtifactStore 生命周期`、D-009、D-016、REF-006
  - 修改：新建 `remote.py`，实现 requests Session、AList auth/fs endpoints、HTTPS/base path、tar.gz safe archive、2/4/8 retry、raw 回读 SHA、checkpoint/run/sweep bundle 与 terminal-last fetch/publish。
  - 约束：401/403/业务参数错误不重试；size 不替代 hash；archive 拒绝 symlink/absolute/..；Secret/认证 URL 不进入错误和审计。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/services/test_alist_store.py`；预期 mock HTTP 发布顺序、恢复、认证、5xx、timeout、checksum 和危险 archive 通过。

- [x] 5.3 实现有界异步同步与 terminal flush
  - 依据：`有界异步 AList 同步`、D-011、D-016
  - 修改：在 `remote.py` 实现主进程唯一非 daemon worker、容量 1 pending checkpoint、pending 合并、active 不取消、边界错误检查、terminal 优先级与 join/flush。
  - 约束：被合并 checkpoint 保留本地；terminal 不合并；后台异常按 policy 可观察；线程未退出不得发布终态。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/services/test_async_artifact_store.py`；预期 pending/active 顺序、异常传播和零残留线程通过。

- [x] 5.4 实现官方企业微信客户端与固定事件模板
  - 依据：`固定企业微信生命周期通知`、D-016、REF-009
  - 修改：新建 `notifications.py`，实现固定 qyapi host、gettoken monotonic cache/lock、message send、token 失效单次刷新、有限 retry、稳定 event_id、run/trial/sweep 模板与 2048 UTF-8 bytes 裁剪。
  - 约束：不导入 py-ext、不持久化 token、不允许 host/custom template；agent ID 为正整数；只 errcode=0 成功；裁剪不得丢 event/status/scope/error type。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/services/test_wecom_client.py tests/services/test_notification_templates.py`；预期 token/cache/刷新/业务码/重试、中文 byte 边界和脱敏通过。

- [x] 5.5 编排 run/sweep 服务顺序与可重入 finalization
  - 依据：`服务失败策略与审计`、`可重入服务终结`、D-016
  - 修改：在 `engine.py`、`sweep.py`、`services.py` 接入 STARTED/terminal 事件、AList bundle、最终 audit service manifest、稳定 checksum/event 去重和 FINALIZING 重入。
  - 约束：required 失败阻止 success/pause，record 明示 degraded；已成功动作不重复；状态漂移失败；核心失败始终为 primary。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_service_lifecycle.py`；预期 run/trial/sweep 全事件、各中断点重入、required/record 与异常优先级通过。

## 6. Multi-Variant Sweeps

- [x] 6.1 实现严格 sweep manifest parser
  - 依据：`严格 sweep manifest 与 trial 派生`、D-017
  - 修改：新建 `sweep.py`，实现 schema、ID/name、至少两 trial、有序路径、父目录边界、派生 run ID、resolved sweep 与唯一 tuning fingerprint。
  - 约束：拒绝 URL/绝对/逃逸/symlink、重复 trial、variant run.id 和仅基础设施不同的重复 tuning。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/sweeps/test_sweep_manifest.py`；预期合法快照与全部路径/ID/fingerprint 负向用例通过。

- [x] 6.2 实现 evaluation contract 子进程预检
  - 依据：`零优化步可比性预检`、`backend-aware Kaggle Doctor`、D-015、D-017
  - 修改：在 `sweep.py`/`cli.py doctor` 实现逐 trial 独立 contract 子进程、DataIdentity/split/Task/label/MetricDefinition 比较和 exact full val metric 门禁。
  - 约束：coordinator 不导入实验/torch/sklearn/CUDA；任一漂移在零拟合 step 失败；test/sampled metric 不可 comparison。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/sweeps/test_comparability_preflight.py`；预期相同合同通过，数据/阈值/权重/formula/direction/test leakage 差异通过。

- [x] 6.3 实现隔离顺序 trial coordinator
  - 依据：`隔离顺序 trial 编排`、`Fail Fast 进程与子进程语义`、D-017
  - 修改：在 `sweep.py` 实现原子 lock、`sys.executable -m ... train` argv、顺序 child、trials.jsonl、0/75/error 分支和停止策略。
  - 约束：不使用 shell 拼接、不并发 trial、不自动搜索；失败立即停止且不产生 best；lock 冲突非零。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/sweeps/test_sweep_coordinator.py`；预期顺序、环境隔离、失败停止、75 和并发 lock 通过。

- [x] 6.4 实现 sweep pause/resume 一致性
  - 依据：`sweep 暂停与严格恢复`、D-007、D-017
  - 修改：在 `sweep.py`/`artifacts.py` 实现 pause manifest、已完成/current/remaining 状态、manifest/base/variant/contract/run checksum 复核、成功 trial 跳过与 paused trial 恢复。
  - 约束：FAILED 不可原 ID resume；任何漂移失败；成功 trial 不重跑；当前 trial 只用自身 checkpoint。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_sweep_resume.py`；预期跨两次进程完成、零重复 trial 与所有漂移/FAILED 负向通过。

- [x] 6.5 实现未舍入排名、best 与 sweep 报告
  - 依据：`科学排名与 best trial`、`离线 sweep 报告`、D-015、D-017
  - 修改：在 `sweep.py`/`reporting.py` 实现 summary 原值读取、direction 排名、YAML 顺序 tie-break、best-trial checksum、成功/进度 HTML、参数差异与 run links。
  - 约束：仅全部成功生成 best/ranking；缺失/NaN/Inf/定义漂移失败；test 仅展示不排名；HTML escape、只读、幂等。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/sweeps/test_ranking.py tests/reporting/test_sweep_report.py`；预期舍入冲突、并列、无效值、暂停/失败页面与链接通过。

## 7. Local And Kaggle Platform

- [x] 7.1 实现 Local/Kaggle 解析与 backend 资源合同
  - 依据：`Kaggle 平台与路径合同`、`Kaggle PyTorch 资源自动利用`、`Kaggle sklearn 增量执行`、`Kaggle CUDA 性能配置`、D-005、D-006、REF-005
  - 修改：新建 `platform.py`，实现环境/路径/symlink 边界、Torch GPU/AMP/worker/effective batch、sklearn 单进程/n_jobs/native thread、环境 manifest。
  - 约束：不选首个 input、不按 GPU 名称分支、不写 working 外；显式资源不满足失败；sklearn batch Kaggle 失败。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/kaggle/test_platform.py tests/training/test_resource_resolution.py`；预期 Local/Kaggle、CPU/1/2 GPU、sklearn CPU、路径和资源矩阵通过。

- [x] 7.2 实现 backend-aware doctor
  - 依据：`backend-aware Kaggle Doctor`、`复现实验 manifest`、D-012、D-014
  - 修改：在 `platform.py`/`cli.py` 实现共享依赖/路径/磁盘/预算/Secret/服务检查，Torch 专属设备/AMP/DDP/DataModule，sklearn 专属 clone/capability/classes/params/persistence 检查和 contract 输出。
  - 约束：doctor 不执行拟合、checkpoint、远程目录或通知；可独立静态错误一次列全；Secret 只显示 key。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/kaggle/test_doctor.py`；预期两 backend 成功、多错误聚合、无副作用和 contract JSON 通过。

- [x] 7.3 实现 monotonic Kaggle 预算与 PREEMPTED
  - 依据：`运行预算与可恢复退出`、D-006、D-008、D-014
  - 修改：在 `platform.py`、两 backend、`checkpoint.py`、`services.py` 和 `artifacts.py` 实现 batch 后 `elapsed>=max-grace`、停止新 step、完整 checkpoint/service/report/pause 和退出 75。
  - 约束：不动态估算或缩短 grace；Torch/partial_fit 均不重复 step；deadline 或 required service 失败为 FAILED；batch sklearn 不进入该流程。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_runtime_budget.py`；预期 fake monotonic 下两 backend 暂停恢复、持久化失败与终态互斥通过。

- [x] 7.4 创建固定 revision Kaggle bootstrap 与 Notebook
  - 依据：`固定 revision 的 Kaggle 启动`、D-012、REF-005
  - 修改：新建 `envs/kaggle_bootstrap.py` 和 `notebook/kaggle_training_template.ipynb`，要求 40 位 SHA、clone/checkout/HEAD、`pip install -e . --no-deps`、doctor、train/sweep 和子进程返回码。
  - 约束：不含 Secret、浮动 master、git pull、框架升级或静默重定向；脚本文本 I/O UTF-8。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/kaggle/test_bootstrap.py tests/kaggle/test_notebook_template.py`；预期结构解析、命令顺序、SHA/Secret/失败传播通过。

## 8. Clean Examples And Configurations

- [x] 8.1 创建无需网络的 PyTorch 通用示例
  - 依据：`显式双后端实验工厂`、`内置与自定义任务`、D-003
  - 修改：新建 `experiments/toy_multiclass.py`、`toy_multi_input.py`、`toy_multilabel.py`、`toy_regression.py`、`toy_custom_task.py` 及完整 base YAML，固定 seed 本地造数。
  - 约束：只用新 API，不访问 Kaggle/AList/微信，不复制旧 LOB 模型；覆盖 Tensor/mapping/tuple/dynamic/custom Task。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_torch_examples.py`；预期每例两 epoch 内指标有限且 Artifact/report 完整。

- [x] 8.2 创建 sklearn batch/incremental/Pipeline 示例
  - 依据：`sklearn 数据与 estimator 合同`、`Kaggle sklearn 增量执行`、D-014、REF-008
  - 修改：新建 `experiments/sklearn_batch.py`、`sklearn_pipeline.py`、`sklearn_incremental.py` 和 base/variant YAML，使用固定本地 ndarray 与 scipy sparse fixture。
  - 约束：项目不下载数据或可选 booster；sample weight 参数显式；batch 只 Local，incremental 可 Kaggle；预处理只在 Pipeline。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_sklearn_examples.py`；预期三示例训练、joblib、指标定义和报告通过。

- [x] 8.3 创建显式数据路径的 MNIST Kaggle 示例
  - 依据：`通用 PyTorch DataModule 合同`、`固定 revision 的 Kaggle 启动`、D-003、D-012
  - 修改：新建 `experiments/mnist.py` 与 Kaggle base config，仅读取显式已挂载数据路径；Notebook 以 Kaggle Dataset 提供数据。
  - 约束：DataModule 构造/测试不联网下载；缺数据失败；核心不解释 MNIST metadata。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_mnist_contract.py`；预期临时 fixture 完成训练，缺路径明确失败。

- [x] 8.4 创建同一实验多 variant 与 sweep 示例
  - 依据：`严格 sweep manifest 与 trial 派生`、`科学排名与 best trial`、D-017
  - 修改：新建 `configs/sweeps/toy-learning-rate/sweep.yaml`、完整 base 和至少三个只含 patch 的 variant，文档引用同一 Experiment 而不复制训练脚本。
  - 约束：comparison 仅 exact/full `val/` 指标；seed/data/Task/定义固定；trial tuning fingerprint 唯一。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_example_sweep.py`；预期顺序完成、稳定 best、聚合报告和 manifest 通过。

## 9. Complete Legacy Removal And Security

- [x] 9.1 删除旧 Python 训练体系和历史实验
  - 依据：`单一新平台与旧 API 移除`、`旧训练体系完整移除`、D-001、D-013、REF-001
  - 修改：替换 `dl_helper/__init__.py`；删除现有 `dl_helper/` 下除新 training/与 `__init__.py` 外全部旧文件，包括 trainer/tester/tracker/train_param/data/models/transforms/tests/other_tests/rl/AutoGluon/压缩/索引/传输；新建 repository 负向测试。
  - 约束：不得保留 shim、重导出、`__getattr__`、改名领域模型或修改 Git 历史；仅删除固定受跟踪项目文件。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/repository/test_removed_python_surface.py`；预期新 import 成功且所有旧 import/受跟踪 surface 失败或为零。

- [x] 9.2 删除旧运行资产、启动器和二进制
  - 依据：`旧训练体系完整移除`、D-001、D-013、REF-001
  - 修改：删除旧 `cpp/`、`参考/`、`envs/`、`notebook/`、setup.py、requirements.txt、受跟踪 wheel/tarball/checkpoint；随后仅保留新 bootstrap/Notebook；新建 tracked inventory 测试。
  - 约束：不删除 `.git`、OpenSpec、Agent/编辑器配置、用户目录、Kaggle Dataset、AList 或外部 runs；不得以 gitignore 隐藏。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/repository/test_tracked_inventory.py`；预期 git ls-files 无禁止路径、旧元数据和二进制。

- [x] 9.3 建立 Secret 扫描、轮换门禁与全链路脱敏测试
  - 依据：`Secret 与凭证安全`、`服务失败策略与审计`、D-013、D-016、REF-001
  - 修改：新建 `tools/scan_secrets.py`、`tests/security/test_no_tracked_secrets.py`、`test_secret_redaction.py` 和发布轮换检查项；覆盖源码/Notebook/YAML/TOML/Markdown、HTTP/joblib/traceback/通知/audit。
  - 约束：扫描规则按 design 精确豁免，不按整文件跳过；不得写出旧凭证值；Apply Agent 不自动重写 Git 历史或调用外部轮换 API。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/security`；预期仓库零明文、所有 error channel 脱敏且轮换确认项存在。

- [x] 9.4 审计隔离 wheel、依赖与 import 删除面
  - 依据：`依赖与导入边界`、`旧训练体系完整移除`、D-001、D-012、D-013
  - 修改：新建 `tests/repository/test_wheel_contents.py` 与 `tools/verify_clean_install.py`，构建 wheel、检查 RECORD/METADATA，并在临时 venv/仓库外 cwd 以 no-deps 探测新旧 import。
  - 约束：wheel 只含新包、版本 1.0.0、无 alist/legacy extra/旧依赖；清空 PYTHONPATH，避免 editable/source 污染；临时目录位于 pytest tmp。
  - 验证：`& 'D:/programs/miniconda3/python.exe' tools/verify_clean_install.py dist`；预期 wheel 内容、metadata、新 API 和全部旧 import 负向探测通过。

## 10. Verification, CI And Documentation

- [x] 10.1 建立两进程梯度与指标归约门禁
  - 依据：`精确的 PyTorch 优化循环`、`全量流式与分布式指标`、D-002、D-015
  - 修改：新建 `tests/distributed/test_gloo_training.py`，用两进程 gloo 覆盖不等尾批、浮点 sample weight、accumulation、weighted moment 和单进程金标。
  - 约束：不需要 CUDA/网络/Secret；spawn 入口兼容 Windows/Linux；超时明确失败。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/distributed/test_gloo_training.py`；预期梯度与单批金标一致，指标误差不超过 `1e-6`，step 完全一致。

- [x] 10.2 建立双后端端到端恢复矩阵
  - 依据：`backend-aware 可审计恢复`、`自动化科学正确性门禁`、D-007、D-014
  - 修改：新建 `tests/integration/test_end_to_end_matrix.py`，参数化 Torch 三任务/自定义/无 val/epoch/mid-epoch 和 sklearn batch/Pipeline/sparse/incremental/resume。
  - 约束：固定小数据、最多两 epoch；断言模型/step/指标定义/checksum/report，不只断言无异常；不加载不可信 joblib。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_end_to_end_matrix.py`；预期全部轨迹和连续/恢复一致性通过。

- [x] 10.3 建立服务与 sweep 故障注入矩阵
  - 依据：`服务失败策略与审计`、`sweep 暂停与严格恢复`、D-016、D-017
  - 修改：新建 `tests/integration/test_failure_injection_matrix.py`，在 AList archive/manifest/latest、微信 token/send、trial 0/75/error、finalization 各边界注入中断并恢复。
  - 约束：不访问真实网络；每例断言 terminal 互斥、audit、primary error、幂等 event/bundle 和无后续 trial。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/integration/test_failure_injection_matrix.py`；预期 required/record、暂停、失败和重入矩阵全部通过。

- [x] 10.4 建立 Windows/Linux CPU CI 与独立覆盖率门禁
  - 依据：`自动化科学正确性门禁`、D-012
  - 修改：新建 `.github/workflows/training-core.yml` 和 `tools/check_coverage.py`，矩阵 Windows/Ubuntu Python 3.10，执行 Secret scan、完整 pytest、branch coverage、wheel 审计。
  - 约束：不使用 GPU/外网/真实 Secret；training line>=85%、branch>=75% 分开计算，任一不足失败；coverage JSON UTF-8 读取。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/ci/test_workflow.py tests/ci/test_coverage_gate.py`；预期 workflow 结构与两个阈值用例通过。

- [x] 10.5 编写公共 API、指标、服务、sweep、Kaggle 与破坏性迁移文档
  - 依据：全部 requirements、D-001、D-014 至 D-017、REF-008、REF-009
  - 修改：新建 `README.md`、`docs/training/{configuration,metrics,custom-task,sklearn,services,sweeps,kaggle,artifacts,breaking-removal}.md`；使用真实命令/示例并记录版本 1.0.0、旧版 commit 和无兼容迁移。
  - 约束：不得包含真实 host/Secret/token、未实现选项或旧凭证值；明确 batch sklearn/Kaggle、sample-weight、sampled/full、joblib trust 和 test 不参与 sweep 排名边界。
  - 验证：`& 'D:/programs/miniconda3/python.exe' -m pytest -q tests/docs/test_documented_contracts.py tests/security/test_no_tracked_secrets.py`；预期路径/CLI/schema/formula/服务字段存在且文档无 Secret。

- [x] 10.6 建立并执行完整本地发布门禁
  - 依据：全部 requirements、D-001 至 D-017
  - 修改：新建 `tools/verify_release.py`，以 subprocess argv 顺序运行 Secret scan、完整 pytest+coverage、coverage 双阈值、wheel build、clean install 和严格 OpenSpec；任一步失败立即抛出 CalledProcessError。
  - 约束：脚本使用 `sys.executable`、所有文本 subprocess `encoding='utf-8'`，不使用 shell；不得跳过测试、降阈值或依赖外网。
  - 验证：`& 'D:/programs/miniconda3/python.exe' tools/verify_release.py`；预期所有本地门禁返回 0，且 OpenSpec strict 无错误。

- [ ] 10.7 执行 Kaggle 真机发布门禁并保存证据
  - 依据：`固定 revision 的 Kaggle 启动`、`运行预算与可恢复退出`、`通用 ArtifactStore 生命周期`、`固定企业微信生命周期通知`、D-002、D-006、D-012、D-016、REF-005、REF-009
  - 修改：使用候选 40 位 commit 在真实 Kaggle 运行模板：Torch toy/MNIST 使用全部可见 GPU并触发一次 PREEMPTED/AList 跨会话恢复；另运行 sklearn incremental CPU smoke；下载不含 Secret 的 doctor、manifest、audit 和 HTML 作为发布工件。
  - 约束：不得用本地模拟替代、不得浮动 revision或提交大运行产物；凭证仅 Kaggle Secrets；仓库所有者同时确认历史 AList/企业微信/其他已识别凭证已轮换。
  - 验证：Notebook 运行 `python -m dl_helper.training.cli doctor`、`train`/`sweep`；预期 doctor=0、首次暂停=75、恢复完成=0、全部 checksum/指标定义/服务审计有效、报告可离线打开。
