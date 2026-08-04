# References

## REF-001 当前仓库训练链路

- 来源：仓库提交 `62ee1e4bbf42065ec07dbd0fc8d5b4f9b642f7fd`，访问日期 `2026-08-01`。
- 关键文件：`dl_helper/trainer.py`、`dl_helper/tester.py`、`dl_helper/tracker.py`、`dl_helper/train_param.py`、`notebook/train.ipynb`、`envs/dl.py`。
- 已核验事实：训练入口使用 Accelerate；正常结束无条件执行金融预测输出和 AList 打包；异常路径存在 `os._exit(0)`；Tracker 保存全量预测；Notebook 下载浮动 `master` 并含明文 Secret。旧微信调用依赖 `D:/code/py_ext/py_ext/wechat.py` 的进程级单例，且该外部模块硬编码企业身份信息；旧值不得在提案、实现、测试或文档中复述，相关凭证必须在发布前轮换。Git 跟踪 370 个 `dl_helper` 文件，其中 `dl_helper/tests` 271 个、`models` 15 个、`rl` 59 个；旧资产还包含 17 个 Notebook、10 个 env 文件、vendored wheel/tarball 和一个约 40 MiB checkpoint。
- 用途：D-001、D-002、D-006、D-010、D-013、D-016；全部 capability、固定删除边界、服务替换和仓库审计任务。

## REF-002 本地受支持运行时基线

- 来源：`D:/programs/miniconda3/python.exe` 的 `importlib.metadata` 和运行时签名，核验日期 `2026-08-01`。
- 版本：Python `3.10.16`、PyTorch `2.7.0+cu126`、Accelerate `1.6.0`、TorchMetrics `1.7.1`、scikit-learn `1.6.1`、joblib `1.4.2`、SciPy `1.15.2`、requests `2.34.2`、Matplotlib `3.10.1`、Pandas `2.2.3`、NumPy `2.2.4`、pytest `8.4.1`、safetensors `0.8.0`、PyYAML `6.0.3`。
- 已核验接口：`Accelerator.prepare`、`Accelerator.accumulate`、`Accelerator.gather_for_metrics`、`Accelerator.register_for_checkpointing(*objects)`、`Accelerator.save_state`、`Accelerator.load_state`、`notebook_launcher`；`DataLoader` 的 `prefetch_factor/persistent_workers/pin_memory`；`torch.amp`、`torch.use_deterministic_algorithms` 和 `torch.compile`。
- 用途：D-002、D-005、D-007、D-012；依赖、训练循环、启动器和 CI 任务。

## REF-003 Hugging Face Accelerate 1.6 官方文档

- 来源：
  - `https://huggingface.co/docs/accelerate/v1.6.0/en/package_reference/accelerator`
  - `https://huggingface.co/docs/accelerate/v1.6.0/en/usage_guides/checkpoint`
  - `https://huggingface.co/docs/accelerate/v1.6.0/en/package_reference/launchers`
- 版本：Accelerate `1.6.0`；本次环境对网页请求超时，接口行为已使用 REF-002 的已安装同版本 docstring 和签名核验。
- 关键约束：对象在训练前统一 `prepare`；梯度累积使用 `accumulate`；模型、优化器、scaler、RNG 和注册对象通过 `save_state/load_state` 保存恢复；多进程由 `notebook_launcher` 启动。
- 用途：D-002、D-007、D-008；`general-training` 和 `kaggle-execution`。

## REF-004 PyTorch 2.7 官方文档

- 来源：
  - `https://docs.pytorch.org/docs/2.7/data.html`
  - `https://docs.pytorch.org/docs/2.7/amp.html`
  - `https://docs.pytorch.org/docs/2.7/notes/randomness.html`
  - `https://docs.pytorch.org/docs/2.7/generated/torch.compile.html`
- 版本：PyTorch `2.7`；本次环境对网页请求超时，相关函数和参数已使用 REF-002 的本地同版本对象核验。
- 关键约束：DataLoader 可以产生任意嵌套批次；AMP 使用 `torch.amp`/Accelerate；严格确定性通过 deterministic algorithms 和显式 seed 控制；`torch.compile` 必须是显式可选优化而不是兼容回退。
- 用途：D-003、D-005、D-012；训练、性能和复现任务。

## REF-005 Kaggle 官方 Python 镜像仓库

- 来源：`https://github.com/Kaggle/docker-python`，目标分支 `main`，记录日期 `2026-08-01`。
- 仓库内运行事实：现有代码使用 `/kaggle/input` 作为输入、`/kaggle/working` 作为工作目录，通过环境变量检测 Kaggle，并面向 P100/T4x2 编写资源分支。
- 采用约束：新实现不锁定 GPU 型号或猜测平台配额；使用 PyTorch 可见设备发现资源；Kaggle profile 要求显式运行时预算、run ID、输入路径和固定 source revision；平台预装包通过 doctor 核验而不是在 Notebook 中静默升级。
- 用途：D-006、D-009、D-012；`kaggle-execution` 和 Kaggle 测试任务。

## REF-006 当前 AList 客户端协议

- 来源：`py-ext 1.0.0`，仓库提交 `9111706377df9b49f237e1258839ce2bf7717e2e`，文件 `D:/code/py_ext/py_ext/alist.py`，核验日期 `2026-08-01`。工作树仅 `py_ext/wechat.py` 有无关本地修改，引用的 `alist.py` 未修改。
- 已核验接口：登录 `/api/auth/login`，目录 `/api/fs/list`，信息 `/api/fs/get`，上传 `/api/fs/put`，建目录 `/api/fs/mkdir`；现有客户端没有请求超时且使用进程级单例。
- 采用约束：新平台不直接复用该单例；在 `dl_helper.training.remote` 实现带超时、重试、校验和和发布顺序的新客户端；host 必须配置，用户名和密码只来自 Secret resolver。
- 用途：D-009、D-011、D-016；远程同步、安全、run/sweep 发布和恢复任务。

## REF-007 指标语义金标

- 来源：scikit-learn `1.6.1` 已安装实现与 API，以及 `https://scikit-learn.org/1.6/modules/model_evaluation.html`，核验日期 `2026-08-02`；仓库当前 `dl_helper/tracker.py` 仅作为已知错误基线。
- 已核验函数：`confusion_matrix`、`accuracy_score`、`balanced_accuracy_score`、`precision_recall_fscore_support`、`multilabel_confusion_matrix`、`hamming_loss`、`mean_absolute_error`、`mean_squared_error`、`r2_score(force_finite=True)`；均支持 design 所用的显式 labels/average/zero_division/sample_weight/multioutput 参数组合。
- 采用约束：金标调用不得依赖默认 labels、average、zero_division 或 multioutput；分类/多标签按 float64 sample weight 评价，常量目标按 force_finite 语义，回归大偏置另与直接 float64 两遍算法核验。公式语义变化必须增加 formula_version。
- 用途：D-004、D-015；`training-reporting`、指标实现与科学正确性门禁。

## REF-008 scikit-learn estimator 与持久化合同

- 来源：
  - `https://scikit-learn.org/1.6/developers/develop.html`
  - `https://scikit-learn.org/1.6/model_persistence.html`
- 版本：scikit-learn `1.6.1`，本地 joblib `1.4.2`；核验日期 `2026-08-02`。
- 已核验接口：`sklearn.base.clone(estimator, safe=True)`、`check_is_fitted`、Pipeline `fit`、`SGDClassifier.partial_fit/classes`、RandomForestClassifier `fit/predict_proba`、estimator `get_params/set_params/n_jobs/random_state`。
- 采用约束：普通 estimator/Pipeline 走 batch fit，明确提供 partial_fit 的 estimator 走 incremental；第三方 wrapper 必须满足相同 estimator/Task capability。pickle/joblib 加载可能执行任意代码，且官方不支持跨不同 sklearn 依赖版本加载，因此只加载当前 run 自产、manifest/checksum/runtime exact-match 的可信 joblib，任何外部模型在反序列化前拒绝。
- 用途：D-014、D-015；`general-training`、`training-reliability`、sklearn backend/checkpoint/doctor/tasks。

## REF-009 企业微信官方 API

- 来源：
  - 获取 access token：`https://developer.work.weixin.qq.com/document/path/91039`
  - 发送应用消息：`https://developer.work.weixin.qq.com/document/path/90236`
- 版本：企业微信当前官方开发者文档，访问日期 `2026-08-02`。
- 已核验合同：token API 使用 `corpid/corpsecret` 并返回 `errcode/access_token/expires_in`；应用消息 API 使用 `access_token`、`touser`、`agentid`、`msgtype=text`、`text.content`，只有 `errcode=0` 表示成功。官方明确 text content 最长 2048 字节并要求 UTF-8，超过会截断。
- 采用约束：客户端固定官方 qyapi host，不允许配置覆盖；平台在发送前自行按 UTF-8 bytes 验证/裁剪，不能依赖服务端静默截断。token 只保存在进程内并提前刷新，Secret 与 access_token 不落盘；token 失效只刷新重放一次，其他业务错误遵循有限重试分类。
- 用途：D-016；`training-services`、notifications、配置、审计、安全和服务测试任务。
