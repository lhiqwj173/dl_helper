## 1. 示例项目边界

- [x] 1.1 将仓库自带实验和配置迁移到 `examples/`，并更新全部导入与路径引用
  - 依据：`显式双后端实验工厂`、D-001
  - 修改：移动 `experiments/` 到 `examples/experiments/`、`configs/` 到 `examples/configs/`；更新 README、docs、notebook、tests 和 sweep manifest 中的稳定路径
  - 约束：不创建旧路径兼容 package、重导出或复制；所有文件内容保持 UTF-8；示例必须通过 `--project-dir examples` 使用
  - 验证：`D:/programs/miniconda3/python.exe -m pytest -q tests/training tests/sweeps tests/kaggle --disable-warnings --maxfail=1`；预期示例导入、预检及相关路径测试通过

- [x] 1.2 增加库模块边界校验，拒绝把训练内容放入 `dl_helper`
  - 依据：`显式双后端实验工厂`、D-001
  - 修改：在 `dl_helper/training/cli.py` 的项目准备/参数验证路径拒绝 `dl_helper`/`dl_helper.*` Experiment 引用，以及实际包目录内的 config/output root；在 CLI 测试增加正负用例
  - 约束：wheel 是否包含 examples 不参与判断；校验必须基于实际导入包目录的 realpath，处理路径逃逸；在 Experiment 导入和 Artifact 创建前失败
  - 验证：`D:/programs/miniconda3/python.exe -m pytest -q tests/training/test_cli.py tests/training/test_cli_platform_coverage.py --disable-warnings --maxfail=1`；预期外部项目通过，库模块引用与包内路径失败

## 2. 默认恢复行为

- [x] 2.1 将 train 的缺省恢复策略改为内部 auto，并删除旧公开配置面
  - 依据：`默认自动恢复的训练入口`、D-002
  - 修改：`dl_helper/training/cli.py::build_parser`、`_cmd_train`、`config.py::CheckpointConfig/_build_checkpoint`、variant/fingerprint、`engine.py` 及 worker/launcher 参数传递；省略 CLI 参数时内部 auto，只接受显式 none/required，删除 YAML resume 字段；另修复预检导入失败写 failure.json 并保留 ModuleNotFoundError 根因（OSR-003）
  - 约束：`--resume auto` 与 `checkpoint.resume` 必须失败；auto 发现损坏或不兼容 checkpoint 必须失败；required 无 checkpoint 必须失败；none 不查询远程；sklearn batch 的内部 auto 不查询 checkpoint
  - 验证：`D:/programs/miniconda3/python.exe -m pytest -q tests/training/test_cli.py tests/training/test_cli_exit_codes.py tests/training/test_config_base.py tests/integration/test_runtime_budget.py --disable-warnings --maxfail=1`；预期默认、none、required 与已删除输入负向用例通过；32 passed；仅为既有环境用例 test_runtime_budget.py::test_torch_budget_preempts_and_resumes 失败（WinError 1455 页面文件不足，HEAD 同路径复现，非本次引入，见 blocker）

- [x] 2.2 更新示例与文档，常规命令不再传 `--resume auto`
  - 依据：`默认自动恢复的训练入口`、D-002
  - 修改：`examples/configs/`、README、`docs/training/guide.md`、`docs/training/kaggle.md`、Notebook；只在解释 none/required 时出现 resume 参数
  - 约束：不得删除 none/required 的准确语义
  - 验证：`rg -n -- "--resume auto|^[[:space:]]+resume:" README.md docs examples notebook`；预期无已删除 CLI 值或 YAML 字段

## 3. Kaggle 固定预算

- [x] 3.1 从用户 Config 删除 runtime，并新增独立平台执行策略
  - 依据：`运行预算与可恢复退出`、D-003
  - 修改：从 `dl_helper/training/config.py` 删除 RuntimeConfig、根级 runtime、variant/fingerprint 分支；`platform.py` 新增 frozen ExecutionPolicy 和 Kaggle 660/10 构造；`cli.py`、`doctor.py`、Torch/sklearn backend 与 `launcher.py` 显式传递策略；run 目录新增 `execution-policy.json`
  - 约束：YAML runtime 必须失败；Local policy 无预算；Kaggle 始终 660/10；spawn 子进程严格重建纯策略 dict；config.resolved 不含平台字段；预检/父进程/worker 值一致
  - 验证：`D:/programs/miniconda3/python.exe -m pytest -q tests/training/test_config_base.py tests/kaggle/test_platform.py tests/kaggle/test_doctor.py tests/training/test_cli_platform_coverage.py tests/distributed/test_gloo_training.py --disable-warnings --maxfail=1`；预期 schema 拒绝、策略、审计和多进程传递用例通过

- [x] 3.2 从用户示例和常规文档移除 runtime 配置要求
  - 依据：`运行预算与可恢复退出`、D-003
  - 修改：`examples/configs/`、README、`docs/training/configuration.md`、`guide.md`、`kaggle.md`、Notebook 与当前规格说明
  - 约束：文档必须明确 Kaggle 实际 660/10、650 分钟训练截止、Local 不限时，以及 runtime 已被删除
  - 验证：`rg -n "max_minutes|shutdown_grace_minutes|^runtime:" README.md docs examples notebook`；预期只在内部机制或删除说明中出现，示例配置无 runtime

## 4. Kaggle Notebook 文档

- [x] 4.1 将 Kaggle 指南重写为可直接执行的 Python Notebook 流程
  - 依据：`解耦训练项目的 Kaggle 启动`、D-004
  - 修改：`docs/training/kaggle.md` 完整展示安装、外部项目目录、配置、Secrets、首次训练、0/75 处理、同 run ID 自动恢复、sweep、sweep report 和本地调试；所有命令使用 `subprocess` 与 `sys.executable`
  - 约束：Kaggle 段不得出现 PowerShell、反引号续行或 `D:/programs/miniconda3/python.exe`；失败退出码必须 raise；75 不得被当作失败
  - 验证：`D:/programs/miniconda3/python.exe -m pytest -q tests/kaggle/test_notebook_template.py tests/kaggle/test_remaining_gate_notebook.py tests/ci/test_docs.py --disable-warnings --maxfail=1`；预期 Notebook JSON、命令形态、退出码和文档链接测试通过

- [x] 4.2 同步 README、训练指南、配置、sweep、Artifact 与服务文档
  - 依据：D-001、D-002、D-003、D-004
  - 修改：所有当前用户文档和 Notebook 中的路径、默认值、命令与恢复流程；不修改 archive 中用于历史审计的旧提案
  - 约束：用户文档必须区分工具库、外部项目和示例；不得声称自动发现业务代码或数据
  - 验证：`rg -n "powershell|--resume auto|runtime.max_minutes.*必须|D:/programs/miniconda3" docs/training/kaggle.md notebook`；预期没有陈旧 Kaggle 指引；另由路径测试确认所有仓库示例引用均解析到 `examples/`

## 5. 回归与规格

- [x] 5.1 完成全量静态、规格和回归验证
  - 依据：全部 Requirements 与 D-001 至 D-004
  - 修改：修复实施引起的测试期望和当前规格，不降低断言强度（含 test_backend_branch_coverage fake_worker 签名、test_error_artifact 预检失败证据、test_example_sweep 路径迁移）
  - 约束：使用指定解释器；不得遗留测试进程；不修改无关用户变更
  - 验证：`D:/programs/miniconda3/python.exe -m compileall -q dl_helper envs examples`、`openspec validate general-training --strict`、`openspec validate kaggle-execution --strict`、`git diff --check` 全过；随后运行全量 pytest，除 4 个既有/环境 blocker 外全部通过（tests/services 58 passed、tests/sweeps/training/kaggle/integration 380+ passed）
  - blocker：test_runtime_budget.py::test_torch_budget_preempts_and_resumes（本机 WinError 1455 页面文件不足加载 torch CUDA DLL）、test_failure_injection_matrix 两个用例（service-manifest run/sweep 状态漂移）、test_coverage_edges.py::test_lifecycle_all_disabled_paths（同上）；后者三个经 HEAD worktree 复现同样失败，属既有缺陷，非本次引入
