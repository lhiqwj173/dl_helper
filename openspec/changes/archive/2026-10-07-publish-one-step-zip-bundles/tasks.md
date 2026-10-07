## Execution Contract

protocol: OPENSPEC_EXECUTION
contract_revision: 1
risk: HIGH

objective:
- 新 run/sweep AList 成果为可一次解压的 ZIP，checkpoint 恢复保持可用。

scope_summary:
- 修改成果归档格式、服务测试和 training-services 规格。

non_goals:
- 不迁移或删除历史远端对象，不改 checkpoint 格式，不修改用户未跟踪的下载 Notebook。

allowed_paths:
- dl_helper/training/remote.py
- tests/services/test_remote_store_coverage.py
- tests/services/test_alist_store.py
- openspec/changes/publish-one-step-zip-bundles/**

conditional_paths:
- path: openspec/specs/training-services/spec.md
  only_if: 用户要求归档规格合并。

forbidden:
- 不访问真实 AList 或外部账户。
- 不修改用户未跟踪文件。

global_invariants:
- checkpoint archive.tar.gz 上传、下载和安全解压路径不变。
- bundle checksum、终态发布顺序与上传回读 SHA 不变。

escalate_if:
- 规格与代码事实冲突。
- 需要更改 checkpoint 协议或用户未跟踪文件。
- 必要验证无法执行。

validation_plan:
  task:
  - 运行对应服务测试文件。
  apply_final:
  - `D:/programs/miniconda3/python.exe -m pytest tests/services/test_remote_store_coverage.py tests/services/test_alist_store.py -q`
  review:
  - 检查 ZIP 成员、远端名称、checkpoint 兼容与 Git diff。
  archive:
  - 校验 OpenSpec，提交并推送。

convergence_policy:
  max_failed_fast_repairs_per_finding: 2

- [x] 1.1 在 `remote.py` 新增确定性 ZIP 成果归档与发布路径，保留 checkpoint TAR/GZIP；以服务测试验证。
  - 依据：通用 ArtifactStore 生命周期；D-001 至 D-003。
  - 读取：`_archive_relative_files`、`_make_tar_gz`、`_publish_tar_gz`、`publish_run_bundle`、`publish_sweep_bundle`。
  - 修改：仅 `remote.py` 的成果归档与调用。
  - 约束：检查点路径和目录 checksum 不变。
  - 验证：服务测试中的归档与发布用例通过。

- [x] 1.2 更新服务测试断言，覆盖 ZIP 内容、过滤、确定性和远端名称，同时保留检查点恢复测试。
  - 依据：通用 ArtifactStore 生命周期；D-001 至 D-004。
  - 读取：`tests/services/test_remote_store_coverage.py`、`tests/services/test_alist_store.py`。
  - 修改：仅相关测试。
  - 约束：不访问真实 AList。
  - 验证：`D:/programs/miniconda3/python.exe -m pytest tests/services/test_remote_store_coverage.py tests/services/test_alist_store.py -q`。
