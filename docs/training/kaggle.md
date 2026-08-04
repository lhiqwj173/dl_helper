# Kaggle

- 任一环境键以 `KAGGLE` 开头即识别 Kaggle。
- 输入必须显式 `/kaggle/input/...`；输出默认 `/kaggle/working/dl-helper-runs`。
- `num_processes=auto` 使用全部可见 CUDA；CPU 为 1。
- `doctor --profile kaggle` 检查固定 revision、预算、磁盘、Secret 与服务。
- 预算 `max_minutes`/`shutdown_grace_minutes` 必填且 grace<max；命中后停止新 step、保存检查点、flush 服务、写 pause manifest 并以 75 退出。
- 模板要求 40 位 commit SHA，clone/checkout/HEAD 校验/`pip install -e . --no-deps`/doctor。

## 发布检查表（任务 10.7，需真机执行）

候选 revision 发布前，仓库所有者须在真实 Kaggle 完成：

1. `python -m dl_helper.training.cli doctor` → 退出码 0。
2. Torch toy/MNIST：使用全部可见 GPU，触发一次 PREEMPTED（预算 75）与 AList/检查点跨会话恢复 → 恢复后完成退出码 0。
3. sklearn incremental CPU smoke → doctor 0、train 0。
4. 下载不含 Secret 的 doctor/manifest/audit/HTML 作为发布工件；校验全部 checksum 与指标定义。
5. 确认历史 AList/企业微信/其他已识别凭证已轮换（外部动作，Apply Agent 不自动执行）。
6. 固定 40 位 commit SHA 由模板写入 `DL_HELPER_GIT_REF`，不允许浮动 revision。
