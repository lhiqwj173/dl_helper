# Kaggle

- 任一环境键以 `KAGGLE` 开头即识别 Kaggle。
- 输入必须显式 `/kaggle/input/...`；输出默认 `/kaggle/working/dl-helper-runs`。
- `num_processes=auto` 使用全部可见 CUDA；CPU 为 1。
- `doctor --profile kaggle` 检查固定 revision、预算、磁盘、Secret 与服务。
- 预算 `max_minutes`/`shutdown_grace_minutes` 必填且 grace<max；命中后停止新 step、保存检查点、flush 服务、写 pause manifest 并以 75 退出。
- 模板要求 40 位 commit SHA，clone/checkout/HEAD 校验/`pip install -e . --no-deps`/doctor。
