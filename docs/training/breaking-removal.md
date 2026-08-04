# 破坏性迁移

版本 `1.0.0` 完整替换旧训练体系：

- 新公共 API 只位于 `dl_helper.training`；旧 `dl_helper.trainer/tester/tracker/train_param/data/scheduler/tool/models/rl/tests/transforms` 全部删除。
- 旧实验配置、`Params` 字典、7z/AList 缓存、旧 checkpoint 不受支持。
- 删除范围：`cpp/`、`参考/`、旧 `envs/`/`notebook/`（仅保留 `envs/kaggle_bootstrap.py` 与 `notebook/kaggle_training_template.ipynb`）、`setup.py`、`requirements.txt`、受跟踪 wheel/tarball/checkpoint。
- 不提供 shim/重导出/`__getattr__`/弃用代理/legacy extra。
- 不删除 `.git`、OpenSpec、Agent/编辑器配置、用户目录、Kaggle Dataset、AList 或外部 runs。
- 回滚整体恢复到旧版 commit；不转换/覆盖新 Artifact。
