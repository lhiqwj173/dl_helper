# 发布可一次解压的成果 ZIP

risk: HIGH

## Why

现有 AList run/sweep 成果以 `.tar.gz` 发布。Windows 用户通常先得到 `.tar`，再解压一次才能查看文件。训练检查点也使用同一归档函数，但检查点恢复依赖现有 `.tar.gz` 路径与安全解压逻辑。

## What Changes

目标是让新发布的 run/sweep 成果分别成为 `run-bundle.zip`、`sweep-bundle.zip`，可直接一次解压；保留归档成员过滤、不可变内容校验、上传回读 SHA 和终态发布顺序。检查点仍使用 `archive.tar.gz`，旧远端成果不迁移也不删除。

## Impact

范围：`dl_helper.training.remote` 的成果归档和发布、相关服务测试与仓库规格。当前未跟踪的下载 Notebook 属于用户工作文件，不纳入本次提交；它固定搜索旧文件名，使用前需要独立更新。

用户可见变化：新运行在 AList 上生成 ZIP 成果包；旧运行的 `.tar.gz` 继续留在原路径。回滚可恢复旧版本发布器，不改变既有检查点和历史对象。
