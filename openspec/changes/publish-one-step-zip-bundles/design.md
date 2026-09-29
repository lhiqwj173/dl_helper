# 设计

## 当前路径

`AListArtifactStore.publish_run_bundle/publish_sweep_bundle` 调用 `_publish_tar_gz`；`publish_checkpoint` 与 `fetch_latest_checkpoint` 也使用 TAR/GZIP。服务终结通过目录内容计算 `bundle_checksum` 并复核 marker。

## 决策

- D-001：只迁移 run/sweep 成果发布路径。检查点上传与恢复继续使用 `archive.tar.gz`，避免改变断点恢复协议。
- D-002：ZIP 使用标准库 `zipfile`、DEFLATE level 1；文件名规范化为 `/`，按稳定顺序写入，固定 ZIP 时间为 1980-01-01，拒绝符号链接与非普通文件。归档内容与既有 bundle 过滤规则一致。
- D-003：上传与 raw 回读 SHA 仍通过既有 AList 调用链，返回的 `bundle_checksum` 仍由目录内容计算；终态 marker 不因容器格式变化而改变。
- D-004：旧 `.tar.gz` 不自动删除或重发；新 run/sweep 只发布 `.zip`。未跟踪的下载 Notebook 不纳入提交，以免覆盖用户未提交的工作。

## 验证

使用服务单元测试验证 ZIP 可直接读取、过滤、确定性、危险成员拒绝和远端命名；使用检查点上传与恢复测试确认旧格式继续生效。不得访问真实 AList。
