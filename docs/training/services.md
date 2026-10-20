# 服务

- **AList**：`remote.type=alist`，显式 HTTP(S) host/base_path 与 Secret key（Kaggle 可使用 `http://IP`）；发布顺序 immutable archive → 回读 SHA → manifest → latest；401/403 不重试，连接/5xx 按 2/4/8 秒有限重试。
- **企业微信**：host 固定 `https://qyapi.weixin.qq.com`；token 单调时钟缓存，失效只刷新重放一次；`errcode=0` 才成功；text content ≤2048 UTF-8 bytes。
- **启动通知**：首次训练发送 `[训练开始]`；Torch 和 sklearn 增量训练只有成功加载检查点后才发送 `[恢复训练]`，包含 run、checkpoint、epoch、step。`resume=auto` 无检查点仍属于首次训练；恢复失败仅走失败通知。多进程训练等待所有 rank 加载完成，由父进程发送一次通知，并在通知确认后开始训练。
- **恢复去重**：每次恢复使用独立的通知 scope；同一次恢复内重复调用不会重复通知，新 Session 即使再次加载同一检查点也会通知。暂停、成功、失败通知沿用该 scope，避免旧 Session 的通知记录压掉本次终态；兼容原有检查点和配置指纹。
- **failure_policy**：`required` 失败阻止 success/pause 终态；`record` 失败写入 degraded 且继续；两种策略都写 service audit。
- 任何 secondary 服务异常不覆盖原训练异常（`failure.json` 保留 primary）。
