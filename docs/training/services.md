# 服务

- **AList**：`remote.type=alist`，显式 HTTP(S) host/base_path 与 Secret key（Kaggle 可使用 `http://IP`）；发布顺序 immutable archive → 回读 SHA → manifest → latest；401/403 不重试，连接/5xx 按 2/4/8 秒有限重试。
- **企业微信**：host 固定 `https://qyapi.weixin.qq.com`；token 单调时钟缓存，失效只刷新重放一次；`errcode=0` 才成功；text content ≤2048 UTF-8 bytes。
- **failure_policy**：`required` 失败阻止 success/pause 终态；`record` 失败写入 degraded 且继续；两种策略都写 service audit。
- 任何 secondary 服务异常不覆盖原训练异常（`failure.json` 保留 primary）。
