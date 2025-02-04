import os
# 设置 NCCL 超时和其他 NCCL 参数
os.environ["NCCL_TIMEOUT"] = "300"