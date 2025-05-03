import tracemalloc, os

# os.environ['PYTHONTRACEMALLOC'] = '30'

# 启动内存追踪，设置堆栈深度为 30 层（关键参数：nframe=30）
tracemalloc.start(30)

# 定义一个多层嵌套调用的函数
def allocate_memory(depth):
    if depth <= 0:
        # 在最后一层分配内存（这里创建一个大列表）
        data = [i for i in range(10000)]
        return data
    else:
        # 递归调用，模拟多层堆栈
        return allocate_memory(depth - 1)

# 第一次快照（基准）
snapshot_1 = tracemalloc.take_snapshot()

# 触发一个深层调用（这里模拟 10 层调用）
result = allocate_memory(10)
del result  # 释放内存，避免干扰后续测试

# 第二次快照（记录内存分配）
snapshot_2 = tracemalloc.take_snapshot()

# 对比快照，获取内存分配统计
stats = snapshot_2.compare_to(snapshot_1, "lineno")
# stats = snapshot_2.compare_to(snapshot_1, "traceback")

# 输出前 3 条内存分配记录及其堆栈
for i, stat in enumerate(stats[:3]):
    print(f"\n[{i}] 内存分配大小: {stat.size / 1024:.2f} KB")
    print("堆栈跟踪（从内到外）:")
    # 格式化工整的堆栈信息
    for line in stat.traceback.format():
        print(f"  {line}")
    print("-" * 60)