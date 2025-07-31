import tracemalloc

# 启用 tracemalloc
tracemalloc.start(25)

# 存储快照的列表
snapshots = []

# 示例循环（模拟内存泄漏）
leaked_objects = []
def leaky_loop():
    for _ in range(1000):
        # 模拟内存泄漏：将对象添加到全局列表，阻止垃圾回收
        leaked_objects.append([1] * 1000)
        # 在每次迭代后保存快照
        snapshots.append(tracemalloc.take_snapshot())

# 运行循环
leaky_loop()

# 比较快照，检测内存泄漏
snapshot1 = snapshots[0]  # 第一次迭代的快照
snapshot2 = snapshots[-1]  # 最后一次迭代的快照

# 获取内存分配差异
# stats = snapshot2.compare_to(snapshot1, 'lineno')
stats = snapshot2.compare_to(snapshot1, 'traceback')

# 打印前几个差异最大的内存分配
print("内存分配差异：")
for stat in stats[:5]:
    print(stat)
    # 打印调用堆栈以定位泄漏来源
    print("\n".join(stat.traceback.format()))

# 停止 tracemalloc
tracemalloc.stop()