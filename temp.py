import tracemalloc

a = []
def func_to_test():
    # 你的可疑代码
    for i in range(5):
        a.append([])

if __name__ == "__main__":
    tracemalloc.start(10)        # 记录最近 10 层调用栈
    snap1 = tracemalloc.take_snapshot()

    func_to_test()               # 执行测试代码

    snap2 = tracemalloc.take_snapshot()
    top_stats = snap2.compare_to(snap1, 'lineno')

    print("[ 内存增长 排名前10 ]")
    for stat in top_stats[:10]:
        print(stat)
