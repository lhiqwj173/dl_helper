
from py_ext.tool import safe_share_memory_queue
import multiprocessing
import time, random

def worker(num):
    """线程函数"""
    print(f'进程 {num}: 开始')
    # 共享队列
    queue = safe_share_memory_queue("test_queue", size=1024, nums=10)
    queue.clear()

    while True:
        rand_int = random.randint(1, 100)
        if_modify = random.randint(0, 1)

        if if_modify and len(queue) < queue.nums * 0.8:
            queue.put(str(rand_int).encode())
            print(f"进程 {num}: 放入数据 {rand_int}, \t\t{len(queue)} / {queue.nums}")
        else:
            if not queue.is_empty():
                print(f"进程 {num}: 获取数据 {queue.get()}, \t\t{len(queue)} / {queue.nums}")

        time.sleep(random.random() * 2)

    print(f'进程 {num}: 结束')

if __name__ == '__main__':
    processes = []

    # 创建三个进程
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("所有进程都完成了！")
