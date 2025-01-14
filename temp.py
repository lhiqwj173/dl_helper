import asyncio
import time
import random
import multiprocessing
from py_ext.tool import share_ndarray, share_ndarray_list
import numpy as np

def cpu_bound_task(lock, event_queue):
    # 模拟 CPU 密集型任务
    print("cpu 开始")

    # 共享任务
    wait_task = share_ndarray_list('wait_task', (1,), 'int64', 20)
    # 共享结果
    results = share_ndarray_list('results', (1,), 'int64', 20)

    # 临时任务 大小与共享数据相同
    temp_task_idx = 0
    temp_task = wait_task.get_blank_same_data_local()
    # 临时结果 大小与共享数据相同
    temp_result_idx = 0
    temp_result = results.get_blank_same_data_local()

    while True:
        with lock:
            if wait_task.size() == 0:
                # 没有新任务，等待
                try:
                    event = event_queue.get(timeout=0.1)
                    if event == 'stop':
                        return
                except:
                    continue

            # 全部拷贝到temp任务, 并清空原任务 > 转移数据
            task_size = wait_task.size()
            wait_task.all_copy_slice(temp_task, temp_task_idx)
            temp_task_idx += task_size

        print(f"cpu 获取 {temp_task_idx} 个任务")
        for idx in range(temp_task_idx):
            print(f"cpu 开始任务 {temp_task[idx]}")
            sleep_time = random.randint(1, 3)
            time.sleep(sleep_time)
            result = sleep_time * temp_task[idx]
            print(f"cpu 完成任务 {temp_task[idx]} > {result}")
            temp_result[temp_result_idx] = result
            temp_result_idx += 1
        print(f"cpu 获取任务全部完成")
        with lock:
            # 将临时结果拷贝到结果队列, 并清空临时结果
            for i in range(temp_result_idx):
                results.append(temp_result[i])
        # 全部处理完成，清理临时数据
        temp_result_idx = 0
        temp_task_idx = 0

async def main():
    # 事件队列
    event_queue = multiprocessing.Queue()

    # 共享任务
    wait_task = share_ndarray_list('wait_task', (1,), 'int64', 20)
    wait_task.reset()
    # 共享结果
    results = share_ndarray_list('results', (1,), 'int64', 20)
    results.reset()

    lock = multiprocessing.Lock()
    p = multiprocessing.Process(target=cpu_bound_task, args=(lock,event_queue))
    p.start()

    for i in range(10):
        with lock:
            wait_task.append(i)
            print(f"main 发送任务 {i}, 任务列表: {wait_task.size()}, 结果列表: {results.size()}")
        await asyncio.sleep(1.5)

    # 完成后结束
    event_queue.put('stop')

    final_result = []
    _temp_result = results.get_blank_same_data_local()
    _temp_result_idx = 0
    while len(final_result) < 10:
        with lock:
            if results.size() == 0:
                # 没有结果，等待
                time.sleep(0.1)
                continue
            # 获取全部结果
            _size = results.size()
            results.all_copy_slice(_temp_result, _temp_result_idx)

        _temp_result_idx += _size
        print(f"main 获取结果 {_size} 个, 任务列表: {wait_task.size()}, 结果列表: {results.size()}")
        # 合并结果
        for i in range(_size):
            final_result.append(_temp_result[i])

    print(f"final_result: {final_result}")
    p.join()  # 等待进程结束

if __name__ == "__main__":
    asyncio.run(main())