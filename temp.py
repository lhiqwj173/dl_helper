import multiprocessing
import asyncio
import random
import time
from multiprocessing.queues import Queue

from dl_helper.tool import AsyncProcessQueueReader

async def worker(name, queue):
    """工作协程"""
    while True:
        task = await queue.get()
        try:
            # 模拟异步网络操作
            await asyncio.sleep(random.uniform(0.5, 2))
            print(f"Coroutine {name}: Completed {task}")
        except Exception as e:
            print(f"Coroutine {name} error: {e}")
        finally:
            queue.task_done()

def task_producer(task_queue):
    """任务生产者进程"""
    task_id = 0
    while True:
        time.sleep(random.uniform(0.1, 0.5))
        task = f"Task-{task_id}"
        task_queue.put(task)
        print(f"Producer: Generated {task}")
        task_id += 1

async def process_queue_consumer(mp_queue: Queue, worker_count: int = 10):
    """处理进程队列的消费者"""
    # 创建队列读取器
    reader = AsyncProcessQueueReader(mp_queue)
    
    # 创建worker协程
    workers = [asyncio.create_task(worker(f"Worker-{i}", reader.async_queue)) 
              for i in range(worker_count)]
    
    await asyncio.gather(*workers)

def process_tasks(task_queue):
    """任务处理进程"""
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(process_queue_consumer(task_queue))
    finally:
        loop.close()

if __name__ == "__main__":
    task_queue = multiprocessing.Queue()
    
    producer = multiprocessing.Process(target=task_producer, args=(task_queue,))
    producer.daemon = True
    producer.start()
    
    consumer = multiprocessing.Process(target=process_tasks, args=(task_queue,))
    consumer.daemon = True
    consumer.start()
    
    try:
        producer.join()
        consumer.join()
    except KeyboardInterrupt:
        print("Shutting down...")