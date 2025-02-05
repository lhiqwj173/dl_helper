from dl_helper.tool import AsyncProcessEventReader
from multiprocessing import Event, Process
import time, asyncio, random

async def event_waiter(waiter_id: int, forwarder):
    """事件等待者"""
    while True:
        print(f"Waiter {waiter_id} waiting for event...")
        await forwarder.wait()
        print(f"Waiter {waiter_id} active")

def worker_process(event):
    """工作进程：定期触发事件"""
    count = 0
    print(f"Worker process started")
    while count < 5:  # 发送5次事件后退出
        time.sleep(random.randint(2, 5))
        print(f"Process setting event")
        event.set()
        event.clear()
        count += 1


async def main():
    # 创建进程事件
    process_event = Event()
    
    # 创建事件转发器
    forwarder = AsyncProcessEventReader(process_event)
    
    # 创建并启动工作进程
    process = Process(target=worker_process, args=(process_event,))
    process.start()
    print(f"Worker process started")
    
    # 创建三个等待者任务
    waiters = [
        asyncio.create_task(event_waiter(i, forwarder))
        for i in range(3)
    ]
    
    # 等待所有任务完成
    await asyncio.gather(*waiters)
    await asyncio.sleep(60)
    
    # 停止并清理
    process.terminate()
    process.join()
    forwarder.stop()

if __name__ == "__main__":
    # 运行主协程
    asyncio.run(main())