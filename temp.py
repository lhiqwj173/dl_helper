import asyncio, random

async def worker(event, name):
    while True:
        print(f"{name} 等待事件...")
        await event.wait()  # 等待事件触发
        print(f"{name} 事件触发，开始执行")
        await asyncio.sleep(random.randint(0, 5))  # 模拟任务执行

async def trigger(event):
    while True:
        await asyncio.sleep(random.randint(0, 5))  # 模拟任务执行
        print("触发事件")
        event.set()  # 触发所有 worker
        event.clear()  # 只有一个 clear()，防止后续任务立即通过

async def main():
    event = asyncio.Event()
    workers = [worker(event, f"任务 {i+1}") for i in range(3)]
    await asyncio.gather(trigger(event), *workers)

asyncio.run(main())
