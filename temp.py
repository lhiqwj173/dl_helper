import asyncio, time

async def my_coroutine():
    await asyncio.sleep(1)

async def my_coroutine2():
    time.sleep(1)

async def main():
    # 模拟协程调用
    await asyncio.gather(my_coroutine(), my_coroutine2())

if __name__ == "__main__":
    asyncio.run(main(), debug=True)
