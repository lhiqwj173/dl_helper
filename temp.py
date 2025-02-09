from dl_helper.rl.socket_base import connect_and_tune, CODE, async_send_msg
import asyncio

ip = '1.1.1.1'
async def main():
    reader, writer = await connect_and_tune('217.142.135.154', 12346)
    print(f'grad_coroutine connect to server done')
    # 发送连接验证
    await async_send_msg(writer, f'{CODE}_{ip}')
    print(f'grad_coroutine send CODE_IP done')
    # 发送指令类型
    await async_send_msg(writer, f'test:update_gradients')
    print(f'grad_coroutine send CMD done')


if __name__ == '__main__':
    asyncio.run(main())

