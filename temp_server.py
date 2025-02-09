import asyncio
import struct
import socket

from dl_helper.rl.socket_base import tune_tcp_socket, async_send_msg, async_recv_msg, ack

# 模拟数据
param_data = b'x' * 8724
grad_data = b'x' * 544454

buffer_size = 4*1024*1024# 4 MB 缓冲区
# buffer_size = 1024*1024# 1 MB 缓冲区
# buffer_size = 512*1024# 512 KB 缓冲区
# buffer_size = 256*1024# 256 KB 缓冲区

class BandwidthServer:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host

        self.port = port
        self.buffer = None

        self.connected_clients = {}

        # 设置socket选项
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tune_tcp_socket(self.sock, buffer_size)
        
    async def handle_client(self, reader, writer):
        need_close = True
        try:
            # 接收 ip_idx
            ip_idx = await async_recv_msg(reader)
            ip, idx = ip_idx.decode().split('_')
            idx = int(idx)
            if ip not in self.connected_clients:
                self.connected_clients[ip] = {}
            self.connected_clients[ip][idx] = (reader, writer)
            if idx > 0:
                # 非第一个连接，只储存不关闭，由第一个连接负责接收数据/关闭连接
                need_close = False
                return

            while True:
                # idx = 0
                # 接收数据
                nums = len(self.connected_clients[ip])
                if nums > 1:
                    tasks = []
                    for _idx in range(nums):
                        tasks.append(async_recv_msg(self.connected_clients[ip][_idx][0]))

                    data = await asyncio.gather(*tasks)
                    data = b''.join(data)
                else:
                    data = await async_recv_msg(reader)

                print(f"Received data: {len(data)} bytes")

                # 发送确认
                await ack(writer)
                print(f"Sent ack")
            
        except Exception as e:
            print(f"Error handling client: {e}")

        finally:
            if need_close:
                # 遍历 self.connected_clients[ip], 关闭所有连接
                for idx, (reader, writer) in self.connected_clients[ip].items():
                    writer.close()
                    await writer.wait_closed()
                del self.connected_clients[ip]

    async def start(self):
        self.sock.bind((self.host, self.port))
        
        # 使用start_server创建服务器
        self.server = await asyncio.start_server(
            self.handle_client,
            sock=self.sock,
        )

        print(f"Server started on {self.host}:{self.port}")
        async with self.server:
            await self.server.serve_forever()

if __name__ == "__main__":
    server = BandwidthServer()
    asyncio.run(server.start())