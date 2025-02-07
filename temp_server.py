import asyncio
import struct
import socket

from dl_helper.rl.socket_base import tune_tcp_socket, async_send_msg, async_recv_msg, ack

# 模拟数据
param_data = b'x' * 8724
grad_data = b'x' * 544454

buffer_size = 4*1024*1024# 4 MB 缓冲区

class BandwidthServer:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.buffer = None

        # 设置socket选项
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tune_tcp_socket(self.sock)
        
    async def handle_client(self, reader, writer):
        try:
            # 接收数据
            data = await async_recv_msg(reader)
            print(f"Received data: {len(data)} bytes")

            # 发送确认
            await ack(writer)
            print(f"Sent ack")
            
        except Exception as e:
            print(f"Error handling client: {e}")

        finally:
            writer.close()
            await writer.wait_closed()
            
    async def start(self):
        self.sock.bind((self.host, self.port))
        
        # 使用start_server创建服务器
        self.server = await asyncio.start_server(
            self.handle_client,
            sock=self.sock,
        )

        print(f"Server started on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    server = BandwidthServer()
    asyncio.run(server.start())