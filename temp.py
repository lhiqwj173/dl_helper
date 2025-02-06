import asyncio
import struct
import socket

def tune_tcp_socket(sock, buffer_size):
    """TCP协议调优"""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 重用地址
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 禁用Nagle算法
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)     # 动态发送缓冲区
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)     # 动态接收缓冲区
    # 开启TCP快速打开（需要内核支持）
    try:
        sock.setsockopt(socket.SOL_TCP, socket.TCP_FASTOPEN, 5)
    except (AttributeError, OSError):
        print("TCP_FASTOPEN not supported, skipping...")
    # 使用BBR拥塞控制算法（需要内核支持）
    try:
        sock.setsockopt(socket.SOL_TCP, socket.TCP_CONGESTION, b'bbr')
    except (AttributeError, OSError):
        print("BBR not supported, using default congestion control...")

async def connect_and_tune(HOST, PORT, buffer_size):
    """创建异步连接并调优TCP参数"""
    reader, writer = await asyncio.open_connection(HOST, PORT)
    
    # 获取底层socket对象
    sock = writer.transport.get_extra_info('socket')
    if sock is not None:
        print("Tuning TCP socket...")
        tune_tcp_socket(sock, buffer_size)
    else:
        print("Warning: Could not get underlying socket for tuning")
    
    return reader, writer

class BandwidthServer:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.buffer = None
        
    async def handle_client(self, reader, writer):
        try:
            size_bytes = await reader.readexactly(4)
            size_mb = struct.unpack('!I', size_bytes)[0]
            total_bytes = size_mb * 1024 * 1024
            
            print(f"Client requested {size_mb}MB of data")

            buffer_size = 4*1024*1024# 4 MB 缓冲区
            self.buffer = b'x' * buffer_size
            
            # 获取socket并优化设置
            sock = writer.transport.get_extra_info('socket')
            tune_tcp_socket(sock, buffer_size)
            
            sent = 0
            # 减少drain的频率
            drain_threshold = buffer_size * 8
            accumulated = 0
            
            while sent < total_bytes:
                remaining = total_bytes - sent
                chunk_size = min(len(self.buffer), remaining)
                writer.write(self.buffer[:chunk_size])
                
                sent += chunk_size
                accumulated += chunk_size
                
                if accumulated >= drain_threshold:
                    await writer.drain()
                    accumulated = 0
            
            if accumulated > 0:
                await writer.drain()
                
            print(f"Finished sending {size_mb}MB")
            
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            
    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, 
            self.host, 
            self.port,
            backlog=4096  # 支持更多并发连接
        )
        print(f"Server started on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    server = BandwidthServer()
    asyncio.run(server.start())