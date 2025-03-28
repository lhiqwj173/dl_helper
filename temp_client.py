%%writefile client.py
import asyncio
import struct
import time
import socket, requests

# 模拟数据
# ack = b'1'
param_data = b'x' * 8724
grad_data = b'x' * 544454
grad_data_m2 = b'x' * 272227

CHUNK_SIZE = 4*1024*1024# 4 MB 缓冲区
# CHUNK_SIZE = 1024*1024# 1 MB 缓冲区
# CHUNK_SIZE = 512*1024# 512 KB 缓冲区
# CHUNK_SIZE = 256*1024# 256 KB 缓冲区
CHUNK_SIZE = 8 * 1024 * 1024

def tune_tcp_socket(sock, buffer_size=CHUNK_SIZE):
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

async def connect_and_tune(HOST, PORT, buffer_size=CHUNK_SIZE):
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

async def async_recvall(reader, n, timeout=10.0, buffer_size=CHUNK_SIZE):  # 增加buffer_size参数
    """异步地从流中读取指定字节数的数据，使用更大的缓冲区提高吞吐量"""
    data = bytearray(n)  # 预分配内存
    view = memoryview(data)  # 使用memoryview避免内存拷贝
    pos = 0
    
    while pos < n:
        try:
            if timeout == -1:
                # 一次性请求尽可能多的数据
                chunk = await reader.read(min(buffer_size, n - pos))
            else:
                chunk = await asyncio.wait_for(
                    reader.read(min(buffer_size, n - pos)),
                    timeout=timeout
                )
            if not chunk:
                raise ConnectionError('Connection closed unexpectedly')
            
            view[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
            
        except asyncio.TimeoutError:
            raise TimeoutError('Receive timeout')
        except Exception as e:
            raise e
            
    return data

async def async_recv_data_length(reader, timeout=-1):
    """读取变长头部"""
    # 读取一个字节
    byte = await async_recvall(reader, 1, timeout)
    
    # 解析已收到的头部字节
    if byte[0] & 0x80 == 0:  # 1字节头
        return byte[0], 1

    elif byte[0] & 0xC0 == 0x80:  # 4字节头
        # 接收剩下的长度
        byte2 = await async_recvall(reader, 3, timeout)
        header_buff = byte + byte2
        return struct.unpack('>I', header_buff)[0] & 0x7FFFFFFF, 4

async def async_recv_msg(reader, timeout=-1):
    """终极优化版消息接收"""
    body_len, header_len = await async_recv_data_length(reader, timeout)
    print(f'recv msg({header_len}) length')
    t = time.time()
    # 接收消息内容
    res = await async_recvall(reader, body_len, timeout)
    cost_time = time.time() - t
    print(f'recv msg({body_len}), cost: {int(1000*cost_time)}ms, speed: {(body_len / cost_time) / (1024*1024):.3f}MB/s')
    return res

async def async_send_msg(writer, msg, chunk_size=CHUNK_SIZE):
    """终极优化版消息发送"""
    # 类型自动转换
    data = msg.encode() if isinstance(msg, str) else msg
    
    # 变长编码长度前缀
    length = len(data)
    if length < 0x80:
        header = struct.pack('B', length)
    else:
        header = struct.pack('>I', length | 0x80000000)
    
    # 创建内存视图
    payload = memoryview(header + data)
    
    # 分块发送
    for offset in range(0, len(payload), chunk_size):
        writer.write(payload[offset:offset+chunk_size].tobytes())
    
    await writer.drain()

async def wait_ack(reader):
    """等待确认消息"""
    # await reader.read(1)
    try:
        chunk = await reader.read(1)
        if not chunk:
            raise ConnectionError('Connection closed unexpectedly')
        
    except asyncio.TimeoutError:
        raise TimeoutError('Receive timeout')
    except Exception as e:
        raise e

async def ack(writer):
    """发送确认消息"""
    writer.write(b'1')

class BandwidthClient:
    def __init__(self, host='217.142.135.154', port=12345):
        self.host = host
        self.port = port
        self.connected_clients = []
        self.ip = requests.get('https://api.ipify.org').text
        
    async def test_data(self):
        nums = 1
        try:
            # 连接, 0 是最后一个（主连接）
            for idx in range(nums-1, -1, -1):
                reader, writer = await connect_and_tune(
                    self.host, 
                    self.port
                )
                # 发送 ip_idx
                await async_send_msg(writer, f"{self.ip}_{idx}")
                self.connected_clients.append((reader, writer))

            total_time = 0
            for i in range(30):
                print(f"send {i} times")
                t = time.time()

                # # 发送数据
                # data = grad_data
                # if nums > 1:
                #     tasks = []
                #     each_length = len(data) // nums
                #     for idx in range(nums):
                #         start = idx * each_length
                #         end = (idx + 1) * each_length if idx < nums - 1 else len(data)
                #         tasks.append(async_send_msg(self.connected_clients[idx][1], data[start:end]))
                #     await asyncio.gather(*tasks)
                # else:
                #     await async_send_msg(writer, data)

                # 发送 ack
                await ack(writer)

                # 等待回复
                await wait_ack(reader)
                total_time += time.time() - t
                
            # 关闭连接
            writer.close()
            await writer.wait_closed()

            # 4MB 
            # grad_data 1 avg time: 1554ms
            # grad_data 1 avg time: 1463ms 
            # grad_data 2 avg time: 775ms
            # ack avg time: 187ms
            print(f"avg time: {int(1000* total_time / 30)}ms")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    client = BandwidthClient()
    asyncio.run(client.test_data())