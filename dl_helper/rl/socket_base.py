# -*- coding: utf-8 -*-
try:
    from py_ext.tool import log, get_exception_msg
except:
    log = print

import asyncio
import socket, time, os, re
import pickle, requests
import struct

CODE = '0QYg9Ky17dWnN4eK'
# PS IP
HOST = '217.142.135.154'
PORT = 12346
CHUNK_SIZE = 8 * 1024 * 1024
# 参数推送间隔
PUSH_INTERVAL = 4

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
        log("TCP_FASTOPEN not supported, skipping...")
    # 使用BBR拥塞控制算法（需要内核支持）
    try:
        sock.setsockopt(socket.SOL_TCP, socket.TCP_CONGESTION, b'bbr')
    except (AttributeError, OSError):
        log("BBR not supported, using default congestion control...")

async def connect_and_tune(HOST, PORT, buffer_size=CHUNK_SIZE):
    """创建异步连接并调优TCP参数"""
    reader, writer = await asyncio.open_connection(HOST, PORT)
    log(f'connected done')

    # 获取底层socket对象
    sock = writer.transport.get_extra_info('socket')
    if sock is not None:
        log("Tuning TCP socket...")
        tune_tcp_socket(sock, buffer_size)
    else:
        log("Warning: Could not get underlying socket for tuning")
        
    log(f'tune done')
    return reader, writer

async def ack(writer):
    """发送确认消息"""
    writer.write(b'1')

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

def recvall(sock, n):
    """接收指定字节数的数据"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    """接收变长编码的消息"""
    # 读取第一个字节
    byte = sock.recv(1)
    if not byte:
        return None

    # 解析头部长度
    if byte[0] & 0x80 == 0:  # 1字节头
        msglen = byte[0]
        header_len = 1
    elif byte[0] & 0xC0 == 0x80:  # 4字节头
        # 接收剩下的长度
        byte2 = recvall(sock, 3)
        if not byte2:

            return None
        header_buff = byte + byte2
        msglen = struct.unpack('>I', header_buff)[0] & 0x7FFFFFFF
        header_len = 4
    t = time.time()
    log(f'recv msg({header_len}) length')
    # 接收消息内容
    data = recvall(sock, msglen)
    cost_time = time.time() - t
    log(f'recv msg({msglen}), cost: {int(1000*cost_time)}ms, speed: {(msglen / cost_time) / (1024*1024):.3f}MB/s')
    return data

def send_msg(sock, msg):
    """发送变长编码的消息"""
    msg = msg.encode() if isinstance(msg, str) else msg
    msg_len = len(msg)
    
    # 使用变长编码
    if msg_len < 0x80:
        # 1字节头
        header = struct.pack('B', msg_len)
    else:
        # 4字节头
        header = struct.pack('>I', msg_len | 0x80000000)
        
    sock.sendall(header + msg)

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
    log(f'recv msg({header_len}) length')
    t = time.time()
    # 接收消息内容
    res = await async_recvall(reader, body_len, timeout)
    cost_time = time.time() - t
    log(f'recv msg({body_len}), cost: {int(1000*cost_time)}ms, speed: {(body_len / cost_time) / (1024*1024):.3f}MB/s')
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

def _connect_server_apply(func, *args, **kwargs):
    """连接到参数中心服务器, 并应用函数
    func: 函数名, 第一个参数为socket连接
    *args: 函数其他参数
    **kwargs: 函数其他参数
    """
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _socket.connect((HOST, PORT))

        # 发送验证
        _ip = requests.get('https://api.ipify.org').text
        send_msg(_socket, f'{CODE}_{_ip}')
        return func(_socket, *args, **kwargs)
    except Exception as e:
        log(f"Failed to connect to server")

        log(get_exception_msg())
        raise e
    finally:

        _socket.close()

def _handle_response_params(response):
    if response is None:
        raise Exception('Failed to receive parameters')
    # 反序列化参数
    try:
        weights, info, version, need_warn_up = pickle.loads(response)
    except Exception as e:
        log(f"Failed to deserialize parameters: {e}")
        raise e
    return weights, info, version, need_warn_up


def _get_server_weights(_socket, train_title, version):
    # 发送获取参数请求
    message = f'{train_title}:get@{version}'
    send_msg(_socket, message.encode())
    
    # 接收参数数据
    response = recv_msg(_socket)
    if response is None:
        raise Exception('Failed to receive parameters')
        
    # 反序列化参数
    return _handle_response_params(response)

async def _async_wait_server_weights(reader, timeout=-1, loads=True):
    # 接收参数数据
    response = await async_recv_msg(reader, timeout)
    if loads:
        return _handle_response_params(response)
    else:
        return response

def get_server_weights(train_title, version=-1):
    """
    获取参数服务器权重
    server返回: (self.learner.get_state(components=COMPONENT_RL_MODULE), self.ver)
    """
    return _connect_server_apply(_get_server_weights, train_title, version)

def _request_need_val(_socket, train_title):
    # 发送请求是否需要验证
    message = f'{train_title}:need_val'
    send_msg(_socket, message.encode())

    # 接收是否需要验证
    response = recv_msg(_socket)
    if response:
        return response.decode() == '1'
    return False

def request_need_val(train_title):
    """请求是否需要验证"""
    return _connect_server_apply(_request_need_val, train_title)

