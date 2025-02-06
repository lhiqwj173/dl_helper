# -*- coding: utf-8 -*-
try:
    from py_ext.tool import log, get_exception_msg
except:
    log = print

import asyncio
import socket, time, os, re
import pickle
import struct

CODE = '0QYg9Ky17dWnN4eK'
# PS IP
HOST = '217.142.135.154'
PORT = 12346
GRAD_BATCH_SIZE = 4

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
    """接收带长度前缀的消息"""
    # 接收4字节的长度前缀
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    # 解析消息长度
    msglen = struct.unpack('>I', raw_msglen)[0]
    # 接收消息内容
    return recvall(sock, msglen)

def send_msg(sock, msg):
    """发送带长度前缀的消息"""
    # 添加4字节的长度前缀
    msg = msg.encode() if isinstance(msg, str) else msg
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

async def async_recvall_0(reader, n, timeout=10.0):
    """异步地从流中读取指定字节数的数据
    timeout: 超时时间，-1表示不设置超时
    返回值: 读取到的数据
    """
    # log(f"开始接收 {n} 字节数据...")  # 添加日志
    data = bytearray()
    while len(data) < n:
        try:
            if timeout == -1:
                packet = await reader.read(n - len(data))
            else:
                packet = await asyncio.wait_for(
                    reader.read(n - len(data)),
                    timeout=timeout  # 添加超时设置
                )
            if not packet:
                log("Connection closed, received empty packet")
                raise Exception('connect closed unexpectedly')

            # log(f"Received packet, size: {len(packet)} bytes")  # Add log
            data.extend(packet)
        except asyncio.TimeoutError:
            log("Receive data timeout")
            raise Exception('recv timeout')
        except Exception as e:
            raise e
    # log(f"成功接收完整数据，总大小: {len(data)} 字节")  # 添加日志
    return data

async def async_recvall(reader, n, timeout=10.0, buffer_size=1024*64):  # 增加buffer_size参数
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

async def async_recv_msg(reader, timeout=-1):
    """异步地接收带长度前缀的消息
    timeout: 超时时间，-1表示不设置超时
    返回值: 读取到的数据
    """
    # log("开始接收消息长度前缀...")  # 添加日志
    # 接收4字节的长度前缀
    raw_msglen = await async_recvall(reader, 4, timeout)
    # 解析消息长度
    msglen = struct.unpack('>I', raw_msglen)[0]
    # log(f"消息长度前缀: {msglen} 字节")  # 添加日志
    # 接收消息内容
    res = await async_recvall(reader, msglen, timeout)
    return res

async def async_send_msg_0(writer, msg):
    """异步地发送带长度前缀的消息"""
    msg = msg.encode() if isinstance(msg, str) else msg
    msg = struct.pack('>I', len(msg)) + msg
    writer.write(msg)
    await writer.drain()

async def async_send_msg(writer, msg, chunk_size=1024*64):
    """异步地分块发送大消息"""
    if isinstance(msg, str):
        msg = msg.encode()
        
    # 发送长度前缀
    length_prefix = struct.pack('>I', len(msg))
    writer.write(length_prefix)
    
    # 分块发送消息体
    for i in range(0, len(msg), chunk_size):
        chunk = msg[i:i + chunk_size]
        writer.write(chunk)
        
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
        send_msg(_socket, f'{CODE}')
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

async def _async_wait_server_weights(reader):
    # 接收参数数据
    response = await async_recv_msg(reader)
    return _handle_response_params(response)

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

