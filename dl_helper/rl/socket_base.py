# -*- coding: utf-8 -*-
try:
    from py_ext.tool import log
except:
    log = print

import asyncio
import socket, time, os, re
import pickle
import struct

CODE = '0QYg9Ky17dWnN4eK'
HOST = '132.226.234.60'
PORT = 12346

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

async def async_recvall(reader, n, timeout=10.0):
    """异步地从流中读取指定字节数的数据
    timeout: 超时时间，-1表示不设置超时
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
                # log("连接已关闭，接收到空数据包")
                return None
            # log(f"接收到数据包，大小: {len(packet)} 字节")  # 添加日志
            data.extend(packet)
        except asyncio.TimeoutError:
            # log("接收数据超时")
            return None
        except Exception as e:
            # log(f"接收数据时发生错误: {str(e)}")
            return None
    # log(f"成功接收完整数据，总大小: {len(data)} 字节")  # 添加日志
    return data

async def async_recv_msg(reader, timeout=-1):
    """异步地接收带长度前缀的消息
    timeout: 超时时间，-1表示不设置超时
    """
    # log("开始接收消息长度前缀...")  # 添加日志
    # 接收4字节的长度前缀
    raw_msglen = await async_recvall(reader, 4, timeout)
    if not raw_msglen:
        # log("未能接收到消息长度前缀")
        return None
    # 解析消息长度
    msglen = struct.unpack('>I', raw_msglen)[0]
    # log(f"消息长度前缀: {msglen} 字节")  # 添加日志
    # 接收消息内容
    return await async_recvall(reader, msglen, timeout)

async def async_send_msg(writer, msg):
    """异步地发送带长度前缀的消息"""
    msg = msg.encode() if isinstance(msg, str) else msg
    msg = struct.pack('>I', len(msg)) + msg
    writer.write(msg)
    await writer.drain()

def _connect_server_apply(func, *args, **kwargs):
    """连接到参数中心服务器, 并应用函数
    func: 函数名, 第一个参数为socket连接
    *args: 函数其他参数
    **kwargs: 函数其他参数
    _type: 连接类型 1次/长连接
    """
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _socket.connect((HOST, PORT))

        # 发送连接类型
        send_msg(_socket, f'{CODE}_one')
        
        return func(_socket, *args, **kwargs)
    except Exception as e:
        log(f"连接服务器失败")
        raise e
    finally:
        _socket.close()

def _get_server_weights(_socket, train_title, version):
    # 发送获取参数请求
    message = f'{train_title}:get@{version}'
    send_msg(_socket, message.encode())
    
    # 接收参数数据
    response = recv_msg(_socket)
    if response is None:
        raise Exception('Failed to receive parameters')
        
    # 反序列化参数
    try:
        weights, info, version = pickle.loads(response)
    except Exception as e:
        print(f"反序列化失败: {e}")
        with open('debug_pickle.pkl', 'wb') as f:
            f.write(response)
        print(f'已保存到 debug_pickle.pkl')
        raise e
    return weights, info, version

def get_server_weights(train_title, version=-1):
    """
    获取参数服务器权重
    server返回: (self.learner.get_state(components=COMPONENT_RL_MODULE), self.ver)
    """
    return _connect_server_apply(_get_server_weights, train_title, version)

def _send_gradients(_socket, train_title, grads, compress_info, version):
    # 发送梯度请求
    message = f'{train_title}:update_gradients'
    send_msg(_socket, message.encode())
    
    # 发送累积梯度
    data = pickle.dumps((grads, compress_info, version))
    send_msg(_socket, data)

async def _async_send_gradients(writer, train_title, grads, compress_info, version):
    # 发送梯度请求
    message = f'{train_title}:update_gradients'
    await async_send_msg(writer, message.encode())
    
    # 发送累积梯度
    data = pickle.dumps((grads, compress_info, version))
    await async_send_msg(writer, data)

def send_gradients(train_title, grads, compress_info, version):
    """发送梯度"""
    return _connect_server_apply(_send_gradients, train_title, grads, compress_info, version)

def _request_client_id(_socket, train_title):
    # 发送请求分配id
    message = f'{CODE}_{train_title}:client_id'
    send_msg(_socket, message.encode())

    # 接收分配的id
    response = recv_msg(_socket)
    if response:
        return int(response.decode())
    return -1

def request_client_id(train_title):
    """请求分配客户端id"""
    return _connect_server_apply(_request_client_id, train_title)


def test_pickle_numpy():
    """测试pickle序列化numpy数组"""
    import numpy as np
    
    # 创建numpy数组
    arr = np.random.rand(3,4)
    print("原始数组:")
    print(arr)
    
    # 序列化
    data = pickle.dumps(arr)
    print("\n序列化后的bytes长度:", len(data))
    
    # 反序列化
    arr2 = pickle.loads(data)
    print("\n还原后的数组:")
    print(arr2)
    
    # 验证是否完全相同
    print("\n两个数组是否完全相同:", np.array_equal(arr, arr2))


if "__main__" == __name__:

    file = r"C:\Users\lh\Desktop\fsdownload\debug_pickle.pkl"
    with open(file, 'rb') as f:
        data = f.read()
    response = pickle.loads(data)
    print(response)

    while 1:
        choose = input("1.get_server_weights\n2.request_client_id\n3.test_pickle_numpy\n")
        if choose == "1":
            print(get_server_weights('test'))
        elif choose == "2":
            print(request_client_id('test'))
        elif choose == "3":
            test_pickle_numpy()
        else:
            print("invalid input")
