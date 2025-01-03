# -*- coding: utf-8 -*-
from py_ext.tool import log

import socket, time, os, re
import pickle
import struct

CODE = '0QYg9Ky17dWnN4eK'
HOST = '146.235.33.108'
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
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def _connect_server_apply(func, *args, **kwargs):
    """连接到参数中心服务器, 并应用函数
    func: 函数名, 第一个参数为socket连接
    *args: 函数其他参数
    **kwargs: 函数其他参数
    """
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _socket.connect((HOST, PORT))
        return func(_socket, *args, **kwargs)
    except Exception as e:
        log(f"连接服务器失败")
        raise e
    finally:
        _socket.close()

def get_net_params(train_title):
    """获取net参数"""
    def _get_net_params(_socket):
        # 发送获取参数请求
        message = f'{CODE}_{train_title}:get'
        send_msg(_socket, message.encode())
        
        # 接收参数数据
        response = recv_msg(_socket)
        if response is None:
            raise Exception('Failed to receive parameters')
            
        # 反序列化参数
        net_params, version = pickle.loads(response)
        return net_params, version

    return _connect_server_apply(_get_net_params)

def check_need_val_test(train_title):
    """查询是否需要验证测试
    返回: 'val' / 'test' / 'no'
    """
    def _check_need_val_test(_socket):
        # 发送查询请求
        message = f'{CODE}_{train_title}:check'
        send_msg(_socket, message.encode())
        
        # 接收响应
        response = recv_msg(_socket)
        if response:
            return response.decode()
        return 'no'

    return _connect_server_apply(_check_need_val_test)

def send_accumulated_gradients(train_title, grads, importance, version, metrics):
    """发送累积的梯度"""
    def _send_accumulated_gradients(_socket):
        # 发送累积梯度请求
        message = f'{CODE}_{train_title}:update_gradients'
        send_msg(_socket, message.encode())
        
        # 发送累积梯度
        data = pickle.dumps((grads, importance, version, metrics))
        send_msg(_socket, data)
        
        # 等待确认
        response = recv_msg(_socket)
        return response == b'ok'

    return _connect_server_apply(_send_accumulated_gradients)

def send_val_test_data(train_title, data_type, metrics):
    """发送训练数据"""
    def _send_val_test_data(_socket):
        # 发送训练数据请求
        message = f'{CODE}_{train_title}:{data_type}'
        send_msg(_socket, message.encode())
        
        # 发送训练数据
        data = pickle.dumps(metrics)
        send_msg(_socket, data)
        
        # 等待确认
        response = recv_msg(_socket)
        return response == b'ok'

    return _connect_server_apply(_send_val_test_data)

def request_client_id(train_title):
    """请求分配客户端id"""
    def _request_client_id(_socket):
        # 发送请求分配id
        message = f'{CODE}_{train_title}:request_id'
        send_msg(_socket, message.encode())
        
        # 接收分配的id
        response = recv_msg(_socket)
        if response:
            return int(response.decode())
        return -1

    return _connect_server_apply(_request_client_id)


