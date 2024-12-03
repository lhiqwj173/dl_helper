# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import socket, time, sys, os, re
import pickle
import struct

CODE = '0QYg9Ky17dWnN4eK'
PORT = 12346

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

def recvall(sock, n):
    """接收指定字节数的数据"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def send_msg(sock, msg):
    """发送带长度前缀的消息"""
    # 添加4字节的长度前缀
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def get_net_params():
    """获取net参数"""
    HOST = '146.235.33.108'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        # 发送获取参数请求
        message = f'{CODE}_get'
        send_msg(client_socket, message.encode())
        
        # 接收参数数据
        response = recv_msg(client_socket)
        if response is None:
            raise Exception('Failed to receive parameters')
            
        # 反序列化参数
        net_params = pickle.loads(response)
        return net_params
    finally:
        client_socket.close()

def update_net_params(params):
    """推送更新net参数"""
    HOST = '146.235.33.108'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        # 发送更新参数请求
        message = f'{CODE}_update'
        send_msg(client_socket, message.encode())
        
        # 发送参数数据
        params_data = pickle.dumps(params)
        send_msg(client_socket, params_data)
        
        # 等待确认
        response = recv_msg(client_socket)
        if response == b'ok':
            return True
    finally:
        client_socket.close()
    return False

# 记录block ip的文件
record_file = os.path.join(os.path.expanduser('~'), 'block_ips_net_center.txt')

def read_block_ips():
    """读取block ip列表"""
    if not os.path.exists(record_file):
        return []
    with open(record_file, 'r') as f:
        ips = f.read().splitlines()
        return ips

def update_block_ips(ips):
    """更新block ip列表"""
    with open(record_file, 'w') as f:
        f.write('\n'.join(ips))

def run_param_center(model, tau= 0.005):
    """参数中心服务器"""
    HOST = '0.0.0.0'
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(20)
    print(f"Parameter center server started: {HOST}:{PORT}")

    # 软更新系数
    block_ips = read_block_ips()

    while True:
        client_socket, client_address = server_socket.accept()
        client_ip = client_address[0]

        # 检查是否是block ip
        if client_ip in block_ips:
            print(f"Blocked connection from {client_ip}")
            client_socket.close()
            continue

        # print(f"Accepted connection from: {client_address}")

        need_block = False
        try:
            # 接收请求消息
            data = recv_msg(client_socket)
            if data:
                try:
                    data_str = data.decode()
                except:
                    need_block = True
                    data_str = ''
                    
                if '_' in data_str:
                    _code, cmd = data_str.split('_', maxsplit=1)

                    if _code == CODE:
                        if cmd == 'get':
                            # 发送模型参数
                            params_data = pickle.dumps(model.state_dict())
                            send_msg(client_socket, params_data)
                            print(f'Parameters sent to {client_ip}')
                        elif cmd == 'update':
                            # 接收新参数
                            params_data = recv_msg(client_socket)
                            if params_data is None:
                                need_block = True
                            else:
                                # 软更新参数
                                new_params = pickle.loads(params_data)
                                for key in model.state_dict():
                                    model.state_dict()[key].copy_(
                                        tau * new_params[key] + (1-tau) * model.state_dict()[key]
                                    )
                                print(f'Parameters updated from {client_ip}')
                                send_msg(client_socket, b'ok')
                        else:
                            need_block = True
                    else:
                        need_block = True
                else:
                    need_block = True
            else:
                need_block = True

        except ConnectionResetError:
            pass
        except Exception as e:
            print(f"Error processing request: {e}")
            need_block = True

        if need_block:
            block_ips.append(client_ip)
            update_block_ips(block_ips)
            print(f'Added block IP: {client_ip}')

        client_socket.close()

if "__main__" == __name__:
    # 测试客户端服务器通信
    import torch
    import torch.nn as nn
    import sys
    
    # 创建一个简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
            
        def forward(self, x):
            return self.fc(x)
            
    # 初始化模型和参数
    model = TestModel()
    tau = 0.1
    
    # 根据命令行参数运行服务端或客户端
    if len(sys.argv) > 1:
        if sys.argv[1] == 'server':
            # 运行服务端
            print('Starting server...')
            run_param_center(model, tau)
        elif sys.argv[1] == 'client':
            # 运行客户端
            print('Starting client...')
            
            # 获取参数
            params = get_net_params()
            print('Parameters retrieved successfully')
            
            # 修改模型参数并推送更新
            print('Modifying and pushing parameters...')
            # 随机修改一个参数
            with torch.no_grad():
                model.fc.weight.data *= 1.1
            # 推送更新
            update_net_params(model.state_dict())
            print('Parameters pushed successfully')
    else:
        print('Please specify mode: python net_center.py [server|client]')