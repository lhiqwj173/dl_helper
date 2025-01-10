# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from py_ext.tool import log, init_logger
from py_ext.wechat import send_wx

from datetime import datetime, timezone, timedelta
import socket, time, sys, os, re
import pickle
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

from dl_helper.rl.socket_base import CODE, PORT, send_msg, recv_msg
from dl_helper.rl.rl_utils import read_train_title_item
from dl_helper.rl.param_keeper import ExperimentHandler

class BlockIPs:
    """管理block ip的类"""
    def __init__(self):
        self.record_file = os.path.join(os.path.expanduser('~'), 'block_ips_net_center.txt')
        self.ips = self.read()

    def read(self):
        """读取block ip列表"""
        if not os.path.exists(self.record_file):
            return []
        with open(self.record_file, 'r') as f:
            ips = f.read().splitlines()
            return list(set(ips)) # 去重

    def _update(self, ips):
        """更新block ip列表"""
        with open(self.record_file, 'w') as f:
            f.write('\n'.join(sorted(set(ips)))) # 排序并去重

    def add(self, ip):
        """添加ip到block ip列表"""
        if ip not in self.ips: # 避免重复添加
            self.ips.append(ip)
            self._update(self.ips)

    def is_blocked(self, ip):
        """检查ip是否被block"""
        return ip in self.ips

def run_param_center():
    """参数中心服务器"""
    HOST = '0.0.0.0'
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(20)
    log(f"Parameter center server started: {HOST}:{PORT}")

    block_ips = BlockIPs()

    # 初始化实验处理器
    handlers = {}
    train_dict = read_train_title_item()
    for title, config in train_dict.items():
        log(f'{title} init')
        handlers[title] = ExperimentHandler(title, config)

    log('all init done')
    while True:
        client_socket, client_address = server_socket.accept()
        client_ip, client_port = client_address

        if block_ips.is_blocked(client_ip):
            log(f"Blocked connection from {client_ip}")
            client_socket.close()
            continue

        # 接收请求数据
        data = recv_msg(client_socket)
        if not data:
            block_ips.add(client_ip)
            client_socket.close()
            continue
            
        try:
            # 请求数据
            data_str = data.decode()
        except:
            block_ips.add(client_ip)
            client_socket.close()
            continue
            
        # 验证CODE
        if ':' not in data_str or '_' not in data_str:
            block_ips.add(client_ip)
            client_socket.close() 
            continue
        _code, a = data_str.split('_', maxsplit=1)
        if _code != CODE:
            block_ips.add(client_ip)
            client_socket.close()
            continue

        # 分解指令
        train_title, cmd = a.split(':', maxsplit=1)
        
        # 获取处理器
        if train_title == 'test':
            handler = handlers.values()[0]
        else:
            if train_title not in handlers:
                # 重新读取 
                train_dict = read_train_title_item()
                if train_title in train_dict:
                    config = train_dict[train_title]
                    handlers[train_title] = ExperimentHandler(train_title, config)
                else:
                    msg = f'{train_title} not found'
                    send_wx(msg)
                    log(msg)
                    client_socket.close()
                    continue
            handler = handlers[train_title]
        
        # 不要影响到其他客户端
        try:
            msg_header = f'[{client_ip:<15} {client_port:<5}][{train_title}][{handler.version}]'
            handler.handle_request(client_socket, msg_header, cmd)
        except Exception as e:
            log(f"Error handling request: {e}")
            error_info = traceback.format_exc()
            log(f"Error details:\n{error_info}")
            client_socket.close()
            continue
        finally:
            client_socket.close()

if __name__ == '__main__':
    init_logger('net_center')

    # 初始化实验处理器
    run_param_center()

