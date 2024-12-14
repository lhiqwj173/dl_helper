# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from py_ext.tool import log
from py_ext.wechat import send_wx

from datetime import datetime, timezone, timedelta
import socket, time, sys, os, re
import pickle
import dill
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dl_helper.rl.socket_base import CODE, PORT, send_msg, recv_msg
from dl_helper.rl.dqn import DQN

class BlockIPs:
    """管理block ip的类"""
    def __init__(self):
        self.record_file = os.path.join(os.path.expanduser('~'), 'block_ips_net_center.txt')

    def read(self):
        """读取block ip列表"""
        if not os.path.exists(self.record_file):
            return []
        with open(self.record_file, 'r') as f:
            ips = f.read().splitlines()
            return ips

    def update(self, ips):
        """更新block ip列表"""
        with open(self.record_file, 'w') as f:
            f.write('\n'.join(ips))

    def is_blocked(self, ip):
        """检查ip是否被block"""
        return ip in self.read()

alist_folder = r'/root/alist_data/rl_learning_process'
root_folder = '' if not os.path.exists(alist_folder) else alist_folder

class ExperimentHandler:
    """处理单个实验的类"""
    def __init__(self, train_title, agent, tau=0.005, simple_test=False):
        self.train_title = train_title
        self.agent = agent
        self.tau = tau
        self.simple_test = simple_test
        
        # 创建实验目录
        self.exp_folder = os.path.join(root_folder, train_title)
        os.makedirs(self.exp_folder, exist_ok=True)
        self.csv_path = os.path.join(self.exp_folder, 'val_test.csv')
        
        # 载入模型数据
        self.agent.load(self.exp_folder)
        
        # 学习进度数据
        self.learn_metrics = {}
        
        # 验证测试数据
        self.train_data = self.init_train_data_from_csv()
        
        # 参数更新计数
        self.update_count = 0
        
        # 是否需要验证测试
        self.need_val = False
        self.need_test = False

    def init_train_data_from_csv(self):
        """从CSV文件初始化训练数据"""
        train_data = {
            'learn': {},
            'val': {},
            'test': {},
            'dt': []
        }
        
        if not os.path.exists(self.csv_path):
            return train_data
            
        df = pd.read_csv(self.csv_path)
        
        if 'dt' in df.columns:
            train_data['dt'] = pd.to_datetime(df['dt']).tolist()
        
        for col in df.columns:
            if col == 'dt':
                continue
                
            dtype, key = col.split('_', 1)
            if dtype in ['val', 'test', 'learn']:
                if key not in train_data[dtype]:
                    train_data[dtype][key] = []
                train_data[dtype][key] = [float(x) if not pd.isna(x) else float('nan') for x in df[col].tolist()]
                
        return train_data

    def save_train_data_to_csv(self):
        """将训练数据保存到CSV文件"""
        latest_data = {}
        for dtype in ['val', 'test', 'learn']:
            for k in self.train_data[dtype]:
                if len(self.train_data[dtype][k]) > 0:
                    latest_data[f'{dtype}_{k}'] = self.train_data[dtype][k][-1]
        
        if len(self.train_data['dt']) > 0:
            latest_data['dt'] = self.train_data['dt'][-1]
        
        file_exists = os.path.exists(self.csv_path)
        
        headers = sorted(latest_data.keys())
        values = [str(latest_data[h]) for h in headers]
        
        if file_exists:
            old_df = pd.read_csv(self.csv_path)
            new_df = pd.DataFrame([values], columns=headers)
            df = pd.concat([old_df, new_df], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
        else:
            df = pd.DataFrame([values], columns=headers)
            df.to_csv(self.csv_path, index=False)

    def handle_val_test_data(self):
        """处理验证测试数据"""
        max_len = 0
        for dtype in ['val', 'test', 'learn']:
            for k in self.train_data[dtype]:
                max_len = max(max_len, len(self.train_data[dtype][k]))
                
        for dtype in ['val', 'test', 'learn']:
            for k in self.train_data[dtype]:
                curr_len = len(self.train_data[dtype][k])
                if curr_len < max_len:
                    pad_value = self.train_data[dtype][k][-1] if curr_len > 0 else float('nan')
                    self.train_data[dtype][k].extend([pad_value] * (max_len - curr_len))

        for dtype in ['val', 'test', 'learn']:
            for k in self.train_data[dtype]:
                self.train_data[dtype][k] = self.train_data[dtype][k][:500]
        self.train_data['dt'] = self.train_data['dt'][:500]

        self.save_train_data_to_csv()
        
        plot_learning_process(self.exp_folder, self.train_data)

        return self.train_data

    def handle_request(self, client_socket, msg_header, cmd):
        """处理客户端请求"""
        try:
            if cmd == 'get':
                params_data = pickle.dumps(self.agent.get_params_to_send())
                send_msg(client_socket, params_data)
                log(f'{msg_header} Parameters sent')

            elif cmd == 'check':
                response = 'no'
                if self.need_test:
                    response = 'test'
                    self.need_test = False
                elif self.need_val:
                    response = 'val' 
                    self.need_val = False
                
                send_msg(client_socket, response.encode())
                msg = f'{msg_header} Check response sent: {response}'
                log(msg)

            elif cmd == 'update':
                update_data = recv_msg(client_socket)
                if update_data is None:
                    return
                    
                new_params, metrics = pickle.loads(update_data)

                self.agent.apply_new_params(new_params, tau=self.tau)
                log(f'{msg_header} Parameters updated')
                send_msg(client_socket, b'ok')
                self.agent.save(self.exp_folder)
                
                for k, v in metrics.items():
                    if k not in self.learn_metrics:
                        self.learn_metrics[k] = []
                    self.learn_metrics[k].append(v)

                self.update_count += 1
                
                if self.simple_test:
                    _test_count = 5
                    _val_count = 3
                else:
                    _val_count = 3000
                    _test_count = _val_count * 5

                if self.update_count % _test_count == 0:
                    self.need_test = True
                elif self.update_count % _val_count == 0:
                    self.need_val = True
                if self.simple_test and self.update_count > _test_count * 3:
                    return True

            elif cmd in ['val', 'test']:
                data_type = cmd
                train_data_new = recv_msg(client_socket)
                if train_data_new is None:
                    return
                    
                metrics = pickle.loads(train_data_new)
                log(f'{msg_header} {cmd}_metrics: {metrics}')
                for k in metrics:
                    if k not in self.train_data[data_type]:
                        self.train_data[data_type][k] = []
                    self.train_data[data_type][k].append(metrics[k])
                
                if cmd == 'val':
                    backup_path = os.path.join(self.exp_folder, 'learn_metrics_backup.pkl')
                    with open(backup_path, 'wb') as f:
                        pickle.dump(self.learn_metrics, f)

                    for k in self.learn_metrics:
                        if k not in self.train_data['learn']:
                            self.train_data['learn'][k] = []
                        length = len(self.learn_metrics[k])
                        if length > 0:
                            self.train_data['learn'][k].append(np.nanmean(self.learn_metrics[k]))

                        log(f'{msg_header} length learn_metrics[{k}]: {length}')
                    self.learn_metrics = {}

                send_msg(client_socket, b'ok')

                dt = datetime.now(timezone(timedelta(hours=8)))
                self.train_data['dt'].append(dt)
                self.train_data = self.handle_val_test_data()
                log(f'{msg_header} handle {cmd}_data done')

        except ConnectionResetError:
            pass

def add_train_title_item(train_title, agent, tau, simple_test):
    with open(os.path.join(root_folder, f'{train_title}.data'), 'wb') as f:
        dill.dump((train_title, agent, tau, simple_test), f)

def read_train_title_item(train_title):
    res = {}
    for file in os.listdir(root_folder):
        if file.endswith('.data'):
            title, agent, tau, simple_test = dill.load(open(os.path.join(root_folder, file), 'rb'))
            res[title] = (agent, tau, simple_test)
    return res

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
    for title, (agent, tau, simple_test) in train_dict.items():
        log(f'{title} init')    
        handlers[title] = ExperimentHandler(title, agent, tau, simple_test)

    while True:
        client_socket, client_address = server_socket.accept()
        client_ip, client_port = client_address

        if block_ips.is_blocked(client_ip):
            log(f"Blocked connection from {client_ip}")
            client_socket.close()
            continue

        try:
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
            if train_title not in handlers:
                # 重新读取 
                train_dict = read_train_title_item()
                if train_title in train_dict:
                    agent, tau, simple_test = train_dict[train_title]
                    handlers[train_title] = ExperimentHandler(train_title, agent, tau, simple_test)
                else:
                    msg = f'{train_title} not found'
                    send_wx(msg)
                    log(msg)
                    client_socket.close()
                    continue
            handler = handlers[train_title]
            
            msg_header = f'[{client_ip:<15} {client_port:<5}][{train_title}][{handler.update_count}]'
            handler.handle_request(client_socket, msg_header, cmd)

        except Exception as e:
            log(f"Error processing request: {e}")
            
        client_socket.close()


if __name__ == '__main__':
    # 初始化实验处理器
    run_param_center()

