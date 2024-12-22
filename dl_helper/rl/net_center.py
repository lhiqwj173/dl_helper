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

from dl_helper.rl.socket_base import CODE, PORT, send_msg, recv_msg
from dl_helper.rl.dqn.dqn import DQN
from dl_helper.rl.dqn.c51 import C51

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

alist_folder = r'/root/alist_data/rl_learning_process'
root_folder = '' if not os.path.exists(alist_folder) else alist_folder

class ExperimentHandler:
    """处理单个实验的类"""
    def __init__(self, train_title, agent_class_name, agent_kwargs, tau=0.005, simple_test=False):
        self.train_title = train_title
        self.agent = globals()[agent_class_name](**agent_kwargs)
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
        t = time.time()
        self.last_val_time = t
        self.last_test_time = t

    def plot_learning_process(self, metrics):
        """
        强化学习评价指标
            图1
            - moving_average_reward: 移动平均奖励

            图2
            - average_td_error: 平均TD误差
            - average_loss: 平均损失值

            图3
            - illegal_ratio: 平均非法动作率

            图4
            - win_ratio: 平均胜率
            - loss_ratio: 平均败率

            图5
            - action_{k}_ratio k: 0-2

            图6
            - hold_length: 平均持仓时间

        交易评价指标
            图7 
            - sortino_ratio
            - sortino_ratio_bm

            图8
            - max_drawdown
            - max_drawdown_bm

            图9
            - total_return
            - total_return_bm

        train_days: 训练天数
        """
        # 检查数据是否存在
        if 'dt' not in metrics or not metrics['dt']:
            log('No dt data found')
            return
            
        # 固定颜色
        colors = {
            'moving_average_reward': '#ff7f0e',
            'average_td_error': '#d62728',
            'average_loss': '#9467bd',
            'illegal_ratio': '#8c564b',
            'win_ratio': '#e377c2',
            'loss_ratio': '#7f7f7f',
            'action_0': '#bcbd22',
            'action_1': '#17becf',
            'action_2': '#1f77b4',
            'hold_length': '#2ca02c',
            'sortino_ratio': '#d62728',
            'sortino_ratio_bm': '#d62728',
            'max_drawdown': '#ff7f0e',
            'max_drawdown_bm': '#ff7f0e',
            'total_return': '#9467bd',
            'total_return_bm': '#9467bd'
        }
        
        # 创建图表,9个子图
        fig, axes = plt.subplots(9, 1, figsize=(12, 36), sharex=True)
        
        # 获取时间变化点的索引
        dt_changes = []
        last_dt = None
        for i, dt in enumerate(metrics['dt']):
            processed_dt = dt.replace(hour=dt.hour - dt.hour % 4, minute=0, second=0, microsecond=0)
            if processed_dt != last_dt:
                dt_changes.append((i, processed_dt, metrics['learn']['train_days'][i]))
                last_dt = processed_dt

        # 设置图表标题
        fig.suptitle(f'Learning Process (Training Days: {int(metrics["learn"]["train_days"][-1])})', fontsize=16)

        # 图1: moving_average_reward
        ax = axes[0]
        for dtype in ['learn', 'val', 'test']:
            if 'moving_average_reward' in metrics[dtype]:
                data = metrics[dtype]['moving_average_reward']
                last_value = data[-1] if len(data) > 0 else 0
                if dtype == 'learn':
                    ax.plot(data, color=colors['moving_average_reward'], alpha=0.3,
                        label=f'{dtype}_moving_average_reward: {last_value:.4f}')
                elif dtype == 'val':
                    ax.plot(data, color=colors['moving_average_reward'],
                        label=f'{dtype}_moving_average_reward: {last_value:.4f}')
                else:  # test
                    ax.plot(data, color=colors['moving_average_reward'], linestyle='--',
                        label=f'{dtype}_moving_average_reward: {last_value:.4f}')
        ax.set_ylabel('Moving Average Reward')
        ax.grid(True)
        ax.legend()

        # 图2: average_td_error & average_loss
        ax = axes[1]
        for dtype in ['learn', 'val', 'test']:
            for key in ['average_td_error', 'average_loss']:
                if key in metrics[dtype]:
                    data = metrics[dtype][key]
                    last_value = data[-1] if len(data) > 0 else 0
                    if dtype == 'learn':
                        ax.plot(data, color=colors[key], alpha=0.3,
                            label=f'{dtype}_{key}: {last_value:.4f}')
                    elif dtype == 'val':
                        ax.plot(data, color=colors[key],
                            label=f'{dtype}_{key}: {last_value:.4f}')
                    else:  # test
                        ax.plot(data, color=colors[key], linestyle='--',
                            label=f'{dtype}_{key}: {last_value:.4f}')
        ax.set_ylabel('Error/Loss')
        ax.grid(True)
        ax.legend()

        # 图3: illegal_ratio
        ax = axes[2]
        for dtype in ['learn', 'val', 'test']:
            if 'illegal_ratio' in metrics[dtype]:
                data = metrics[dtype]['illegal_ratio']
                last_value = data[-1] if len(data) > 0 else 0
                if dtype == 'learn':
                    ax.plot(data, color=colors['illegal_ratio'], alpha=0.3,
                        label=f'{dtype}_illegal_ratio: {last_value:.4f}')
                elif dtype == 'val':
                    ax.plot(data, color=colors['illegal_ratio'],
                        label=f'{dtype}_illegal_ratio: {last_value:.4f}')
                else:  # test
                    ax.plot(data, color=colors['illegal_ratio'], linestyle='--',
                        label=f'{dtype}_illegal_ratio: {last_value:.4f}')
        ax.set_ylabel('Illegal Ratio')
        ax.grid(True)
        ax.legend()

        # 图4: win_ratio & loss_ratio
        ax = axes[3]
        for dtype in ['learn', 'val', 'test']:
            for key in ['win_ratio', 'loss_ratio']:
                if key in metrics[dtype]:
                    data = metrics[dtype][key]
                    last_value = data[-1] if len(data) > 0 else 0
                    if dtype == 'learn':
                        ax.plot(data, color=colors[key], alpha=0.3,
                            label=f'{dtype}_{key}: {last_value:.4f}')
                    elif dtype == 'val':
                        ax.plot(data, color=colors[key],
                            label=f'{dtype}_{key}: {last_value:.4f}')
                    else:  # test
                        ax.plot(data, color=colors[key], linestyle='--',
                            label=f'{dtype}_{key}: {last_value:.4f}')
        ax.set_ylabel('Win/Loss Ratio')
        ax.grid(True)
        ax.legend()

        # 图5: action ratios
        ax = axes[4]
        for dtype in ['learn', 'val', 'test']:
            for i in range(3):
                key = f'action_{i}_ratio'
                if key in metrics[dtype]:
                    data = metrics[dtype][key]
                    last_value = data[-1] if len(data) > 0 else 0
                    if dtype == 'learn':
                        ax.plot(data, color=colors[f'action_{i}'], alpha=0.3,
                            label=f'{dtype}_{key}: {last_value:.4f}')
                    elif dtype == 'val':
                        ax.plot(data, color=colors[f'action_{i}'],
                            label=f'{dtype}_{key}: {last_value:.4f}')
                    else:  # test
                        ax.plot(data, color=colors[f'action_{i}'], linestyle='--',
                            label=f'{dtype}_{key}: {last_value:.4f}')
        ax.set_ylabel('Action Ratio')
        ax.grid(True)
        ax.legend()

        # 图6: hold_length
        ax = axes[5]
        for dtype in ['learn', 'val', 'test']:
            if 'hold_length' in metrics[dtype]:
                data = metrics[dtype]['hold_length']
                last_value = data[-1] if len(data) > 0 else 0
                if dtype == 'learn':
                    ax.plot(data, color=colors['hold_length'], alpha=0.3,
                        label=f'{dtype}_hold_length: {last_value:.4f}')
                elif dtype == 'val':
                    ax.plot(data, color=colors['hold_length'],
                        label=f'{dtype}_hold_length: {last_value:.4f}')
                else:  # test
                    ax.plot(data, color=colors['hold_length'], linestyle='--',
                        label=f'{dtype}_hold_length: {last_value:.4f}')
        ax.set_ylabel('Hold Length')
        ax.grid(True)
        ax.legend()

        # 图7: sortino_ratio
        ax = axes[6]
        for dtype in ['learn', 'val', 'test']:
            if 'sortino_ratio' in metrics[dtype]:
                data = metrics[dtype]['sortino_ratio']
                last_value = data[-1] if len(data) > 0 else 0
                if dtype == 'learn':
                    ax.plot(data, color=colors['sortino_ratio'], alpha=0.3,
                        label=f'{dtype}_sortino_ratio: {last_value:.4f}')
                elif dtype == 'val':
                    ax.plot(data, color=colors['sortino_ratio'],
                        label=f'{dtype}_sortino_ratio: {last_value:.4f}')
                else:  # test
                    ax.plot(data, color=colors['sortino_ratio'], linestyle='--',
                        label=f'{dtype}_sortino_ratio: {last_value:.4f}')
                    if 'sortino_ratio_bm' in metrics[dtype]:
                        bm_data = metrics[dtype]['sortino_ratio_bm']
                        last_value = bm_data[-1] if len(bm_data) > 0 else 0
                        ax.fill_between(range(len(bm_data)), bm_data, alpha=0.1,
                                    color=colors['sortino_ratio_bm'],
                                    label=f'{dtype}_sortino_ratio_bm: {last_value:.4f}')
        ax.set_ylabel('Sortino Ratio')
        ax.grid(True)
        ax.legend()

        # 图8: max_drawdown
        ax = axes[7]
        for dtype in ['learn', 'val', 'test']:
            if 'max_drawdown' in metrics[dtype]:
                data = metrics[dtype]['max_drawdown']
                last_value = data[-1] if len(data) > 0 else 0
                if dtype == 'learn':
                    ax.plot(data, color=colors['max_drawdown'], alpha=0.3,
                        label=f'{dtype}_max_drawdown: {last_value:.4f}')
                elif dtype == 'val':
                    ax.plot(data, color=colors['max_drawdown'],
                        label=f'{dtype}_max_drawdown: {last_value:.4f}')
                else:  # test
                    ax.plot(data, color=colors['max_drawdown'], linestyle='--',
                        label=f'{dtype}_max_drawdown: {last_value:.4f}')
                    if 'max_drawdown_bm' in metrics[dtype]:
                        bm_data = metrics[dtype]['max_drawdown_bm']
                        last_value = bm_data[-1] if len(bm_data) > 0 else 0
                        ax.fill_between(range(len(bm_data)), bm_data, alpha=0.1,
                                    color=colors['max_drawdown_bm'],
                                    label=f'{dtype}_max_drawdown_bm: {last_value:.4f}')
        ax.set_ylabel('Max Drawdown')
        ax.grid(True)
        ax.legend()

        # 图9: total_return
        ax = axes[8]
        for dtype in ['learn', 'val', 'test']:
            if 'total_return' in metrics[dtype]:
                data = metrics[dtype]['total_return']
                last_value = data[-1] if len(data) > 0 else 0
                if dtype == 'learn':
                    ax.plot(data, color=colors['total_return'], alpha=0.3,
                        label=f'{dtype}_total_return: {last_value:.4f}')
                elif dtype == 'val':
                    ax.plot(data, color=colors['total_return'],
                        label=f'{dtype}_total_return: {last_value:.4f}')
                else:  # test
                    ax.plot(data, color=colors['total_return'], linestyle='--',
                        label=f'{dtype}_total_return: {last_value:.4f}')
                    if 'total_return_bm' in metrics[dtype]:
                        bm_data = metrics[dtype]['total_return_bm']
                        last_value = bm_data[-1] if len(bm_data) > 0 else 0
                        ax.fill_between(range(len(bm_data)), bm_data, alpha=0.1,
                                    color=colors['total_return_bm'],
                                    label=f'{dtype}_total_return_bm: {last_value:.4f}')
        ax.set_ylabel('Total Return')
        ax.grid(True)
        ax.legend()

        # 设置x轴刻度和标签
        for ax in axes:
            ax.set_xticks([i for i, _, _ in dt_changes])
            ax.set_xticklabels([f"{dt.strftime('%d %H')}({days:.1f})" for _, dt, days in dt_changes], rotation=45)
        
        # 设置共享的x轴标签
        fig.text(0.5, 0.02, 'Episode', ha='center')
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.exp_folder, 'learning_process.png'))
        plt.close()

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
                if hasattr(self.train_data[dtype][k], '__len__') and len(self.train_data[dtype][k]) > 0:
                    latest_data[f'{dtype}_{k}'] = self.train_data[dtype][k][-1]
                else:
                    latest_data[f'{dtype}_{k}'] = self.train_data[dtype][k]

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
                # 只对列表/数组类型的数据进行max_len判断
                if hasattr(self.train_data[dtype][k], '__len__'):
                    max_len = max(max_len, len(self.train_data[dtype][k]))
                
        for dtype in ['val', 'test', 'learn']:
            for k in self.train_data[dtype]:
                if hasattr(self.train_data[dtype][k], '__len__'):
                    curr_len = len(self.train_data[dtype][k])
                    if curr_len < max_len:
                        pad_value = self.train_data[dtype][k][-1] if curr_len > 0 else float('nan')
                        self.train_data[dtype][k].extend([pad_value] * (max_len - curr_len))

        for dtype in ['val', 'test', 'learn']:
            for k in self.train_data[dtype]:
                if hasattr(self.train_data[dtype][k], '__len__'):
                    self.train_data[dtype][k] = self.train_data[dtype][k][:500]
        self.train_data['dt'] = self.train_data['dt'][:500]

        self.save_train_data_to_csv()
        
        self.plot_learning_process(self.train_data)

        return self.train_data

    def handle_request(self, client_socket, msg_header, cmd):
        """处理客户端请求"""
        try:
            if cmd == 'get':
                params_data = pickle.dumps(self.agent.get_params_to_send())
                send_msg(client_socket, params_data)
                log(f'{msg_header} Parameters sent')

            elif cmd == 'check':
                # 30min一次val, 2小时一次test
                response = 'no'

                t = time.time()
                if t - self.last_val_time > 1800:
                # # FOR TEST
                # if t - self.last_val_time > 60 * 5:
                    response = 'val'
                    self.last_val_time = t
                elif t - self.last_test_time > 7200:
                # # FOR TEST
                # elif t - self.last_test_time > 60 * 5:
                    response = 'test'
                    self.last_test_time = t
                
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
                
                if cmd in ['val', 'test']:
                    backup_path = os.path.join(self.exp_folder, 'learn_metrics_backup.pkl')
                    with open(backup_path, 'wb') as f:
                        pickle.dump(self.learn_metrics, f)

                    for k in self.learn_metrics:
                        if k not in self.train_data['learn']:
                            self.train_data['learn'][k] = []

                        length = len(self.learn_metrics[k])
                        if length > 0:
                            if k == 'train_days':
                                add_days = np.nansum(self.learn_metrics[k])
                                if len(self.train_data['learn'][k]) > 0:
                                    add_days += self.train_data['learn'][k][-1]
                                self.train_data['learn'][k].append(add_days)
                            else:
                                self.train_data['learn'][k].append(np.nanmean(self.learn_metrics[k]))

                        # log(f'{msg_header} length learn_metrics[{k}]: {length}')
                    self.learn_metrics = {}

                send_msg(client_socket, b'ok')

                dt = datetime.now(timezone(timedelta(hours=8)))
                self.train_data['dt'].append(dt)
                self.train_data = self.handle_val_test_data()
                log(f'{msg_header} handle {cmd}_data done')

        except ConnectionResetError:
            pass

def add_train_title_item(train_title, agent_class, agent_kwargs, tau, simple_test):
    file = os.path.join(root_folder, f'{train_title}.data')
    if os.path.exists(file):
        return
    with open(file, 'wb') as f:
        pickle.dump((agent_class.__name__, agent_kwargs, tau, simple_test), f)

def read_train_title_item():
    res = {}
    for file in os.listdir(root_folder):
        if file.endswith('.data'):
            title = file.replace('.data', '')
            agent_class_name, agent_kwargs, tau, simple_test = pickle.load(open(os.path.join(root_folder, file), 'rb'))
            res[title] = (agent_class_name, agent_kwargs, tau, simple_test)
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
    for title, (agent_class_name, agent_kwargs, tau, simple_test) in train_dict.items():
        log(f'{title} init')
        handlers[title] = ExperimentHandler(title, agent_class_name, agent_kwargs, tau, simple_test)

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
        if train_title not in handlers:
            # 重新读取 
            train_dict = read_train_title_item()
            if train_title in train_dict:
                agent_class_name, agent_kwargs, tau, simple_test = train_dict[train_title]
                handlers[train_title] = ExperimentHandler(train_title, agent_class_name, agent_kwargs, tau, simple_test)
            else:
                msg = f'{train_title} not found'
                send_wx(msg)
                log(msg)
                client_socket.close()
                continue
        handler = handlers[train_title]
        
        msg_header = f'[{client_ip:<15} {client_port:<5}][{train_title}][{handler.update_count}]'
        handler.handle_request(client_socket, msg_header, cmd)

        client_socket.close()


if __name__ == '__main__':
    init_logger('net_center')

    # 初始化实验处理器
    run_param_center()

