# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from py_ext.tool import log

import socket, time, sys, os, re
import pickle
import struct
import numpy as np
import matplotlib.pyplot as plt

CODE = '0QYg9Ky17dWnN4eK'
PORT = 12346
alist_folder = r'/root/alist_data/rl_learning_process'

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

def send_net_params(params):
    """推送更新net参数"""
    # 将GPU上的参数转移到CPU
    cpu_params = {
        k: v.cpu() if hasattr(v, 'cpu') else v 
        for k, v in params.items()
    }

    HOST = '146.235.33.108'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        # 发送更新参数请求
        message = f'{CODE}_update'
        send_msg(client_socket, message.encode())
        
        # 发送参数数据
        params_data = pickle.dumps(cpu_params)
        send_msg(client_socket, params_data)
        
        # 等待确认
        response = recv_msg(client_socket)
        if response == b'ok':
            return True
    finally:
        client_socket.close()
    return False

def send_val_test_data(data_type, watch_data):
    """发送训练数据"""
    HOST = '146.235.33.108'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        # 发送训练数据请求
        message = f'{CODE}_{data_type}'
        send_msg(client_socket, message.encode())
        
        # 发送训练数据
        data = pickle.dumps(watch_data)
        send_msg(client_socket, data)
        
        # 等待确认
        response = recv_msg(client_socket)
        if response == b'ok':
            return True
    finally:
        client_socket.close()
    return False

def update_model_params(model, new_params, tau=0.005):
    """使用新参数软更新模型参数"""
    params = model.state_dict()
    for name, param in params.items():
        if name in new_params:
            # 确保新参数在同一设备上
            new_param = new_params[name]
            if new_param.device != param.device:
                new_param = new_param.to(param.device)
            param.copy_((1 - tau) * param + tau * new_param)
    return model

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

def plot_learning_process(root, watch_data):
    """
    强化学习性能指标:
        return_list: 回合累计奖励
        avg_return_list: 回合平均长度奖励
        episode_lens: 回合长度
        max_q_value_list: 最大Q值
    
    交易评价指标:
        sharpe_ratio: 夏普比率
        sortino_ratio: 索提诺比率
        max_drawdown: 最大回撤
        total_return: 总回报
    """
    # 获取数据键,将交易评价指标单独处理
    rl_keys = ['return_list', 'avg_return_list', 'episode_lens', 'max_q_value_list']
    trade_keys = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'total_return']
    
    # 固定颜色
    colors = {
        'return': '#1f77b4',      # 蓝色
        'avg_return': '#2ca02c',  # 绿色
        'episode_lens': '#ff7f0e', # 橙色
        'max_q_value': '#9467bd', # 紫色
        'sharpe_ratio': '#8c564b', # 棕色
        'sortino_ratio': '#e377c2', # 粉色
        'max_drawdown': '#7f7f7f', # 灰色
        'total_return': '#bcbd22'  # 黄绿色
    }
    
    # 计算总图数
    n_rl = 3  # return合并为1个,其他2个
    n_trade = 2  # ratio合并为1个,其他1个
    n_total = n_rl + n_trade
    
    # 创建图表,每行一个子图
    fig, axes = plt.subplots(n_total, 1, figsize=(12, 4*n_total), sharex=True)
    if n_total == 1:
        axes = np.array([axes])
    
    # 获取时间变化点的索引
    dt_changes = []
    last_dt = None
    for i, dt in enumerate(watch_data['dt']):
        if dt != last_dt:
            dt_changes.append((i, dt))
            last_dt = dt
            
    # 绘制强化学习指标
    # 1. return_list和avg_return_list
    ax = axes[0]
    ax.plot(watch_data['val']['return_list'], color=colors['return'], alpha=0.3, label='val_return')
    ax.plot(watch_data['val']['avg_return_list'], color=colors['avg_return'], alpha=0.3, label='val_avg_return')
    ax.plot(watch_data['test']['return_list'], color=colors['return'], label='test_return')
    ax.plot(watch_data['test']['avg_return_list'], color=colors['avg_return'], label='test_avg_return')
    ax.set_ylabel('Return')
    ax.grid(True)
    ax.legend()
    
    # 2. episode_lens
    ax = axes[1]
    ax.plot(watch_data['val']['episode_lens'], color=colors['episode_lens'], alpha=0.3, label='val_episode_lens')
    ax.plot(watch_data['test']['episode_lens'], color=colors['episode_lens'], label='test_episode_lens')
    ax.set_ylabel('Episode Length')
    ax.grid(True)
    ax.legend()
    
    # 3. max_q_value_list
    ax = axes[2]
    ax.plot(watch_data['val']['max_q_value_list'], color=colors['max_q_value'], alpha=0.3, label='val_max_q_value')
    ax.plot(watch_data['test']['max_q_value_list'], color=colors['max_q_value'], label='test_max_q_value')
    ax.set_ylabel('Max Q Value')
    ax.grid(True)
    ax.legend()
    
    # 绘制交易评价指标
    # 1. sharpe_ratio和sortino_ratio
    ax = axes[3]
    ax.plot(watch_data['val']['sharpe_ratio'], color=colors['sharpe_ratio'], alpha=0.3, label='val_sharpe_ratio')
    ax.plot(watch_data['val']['sortino_ratio'], color=colors['sortino_ratio'], alpha=0.3, label='val_sortino_ratio')
    ax.plot(watch_data['test']['sharpe_ratio'], color=colors['sharpe_ratio'], label='test_sharpe_ratio')
    ax.plot(watch_data['test']['sortino_ratio'], color=colors['sortino_ratio'], label='test_sortino_ratio')
    ax.set_ylabel('Ratio')
    ax.grid(True)
    ax.legend()
    
    # 2. max_drawdown和total_return (双y轴)
    ax = axes[4]
    ax2 = ax.twinx()  # 创建共享x轴的第二个y轴
    
    lines = []
    # 在左轴绘制max_drawdown
    l1 = ax.plot(watch_data['val']['max_drawdown'], color=colors['max_drawdown'], alpha=0.3, label='val_max_drawdown')[0]
    l2 = ax.plot(watch_data['test']['max_drawdown'], color=colors['max_drawdown'], label='test_max_drawdown')[0]
    lines.extend([l1, l2])
    ax.set_ylabel('Max Drawdown')
    
    # 在右轴绘制total_return
    l3 = ax2.plot(watch_data['val']['total_return'], color=colors['total_return'], alpha=0.3, label='val_total_return')[0]
    l4 = ax2.plot(watch_data['test']['total_return'], color=colors['total_return'], label='test_total_return')[0]
    lines.extend([l3, l4])
    ax2.set_ylabel('Total Return')
    
    # 合并两个轴的图例
    ax.legend(handles=lines, loc='upper left')
    ax.grid(True)
    
    # 设置x轴刻度和标签
    for ax in axes:
        ax.set_xticks([i for i, _ in dt_changes])
        ax.set_xticklabels([dt.strftime('%d %H') for _, dt in dt_changes], rotation=45)
    
    # 设置共享的x轴标签
    fig.text(0.5, 0.02, 'Episode', ha='center')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(root, 'learning_process.png'))
    plt.close()

def handle_val_test_data(train_data):
    """处理验证测试数据"""
    # 数据补齐
    # 找出所有类型中最长的列表长度
    max_len = 0
    for dtype in ['val', 'test']:
        for k in train_data[dtype]:
            max_len = max(max_len, len(train_data[dtype][k]))
    # 对齐所有类型的所有列表到最大长度
    for dtype in ['val', 'test']:
        for k in train_data[dtype]:
            curr_len = len(train_data[dtype][k])
            if curr_len < max_len:
                # 获取前一个值,若无则用nan
                pad_value = train_data[dtype][k][-1] if curr_len > 0 else float('nan')
                train_data[dtype][k].extend([pad_value] * (max_len - curr_len))

    # 数据截断， 最多允许 500 个数据
    for dtype in ['val', 'test']:
        for k in train_data[dtype]:
            train_data[dtype][k] = train_data[dtype][k][:500]
    train_data['dt'] = train_data['dt'][:500]
    
    # 绘制数据
    plot_learning_process('' if not os.path.exists(alist_folder) else alist_folder, train_data)

    return train_data

def blank_watch_data():
    """返回空白watch_data"""
    return {
        'return_list': [],
        'avg_return_list': [],
        'episode_lens': [],
        'max_q_value_list': [],
        'sharpe_ratio': [],
        'sortino_ratio': [],
        'max_drawdown': [],
        'total_return': []
    }

def run_param_center(agent, tau= 0.005):
    """参数中心服务器"""
    # 载入模型数据，若有
    agent.load(alist_folder)
    model = agent.q_net

    HOST = '0.0.0.0'
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(20)
    log(f"Parameter center server started: {HOST}:{PORT}")

    # 软更新系数
    block_ips = read_block_ips()
    
    # 训练数据
    train_data = {
        'val': blank_watch_data(),
        'test': blank_watch_data(),
        'dt': [],
    }

    while True:
        client_socket, client_address = server_socket.accept()
        client_ip = client_address[0]

        # 检查是否是block ip
        if client_ip in block_ips:
            log(f"Blocked connection from {client_ip}")
            client_socket.close()
            continue

        log(f"Accepted connection from: {client_address}")

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
                            log(f'Parameters sent to {client_ip}')

                        elif cmd == 'update':
                            # 接收新参数
                            params_data = recv_msg(client_socket)
                            if params_data is None:
                                need_block = True
                            else:
                                # 软更新参数
                                new_params = pickle.loads(params_data)
                                model = update_model_params(model, new_params)
                                # 也更新 target 模型
                                agent.target_q_net = update_model_params(agent.target_q_net, new_params)
                                log(f'Parameters updated from {client_ip}')
                                send_msg(client_socket, b'ok')
                                # 保存最新参数
                                agent.save(alist_folder)
                                log(f'agent saved to {alist_folder}')

                        elif cmd in ['val', 'test']:
                            # 接收训练数据
                            data_type = cmd
                            train_data_new = recv_msg(client_socket)
                            log(f'train_data: {train_data_new}')
                            if train_data_new is None:
                                need_block = True
                            else:
                                # 更新训练数据
                                watch_data = pickle.loads(train_data_new)
                                for k in train_data[data_type]:
                                    if k in watch_data:
                                        train_data[data_type][k].append(watch_data[k])
                                                                
                                log(f'Train data updated from {client_ip}')
                                send_msg(client_socket, b'ok')

                                # 增加北京时间(每4小时)
                                dt = datetime.now(timezone(timedelta(hours=8)))  # 使用timezone确保是北京时间
                                dt = dt.replace(hour=dt.hour - dt.hour % 4, minute=0, second=0, microsecond=0)
                                train_data['dt'].append(dt)
                                train_data = handle_val_test_data(train_data)

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
            log(f"Error processing request: {e}")
            need_block = True

        if need_block:
            block_ips.append(client_ip)
            update_block_ips(block_ips)
            log(f'Added block IP: {client_ip}')

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
            send_net_params(model.state_dict())
            print('Parameters pushed successfully')
            
            # 发送训练数据
            watch_data = {
                'return_list': 1.0,
                'avg_return_list': 0.5,
                'episode_lens': 10,
                'max_q_value_list': 0.8,
                'sharpe_ratio': 1.2,
                'sortino_ratio': 0.9,
                'max_drawdown': 0.15,
                'total_return': 2.5
            }
            send_train_data('val', watch_data)
            print('Train data sent successfully')
    else:
        print('Please specify mode: python net_center.py [server|client]')