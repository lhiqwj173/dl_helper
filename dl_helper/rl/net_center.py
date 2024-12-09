# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from py_ext.tool import log

from datetime import datetime, timezone, timedelta
import socket, time, sys, os, re
import pickle
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CODE = '0QYg9Ky17dWnN4eK'
PORT = 12346
alist_folder = r'/root/alist_data/rl_learning_process'
root_folder = '' if not os.path.exists(alist_folder) else alist_folder
csv_path = os.path.join(root_folder, 'val_test.csv')

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

def connect_client():
    """连接到参数中心服务器并返回socket连接"""
    HOST = '146.235.33.108'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        return client_socket
    except Exception as e:
        log(f"连接服务器失败: {e}")
        return None

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

def check_need_val_test():
    """查询是否需要验证测试
    返回:
        {'val': 需要验证
        'test': 需要测试}
    """
    HOST = '146.235.33.108'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        # 发送查询请求
        message = f'{CODE}_check'
        send_msg(client_socket, message.encode())
        
        # 接收响应
        response = recv_msg(client_socket)
        if response:
            return pickle.loads(response)
    finally:
        client_socket.close()
    
    return {'val': False, 'test': False}

def send_net_updates(params, metrics):
    """
    推送更新net
    包含 (模型参数, 学习监控指标)
    """
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
        params_data = pickle.dumps((cpu_params, metrics))
        send_msg(client_socket, params_data)
        
        # 等待确认
        response = recv_msg(client_socket)
        if response == b'ok':
            return True
    finally:
        client_socket.close()
    return False

def send_val_test_data(data_type, metrics):
    """发送训练数据"""
    HOST = '146.235.33.108'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        # 发送训练数据请求
        message = f'{CODE}_{data_type}'
        send_msg(client_socket, message.encode())
        
        # 发送训练数据
        data = pickle.dumps(metrics)
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
        return:         交易对奖励
    
    交易评价指标:
        sharpe_ratio:   夏普比率
        sortino_ratio:  索提诺比率
        max_drawdown:   最大回撤
        total_return:   总回报
        win:           胜率
        loss:          负率
        illegal:       非法率
    """
    # 检查数据是否存在
    if 'dt' not in watch_data or not watch_data['dt']:
        log('No dt data found')
        return
        
    for dtype in ['val', 'test']:
        if dtype not in watch_data:
            log(f'No {dtype} data found')
            return
            
    # 获取数据键,将交易评价指标单独处理
    rl_keys = ['return']
    trade_keys = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'total_return']
    rate_keys = ['win', 'loss', 'illegal']
    
    # 固定颜色
    colors = {
        'return': '#1f77b4',      # 蓝色
        'sharpe_ratio': '#8c564b', # 棕色
        'sortino_ratio': '#e377c2', # 粉色
        'max_drawdown': '#7f7f7f', # 灰色
        'total_return': '#bcbd22', # 黄绿色
        'win': '#2ca02c',         # 绿色
        'loss': '#d62728',        # 红色
        'illegal': '#ff7f0e'      # 橙色
    }
    
    # 计算总图数
    n_rl = 1  # return
    n_trade = 2  # ratio合并为1个,其他1个
    n_rate = 1   # win/loss/illegal合并为1个
    n_total = n_rl + n_trade + n_rate
    
    # 创建图表,每行一个子图
    fig, axes = plt.subplots(n_total, 1, figsize=(12, 4*n_total), sharex=True)
    if n_total == 1:
        axes = np.array([axes])
    
    # 获取时间变化点的索引
    dt_changes = []
    last_dt = None
    # 先处理dt列表,但不修改原数据
    for i, dt in enumerate(watch_data['dt']):
        # 按4小时对齐
        processed_dt = dt.replace(hour=dt.hour - dt.hour % 4, minute=0, second=0, microsecond=0)
        if processed_dt != last_dt:
            dt_changes.append((i, processed_dt))
            last_dt = processed_dt
            
    # 绘制强化学习指标
    # 1. return 
    ax = axes[0]
    for dtype in ['val', 'test']:
        if 'return' in watch_data[dtype]:
            alpha = 0.3 if dtype == 'val' else 1.0
            data = watch_data[dtype]['return']
            last_value = data[-1] if len(data) > 0 else 0
            ax.plot(data, color=colors['return'], alpha=alpha, label=f'{dtype}_return: {last_value:.4f}')
    ax.set_ylabel('Return')
    ax.grid(True)
    ax.legend()
    
    # 绘制交易评价指标
    # 1. sharpe_ratio和sortino_ratio
    ax = axes[1]
    for dtype in ['val', 'test']:
        for key in ['sharpe_ratio', 'sortino_ratio']:
            if key in watch_data[dtype]:
                alpha = 0.3 if dtype == 'val' else 1.0
                data = watch_data[dtype][key]
                last_value = data[-1] if len(data) > 0 else 0
                ax.plot(data, color=colors[key], alpha=alpha, label=f'{dtype}_{key}: {last_value:.4f}')
    ax.set_ylabel('Ratio')
    ax.grid(True)
    ax.legend()
    
    # 2. max_drawdown和total_return (双y轴)
    ax = axes[2]
    ax2 = ax.twinx()  # 创建共享x轴的第二个y轴
    
    lines = []
    # 在左轴绘制max_drawdown
    for dtype in ['val', 'test']:
        if 'max_drawdown' in watch_data[dtype]:
            alpha = 0.3 if dtype == 'val' else 1.0
            data = watch_data[dtype]['max_drawdown']
            last_value = data[-1] if len(data) > 0 else 0
            l = ax.plot(data, color=colors['max_drawdown'], alpha=alpha, label=f'{dtype}_max_drawdown: {last_value:.4f}')[0]
            lines.append(l)
    ax.set_ylabel('Max Drawdown')
    
    # 在右轴绘制total_return
    for dtype in ['val', 'test']:
        if 'total_return' in watch_data[dtype]:
            alpha = 0.3 if dtype == 'val' else 1.0
            data = watch_data[dtype]['total_return']
            last_value = data[-1] if len(data) > 0 else 0
            l = ax2.plot(data, color=colors['total_return'], alpha=alpha, label=f'{dtype}_total_return: {last_value:.4f}')[0]
            lines.append(l)
    ax2.set_ylabel('Total Return')
    
    # 合并两个轴的图例
    ax.legend(handles=lines, loc='upper left')
    ax.grid(True)

    # 3. win/loss/illegal rates
    ax = axes[3]
    for dtype in ['val', 'test']:
        for key in rate_keys:
            if key in watch_data[dtype]:
                alpha = 0.3 if dtype == 'val' else 1.0
                data = watch_data[dtype][key]
                last_value = data[-1] if len(data) > 0 else 0
                ax.plot(data, color=colors[key], alpha=alpha, label=f'{dtype}_{key}: {last_value:.4f}')
    ax.set_ylabel('Rate (%)')
    ax.grid(True)
    ax.legend()
    
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

def save_train_data_to_csv(train_data, csv_path):
    """将训练数据保存到CSV文件"""
    # 准备最新数据
    latest_data = {}
    for dtype in ['val', 'test', 'learn']:
        for k in train_data[dtype]:
            if len(train_data[dtype][k]) > 0:
                latest_data[f'{dtype}_{k}'] = train_data[dtype][k][-1]
    
    # 添加时间
    if len(train_data['dt']) > 0:
        latest_data['dt'] = train_data['dt'][-1]
    
    # 检查文件是否存在
    file_exists = os.path.exists(csv_path)
    
    # 获取表头和数据行
    headers = sorted(latest_data.keys())
    values = [str(latest_data[h]) for h in headers]
    
    if file_exists:
        # 读取已有数据
        old_df = pd.read_csv(csv_path)
        # 构建新数据行
        new_df = pd.DataFrame([values], columns=headers)
        # 合并数据
        df = pd.concat([old_df, new_df], ignore_index=True)
        # 保存到CSV
        df.to_csv(csv_path, index=False)
    else:
        # 创建新文件
        df = pd.DataFrame([values], columns=headers)
        df.to_csv(csv_path, index=False)

def init_train_data_from_csv(csv_path):
    """从CSV文件初始化训练数据"""
    train_data = {
        'learn': {},
        'val': {},
        'test': {},
        'dt': []
    }
    
    if not os.path.exists(csv_path):
        return train_data
        
    df = pd.read_csv(csv_path)
    
    # 读取时间列并转换为datetime格式
    if 'dt' in df.columns:
        train_data['dt'] = pd.to_datetime(df['dt']).tolist()
    
    # 读取val和test数据
    for col in df.columns:
        if col == 'dt':
            continue
            
        dtype, key = col.split('_', 1)  # 例如: 'val_win_rate' -> ('val', 'win_rate')
        if dtype in ['val', 'test', 'learn']:
            if key not in train_data[dtype]:
                train_data[dtype][key] = []
            train_data[dtype][key] = [float(x) if not pd.isna(x) else float('nan') for x in df[col].tolist()]
            
    return train_data


def handle_val_test_data(train_data):
    """处理验证测试数据"""
    # 数据补齐
    # 找出所有类型中最长的列表长度
    max_len = 0
    for dtype in ['val', 'test', 'learn']:
        for k in train_data[dtype]:
            max_len = max(max_len, len(train_data[dtype][k]))
    # 对齐所有类型的所有列表到最大长度
    for dtype in ['val', 'test', 'learn']:
        for k in train_data[dtype]:
            curr_len = len(train_data[dtype][k])
            if curr_len < max_len:
                # 获取前一个值,若无则用nan
                pad_value = train_data[dtype][k][-1] if curr_len > 0 else float('nan')
                train_data[dtype][k].extend([pad_value] * (max_len - curr_len))

    # 数据截断， 最多允许 500 个数据
    for dtype in ['val', 'test', 'learn']:
        for k in train_data[dtype]:
            train_data[dtype][k] = train_data[dtype][k][:500]
    train_data['dt'] = train_data['dt'][:500]

    # 增量保存最新的验证测试数据到csv
    save_train_data_to_csv(train_data, csv_path)
    
    # 绘制数据
    plot_learning_process(root_folder, train_data)

    return train_data

def run_param_center(agent, tau= 0.005, simple_test=False):
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
    
    # 学习进度数据
    learn_metrics = {}

    # 验证测试数据
    train_data = init_train_data_from_csv(csv_path)
    
    # 参数更新计数
    update_count = 0

    # 是否需要验证测试
    need_val = False
    need_test = False

    while True:
        client_socket, client_address = server_socket.accept()
        client_ip, client_port = client_address

        # 检查是否是block ip
        if client_ip in block_ips:
            log(f"Blocked connection from {client_ip}")
            client_socket.close()
            continue

        msg_header = f'[{client_ip:<15} {client_port:<5}][{update_count}]'
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
                            log(f'{msg_header} Parameters sent')

                        elif cmd == 'check':
                            # 检查是否需要验证测试
                            send_msg(client_socket, pickle.dumps({'val': need_val, 'test': need_test}))
                            msg = f'{msg_header} Check response sent'
                            if need_val:
                                msg += ' val:True'
                            if need_test:
                                msg += ' test:True'
                            log(msg)
                            # 重置
                            if need_val:
                                need_val = False
                            if need_test:
                                need_test = False

                        elif cmd == 'update':
                            # 接收新参数
                            update_data = recv_msg(client_socket)
                            if update_data is None:
                                need_block = True
                            else:
                                # 软更新参数
                                # metrics 学习监控指标:
                                #     {
                                #         'total_reward': 2.5, 
                                #         'average_reward': 0.5, 
                                #         'moving_average_reward': 0.5, 
                                #         'action_distribution': {2: 0.6, 1: 0.2, 0: 0.2}, 
                                #         'total_td_error': 0.5, 
                                #         'total_loss': 0.25, 
                                #         'average_illegal_ratio': 0.0, 
                                #         'average_win_ratio': 0.2, 
                                #         'average_loss_ratio': 0.8,
                                #     }
                                new_params, metrics = pickle.loads(update_data)

                                # 处理新参数
                                model = update_model_params(model, new_params)
                                # target模型 与 q_net 模型参数同步
                                agent.target_q_net = update_model_params(agent.target_q_net, model.state_dict(), tau=1)
                                log(f'{msg_header} Parameters updated')
                                send_msg(client_socket, b'ok')
                                # 保存最新参数
                                agent.save(alist_folder)
                                
                                # 处理学习评价指标
                                for k, v in metrics.items():
                                    if k not in learn_metrics:
                                        learn_metrics[k] = []
                                    learn_metrics[k].append(v)

                                # 更新计数
                                update_count += 1
                                # 更新是否需要验证测试
                                if simple_test:
                                    _test_count = 20
                                    _val_count = 10
                                else:
                                    _val_count = 5000
                                    _test_count = _val_count * 50

                                if update_count % _test_count == 0:
                                    need_test = True
                                elif update_count % _val_count == 0:
                                    need_val = True

                        elif cmd in ['val', 'test']:
                            # 接收训练数据
                            data_type = cmd
                            train_data_new = recv_msg(client_socket)
                            if train_data_new is None:
                                need_block = True
                            else:
                                # 更新训练数据
                                metrics = pickle.loads(train_data_new)
                                log(f'{msg_header} {cmd}_metrics: {metrics}')
                                for k in metrics:
                                    if k not in train_data[data_type]:
                                        train_data[data_type][k] = []
                                    train_data[data_type][k].append(metrics[k])
                                
                                if cmd == 'val':
                                    # learn_metrics取均值新增到 train_data 中
                                    for k in learn_metrics:
                                        if k not in train_data['learn']:
                                            train_data['learn'][k] = []
                                        length = len(learn_metrics[k])
                                        if length > 0:
                                            train_data['learn'][k].append(sum(learn_metrics[k]) / length)
                                        log(f'{msg_header} length learn_metrics[{k}]: {length}')
                                    # 清空learn_metrics
                                    learn_metrics = {}

                                send_msg(client_socket, b'ok')

                                # 增加北京时间(每4小时)
                                dt = datetime.now(timezone(timedelta(hours=8)))  # 使用timezone确保是北京时间
                                train_data['dt'].append(dt)
                                train_data = handle_val_test_data(train_data)
                                log(f'{msg_header} handle {cmd}_data done')

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
        # except Exception as e:
        #     log(f"Error processing request: {e}")

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
            send_net_updates(model.state_dict())
            print('Parameters pushed successfully')
            
            # 发送训练数据
            watch_data = {
                'return': 1.0,
                'sharpe_ratio': 1.2,
                'sortino_ratio': 0.9,
                'max_drawdown': 0.15,
                'total_return': 2.5
            }
            send_train_data('val', watch_data)
            print('Train data sent successfully')
    else:
        print('Please specify mode: python net_center.py [server|client]')