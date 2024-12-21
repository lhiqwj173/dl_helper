from dl_helper.train_param import in_kaggle
from dl_helper.rl.rl_env.lob_env import data_producer, LOB_trade_env, ILLEGAL_REWARD, USE_CODES
from dl_helper.rl.rl_utils import ReplayBufferWaitClose
import matplotlib.pyplot as plt
import os
import time
import queue
import threading
from typing import Optional
from py_ext.tool import log

def plot_net(env):
    net = env.acc.net.copy()
    net_bm = env.acc.net_bm.copy()

    # 统一起始净值 1
    net = net / net[0]
    net_bm = net_bm / net_bm[0]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制策略净值和基准净值曲线
    ax.plot(net, label='Strategy', color='blue')
    ax.plot(net_bm, label='Benchmark', color='orange', linestyle='--')
    
    # 添加标题和标签
    ax.set_title('Net Value Comparison')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Net Value') 
    
    # 添加网格和图例
    ax.grid(True)
    ax.legend()
    
    return fig  # 直接返回图形对象，而不是保存文件

def produce_msg(info, reward):
    # 生成消息内容
    msg = ["-------------------------------"]
    
    # 更新跟踪器 非法/win/loss
    if info['act_criteria'] == -1:
        msg.append('illegal')
    elif info['act_criteria'] == 0:
        msg.append('win') 
    else:
        msg.append('loss')
    
    msg.append(f'reward: {reward:.6f}')

    # 更新评价指标
    for k, v in info.items():
        if k not in ['close', 'date_done', 'act_criteria']:
            msg.append(f'{k}: {v:.6f}')
    msg.append("-------------------------------")

    return '\n'.join(msg)

def paly():
    # 训练数据
    if in_kaggle:
        input_folder = r'/kaggle/input'
        data_folder_name = os.listdir(input_folder)[0]
        data_folder = os.path.join(input_folder, data_folder_name)
    else:
        data_folder = r'D:\L2_DATA_T0_ETF\train_data\RL_combine_data_test'

    # 初始化环境
    dp = data_producer(data_folder=data_folder, file_num=10)
    env = LOB_trade_env(data_producer=dp)
    
    buffer = ReplayBufferWaitClose(capacity=10000)

    # 回合的评价指标
    state, info = env.reset()
    done = False
    while not done:
        # 动作
        action = env.plot(state)
        log(f'action: {action}')

        # 环境交互
        next_state, reward, done1, done2, info = env.step(action)
        done = done1 or done2

        # 添加到回放池
        buffer.add(state, action, reward, next_state, done)

        # 如果 交易close 则需要回溯更新所有 reward 为最终close时的reward
        if info.get('close', False):
            buffer.update_reward(reward if reward!=ILLEGAL_REWARD else None)

            # 绘制净值
            fig = plot_net(env)
            
            # 生成消息
            msg = produce_msg(info, reward)

            # 使用GUI管理器显示通知
            env.show_notification('Trade Result', msg, fig)
            plt.close(fig)

            # 记录日志
            for line in msg.split('\n'):
                log(line)

            # 清空buffer
            buffer.reset()

        state = next_state
    

if __name__ == '__main__':
    paly()