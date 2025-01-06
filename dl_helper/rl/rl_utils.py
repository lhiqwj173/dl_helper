from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import collections
import random, pickle
import cProfile
import time
import os
import sys
import pstats
from datetime import datetime, timedelta, timezone
import threading
import matplotlib.pyplot as plt

from dl_helper.rl.run import run_client_learning, run_client_learning_device_breakout
from dl_helper.tool import keep_upload_log_file, init_logger_by_ip, in_windows
from py_ext.tool import log

alist_folder = r'/root/alist_data/rl_learning_process'
root_folder = os.path.expanduser("~") if (in_windows() or (not os.path.exists(alist_folder))) else alist_folder

def simplify_rllib_metrics(data, out_func=print):
    important_metrics = {
        "环境运行器": {},
        "评估": {},
        "学习者": {},
    }

    if 'counters' in data:
        if 'num_env_steps_sampled' in data["counters"]:
            important_metrics["环境运行器"]["采样环境总步数"] = data["counters"]["num_env_steps_sampled"]

    if 'env_runners' in data:
        if 'episode_return_mean' in data["env_runners"]:
            important_metrics["环境运行器"]["episode平均回报"] = data["env_runners"]["episode_return_mean"]
        if 'episode_return_max' in data["env_runners"]:
            important_metrics["环境运行器"]["episode最大回报"] = data["env_runners"]["episode_return_max"]
        if 'episode_len_mean' in data["env_runners"]:
            important_metrics["环境运行器"]["episode平均步数"] = data["env_runners"]["episode_len_mean"]
        if 'episode_len_max' in data["env_runners"]:
            important_metrics["环境运行器"]["episode最大步数"] = data["env_runners"]["episode_len_max"]
        if 'num_env_steps_sampled' in data["env_runners"]:
            important_metrics["环境运行器"]["采样环境总步数"] = data["env_runners"]["num_env_steps_sampled"]
        if 'num_episodes' in data["env_runners"]:
            important_metrics["环境运行器"]["episodes计数"] = data["env_runners"]["num_episodes"]

    if 'evaluation' in data:
        if 'env_runners' in data["evaluation"]:
            if 'episode_return_mean' in data["evaluation"]["env_runners"]:
                important_metrics["评估"]["episode平均回报"] = data["evaluation"]["env_runners"]["episode_return_mean"]
            if 'episode_return_max' in data["evaluation"]["env_runners"]:
                important_metrics["评估"]["episode最大回报"] = data["evaluation"]["env_runners"]["episode_return_max"]
            if 'episode_len_mean' in data["evaluation"]["env_runners"]:
                important_metrics["评估"]["episode平均步数"] = data["evaluation"]["env_runners"]["episode_len_mean"]
            if 'episode_len_max' in data["evaluation"]["env_runners"]:
                important_metrics["评估"]["episode最大步数"] = data["evaluation"]["env_runners"]["episode_len_max"]

    if 'learners' in data:
        if 'default_policy' in data["learners"]:
            if 'entropy' in data["learners"]["default_policy"]:
                important_metrics["学习者"]["默认策略"]["熵"] = data["learners"]["default_policy"]["entropy"]
            if 'policy_loss' in data["learners"]["default_policy"]:
                important_metrics["学习者"]["默认策略"]["策略损失"] = data["learners"]["default_policy"]["policy_loss"]
            if 'vf_loss' in data["learners"]["default_policy"]:
                important_metrics["学习者"]["默认策略"]["值函数损失"] = data["learners"]["default_policy"]["vf_loss"]
            if 'total_loss' in data["learners"]["default_policy"]:
                important_metrics["学习者"]["默认策略"]["总损失"] = data["learners"]["default_policy"]["total_loss"]

    if 'time_this_iter_s' in data:
        important_metrics["本轮时间"] = data["time_this_iter_s"]
    if 'num_training_step_calls_per_iteration' in data:
        important_metrics["每轮训练步数"] = data["num_training_step_calls_per_iteration"]
    if 'training_iteration' in data:
        important_metrics["训练迭代次数"] = data["training_iteration"]
            
    out_func(f"--------- 训练迭代: {important_metrics['训练迭代次数']} ---------")
    out_func("环境运行器:")
    if '环境运行器' in important_metrics and important_metrics['环境运行器']:
        if 'episode平均回报' in important_metrics['环境运行器']:
            out_func(f"  episode平均回报: {important_metrics['环境运行器']['episode平均回报']:.4f}")
        if 'episode最大回报' in important_metrics['环境运行器']:
            out_func(f"  episode最大回报: {important_metrics['环境运行器']['episode最大回报']:.4f}")
        if 'episode平均步数' in important_metrics['环境运行器']:
            out_func(f"  episode平均步数: {important_metrics['环境运行器']['episode平均步数']:.4f}")
        if 'episode最大步数' in important_metrics['环境运行器'] and important_metrics['环境运行器']['episode最大步数'] is not None:
            out_func(f"  episode最大步数: {important_metrics['环境运行器']['episode最大步数']}")
        if '采样环境总步数' in important_metrics['环境运行器']:
            out_func(f"  采样环境总步数: {important_metrics['环境运行器']['采样环境总步数']}")
        if 'episodes计数' in important_metrics['环境运行器']:
            out_func(f"  episodes计数: {important_metrics['环境运行器']['episodes计数']}")
    else:
        out_func("  无环境运行器数据")
    
    out_func("\n评估:")
    if '评估' in important_metrics and important_metrics['评估']:
        if 'episode平均回报' in important_metrics['评估'] and important_metrics['评估']['episode平均回报'] is not None:
            out_func(f"  episode平均回报: {important_metrics['评估']['episode平均回报']:.4f}")
            if 'episode最大回报' in important_metrics['评估']:
                out_func(f"  episode最大回报: {important_metrics['评估']['episode最大回报']:.4f}")
            if 'episode平均步数' in important_metrics['评估']:
                out_func(f"  episode平均步数: {important_metrics['评估']['episode平均步数']:.4f}")
            if 'episode最大步数' in important_metrics['评估'] and important_metrics['评估']['episode最大步数'] is not None:
                out_func(f"  episode最大步数: {important_metrics['评估']['episode最大步数']}")
        else:
            out_func("  无评估数据")
    else:
        out_func("  无评估数据")
        
    out_func("\n学习者(默认策略):")
    if '学习者' in important_metrics and important_metrics['学习者'] and '默认策略' in important_metrics['学习者']:
        if '熵' in important_metrics['学习者']['默认策略']:
            out_func(f"  熵: {important_metrics['学习者']['默认策略']['熵']:.4f}")
        if '策略损失' in important_metrics['学习者']['默认策略']:
            out_func(f"  策略损失: {important_metrics['学习者']['默认策略']['策略损失']:.4f}")
        if '值函数损失' in important_metrics['学习者']['默认策略']:
            out_func(f"  值函数损失: {important_metrics['学习者']['默认策略']['值函数损失']:.4f}")
        if '总损失' in important_metrics['学习者']['默认策略']:
            out_func(f"  总损失: {important_metrics['学习者']['默认策略']['总损失']:.4f}")
    else:
        out_func("  无学习者数据")
    
    if '本轮时间' in important_metrics:
        out_func(f"\n本轮时间: {important_metrics['本轮时间']:.4f}")
    if '每轮训练步数' in important_metrics:
        out_func(f"每轮训练步数: {important_metrics['每轮训练步数']}")
    out_func('-'*30)

class ExperimentHandler:
    """处理单个实验的类"""
    def __init__(self, train_title, agent_class_name=None, agent_kwargs=None, simple_test=False, period_day=True):
        """
        train_title: 训练标题
        agent_class_name: 代理类名
        agent_kwargs: 代理参数
        simple_test: 是否简单测试
        period_day: train_periods 是否按天统计
        """
        self.train_title = train_title

        self.simple_test = simple_test
        self.period_day = period_day

        # 创建实验目录
        self.exp_folder = os.path.join(root_folder, train_title)
        os.makedirs(self.exp_folder, exist_ok=True)
        self.csv_path = os.path.join(self.exp_folder, 'val_test.csv')
        
        self.agent = None
        self.param_server = None
        if agent_class_name is not None:    
            self.agent = globals()[agent_class_name](**agent_kwargs)
            # 载入模型数据
            self.agent.load(self.exp_folder)
            # 参数服务器
            self.param_server = AsyncRLParameterServer(self.agent,
                                                    learning_rate=agent_kwargs.get('learning_rate', 0.001),
                                                    staleness_threshold=20,
                                                    momentum=0.9,
                                                    importance_decay=0.8,
                                                    max_version_delay=100)
        
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
            - average_loss: 平均损失值

            图3 (若方差为0, 则不进行绘制)
            - illegal_ratio: 平均非法动作率

            图4 (若方差为0, 则不进行绘制)
            - win_ratio: 平均胜率
            - loss_ratio: 平均败率

            图5
            - action_{k}_ratio k: 0-2

            图6 (不存在字段, 则不进行绘制)
            - hold_length: 平均持仓时间

        交易评价指标
            图7 (不存在字段, 则不进行绘制)
            - sortino_ratio
            - sortino_ratio_bm

            图8 (不存在字段, 则不进行绘制)
            - max_drawdown
            - max_drawdown_bm

            图9 (不存在字段, 则不进行绘制)
            - total_return
            - total_return_bm

        train_periods: 训练周期数
        """
        # 检查数据是否存在
        if 'dt' not in metrics or not metrics['dt']:
            log('No dt data found')
            return
            
        # 固定颜色
        colors = {
            'moving_average_reward': '#ff7f0e',
            'average_loss': '#9467bd',
            'illegal_ratio': '#8c564b',
            'win_ratio': '#e377c2',
            'loss_ratio': '#7f7f7f',
            'action_0': '#bcbd22',
            'action_1': '#17becf',
            'action_2': '#1f77b4',
            'action_3': '#ff7f0e',
            'action_4': '#2ca02c',
            'action_5': '#d62728',
            'action_6': '#9467bd',
            'hold_length': '#2ca02c',
            'sortino_ratio': '#d62728',
            'sortino_ratio_bm': '#d62728',
            'max_drawdown': '#ff7f0e',
            'max_drawdown_bm': '#ff7f0e',
            'total_return': '#9467bd',
            'total_return_bm': '#9467bd'
        }

        # 确定需要绘制的图表数量
        plots_to_draw = []
        
        # 图1-2总是绘制
        plots_to_draw.extend([0, 1])
        
        # 检查图3-4是否有非零方差
        for metric in ['illegal_ratio']:
            for dtype in ['learn', 'val', 'test']:
                if metric in metrics[dtype] and np.var(metrics[dtype][metric]) > 0:
                    plots_to_draw.append(2)
                    break
                    
        for metric in ['win_ratio', 'loss_ratio']:
            for dtype in ['learn', 'val', 'test']:
                if metric in metrics[dtype] and np.var(metrics[dtype][metric]) > 0:
                    plots_to_draw.append(3)
                    break
                    
        # 图5总是绘制
        plots_to_draw.append(4)
        
        # 检查图6-9是否存在相应字段
        metrics_to_check = [
            ['hold_length'],
            ['sortino_ratio'],
            ['max_drawdown'],
            ['total_return']
        ]
        
        for i, metric_group in enumerate(metrics_to_check):
            for metric in metric_group:
                for dtype in ['learn', 'val', 'test']:
                    if metric in metrics[dtype]:
                        plots_to_draw.append(i + 5)
                        break
                if i + 5 in plots_to_draw:
                    break
                    
        plots_to_draw = sorted(list(set(plots_to_draw)))
        
        # 创建图表
        fig, axes = plt.subplots(len(plots_to_draw), 1, figsize=(12, 4*len(plots_to_draw)), sharex=True)
        if len(plots_to_draw) == 1:
            axes = [axes]
        
        # 获取时间变化点的索引
        dt_changes = []
        last_dt = None
        for i, dt in enumerate(metrics['dt']):
            processed_dt = dt.replace(hour=dt.hour - dt.hour % 4, minute=0, second=0, microsecond=0)
            if processed_dt != last_dt:
                dt_changes.append((i, processed_dt, metrics['learn']['train_periods'][i]))
                last_dt = processed_dt

        # 设置图表标题
        periods = metrics["learn"]["train_periods"][-1]
        if self.period_day:
            fig.suptitle(f'Learning Process ({f"{int(periods/365):.2e}" if int(periods/365)>=1000 else int(periods/365)}Ys {int(periods%365)}ds)', fontsize=16)
        else:
            fig.suptitle(f'Learning Process ({f"{periods:.2e}" if periods >= 1000 else int(periods)} periods)', fontsize=16)

        plot_idx = 0
        
        # 图1: moving_average_reward
        if 0 in plots_to_draw:
            ax = axes[plot_idx]
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
            plot_idx += 1

        # 图2: average_loss
        if 1 in plots_to_draw:
            ax = axes[plot_idx]
            for dtype in ['learn', 'val', 'test']:
                key = 'average_loss'
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
            plot_idx += 1

        # 图3: illegal_ratio
        if 2 in plots_to_draw:
            ax = axes[plot_idx]
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
            plot_idx += 1

        # 图4: win_ratio & loss_ratio
        if 3 in plots_to_draw:
            ax = axes[plot_idx]
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
            plot_idx += 1

        # 图5: action ratios
        if 4 in plots_to_draw:
            ax = axes[plot_idx]
            action_dim = len([i for i in metrics['learn'] if i.startswith('action_')])
            for dtype in ['learn', 'val', 'test']:
                for i in range(action_dim):
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
            plot_idx += 1

        # 图6: hold_length
        if 5 in plots_to_draw:
            ax = axes[plot_idx]
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
            plot_idx += 1

        # 图7: sortino_ratio
        if 6 in plots_to_draw:
            ax = axes[plot_idx]
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
            plot_idx += 1

        # 图8: max_drawdown
        if 7 in plots_to_draw:
            ax = axes[plot_idx]
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
            plot_idx += 1

        # 图9: total_return
        if 8 in plots_to_draw:
            ax = axes[plot_idx]
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
            if self.period_day:
                ax.set_xticklabels([f"{dt.strftime('%d %H')}({f'{int(periods/365):.2e}' if int(periods/365)>=1000 else int(periods/365)}Ys {int(periods%365)}ds)" if periods >= 365 else f"{dt.strftime('%d %H')}({int(periods)}ds)" for _, dt, periods in dt_changes], rotation=45)
            else:
                ax.set_xticklabels([f"{dt.strftime('%d %H')}({f'{periods:.2e}' if periods >= 1000 else int(periods)} periods)" for _, dt, periods in dt_changes], rotation=45)
        
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

    def check_need_val_test(self):
        """检查是否需要进行验证/测试"""
        # 30min一次val, 2小时一次test
        response = 'no'

        t = time.time()
        if t - self.last_val_time > 1800:
        # # FOR TEST
        # if t - self.last_val_time > 60 * 3:
            response = 'val'
            self.last_val_time = t
        elif t - self.last_test_time > 7200:
        # # FOR TEST
        # elif t - self.last_test_time > 60 * 3:
            response = 'test'
            self.last_test_time = t
        return response

    def update_learn_metrics(self, metrics):
        for k, v in metrics.items():
            if k not in self.learn_metrics:
                self.learn_metrics[k] = []
            self.learn_metrics[k].append(v)

    def update_val_test_metrics(self, data_type, metrics):
        for k in metrics:
            if k not in self.train_data[data_type]:
                self.train_data[data_type][k] = []
            self.train_data[data_type][k].append(metrics[k])

        backup_path = os.path.join(self.exp_folder, 'learn_metrics_backup.pkl')
        with open(backup_path, 'wb') as f:
            pickle.dump(self.learn_metrics, f)

        action_dim = len([i for i in self.learn_metrics if i.startswith('action_')])
        for k in self.learn_metrics:
            if k not in self.train_data['learn']:
                self.train_data['learn'][k] = []

            length = len(self.learn_metrics[k])
            if length > 0:
                if k == 'train_periods':
                    add_days = np.nansum(self.learn_metrics[k])
                    if len(self.train_data['learn'][k]) > 0:
                        add_days += self.train_data['learn'][k][-1]
                    self.train_data['learn'][k].append(add_days)

                elif k.startswith('action_'):
                    continue

                else:
                    self.train_data['learn'][k].append(np.nanmean(self.learn_metrics[k]))

        # 处理 action_ratio
        # For action probabilities, take the mean of each action separately
        # and normalize to ensure they sum to 1
        action_probs = np.array([np.nanmean(self.learn_metrics[f'action_{act}_ratio']) for act in range(action_dim)])
        action_probs = action_probs / np.sum(action_probs)  # Normalize
        for i in range(action_dim):
            self.train_data['learn'][f'action_{i}_ratio'].append(action_probs[i])

        # 重置学习指标
        self.learn_metrics = {}

        dt = datetime.now(timezone(timedelta(hours=8)))
        self.train_data['dt'].append(dt)
        self.train_data = self.handle_val_test_data()

    def handle_request(self, client_socket, msg_header, cmd):
        """处理客户端请求"""
        try:
            if cmd == 'get':
                assert self.agent is not None, 'agent is not initialized'
                params_data = pickle.dumps((self.agent.get_params_to_send(), self.agent.version))
                send_msg(client_socket, params_data)
                log(f'{msg_header} Parameters sent, version: {self.agent.version}')

            elif cmd == 'check':
                response = self.check_need_val_test()
                send_msg(client_socket, response.encode())
                msg = f'{msg_header} Check response sent: {response}'
                log(msg)

            elif cmd == 'update_gradients':
                assert self.param_server is not None, 'param_server is not initialized'
                update_data = recv_msg(client_socket)
                if update_data is None:
                    return
                grads, importance, version, metrics = pickle.loads(update_data)

                # 更新梯度
                res = self.param_server.process_update(grads, importance, version)
                if not res:
                    return

                log(f'{msg_header} Parameters updated, version: {self.agent.version}')

                send_msg(client_socket, b'ok')
                self.agent.save(self.exp_folder)
                
                # 更新学习指标
                self.update_learn_metrics(metrics)

                self.update_count += 1

            elif cmd in ['val', 'test']:
                data_type = cmd
                train_data_new = recv_msg(client_socket)
                if train_data_new is None:
                    return
                    
                metrics = pickle.loads(train_data_new)
                log(f'{msg_header} {cmd}_metrics: {metrics}')
                self.update_val_test_metrics(data_type, metrics)   
                log(f'{msg_header} handle {cmd}_data done')

                send_msg(client_socket, b'ok')

        except ConnectionResetError:
            pass


def add_train_title_item(train_title, agent_class, agent_kwargs, simple_test, period_day=True):
    file = os.path.join(root_folder, f'{train_title}.data')
    if os.path.exists(file):
        return
    with open(file, 'wb') as f:
        pickle.dump((agent_class.__name__, agent_kwargs, simple_test, period_day), f)

def read_train_title_item():
    res = {}
    for file in os.listdir(root_folder):
        if file.endswith('.data'):
            title = file.replace('.data', '')
            agent_class_name, agent_kwargs, simple_test, period_day = pickle.load(open(os.path.join(root_folder, file), 'rb'))
            res[title] = (agent_class_name, agent_kwargs, simple_test, period_day)
    return res

class LRTrainParams:
    help_doc = """
    命令行参数说明:
    
    训练参数:
        train_title=<str>           训练标题
        run_client_learning_func=<func>  运行客户端学习函数
        agent_class=<class>         智能体类
        agent_kwargs=<dict>         智能体初始化参数
        lr=<float>                  学习率, 默认1e-4
        num_episodes=<int>          训练回合数, 默认5000
        hidden_dim=<int>            隐藏层维度, 默认128
        gamma=<float>               折扣因子, 默认0.98
        epsilon=<float>             探索率, 默认0.5
        target_update=<int>         目标网络更新频率, 默认50
        buffer_size=<int>           经验回放池大小, 默认3000
        minimal_size=<int>          最小训练样本数, 默认3000
        batch_size=<int>            批次大小, 默认256
        sync_interval_learn_step=<int>  同步参数间隔, 默认150
        learn_interval_step=<int>   学习更新间隔, 默认4
        n_step=<int>                多步学习步数, 默认1
        use_noisy=<bool>            是否使用noisy网络, 默认True
        train_buffer_class=<class>  训练经验回放池类

    模型参数:
        need_reshape=<tuple>        是否需要reshape, 默认None
        features_dim=<int>          特征维度
        action_dim=<int>            动作维度
        net_arch=<list>             网络结构
        features_extractor_class=<class>  特征提取器类
        features_extractor_kwargs=<dict>  特征提取器参数, 默认{}

    运行参数:
        period_day=<bool>           是否是周期性训练, 默认False
        simple_test=<bool>          启用简单测试模式, 默认False
        test_val=<val/test/all>     验证/测试模式
        local=<bool>                是否是本地训练, 默认False
        server                      以服务端模式运行
        client                      以客户端模式运行(默认)

    性能分析参数:
        profile                     启用性能分析, 默认False
        profile_stats_n=<int>       显示前N个耗时函数, 默认30
        profile_output_dir=<str>    性能分析结果保存目录, 默认'profile_results'

    使用示例:
        python script.py lr=0.001 num_episodes=1000 server
    """

    def __init__(self, 
        train_title, 
        run_client_learning_func,
        agent_class, 
        need_reshape, 
        features_dim, 
        action_dim, 
        net_arch, 
        train_buffer_class, 
        features_extractor_class, 
        features_extractor_kwargs,
        **kwargs
    ):
        self.kwargs = kwargs

        # 训练参数
        self.train_title = train_title
        self.run_client_learning_func = run_client_learning_func
        self.agent_class = agent_class
        self.lr = 1e-4
        self.num_episodes = 5000
        self.hidden_dim = 128
        self.gamma = 0.98
        self.epsilon = 0.5
        self.target_update = 50
        self.buffer_size = 3000
        self.minimal_size = 3000
        self.batch_size = 256
        self.sync_interval_learn_step = 150
        self.learn_interval_step = 4
        self.n_step = 1
        self.use_noisy = True
        self.train_buffer_class = train_buffer_class

        # 模型参数
        self.need_reshape = need_reshape
        self.features_dim = features_dim
        self.action_dim = action_dim
        self.net_arch = net_arch
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        # 运行参数
        self.period_day = False
        self.simple_test = False
        self.val_test = ''
        self.local = False
        self.is_server = False

        # 性能分析参数
        self.enable_profiling = False
        self.profile_stats_n = 30  # 显示前N个耗时函数
        self.profile_output_dir = 'profile_results'  # 性能分析结果保存目录

    def run(self):
        """运行训练"""
        if not self.is_server:
            # 训练者客户端
            # 保持上传日志文件
            upload_thread = threading.Thread(target=keep_upload_log_file, args=(self.train_title,), daemon=True)
            upload_thread.start()

            # 初始化agent
            agent = self.agent_class(**self.get_agent_kwargs())

            args = (agent, self.num_episodes, self.minimal_size, self.batch_size, 
                self.sync_interval_learn_step, self.learn_interval_step, self.local)
            kwargs = {'simple_test': self.simple_test, 'val_test': self.val_test, 
                    'enable_profiling': self.enable_profiling}
            run_client_learning(self.run_client_learning_func, args, kwargs)
        else:
            # 服务端
            add_train_title_item(
                self.train_title, 
                self.agent_class, 
                self.get_agent_kwargs(), 
                self.simple_test, 
                period_day=self.period_day
            )

    def update_from_args(self, args):
        """从命令行参数更新配置"""
        # 如果输入help,打印帮助文档并退出
        if 'help' in args:
            print(self.help_doc)
            sys.exit(0)

        for arg in args:
            if arg.startswith('train_title='):
                self.train_title = arg.split('=')[1]
            elif arg.startswith('lr='):
                self.lr = float(arg.split('=')[1])
            elif arg.startswith('num_episodes='):
                self.num_episodes = int(arg.split('=')[1])
            elif arg.startswith('hidden_dim='):
                self.hidden_dim = int(arg.split('=')[1])
            elif arg.startswith('gamma='):
                self.gamma = float(arg.split('=')[1])
            elif arg.startswith('epsilon='):
                self.epsilon = float(arg.split('=')[1])
            elif arg.startswith('target_update='):
                self.target_update = int(arg.split('=')[1])
            elif arg.startswith('buffer_size='):
                self.buffer_size = int(arg.split('=')[1])
            elif arg.startswith('minimal_size='):
                self.minimal_size = int(arg.split('=')[1])
            elif arg.startswith('batch_size='):
                self.batch_size = int(arg.split('=')[1])
            elif arg.startswith('sync_interval_learn_step='):
                self.sync_interval_learn_step = int(arg.split('=')[1])
            elif arg.startswith('learn_interval_step='):
                self.learn_interval_step = int(arg.split('=')[1])
            elif arg == 'local':
                self.local = True
            elif arg == 'simple_test':
                self.simple_test = True
            elif arg.startswith('test_val='):
                self.val_test = arg.split('=')[1]
            elif arg == 'profile':
                self.enable_profiling = True
            elif arg.startswith('profile_stats_n='):
                self.profile_stats_n = int(arg.split('=')[1])
            elif arg.startswith('profile_output_dir='):
                self.profile_output_dir = arg.split('=')[1]
            elif arg == 'server':
                self.is_server = True

    def get_agent_kwargs(self):
        """获取agent初始化参数"""
        # # 基础参数
        # learning_rate,
        # gamma,
        # epsilon,
        # target_update,
        # # 基类参数
        # buffer_size,
        # train_buffer_class,
        # train_title,
        # action_dim,
        # features_dim,
        # features_extractor_class,
        # features_extractor_kwargs=None,
        # use_noisy=True,
        # n_step=1,
        # net_arch=None,
        # need_reshape=None,
        agent_kwarg = {
            'learning_rate': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'target_update': self.target_update,
            'buffer_size': self.buffer_size,
            'train_buffer_class': self.train_buffer_class,
            'train_title': self.train_title,
            'action_dim': self.action_dim,
            'features_dim': self.features_dim,
            'features_extractor_class': self.features_extractor_class,
            'features_extractor_kwargs': self.features_extractor_kwargs,
            'use_noisy': self.use_noisy,
            'n_step': self.n_step,
            'net_arch': self.net_arch,
            'need_reshape': self.need_reshape,
        }

        # 特定的参数
        if self.agent_class.__name__ == 'C51':
            extra_kwargs = {'n_atoms':51, 'v_min':-10, 'v_max':10}
        elif self.agent_class.__name__ == 'DQN':
            extra_kwargs = {'dqn_type':DD_DQN}
        else:
            extra_kwargs = {}

        for k, v in extra_kwargs.items():
            if k in self.kwargs:
                agent_kwarg[k] = self.kwargs[k]
            else:
                log(f"{self.agent_class.__name__}使用默认参数{k}: {v}")

        return agent_kwarg

class Profiler:
    def __init__(self, params):
        self.params = params
        self.profiler = None
        self.start_time = None

    def before_train(self):
        # 根据参数决定是否启用性能分析
        if self.params.enable_profiling:
            # 创建性能分析结果保存目录
            os.makedirs(self.params.profile_output_dir, exist_ok=True)
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.start_time = time.time()

    def after_train(self):
        # 如果启用了性能分析，输出并保存结果
        if self.params.enable_profiling:
            self.profiler.disable()
            end_time = time.time()

            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 创建结果文件名
            stats_file = os.path.join(self.params.profile_output_dir, f'profile_stats_{timestamp}.txt')
            
            # 打开文件并重定向stdout
            with open(stats_file, 'w') as f:
                # 记录总运行时间
                total_time = end_time - self.start_time
                f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
                
                # 创建性能分析报告
                stats = pstats.Stats(self.profiler, stream=f)
                stats.sort_stats('cumulative')  # 按累计时间排序
                stats.print_stats(self.params.profile_stats_n)  # 显示前N个耗时最多的函数
                
                # 保存调用关系图
                stats.print_callers()
                stats.print_callees()
            
            # 同时在控制台显示结果
            log(f"Total execution time: {total_time:.2f} seconds")
            log(f"Profile results saved to: {stats_file}")
            
            # 创建二进制统计文件以供后续分析
            stats.dump_stats(os.path.join(self.params.profile_output_dir, f'profile_stats_{timestamp}.prof'))


def _get_n_step_info(n_step_buffer, gamma):
    """计算n步return"""
    reward, next_state, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]
        reward = r + gamma * reward * (1 - d)
        next_state = n_s if d else next_state
        done = d

    return reward, next_state, done

class ReplayBuffer:
    def __init__(self, capacity, n_step=1, gamma=0.99):
        self.buffer = collections.deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        # n步缓存
        self.n_step_buffer = collections.deque(maxlen=n_step) if n_step > 1 else None
        # 预定义数据类型
        self.dtypes = [np.float32, np.int64, np.float32, np.float32, np.float32]

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self.n_step > 1:
            self.n_step_buffer.append(transition)

            # 只有当n步缓存满了才添加到主缓存
            if len(self.n_step_buffer) == self.n_step:
                # 计算n步return
                n_reward, n_next_state, n_done = _get_n_step_info(self.n_step_buffer, self.gamma)
                state, action, reward, next_state, done = self.n_step_buffer[0]
                
                # 存储原始transition和n步信息
                self.buffer.append((
                    state, action, reward, next_state, done,  # 原始数据
                    n_reward, n_next_state, n_done  # n步数据
                ))
        else:
            self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]

        # 分离原始数据和n步数据
        states = np.array([t[0] for t in transitions], dtype=self.dtypes[0])
        actions = np.array([t[1] for t in transitions], dtype=self.dtypes[1])
        rewards = np.array([t[2] for t in transitions], dtype=self.dtypes[2])
        next_states = np.array([t[3] for t in transitions], dtype=self.dtypes[3])
        dones = np.array([t[4] for t in transitions], dtype=self.dtypes[4])

        n_rewards = None
        n_next_states = None
        n_dones = None
        # 如果使用n步学习，添加n步数据
        if self.n_step > 1:
            n_rewards = np.array([t[5] for t in transitions], dtype=self.dtypes[2])
            n_next_states = np.array([t[6] for t in transitions], dtype=self.dtypes[3])
            n_dones = np.array([t[7] for t in transitions], dtype=self.dtypes[4])

        return (states, actions, rewards, next_states, dones,
                n_rewards, n_next_states, n_dones)

    def get(self, batch_size):
        # 只针对1步 进行验证
        n = min(batch_size, len(self.buffer))
        # 预分配列表空间
        transitions = []
        transitions.extend(self.buffer.popleft() for _ in range(n))
        # 预分配numpy数组
        return tuple(np.array([t[i] for t in transitions], dtype=self.dtypes[i])
                    for i in range(5))

    def size(self):
        return len(self.buffer)

    def clear_n_step_buffer(self):
        if self.n_step > 1:
            self.n_step_buffer.clear()

    def reset(self):
        self.buffer.clear()
        self.clear_n_step_buffer()

class ReplayBufferWaitClose(ReplayBuffer):
    def __init__(self, capacity, n_step=1, gamma=0.99):
        super().__init__(capacity, n_step, gamma)
        # 使用deque替代list,提高append和extend性能
        self.buffer_temp = collections.deque()

    def add(self, state, action, reward, next_state, done):
        # 使用元组存储,减少内存使用
        self.buffer_temp.append((state, action, reward, next_state, done))

    def update_reward(self, reward=None):
        if reward is not None:
            # 使用列表推导式替代循环,性能更好
            self.buffer_temp = collections.deque(
                (t[0], t[1], reward, t[3], t[4]) for t in self.buffer_temp
            )

        if self.n_step > 1:
            # 使用父类add方法
            for t in self.buffer_temp:
                super().add(t[0], t[1], t[2], t[3], t[4])
            # 清空n步缓冲区
            self.clear_n_step_buffer()
        else:
            # 批量添加到buffer， 效率更高
            self.buffer.extend(self.buffer_temp)

        # 清空临时缓冲区
        self.buffer_temp.clear()

    def reset(self):
        super().reset()
        self.buffer_temp.clear()

class SumTree:
    """
    SumTree数据结构，用于高效存储和采样
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # 修改初始化方式，使用 None 而不是 0
        self.data = np.array([None] * capacity, dtype=object)
        self.data_pointer = 0
        self.is_full = False

    def add(self, priority, data):
        """
        添加新的经验
        """
        if not isinstance(data, (tuple, list)) or len(data) not in [5, 8]:
            pickle.dump((priority, data), open("error_SumTree_add_data.pkl", "wb"))
            raise ValueError(f"Invalid data format: expected tuple/list of length 5 or 8, got {data}({type(data)})")

        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.is_full = True
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        """
        更新优先级
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        根据优先级值获取叶子节点
        """
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            # 如果到达叶子节点
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break

            # 否则继续向下搜索
            if v <= self.tree[left_child]:
                parent_idx = left_child
            else:
                v -= self.tree[left_child]
                parent_idx = right_child

        data_idx = leaf_idx - self.capacity + 1
        if self.data[data_idx] is None:
            raise ValueError("Trying to access empty data slot")
            
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        """
        返回总优先级
        """
        return self.tree[0]

class PrioritizedReplayBuffer:
    """
    优先级经验回放
    """
    def __init__(
        self, 
        capacity=10000, 
        alpha=0.6,  # 决定优先级的指数
        beta=0.4,   # 重要性采样权重的初始值
        beta_increment_per_sampling=0.001,
        max_priority=1.0,
        n_step=1,
        gamma=0.99
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self._beta = beta
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.max_priority = max_priority
        self.epsilon = 1e-6  # 避免零优先级

        self.n_step = n_step
        self.gamma = gamma
        # n步缓存
        self.n_step_buffer = collections.deque(maxlen=n_step) if n_step > 1 else None

        # 预定义数据类型
        self.dtypes = [np.float32, np.int64, np.float32, np.float32, np.float32]    

    def add(self, state, action, reward, next_state, done):
        """
        添加新的经验
        默认给最大优先级
        """
        transition = (state, action, reward, next_state, done)
        if self.n_step > 1:
            self.n_step_buffer.append(transition)

            # 只有当n步缓存满了才添加到主缓存
            if len(self.n_step_buffer) == self.n_step:
                # 计算n步return
                n_reward, n_next_state, n_done = _get_n_step_info(self.n_step_buffer, self.gamma)
                state, action, reward, next_state, done = self.n_step_buffer[0]
                
                # 存储原始transition和n步信息
                experience = (
                    state, action, reward, next_state, done,  # 原始数据
                    n_reward, n_next_state, n_done  # n步数据
                )
                max_priority = self.max_priority if not self.tree.is_full else self.tree.tree[0]
                self.tree.add(max_priority, experience)
        else:
            max_priority = self.max_priority if not self.tree.is_full else self.tree.tree[0]
            self.tree.add(max_priority, transition)

    def sample(self, batch_size):
        """
        采样
        """
        if self.size() < batch_size:
            raise ValueError(f"Not enough samples in buffer. Current size: {self.size()}, requested: {batch_size}")

        batch = []
        batch_indices = []
        batch_priorities = []
        segment = self.tree.total_priority() / batch_size

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            value = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)
            
            batch.append(data)
            batch_indices.append(idx)
            batch_priorities.append(priority)

        # 计算重要性采样权重
        total_priority = self.tree.total_priority()
        # 确保总优先级不为零
        if total_priority == 0:
            probabilities = np.ones_like(batch_priorities) / len(batch_priorities)
        else:
            probabilities = batch_priorities / total_priority

        # 添加数值稳定性检查
        weights = np.zeros_like(probabilities)
        valid_probs = probabilities > 0
        if np.any(valid_probs):
            weights[valid_probs] = np.power(self.tree.capacity * probabilities[valid_probs], -self.beta)
            # 避免除以零，使用 np.maximum 确保分母不为零
            weights /= np.maximum(weights.max(), 1e-8)
        else:
            weights = np.ones_like(probabilities)  # 如果所有概率都为零，返回均匀权重

        # batch 内转为numpy数组
        try:
            # 原始数据
            states = np.array([t[0] for t in batch], dtype=self.dtypes[0])
            actions = np.array([t[1] for t in batch], dtype=self.dtypes[1])
            rewards = np.array([t[2] for t in batch], dtype=self.dtypes[2])
            next_states = np.array([t[3] for t in batch], dtype=self.dtypes[3])
            dones = np.array([t[4] for t in batch], dtype=self.dtypes[4])

            # n步数据
            n_rewards = None
            n_next_states = None
            n_dones = None
            if self.n_step > 1:
                n_rewards = np.array([t[5] for t in batch], dtype=self.dtypes[2])
                n_next_states = np.array([t[6] for t in batch], dtype=self.dtypes[3])
                n_dones = np.array([t[7] for t in batch], dtype=self.dtypes[4])

            # 合并数据  
            batch = (states, actions, rewards, next_states, dones,
                    n_rewards, n_next_states, n_dones)

        except Exception as e:
            print(f"Error converting batch to numpy arrays: {str(e)}")
            print(f"Batch content: {batch}")
            pickle.dump(batch, open("error_batch.pkl", "wb"))
            raise e

        return batch, batch_indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """
        更新优先级
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            # 限制优先级范围,防止过大
            clipped_errors = np.minimum(priority, self.max_priority)
            # 确保优先级非零
            priority = np.power(clipped_errors + self.epsilon, self.alpha)
            self.tree.update(idx, priority)

    def get(self, batch_size):
        # get 返回按顺序的batch, 在优先级经验回放中, 性能较差
        raise "should not use this function, use ReplayBufferWaitClose/ReplayBuffer get function instead"

    def size(self):
        """
        返回当前缓冲区中的经验数量
        """
        if self.tree.is_full:
            return self.capacity
        return self.tree.data_pointer

    def clear_n_step_buffer(self):
        if self.n_step > 1:
            self.n_step_buffer.clear()

    def reset(self):
        """
        重置缓冲区
        """
        self.tree = SumTree(self.capacity)
        self.beta = self._beta   # 重置 beta 到初始值
        self.clear_n_step_buffer()

class PrioritizedReplayBufferWaitClose(PrioritizedReplayBuffer):
    """
    支持延迟更新 reward 的优先级经验回放
    """
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, 
                 beta_increment_per_sampling=0.001, max_priority=1.0, n_step=1, gamma=0.99):
        super().__init__(capacity, alpha, beta, beta_increment_per_sampling, max_priority, n_step, gamma)
        self.temp_experiences = collections.deque()  # 临时存储经验
        self.temp_indices = collections.deque()      # 临时存储对应的树索引

    def add(self, state, action, reward, next_state, done):
        """
        临时存储经验
        """
        experience = (state, action, reward, next_state, done)
        self.temp_experiences.append(experience)

    def update_reward(self, reward=None):
        """
        更新 reward 并将临时经验转移到主缓冲区
        """
        if not self.temp_experiences:
            return
            
        if reward is not None:
            # 更新所有临时经验的 reward
            updated_experiences = collections.deque()
            for exp in self.temp_experiences:
                state, action, _, next_state, done = exp
                updated_experiences.append((state, action, reward, next_state, done))
            self.temp_experiences = updated_experiences

        # 将所有临时经验添加到主缓冲区
        for experience in self.temp_experiences:
            if not isinstance(experience, (tuple, list)) or len(experience) != 5:
                pickle.dump(experience, open("error_update_reward.pkl", "wb"))
                raise ValueError(f"Invalid experience format before adding to buffer: {experience}")
            super().add(*experience)

        # 清空n步缓冲区
        self.clear_n_step_buffer()
        # 清空临时缓冲区    
        self.temp_experiences.clear()
        self.temp_indices.clear()

    def reset(self):
        """
        重置缓冲区
        """
        super().reset()
        self.temp_experiences.clear()
        self.temp_indices.clear()

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                
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

def calculate_importance_loss(loss: torch.Tensor) -> float:
    """计算更新的重要性权重
    
    Args:
        loss: 当前批次的损失值
    
    Returns:
        float: 重要性权重
    """
    # 基于loss值计算重要性
    importance = float(loss.item())
    # 归一化到[0, 1]范围
    importance = np.clip(importance / 10.0, 0, 1)
    return importance