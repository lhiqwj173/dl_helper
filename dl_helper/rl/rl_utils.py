from tqdm import tqdm
import numpy as np
import math
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
from collections import OrderedDict

from dl_helper.rl.run import run_client_learning, run_client_learning_device_breakout
from dl_helper.tool import keep_upload_log_file, init_logger_by_ip, in_windows
from dl_helper.train_param import is_kaggle
from py_ext.tool import log, get_log_file
from py_ext.lzma import compress_folder, decompress
from py_ext.alist import alist
from py_ext.datetime import beijing_time

rl_folder = r'/root/rl_learning' if not is_kaggle() else r'/kaggle/working/rl_learning'
root_folder = os.path.expanduser("~") if (in_windows() or (not os.path.exists(rl_folder))) else rl_folder

def stop():
    import signal
    os.kill(os.getpid(), signal.SIGKILL)  # 强制终止当前进程

def simplify_rllib_metrics(data, out_func=print, out_file=''):
    """
    自定义指标统一放在 custom_metrics 下

    """
    important_metrics = {
        "env_runner": {},
        "val": {},
        "learner": {},
    }

    ks = [
        'env_runner_episode_return_mean',
        'env_runner_episode_return_max',
        'env_runner_episode_len_mean',
        'env_runner_episode_len_max',
        'env_runner_num_env_steps_sampled',
        'env_runner_num_episodes',
        'env_runner_num_env_steps_sampled_lifetime',
        'val_episode_return_mean',
        'val_episode_return_max',
        'val_episode_len_mean',
        'val_episode_len_max',
        'learner_default_policy_entropy',
        'learner_default_policy_policy_loss',
        'learner_default_policy_vf_loss',
        'learner_default_policy_total_loss',
        'time_this_iter_s',
        'num_training_step_calls_per_iteration',
        'training_iteration',
        'learning_rate',
    ]

    data_dict = OrderedDict.fromkeys(ks, '')

    if 'counters' in data:
        if 'num_env_steps_sampled' in data["counters"]:
            important_metrics["env_runner"]["num_env_steps_sampled"] = data["counters"]["num_env_steps_sampled"]

    if 'env_runners' in data:
        if 'episode_return_mean' in data["env_runners"]:
            important_metrics["env_runner"]["episode_return_mean"] = data["env_runners"]["episode_return_mean"]
        if 'episode_return_max' in data["env_runners"]:
            important_metrics["env_runner"]["episode_return_max"] = data["env_runners"]["episode_return_max"]
        if 'episode_len_mean' in data["env_runners"]:
            important_metrics["env_runner"]["episode_len_mean"] = data["env_runners"]["episode_len_mean"]
        if 'episode_len_max' in data["env_runners"]:
            important_metrics["env_runner"]["episode_len_max"] = data["env_runners"]["episode_len_max"]
        if 'num_env_steps_sampled' in data["env_runners"]:
            important_metrics["env_runner"]["num_env_steps_sampled"] = data["env_runners"]["num_env_steps_sampled"]
        if 'num_episodes' in data["env_runners"]:
            important_metrics["env_runner"]["num_episodes"] = data["env_runners"]["num_episodes"]
        if 'num_env_steps_sampled_lifetime' in data["env_runners"]:
            important_metrics["env_runner"]["num_env_steps_sampled_lifetime"] = data["env_runners"]["num_env_steps_sampled_lifetime"]

    if 'evaluation' in data:
        if 'env_runners' in data["evaluation"]:
            if 'episode_return_mean' in data["evaluation"]["env_runners"]:
                important_metrics["val"]["episode_return_mean"] = data["evaluation"]["env_runners"]["episode_return_mean"]
            if 'episode_return_max' in data["evaluation"]["env_runners"]:
                important_metrics["val"]["episode_return_max"] = data["evaluation"]["env_runners"]["episode_return_max"]
            if 'episode_len_mean' in data["evaluation"]["env_runners"]:
                important_metrics["val"]["episode_len_mean"] = data["evaluation"]["env_runners"]["episode_len_mean"]
            if 'episode_len_max' in data["evaluation"]["env_runners"]:
                important_metrics["val"]["episode_len_max"] = data["evaluation"]["env_runners"]["episode_len_max"]

    if 'learners' in data:
        if 'default_policy' in data["learners"]:
            important_metrics["learner"]["default_policy"] = {}
            if 'entropy' in data["learners"]["default_policy"]:
                important_metrics["learner"]["default_policy"]["entropy"] = data["learners"]["default_policy"]["entropy"]
            if 'policy_loss' in data["learners"]["default_policy"]:
                important_metrics["learner"]["default_policy"]["policy_loss"] = data["learners"]["default_policy"]["policy_loss"]
            if 'vf_loss' in data["learners"]["default_policy"]:
                important_metrics["learner"]["default_policy"]["vf_loss"] = data["learners"]["default_policy"]["vf_loss"]
            if 'total_loss' in data["learners"]["default_policy"]:
                important_metrics["learner"]["default_policy"]["total_loss"] = data["learners"]["default_policy"]["total_loss"]

    if 'time_this_iter_s' in data:
        important_metrics["time_this_iter_s"] = data["time_this_iter_s"]
    if 'num_training_step_calls_per_iteration' in data:
        important_metrics["num_training_step_calls_per_iteration"] = data["num_training_step_calls_per_iteration"]
    if 'training_iteration' in data:
        important_metrics["training_iteration"] = data["training_iteration"]

    try:
        important_metrics["learning_rate"] = data['learners']['default_policy']['default_optimizer_learning_rate']
    except:
        pass

    # 搜集自定义指标
    if 'custom_metrics' in data:
        important_metrics['custom_metrics'] = {}
        for k, v in data['custom_metrics'].items():
            ks.append(f'custom_metrics_{k}')
            important_metrics['custom_metrics'][k] = v
            
    out_func(f"--------- training iteration: {important_metrics['training_iteration']} ---------")
    out_func("env_runner:")
    if important_metrics['env_runner']:
        for k, v in important_metrics['env_runner'].items():
            out_func(f"  {k}: {v:.4f}")
    else:
        out_func("  no env_runner data")
    
    out_func("val:")
    if 'val' in important_metrics:
        for k, v in important_metrics['val'].items():
            out_func(f"  {k}: {v:.4f}")
    else:
        out_func("  no val data")

    out_func("learner(default_policy):")
    if 'default_policy' in important_metrics['learner'] and important_metrics['learner']['default_policy']:
        for k, v in important_metrics['learner']['default_policy'].items():
            out_func(f"  {k}: {v:.4f}")
    else:
        out_func("  no learner data")
    
    if 'time_this_iter_s' in important_metrics:
        out_func(f"time_this_iter_s: {important_metrics['time_this_iter_s']:.4f}")
    if 'num_training_step_calls_per_iteration' in important_metrics:
        out_func(f"num_training_step_calls_per_iteration: {important_metrics['num_training_step_calls_per_iteration']}")
    if 'learning_rate' in important_metrics:
        out_func(f"learning_rate: {important_metrics['learning_rate']}")

    if 'custom_metrics' in important_metrics:
        out_func(f"custom_metrics:")
        for k, v in important_metrics['custom_metrics'].items():
            out_func(f"  {k}: {v}")

    out_func('-'*30)

    if out_file:
        # 写入列名
        if not os.path.exists(out_file):
            with open(out_file, 'w') as f:
                f.write('datetime,')
                f.write(','.join(ks) + '\n')
        
        # 遍历提取数值
        def get_k_v(data_dict, d, pk=''):
            for k, v in d.items():
                _k = f'{pk}_{k}' if pk else k
                if isinstance(v, dict):
                    get_k_v(data_dict, v, _k)
                else:
                    data_dict[_k] = str(v)
        get_k_v(data_dict, important_metrics)

        # 写入数据
        with open(out_file, 'a') as f:
            f.write(beijing_time().strftime('%Y-%m-%d %H:%M:%S') + ',')
            f.write(','.join(data_dict.values()) + '\n')

    # 异常采用检查
    if 'num_env_steps_sampled' in important_metrics['env_runner'] and important_metrics['env_runner']['num_env_steps_sampled'] == 0:
        raise ValueError(f'num_env_steps_sampled is 0, please check the data')

    return important_metrics

class BaseCustomPlotter:
    def get_additional_plot_count(self):
        """
        子类必须实现
        返回需要额外绘制的图表数量
        """
        raise NotImplementedError("子类必须实现 get_additional_plot_count 方法")
    
        return 2
    
    def plot(self, out_data, axes_list):
        """
        子类必须实现
        绘制额外图表
        """
        raise NotImplementedError("子类必须实现 plot 方法")

        # axes_list will contain 2 axes since we requested 2 additional plots
        datetime = pd.to_datetime(out_data['datetime'])
        
        # Plot on first additional subplot (axes_list[0])
        axes_list[0].plot(datetime, out_data['some_column'], 'g-', label='Custom Data 1')
        axes_list[0].legend()
        
        # Plot on second additional subplot (axes_list[1])
        axes_list[1].plot(datetime, out_data['another_column'], 'r-', label='Custom Data 2')
        axes_list[1].legend()

def plot_training_curve(train_title, train_folder, out_file, total_time=None, pic_name=None, y_axis_max=None, custom_plotter=None):
    """
    total_time: 训练总时间（秒）
    custom_plotter: 自定义绘图对象，需包含两个方法：
        - get_additional_plot_count(): 返回所需的额外子图数量
        - plot(out_data, axes_list): 在给定的轴列表上执行自定义绘图
    """
    out_data = pd.read_csv(out_file)

    # 填充 NaN 值
    out_data = out_data.ffill()
    
    # 提取数据
    datetime = pd.to_datetime(out_data['datetime'])  # 转换为日期时间对象
    mean_reward = out_data['env_runner_episode_return_mean'].tolist()  # 平均奖励
    max_reward = out_data['env_runner_episode_return_max'].tolist()    # 最大奖励
    val_mean_reward = out_data['val_episode_return_mean'].tolist()     # 验证集平均奖励
    val_max_reward = out_data['val_episode_return_max'].tolist()       # 验证集最大奖励
    learning_rates = out_data['learning_rate'].values                  # 学习率
    loss = out_data.get('learner_default_policy_total_loss', pd.Series()).tolist()                  # 损失
    loss_ma = smooth_metrics(loss)
    mean_steps = out_data['env_runner_episode_len_mean'].tolist()       # 平均步数
    val_mean_steps = out_data['val_episode_len_mean'].tolist()         # 验证集平均步数
    entropy = out_data['learner_default_policy_entropy'].tolist()     # 熵
    entropy_ma = smooth_metrics(entropy)
    
    if len(mean_reward) == 0:
        log('未找到 mean_reward 数据')
        return

    # 过滤 NaN 值
    _mean_reward = [x for x in mean_reward if not np.isnan(x)]
    _max_reward = [x for x in max_reward if not np.isnan(x)]
    _val_mean_reward = [x for x in val_mean_reward if not np.isnan(x)]
    _val_max_reward = [x for x in val_max_reward if not np.isnan(x)]
    _learning_rates = [x for x in learning_rates if not np.isnan(x)]
    _loss = [x for x in loss if not np.isnan(x)]
    _loss_ma = [x for x in loss_ma if not np.isnan(x)]
    _mean_steps = [x for x in mean_steps if not np.isnan(x)]
    _val_mean_steps = [x for x in val_mean_steps if not np.isnan(x)]
    _entropy = [x for x in entropy if not np.isnan(x)]
    _entropy_ma = [x for x in entropy_ma if not np.isnan(x)]

    # 获取最新值
    mean_reward_latest = _mean_reward[-1] if len(_mean_reward) > 0 else np.nan
    max_reward_latest = _max_reward[-1] if len(_max_reward) > 0 else np.nan
    val_mean_reward_latest = _val_mean_reward[-1] if len(_val_mean_reward) > 0 else np.nan
    val_max_reward_latest = _val_max_reward[-1] if len(_val_max_reward) > 0 else np.nan
    lr_latest = _learning_rates[-1] if len(_learning_rates) > 0 else np.nan
    loss_latest = _loss[-1] if len(_loss) > 0 else np.nan
    loss_ma_latest = _loss_ma[-1] if len(_loss_ma) > 0 else np.nan
    mean_steps_latest = _mean_steps[-1] if len(_mean_steps) > 0 else np.nan
    val_mean_steps_latest = _val_mean_steps[-1] if len(_val_mean_steps) > 0 else np.nan
    entropy_latest = _entropy[-1] if len(_entropy) > 0 else np.nan
    entropy_ma_latest = _entropy_ma[-1] if len(_entropy_ma) > 0 else np.nan

    # 确定子图数量（4个固定子图：奖励、学习率/损失，加上自定义子图）
    additional_plots = custom_plotter.get_additional_plot_count() if custom_plotter else 0
    total_plots = 4 + additional_plots  # 主奖励图、学习率图/损失图, 平均步数, 熵，加上自定义图

    # 创建带有子图的图形
    heights = [3] + [2] * (total_plots-1)  # 主图高度4，其他子图高度2
    fig = plt.figure(figsize=(10, sum(heights)))
    gs = plt.GridSpec(total_plots, 1, height_ratios=heights)
    axes = [plt.subplot(gs[i]) for i in range(total_plots)]
    
    # 主训练曲线图（第一个子图）- 仅显示奖励
    ax = axes[0]
    l1 = ax.plot(mean_reward, color='blue', alpha=0.4, label=f'mean_reward({mean_reward_latest:.4f})')
    l2 = ax.plot(max_reward, color='orange', alpha=0.4, label=f'max_reward({max_reward_latest:.4f})')
    l3 = ax.plot(val_mean_reward, color='blue', label=f'val_mean_reward({val_mean_reward_latest:.4f})')
    l4 = ax.plot(val_max_reward, color='orange', label=f'val_max_reward({val_max_reward_latest:.4f})')
    
    ax.legend(loc='upper left')
    ax.set_title(f'{train_title} ' + (f' {total_time/3600:.2f} H' if total_time is not None else ''))
    if y_axis_max is not None:
        ax.set_ylim(0, y_axis_max)
    else:
        values = _mean_reward + _val_mean_reward
        min_y = min(values) if len(values) > 0 else 0
        max_y = max(values) if len(values) > 0 else 1
        ax.set_ylim(min_y, max_y)

    # 损失图和学习率图（第二个子图）
    ax_loss = axes[1]
    # 绘制损失曲线(左y轴)
    l1 = ax_loss.plot(loss, color='red', label=f'loss({loss_latest:.4f})', alpha=0.4)
    l2 = ax_loss.plot(loss_ma, color='red', label=f'loss_ma({loss_ma_latest:.4f})')
    ax_loss.set_ylabel('loss')
    ax_loss.tick_params(axis='y')
    
    # 创建右侧y轴并绘制学习率曲线
    ax_lr = ax_loss.twinx()
    l3 = ax_lr.plot(learning_rates, color='blue', label=f'lr({lr_latest:.6f})', alpha=0.4)
    ax_lr.set_ylabel('lr', color='blue')
    ax_lr.tick_params(axis='y', labelcolor='blue')
    
    # 合并两个图例
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, labels, loc='upper left')

    # 平均步数图（第三个子图）
    ax_mean_steps = axes[2]
    l1 = ax_mean_steps.plot(mean_steps, color='green', label=f'mean_steps({mean_steps_latest:.4f})', alpha=0.4)
    l2 = ax_mean_steps.plot(val_mean_steps, color='green', label=f'val_mean_steps({val_mean_steps_latest:.4f})')
    ax_mean_steps.set_ylabel('mean_steps')
    ax_mean_steps.tick_params(axis='y')
    ax_mean_steps.legend(loc='upper left')

    # 熵图（第四个子图）
    ax_entropy = axes[3]
    l1 = ax_entropy.plot(entropy, color='purple', label=f'entropy({entropy_latest:.4f})', alpha=0.4)
    l2 = ax_entropy.plot(entropy_ma, color='purple', label=f'entropy_ma({entropy_ma_latest:.4f})')
    ax_entropy.set_ylabel('entropy')
    ax_entropy.tick_params(axis='y')
    ax_entropy.legend(loc='upper left')

    # 计算每4小时的时间点
    min_time = datetime.min()
    max_time = datetime.max()
    time_range = pd.date_range(start=min_time, end=max_time, freq='4h')  # 将 'H' 改为 'h'
    indices = [(datetime - t).abs().idxmin() for t in time_range]

    # 在所有子图上添加竖线和时间标注
    for ax_i in axes:
        for	idx in indices:
            ax_i.axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
        if ax_i == axes[0]:  # 仅在主图上添加时间标签
            y_max = ax_i.get_ylim()[1]
            for idx in indices:
                ax_i.text(idx, y_max, datetime[idx].strftime('%Y-%m-%d %H:%M'), 
                         rotation=90, verticalalignment='top', horizontalalignment='right', 
                         fontsize=8, color='gray')

    # 添加自定义绘图（若提供）
    if custom_plotter and additional_plots > 0:
        custom_plotter.plot(out_data, axes[2:])

    # 仅在最下方子图设置x轴标签
    axes[-1].set_xlabel('iteration')
    
    # 调整布局以防止重叠
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(train_folder, 
                           f'训练曲线_{beijing_time().strftime("%Y%m%d")}.png' 
                           if pic_name is None else pic_name)
    plt.savefig(save_path)
    
    plt.close(fig)  # 关闭图形

class GradientAccumulator:
    def __init__(self, momentum=0.9, eps=1e-5):
        self.momentum = momentum
        self.eps = eps
        self.gradient_buffer = []
        self.accumulated_grads = None
        self.count = 0
    
    def add_gradients(self, cpu_gradients):
        """添加新的梯度列表到缓冲区
        Args:
            cpu_gradients: List[np.ndarray] - CPU上的梯度列表
        """
        self.gradient_buffer.append(cpu_gradients)
        self.count += 1
    
    def merge_gradients(self):
        """合并所有累积的梯度列表"""
        if not self.gradient_buffer:
            log('gradient_buffer is empty')
            return None

        # 第一次累积，初始化accumulated_grads
        if self.accumulated_grads is None:
            self.accumulated_grads = [
                torch.zeros_like(grad, dtype=torch.float32)
                for grad in self.gradient_buffer[0]
            ]
        
        # 使用动量累积处理每批梯度
        for grads in self.gradient_buffer:
            for i, grad in enumerate(grads):
                # 动量更新
                self.accumulated_grads[i] = (
                    self.momentum * self.accumulated_grads[i] + 
                    (1 - self.momentum) * grad
                )
        
        # 计算最终梯度（加入eps避免除零）
        bias_correction = 1 - self.momentum ** self.count
        final_grads = [
            grad / (bias_correction + self.eps)
            for grad in self.accumulated_grads
        ]
        
        # 清空缓冲区但保持累积梯度
        self.gradient_buffer = []
        
        return final_grads
    
    def reset(self):
        """重置累积器"""
        self.gradient_buffer = []
        self.accumulated_grads = None
        self.count = 0

class ParamCompressor:
    """
    参数压缩器
    输入输出都是 torch.Tensor
    """
    def __init__(self, param_keys=None, quantize_bits=8):
        self.quantize_bits = quantize_bits
        self.param_keys = param_keys

    def compress_param(self, param):
        """压缩单个参数张量
        参数:
            param: torch.Tensor

        返回:
            压缩后的参数[torch.Tensor]，以及压缩信息字典
        """
        # 记录原始形状
        original_shape = param.shape
        
        # 展平数组以便处理
        flat_param = param.reshape(-1)
        
        # 计算量化参数
        min_val = flat_param.min().float()
        max_val = flat_param.max().float()
        
        # 避免零除错误
        if max_val == min_val:
            scale = torch.tensor(1.0, dtype=torch.float32)
        else:
            scale = ((max_val - min_val) / (2**self.quantize_bits - 1)).float()
        
        # 确保 scale 有一个最小值，以避免数值溢出
        min_scale = 1e-8  # 你可以根据需要调整这个值
        scale = torch.max(scale, torch.tensor(min_scale, dtype=torch.float32))
        
        # 量化并使用 torch.clamp 确保结果在 uint8 范围内
        quantized = torch.round((flat_param - min_val) / scale)
        quantized = torch.clamp(quantized, 0, 2**self.quantize_bits - 1).byte()
        
        compress_info = {
            'shape': original_shape,
            'min_val': min_val,
            'scale': scale
        }
        
        return quantized, compress_info
    
    def decompress_param(self, quantized, compress_info):
        """解压单个参数张量"""
        # 反量化
        decompressed = (quantized.float() * compress_info['scale'] + 
                       compress_info['min_val'])
        
        # 恢复原始形状
        decompressed = decompressed.view(compress_info['shape'])
        
        return decompressed
    
    def compress_params_dict(self, params):
        """压缩整个参数字典 
        参数:
            params: 参数字典{k:torch.Tensor} / 参数张量的列表[torch.Tensor]

        返回是 
            压缩后的参数列表[torch.Tensor]，以及压缩信息字典
        """
        compressed_list = []
        info_list = []

        if isinstance(params, dict):
            iters = list(params.values())
        else:
            iters = params

        # for debug
        return iters, []

        for param in iters:
            quantized, compress_info = self.compress_param(param)
            compressed_list.append(quantized)
            info_list.append(compress_info)
        
        return compressed_list, info_list
    
    def decompress_params_dict(self, compressed_list, info_list):
        """
        根据 解压参数列表，压缩信息字典
        解压整个参数
        
        返回的是 解压后的参数字典[torch.Tensor]
        """
        decompressed_dict = OrderedDict()

        # for debug
        for idx, k in enumerate(self.param_keys):
            decompressed_dict[k] = compressed_list[idx]
        return decompressed_dict
        
        for idx, (k, info) in enumerate(zip(self.param_keys, info_list)):
            decompressed_dict[k] = self.decompress_param(compressed_list[idx], info)
            
        return decompressed_dict
    
def add_train_title_item(train_title, config):
    file = os.path.join(root_folder, f'{train_title}.data')
    if os.path.exists(file):
        return
    with open(file, 'wb') as f:
        pickle.dump(config, f)

def read_train_title_item():
    res = {}
    for file in os.listdir(root_folder):
        if file.endswith('.data'):
            title = file.replace('.data', '')
            config = pickle.load(open(os.path.join(root_folder, file), 'rb'))
            res[title] = config
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
            log(f"Error converting batch to numpy arrays: {str(e)}")
            log(f"Batch content: {batch}")
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

def smooth_metrics(data, window_size=50):
    """
    平滑训练指标的函数，使用滑动窗口均值。
    
    参数:
        data (list or array): 输入的训练指标序列
        window_size (int): 窗口大小，默认为 50
    
    返回:
        np.ndarray: 平滑后的指标序列
    """
    data = np.asarray(data)
    if len(data) == 0 or window_size <= 0:
        return np.array([])
    
    # 计算累积和
    cumsum = np.cumsum(np.insert(data, 0, 0))
    # 完整窗口的均值
    full_window_means = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # 处理前缀（不足窗口大小的部分）
    prefix = np.array([data[:i + 1].mean() for i in range(min(window_size - 1, len(data)))])
    
    # 合并结果
    return np.concatenate([prefix, full_window_means[:len(data) - len(prefix)]])

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

if __name__ == "__main__":
    # plot_training_curve(r"C:\Users\lh\Desktop\temp", out_file=r"C:\Users\lh\Downloads\out_20250304.csv")
    plot_training_curve('temp', r"C:\Users\lh\Desktop\temp", out_file=r"C:\Users\lh\Desktop\temp\out_20250324.csv")
