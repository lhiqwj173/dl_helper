# !pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz > /dev/null 2>&1
# !mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git > /dev/null 2>&1
# !cd /kaggle/working/3rd/dl_helper && pip install -e . > /dev/null 2>&1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import safe_mean

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy

import pandas as pd
import torch
import torch.nn as nn
import time
import numpy as np
import random, psutil
import sys, os, shutil
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ['ALIST_USER'] = 'admin'
os.environ['ALIST_PWD'] = 'LHss6632673'

from py_ext.lzma import decompress, compress_folder
from py_ext.alist import alist
from py_ext.tool import init_logger,log
from py_ext.wechat import send_wx

from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_env.lob_trade.lob_expert import LobExpert
from dl_helper.tool import report_memory_usage

train_folder = train_title = f'20250406_lob_trade_bc'
init_logger(train_title, home=train_folder, timestamp=False)

# 自定义特征提取器
class CausalConvLSTM(BaseFeaturesExtractor):
    def __init__(
            self, 
            observation_space, 
            features_dim: int = 64,
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
        ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims    

        # 因果卷积网络
        self.causal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_dims[1],
                out_channels=64,
                kernel_size=3,
                padding=2,  # 保持时间维度不变
                dilation=2
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=4,
                dilation=4
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True,
            bidirectional=False
        )
        
        # 静态特征处理
        self.static_net = nn.Sequential(
            nn.Linear(self.extra_input_dims, self.extra_input_dims * 4),
            nn.LayerNorm(self.extra_input_dims * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + self.extra_input_dims * 4, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, features_dim)
        )


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        extra_x = observations[:, -self.extra_input_dims:]
        x = observations[:,:-self.extra_input_dims].reshape(-1, *self.input_dims)

        # 因果卷积处理
        conv_in = x.permute(0, 2, 1)  # [B,20,10]
        conv_out = self.causal_conv(conv_in)     # [B,32,10]
        conv_out = conv_out.permute(0, 2, 1)     # [B,10,32]

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(conv_out)
        temporal_feat = lstm_out[:, -1, :]  # 取最后一个时间步 [B,64]

        # 静态特征处理
        static_out = self.static_net(extra_x)  # [B,16]

        # 融合层
        fused_out = torch.cat([temporal_feat, static_out], dim=1)  # [B,80]
        fused_out = self.fusion(fused_out)  # [B,self.output_dims]

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            raise ValueError(f'fused_out is nan or inf')

        return fused_out

class CustomCheckpointCallback(BaseCallback):

    def __init__(self, train_folder: str):
        super().__init__()

        self.train_folder = train_folder

        self.metrics = []

        self.checkpoint_path = os.path.join(self.train_folder, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def _on_rollout_end(self) -> None:
        # This method is called after each rollout
        # Collect metrics from logger
        metrics_dict = {}
        
        # Rollout metrics
        metrics_dict['rollout/ep_len_mean'] = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        metrics_dict['rollout/ep_rew_mean'] = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        
        # Training metrics from the last update
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            for key in ['train/approx_kl', 'train/clip_fraction', 'train/clip_range', 
                        'train/entropy_loss', 'train/explained_variance', 'train/learning_rate',
                        'train/loss', 'train/n_updates', 'train/policy_gradient_loss', 'train/value_loss']:
                if key in self.model.logger.name_to_value:
                    metrics_dict[key] = self.model.logger.name_to_value[key]

        # 增加时间戳
        metrics_dict['timestamp'] = int(time.time())
        self.metrics.append(metrics_dict)

        # 如果存在历史指标文件,读取并合并去重
        metrics_file = os.path.join(self.train_folder, "training_metrics.csv")
        if os.path.exists(metrics_file):
            # 读取历史数据
            history_df = pd.read_csv(metrics_file)
            # 将当前指标转为DataFrame
            current_df = pd.DataFrame(self.metrics)
            # 基于timestamp去重合并
            merged_df = pd.concat([history_df, current_df]).drop_duplicates(subset=['timestamp'], keep='last')
            # 按timestamp排序
            merged_df = merged_df.sort_values('timestamp')
            df = merged_df
        else:
            df = pd.DataFrame(self.metrics)
        df.to_csv(metrics_file, index=False)
        self._plot(df)

    def _plot(self, df):
        """
        图1 绘制 ep_len_mean / ep_len_mean平滑
        图2 绘制 ep_rew_mean / ep_rew_mean平滑
        图3 绘制 train/loss / train/loss平滑
        图4 绘制 train/entropy_loss / train/entropy_loss平滑
        竖向排列，对齐 x 轴
        """
        # 定义平滑函数
        def smooth_data(data, window_size=10):
            return data.rolling(window=window_size, min_periods=1).mean()

        # 创建绘图，4 个子图竖向排列，共享 x 轴
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)  # 宽度 10，高度 12

        # 数据长度，用于对齐 x 轴
        data_len = len(df)

        # 图 1: ep_len_mean
        if 'rollout/ep_len_mean' in df.columns:
            axs[0].plot(df['rollout/ep_len_mean'], label=f'ep_len_mean({df.iloc[-1]["rollout/ep_len_mean"]:.2f})', alpha=0.5)
            axs[0].plot(smooth_data(df['rollout/ep_len_mean']), label='smoothed', linewidth=2)
            axs[0].set_title('Episode Length Mean')
            axs[0].set_ylabel('Length')
            axs[0].legend()
            axs[0].grid(True)

        # 图 2: ep_rew_mean
        if 'rollout/ep_rew_mean' in df.columns:
            axs[1].plot(df['rollout/ep_rew_mean'], label=f'ep_rew_mean({df.iloc[-1]["rollout/ep_rew_mean"]:.2f})', alpha=0.5)
            axs[1].plot(smooth_data(df['rollout/ep_rew_mean']), label='smoothed', linewidth=2)
            axs[1].set_title('Episode Reward Mean')
            axs[1].set_ylabel('Reward')
            axs[1].legend()
            axs[1].grid(True)

        # 图 3: train/loss
        if 'train/loss' in df.columns:
            axs[2].plot(df['train/loss'], label=f'loss({df.iloc[-1]["train/loss"]:.2f})', alpha=0.5)
            axs[2].plot(smooth_data(df['train/loss']), label='smoothed', linewidth=2)
            axs[2].set_title('Training Loss')
            axs[2].set_ylabel('Loss')
            axs[2].legend()
            axs[2].grid(True)

        # 图 4: train/entropy_loss
        if 'train/entropy_loss' in df.columns:
            axs[3].plot(df['train/entropy_loss'], label=f'entropy_loss({df.iloc[-1]["train/entropy_loss"]:.2f})', alpha=0.5)
            axs[3].plot(smooth_data(df['train/entropy_loss']), label='smoothed', linewidth=2)
            axs[3].set_title('Entropy Loss')
            axs[3].set_xlabel('Rollout')  # 只有最下方子图显示 x 轴标签
            axs[3].set_ylabel('Entropy Loss')
            axs[3].legend()
            axs[3].grid(True)

        # 设置统一的 x 轴范围（可选，手动设置）
        for ax in axs:
            ax.set_xlim(0, data_len - 1)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(self.train_folder, 'training_plots.png'), dpi=300)
        plt.close()

        # # 分别保存每个图表（可选）
        # for key, title in [
        #     ('rollout/ep_len_mean', 'Episode_Length_Mean'),
        #     ('rollout/ep_rew_mean', 'Episode_Reward_Mean'),
        #     ('train/loss', 'Training_Loss'),
        #     ('train/entropy_loss', 'Entropy_Loss')
        # ]:
        #     if key in df.columns:
        #         plt.figure(figsize=(10, 3))  # 单独图表尺寸
        #         plt.plot(df[key], label=key.split('/')[-1], alpha=0.5)
        #         plt.plot(smooth_data(df[key]), label='smoothed', linewidth=2)
        #         plt.title(title)
        #         plt.xlabel('Rollout')
        #         plt.ylabel(key.split('/')[-1].replace('_', ' ').title())
        #         plt.legend()
        #         plt.grid(True)
        #         plt.xlim(0, data_len - 1)  # 对齐 x 轴
        #         plt.savefig(os.path.join(self.plot_path, f'{title}.png'), dpi=300)
        #         plt.close()

    def _on_step(self):
        return True


model_type = 'CnnPolicy'
run_type = 'train'# 'train' or 'test'
# run_type = 'test'# 'train' or 'test'
train_folder = f'lob_trade_bc_{model_type}'
os.makedirs(train_folder, exist_ok=True)

model_config={
    # 自定义编码器参数  
    'input_dims' : (10, 20),
    'extra_input_dims' : 3,
    'features_dim' : 32,
}
env_config ={
    'data_type': 'train',# 训练/测试
    'his_len': 10,# 每个样本的 历史数据长度
    'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
    'train_folder': train_folder,
    'train_title': train_folder,
}

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

# 配置 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=CausalConvLSTM,
    features_extractor_kwargs=model_config,
    net_arch = [32,16]
)

if run_type == 'train':

    # 创建环境
    env = LOB_trade_env(env_config)

    # 专家
    expert = LobExpert(env)

    # 指定设备为 GPU (cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 1e-3
    # lr = 5e-3
    # lr = 1e-4
    # lr = 5e-4
    # lr = 1e-5
    # lr = 5e-5

    # 生成模型实例    
    kaggle_keep_trained_file = rf'/kaggle/input/checkpoint/{train_folder}.checkpoint'
    if os.path.exists(kaggle_keep_trained_file):
        log(f'加载模型参数, 继续训练')
        # 从kaggle保存的模型文件中拷贝到当前目录
        shutil.copy(kaggle_keep_trained_file, f'./{train_folder}_checkpoint.zip')
        # load model
        model = PPO.load(f'./{train_folder}_checkpoint.zip', custom_objects= {"policy_kwargs": policy_kwargs} if model_type == 'CnnPolicy' else None)
        # 提取权重
        policy_state_dict = model.policy.state_dict()  
        # 新建模型
        model = PPO(
            model_type, 
            env, 
            verbose=1, 
            learning_rate=5e-4,
            ent_coef=0.2,
            gamma=0.98,
            clip_range=0.3,
            device=device, 
            policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
        )
        # 加载权重
        model.policy.load_state_dict(policy_state_dict)  

    else:
        log(f'新训练 {model_type}')
        lr_schedule = linear_schedule(1e-3, 5e-4)
        clip_range_schedule = linear_schedule(0.15, 0.3)
        model = PPO(
            model_type, 
            env, 
            verbose=1, 
            learning_rate=1e-4,
            ent_coef=0.2,
            gamma=0.97,
            device=device, 
            policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
        )

    # 打印模型结构
    log("模型结构:")
    log(model.policy)
    log(f'参数量: {sum(p.numel() for p in model.policy.parameters())}')
    # sys.exit()

    # 生成专家数据
    # 包装为 DummyVecEnv
    vec_env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
    rng = np.random.default_rng(0)
    t = time.time()
    memory_usage = psutil.virtual_memory()
    rollouts = rollout.rollout(
        expert,
        vec_env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=int(5000*2.5)),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)
    memory_usage2 = psutil.virtual_memory()
    msg = ''
    cost_msg = f'生成专家数据耗时: {time.time() - t:.2f} 秒'
    log(cost_msg)
    msg += cost_msg + '\n'
    mem_pct_msg = f"CPU 内存占用：{memory_usage2.percent}% ({memory_usage2.used/1024**3:.3f}GB/{memory_usage2.total/1024**3:.3f}GB)"
    log(mem_pct_msg)
    msg += mem_pct_msg + '\n'
    mem_expert_msg = f"专家数据内存占用：{(memory_usage2.used - memory_usage.used)/1024**3:.3f}GB"
    log(mem_expert_msg)
    msg += mem_expert_msg + '\n'
    send_wx(msg)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=model.policy,
        rng=rng,
    )

    env.test()
    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    log(f"Reward before training: {reward_before_training}")

    bc_trainer.train(n_epochs=30)

    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    log(f"Reward after training: {reward_after_training}")
else:
    pass