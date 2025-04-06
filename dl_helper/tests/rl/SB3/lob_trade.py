# pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import safe_mean
import pandas as pd
import torch
import torch.nn as nn
import time
import numpy as np
import random
import sys, os, shutil
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ['ALIST_USER'] = 'admin'
os.environ['ALIST_PWD'] = 'LHss6632673'

from py_ext.lzma import decompress, compress_folder
from py_ext.alist import alist

from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env

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
train_folder = f'lob_trade_{model_type}'
os.makedirs(train_folder, exist_ok=True)

model_config={
    # 自定义编码器参数  
    'input_dims' : (10, 20),
    'extra_input_dims' : 3,
    'features_dim' : 32,
}
env_config ={
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

# 创建带 Monitor 的环境函数
def make_env():
    env = LOB_trade_env(env_config)
    return env

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

# 配置 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=CausalConvLSTM,
    features_extractor_kwargs=model_config,
    net_arch = [32,16]
)

if run_type == 'train':

    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecMonitor(env)  # 添加监控器

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
        print(f'加载模型参数, 继续训练')
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
        print(f'新训练 {model_type}')
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
    print("模型结构:")
    print(model.policy)
    print(f'参数量: {sum(p.numel() for p in model.policy.parameters())}')
    sys.exit()

    # model.learn(total_timesteps=500000, callback=[checkpoint_callback])
    # model.save(os.path.join(train_folder,f"{train_folder}.zip"))

    for i in range(10000):
        model.learn(total_timesteps=1000000, callback=[checkpoint_callback])
        model.save(f"{train_folder}_{i}")
        shutil.copy(f"{train_folder}_{i}.zip", os.path.join(train_folder, f"{train_folder}.zip"))

        # 打包文训练件夹，并上传到alist
        zip_file = f'{train_folder}.7z'
        if os.path.exists(zip_file):
            os.remove(zip_file)
        compress_folder(train_folder, zip_file, 9, inplace=False)
        print('compress_folder done')
        # 上传更新到alist
        ALIST_UPLOAD_FOLDER = 'rl_learning_process'
        client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
        upload_folder = f'/{ALIST_UPLOAD_FOLDER}/'
        client.mkdir(upload_folder)
        client.upload(zip_file, upload_folder)
        print('upload done')

else:
    pass