# !pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz > /dev/null 2>&1
# !mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git > /dev/null 2>&1
# !cd /kaggle/working/3rd/dl_helper && pip install -e . > /dev/null 2>&1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecCheckNan
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger as imit_logger

import pandas as pd
import torch
import torch.nn as nn
import torch as th
th.autograd.set_detect_anomaly(True)
import time, pickle
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
from py_ext.datetime import beijing_time

from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_env.lob_trade.lob_expert import LobExpert_file
from dl_helper.rl.rl_utils import plot_bc_train_progress
from dl_helper.tool import report_memory_usage, in_windows
from dl_helper.train_folder_manager import TrainFolderManagerBC

train_folder = train_title = f'20250412_lob_trade_bc'
log_name = f'{train_title}_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)

# FOR DEBUG
# df_progress = pd.read_csv(r"C:\Users\lh\Downloads\progress_all (1).csv")
# plot_bc_train_progress(train_folder, df_progress=df_progress)

custom_logger = imit_logger.configure(
    folder=train_folder,
    format_strs=["csv", "stdout"],
)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

class ImprovedSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_output, _ = self.multihead_attn(x, x, x)  # 使用多头注意力
        attn_output = self.dropout(attn_output)
        out = self.norm(x + attn_output)  # 残差连接 + LayerNorm
        return out

# 自定义特征提取器
# 参数量: 735979
class DeepLob(BaseFeaturesExtractor):
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

        # 卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(2, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(2, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(2, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(2, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 5)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(2, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(2, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # Inception 模块
        self.inp1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
        )

        # LSTM 层 
        self.lstm = nn.LSTM(input_size=128*3, hidden_size=128, num_layers=2, batch_first=True)

        # 自注意力层
        # self.attention = SelfAttention(128)
        self.improved_attention = ImprovedSelfAttention(128)

        # 静态特征处理
        self.static_net = nn.Sequential(
            nn.Linear(self.extra_input_dims, self.extra_input_dims * 4),
            nn.LayerNorm(self.extra_input_dims * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128 + self.extra_input_dims * 4, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]# 最后一维是 日期信息， 不参与模型计算，用于专家透视未来数据
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)  # (B, 1, 10, 20)

        # 卷积块
        x = self.conv1(x)  # (B, 32, 28, 10)
        x = self.conv2(x)  # (B, 32, 26, 5)
        x = self.conv3(x)  # (B, 32, 24, 1)

        # Inception 模块
        x_inp1 = self.inp1(x)  # (B, 64, 24, 1)
        x_inp2 = self.inp2(x)  # (B, 64, 24, 1)
        x_inp3 = self.inp3(x)  # (B, 64, 24, 1)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)  # (B, 192, 24, 1)

        # 调整形状以适配 LSTM
        x = x.permute(0, 2, 1, 3).squeeze(3)  # (B, 24, 192)

        # LSTM 处理 取最后一个时间步
        lstm_out, _ = self.lstm(x)  # (B, 24, 64)

        # # 自注意力
        # attn_out = self.attention(lstm_out)  # (B, ?, 128)
        # temporal_feat = attn_out[:, -1, :]  # 取最后一个时间步 (B, 128)
        attn_out = self.improved_attention(lstm_out)
        temporal_feat = attn_out.mean(dim=1)  # 平均池化

        # 静态特征处理
        static_out = self.static_net(extra_x)  # (B, self.extra_input_dims * 4)

        # 融合层
        fused_out = torch.cat([temporal_feat, static_out], dim=1)  # (B, 64 + self.extra_input_dims * 4)
        fused_out = self.fusion(fused_out)  # (B, features_dim)

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            raise ValueError("fused_out is nan or inf")

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

    def _on_step(self):
        return True

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

model_type = 'CnnPolicy'
run_type = 'train'# 'train' or 'test'

model_config={
    # 自定义编码器参数  
    'input_dims' : (30, 20),
    'extra_input_dims' : 3,
    'features_dim' : 64,
}
env_config ={
    'data_type': 'train',# 训练/测试
    'his_len': 30,# 每个样本的 历史数据长度
    'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
    'train_folder': train_folder,
    'train_title': train_folder,
}

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

# 配置 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=DeepLob,
    features_extractor_kwargs=model_config,
    net_arch = [32,32]
)

def make_env():
    return RolloutInfoWrapper(LOB_trade_env(env_config))

if run_type == 'train':

    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecCheckNan(env, raise_exception=True)  # 添加nan检查
    env = VecMonitor(env)  # 添加监控器

    # # 创建单个环境
    # env = LOB_trade_env(env_config)
    # vec_env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

    # 专家
    expert = LobExpert_file(pre_cache=True)

    # 指定设备为 GPU (cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # lr_schedule = linear_schedule(1e-3, 5e-4)
    # clip_range_schedule = linear_schedule(0.15, 0.3)
    model = PPO(
        model_type, 
        env, 
        verbose=1, 
        learning_rate=1e-3,
        ent_coef=0.2,
        gamma=0.97,
        device=device, 
        policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
    )

    # 打印模型结构
    log("模型结构:")
    log(model.policy)
    log(f'参数量: {sum(p.numel() for p in model.policy.parameters())}')

    # test_x = env.observation_space.sample()
    # test_x = torch.from_numpy(test_x).unsqueeze(0)
    # print(test_x.shape)
    # test_x = test_x.float().to(device)
    # out = model.policy.features_extractor(test_x)
    # print(out.shape)
    # sys.exit()

    # 训练文件夹管理
    if not in_windows():
        train_folder_manager = TrainFolderManagerBC(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(model.policy)

    # 生成专家数据
    vec_env = env
    rng = np.random.default_rng()
    t = time.time()
    memory_usage = psutil.virtual_memory()
    rollouts = rollout.rollout(
        expert,
        vec_env,
        rollout.make_sample_until(min_timesteps=10000),
        # rollout.make_sample_until(min_timesteps=1.3e6),
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
        custom_logger=custom_logger,
    )

    test_env = make_env()
    test_env.test()

    total_epochs = 500000
    checkpoint_interval = 1
    begin = 0
    # 读取之前的i
    loop_i_file = os.path.join(train_folder, 'checkpoint', 'loop_i')
    if os.path.exists(loop_i_file):
        begin = int(open(loop_i_file, 'r').read()) + 1
    for i in range(begin, total_epochs // checkpoint_interval):
        _t = time.time()
        bc_trainer.policy.train()
        bc_trainer.train(n_epochs=checkpoint_interval)
        log(f'训练耗时: {time.time() - _t:.2f} 秒')
        # 保存模型
        bc_trainer.policy.save(os.path.join(train_folder, 'checkpoint', f"bc_policy"))
        progress_file = os.path.join(train_folder, f"progress.csv")
        progress_file_all = os.path.join(train_folder, f"progress_all.csv")
        # 验证模型
        _t = time.time()
        reward_after_training, _ = evaluate_policy(bc_trainer.policy, test_env)
        log(f"训练后的平均reward: {reward_after_training}, 验证耗时: {time.time() - _t:.2f} 秒")
        if os.path.exists(progress_file_all):
            df_progress = pd.read_csv(progress_file_all)
        else:
            df_progress = pd.DataFrame()
        df_new = pd.read_csv(progress_file).iloc[len(df_progress):]
        df_new['bc/epoch'] += i * checkpoint_interval
        df_new['val/mean_reward'] = np.nan
        df_new['val/mean_reward'].iloc[-1] = reward_after_training
        df_progress = pd.concat([df_progress, df_new])
        df_progress.ffill(inplace=True)
        df_progress.to_csv(progress_file_all, index=False)
        # 训练进度可视化
        try:
            plot_bc_train_progress(train_folder, df_progress=df_progress)
        except Exception as e:
            pickle.dump(df_progress, open('df_progress.pkl', 'wb'))
            log(f"训练进度可视化失败")
            raise e
        # 记录当前训练进度
        open(loop_i_file, 'w').write(str(i))
        # 上传
        if not in_windows():
            train_folder_manager.push()
else:
    pass