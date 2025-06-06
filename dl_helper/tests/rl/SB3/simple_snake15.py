import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecCheckNan
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pandas as pd
import torch
import torch.nn as nn
import torch as th
th.autograd.set_detect_anomaly(True)
import pygame
import time
import numpy as np
import random
import sys, shutil
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ['ALIST_USER'] = 'admin'
os.environ['ALIST_PWD'] = 'LHss6632673'

from py_ext.lzma import decompress, compress_folder
from py_ext.alist import alist
from py_ext.tool import log, init_logger
from py_ext.datetime import beijing_time

from dl_helper.rl.rl_env.snake3.snake_env import SnakeEnv
from dl_helper.rl.rl_utils import CustomCheckpointCallback
from dl_helper.tool import report_memory_usage, in_windows
from dl_helper.train_folder_manager import TrainFolderManagerSB3

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

# 自定义 CNN 特征提取器
class CNN_0(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # 获取输入通道数，例如 (3, 10, 10) 的 RGB 图像，n_input_channels = 3
        n_input_channels = observation_space.shape[0]
        # 定义 CNN 架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # 展平特征图
        )

        # 计算展平后的维度
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        # 全连接层，将展平后的特征映射到指定维度
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CNN_1(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # 获取输入通道数，例如 (3, 10, 10) 的 RGB 图像，n_input_channels = 3
        n_input_channels = observation_space.shape[0]
        # 定义 CNN 架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # 展平特征图
        )

        # 计算展平后的维度
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        # 全连接层，将展平后的特征映射到指定维度
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


model_type = 'CnnPolicy'

model_cls = CNN_0
# model_cls = CNN_1
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == '0':
            model_cls = CNN_0# 参数量: 115340
        elif arg == '1':
            model_cls = CNN_1# 参数量: 288964

run_type = 'train'# 'train' or 'test'
# run_type = 'test'# 'train' or 'test'
train_folder = train_title = f'snake_small_{model_cls.__name__}'
os.makedirs(train_folder, exist_ok=True)
log_name = f'{train_title}_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)

# 创建带 Monitor 的环境函数
def make_env():
    env = SnakeEnv({
        'obs_type': 'image' if model_type == 'CnnPolicy' else 'state',
    })
    return env

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

# 配置 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=model_cls,
    # features_extractor_kwargs=dict(features_dim=256),
    net_arch = [128, 64]
)

if run_type == 'train':
    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecCheckNan(env)  # 添加nan检查
    env = VecMonitor(env)  # 添加监控器

    # lr_schedule = linear_schedule(1e-3, 5e-4)
    # clip_range_schedule = linear_schedule(0.15, 0.3)
    # model = PPO(
    #     model_type, 
    #     env, 
    #     verbose=1, 
    #     learning_rate=1e-4,
    #     ent_coef=0.2,
    #     gamma=0.97,
    #     policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
    # )
    model = PPO(
        model_type, 
        env, 
        learning_rate=3e-4,
        batch_size=512,
        gamma=0.995,
        clip_range_vf=0.2,
        ent_coef=0.01,
        verbose=1, 
        policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
    )

    # 训练文件夹管理
    if not in_windows():
        train_folder_manager = TrainFolderManagerSB3(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(model, custom_objects= {"policy_kwargs": policy_kwargs} if model_type == 'CnnPolicy' else None)

    # 打印模型结构
    log("模型结构:")
    log(model.policy)
    log(f'参数量: {sum(p.numel() for p in model.policy.parameters())}')
    # sys.exit()

    for i in range(1000000):
        model.learn(total_timesteps=100_000, callback=[checkpoint_callback])
        model.save(os.path.join(train_folder, 'checkpoint', f"{train_folder}.zip"))

        # 打包文训练件夹，并上传到alist
        if not in_windows():
            train_folder_manager.push()

else:
    env = SnakeEnv({
        'obs_type': 'image' if model_type == 'CnnPolicy' else 'state',
        'render_mode': 'human',
    })

    model = PPO.load(
        rf"D:\code\dl_helper\dl_helper\tests\rl\SB3\{train_folder}", 
        custom_objects= {"policy_kwargs": policy_kwargs} if model_type == 'CnnPolicy' else None
    )

    obs = env.reset()
    for i in range(2000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()