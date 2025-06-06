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

###########################################
# 20250412 测试1
# 加大学习率    3e-4      >    5e-3
# 
# 20250412 测试2
# 加大学习率    5e-3      >    1E-2
# 
# 20250412 测试3
# 增大ent_coef    0.01      >    0.1
# 
# 20250415 测试4
# 调整奖励函数
# 
# 20250416 测试5
# 调整奖励函数
# 
# 20250416 测试6
# 降低 ent_coef    0.1      >    0.01

###########################################

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
class CustomCNN(BaseFeaturesExtractor):
    # 参数量: 167172
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        
        # 计算全连接层输入维度
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

model_type = 'MlpPolicy'
model_type = 'CnnPolicy'#参数量: 167172

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == 'cnn':
            model_type = 'CnnPolicy'
        elif arg == 'mlp':
            model_type = 'MlpPolicy'

run_type = 'train'# 'train' or 'test'
run_type = 'test'# 'train' or 'test'
train_folder = train_title = f'snake_base_6_{model_type}'
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
policy_kwargs = {
    "features_extractor_class": CustomCNN,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [dict(pi=[128, 64], vf=[128, 64])],  # 自定义策略网络架构
    "activation_fn": nn.ReLU
}

if run_type == 'train':
    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecCheckNan(env)  # 添加nan检查
    env = VecMonitor(env)  # 添加监控器

    model = PPO(
        model_type, 
        env, 
        learning_rate=1e-2,
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

    model.learn(total_timesteps=5_000_000, callback=[checkpoint_callback])
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