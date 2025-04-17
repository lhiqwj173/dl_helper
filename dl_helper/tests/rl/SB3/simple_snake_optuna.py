import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecCheckNan
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import torch
import torch.nn as nn
import torch as th
import pygame
import time
import numpy as np
import random, pickle
import sys, shutil
import matplotlib.pyplot as plt
import optuna
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
from dl_helper.train_folder_manager import TrainFolderManagerOptuna

import logging
def set_optuna_log_file(file):
    # 获取 Optuna 的 logger
    logger = logging.getLogger('optuna')
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

    # 清空默认处理器（防止重复输出）
    logger.handlers.clear()

    # 设置日志输出位置：保存到文件
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 自定义 CNN 特征提取器
class CustomCNN(BaseFeaturesExtractor):
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
        
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# 定义全局变量
model_type = 'CnnPolicy'
train_folder = train_title = f'simple_snake_optuna_test'
os.makedirs(train_folder, exist_ok=True)
log_name = f'{train_title}_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)
set_optuna_log_file(os.path.join(train_folder, "logs", 'optuna_log.txt'))

# 创建带 Monitor 的环境函数
def make_env():
    env = SnakeEnv({'obs_type': 'image'})
    return env

# 配置 policy_kwargs
policy_kwargs = {
    "features_extractor_class": CustomCNN,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
    "activation_fn": nn.ReLU
}

# 定义 Optuna 的目标函数
def objective(trial):
    # 建议超参数
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    # ent_coef = trial.suggest_uniform('ent_coef', 0.0, 0.1)
    # n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    # gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 1.0)
    # clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)

    # 创建并行环境
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecCheckNan(env)
    env = VecMonitor(env)

    # 创建 PPO 模型
    model = PPO(
        model_type,
        env,
        learning_rate=learning_rate,
        # batch_size=batch_size,
        # n_steps=n_steps,
        # ent_coef=ent_coef,
        gamma=gamma,
        # gae_lambda=gae_lambda,
        # clip_range=clip_range,
        verbose=0,
        policy_kwargs=policy_kwargs
    )

    # 训练模型
    model.learn(total_timesteps=500_000)

    # 评估模型
    eval_env = SnakeEnv({'obs_type': 'image'})
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)

    return mean_reward

# 主逻辑
study = None
if not in_windows():
    # 训练文件夹管理
    train_folder_manager = TrainFolderManagerOptuna(train_folder)
    if train_folder_manager.exists():
        log(f"restore from {train_folder_manager.checkpoint_folder}")
        study = train_folder_manager.load_checkpoint()
if study is None:
    # 创建新的 study
    study = optuna.create_study(direction='maximize')

# 定义回调函数
def on_trial_end(study, trial):
    # 保存当前 study 状态到文件
    with open(os.path.join(train_folder_manager.checkpoint_folder, train_folder_manager.checkpoint_name), 'wb') as f:
        pickle.dump(study, f)

    # 推送上传到 alist
    if not in_windows():
        train_folder_manager.push()

# 运行优化
study.optimize(objective, n_trials=30, callbacks=[on_trial_end])

# 获取所有 trial 的数据
df = study.trials_dataframe()
df.to_csv(os.path.join(train_folder, 'trials_dataframe.csv'), index=False)
