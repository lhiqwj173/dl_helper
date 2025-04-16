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
import random
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
from dl_helper.train_folder_manager import TrainFolderManagerSB3

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
run_type = 'train_optuna'  # 修改为 'train_optuna' 以运行 Optuna 优化
train_folder = train_title = f'simple_snake_optuna_test'
os.makedirs(train_folder, exist_ok=True)
log_name = f'{train_title}_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)

# 创建带 Monitor 的环境函数
def make_env():
    env = SnakeEnv({'obs_type': 'image'})
    return env

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

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
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    ent_coef = trial.suggest_uniform('ent_coef', 0.0, 0.1)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 1.0)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)

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
        batch_size=batch_size,
        n_steps=n_steps,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=0,
        policy_kwargs=policy_kwargs
    )

    # 训练模型
    model.learn(total_timesteps=100_000)  # 每个试验训练 100,000 步

    # 评估模型
    eval_env = SnakeEnv({'obs_type': 'image'})
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)

    # 保存试验模型（可选）
    trial_folder = os.path.join(train_folder, f'trial_{trial.number}')
    os.makedirs(trial_folder, exist_ok=True)
    model.save(os.path.join(trial_folder, 'model.zip'))

    return mean_reward

# 主逻辑
if run_type == 'train_optuna':
    # 创建 Optuna study 并运行优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # 运行 50 个试验

    # 记录最佳超参数
    log(f"Best trial: {study.best_trial.params}")

    # 使用最佳超参数进行最终训练（可选）
    best_params = study.best_trial.params
    env = DummyVecEnv([make_env for _ in range(4)])
    env = VecCheckNan(env)
    env = VecMonitor(env)

    model = PPO(
        model_type,
        env,
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        n_steps=best_params['n_steps'],
        ent_coef=best_params['ent_coef'],
        gamma=best_params['gamma'],
        gae_lambda=best_params['gae_lambda'],
        clip_range=best_params['clip_range'],
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback])
    model.save(os.path.join(train_folder, 'final_model.zip'))

elif run_type == 'train':
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecCheckNan(env)
    env = VecMonitor(env)

    model = PPO(
        model_type,
        env,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    if not in_windows():
        train_folder_manager = TrainFolderManagerSB3(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(model, custom_objects={"policy_kwargs": policy_kwargs})

    log("模型结构:")
    log(model.policy)
    log(f'参数量: {sum(p.numel() for p in model.policy.parameters())}')

    for i in range(10000000000000):
        model.learn(total_timesteps=10_000, callback=[checkpoint_callback])
        model.save(os.path.join(train_folder, 'checkpoint', f"{train_folder}.zip"))

        if not in_windows():
            train_folder_manager.push()

else:  # run_type == 'test'
    env = SnakeEnv({'obs_type': 'image', 'render_mode': 'human'})
    model = PPO.load(
        rf"D:\code\dl_helper\dl_helper\tests\ RL\SB3\{train_folder}",
        custom_objects={"policy_kwargs": policy_kwargs}
    )

    obs = env.reset()
    for i in range(2000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()