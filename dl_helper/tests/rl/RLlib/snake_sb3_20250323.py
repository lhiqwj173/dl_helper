import os
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 自定义 CNN 特征提取器
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # 获取输入通道数，例如 (3, 10, 10) 的 RGB 图像，n_input_channels = 3
        n_input_channels = observation_space.shape[0]
        # 定义 CNN 架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
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

from dl_helper.rl.rl_env.snake.snake_env import SnakeEnv
from py_ext.tool import init_logger, log
import sys

model_type = 'mlp'
for arg in sys.argv:
    if arg == 'cnn':
        model_type = 'cnn'

train_folder = train_title = f'20250322_snake_sb3' + f'_{model_type}'
init_logger(train_title, home=train_folder, timestamp=False)

# 吃到食物标准奖励
STD_EAT_FOOD_REWARD = 100

# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_EAT_FOOD_REWARD / 2

"""
激励函数

# 最大吃食物数量
MAX_EAT_FOOD_NUM = 10 *10 - 1

# 奖励函数
shaping = -(距离²/(10² + 10²)) * STD_MOVE_REWARD - (MAX_EAT_FOOD_NUM - 吃到食物数量) * STD_EAT_FOOD_REWARD
"""

def crash_reward(snake, food, grid_size):
    # 10 * 10 的网格, 最大惩罚: -10000
    MAX_EAT_FOOD_NUM = grid_size[0] * grid_size[1] - 1
    return -(MAX_EAT_FOOD_NUM + 1) * STD_EAT_FOOD_REWARD

def keep_alive_reward(snake, food, grid_size):
    MAX_EAT_FOOD_NUM = grid_size[0] * grid_size[1] - 1
    eat_food_num = len(snake) - 1
    distance_sqrt = (snake[0][0] - food[0])**2 + (snake[0][1] - food[1])**2
    return -(distance_sqrt/(grid_size[0]**2 + grid_size[1]**2)) * STD_MOVE_REWARD - (MAX_EAT_FOOD_NUM - eat_food_num) * STD_EAT_FOOD_REWARD

if __name__ == "__main__":
    env_config = {
        'grid_size': (10, 10),
        'crash_reward': crash_reward,
        'eat_reward': keep_alive_reward,
        'move_reward': keep_alive_reward,
        'model_type': f'sb3_{model_type}',
    }

    env = SnakeEnv(
        config=env_config,
    )

    # 设置保存参数
    save_path = os.path.join(train_folder, 'checkpoint')
    save_freq = 10000           # 每 10000 步保存一次

    # 创建回调实例
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path)

    # 配置 policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=24),
    )

    model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    log(model.policy)

    model.learn(
        total_timesteps=int(100000000),
        callback=[checkpoint_callback]
    )
    env.close()

    # Save the final model
    model.save(os.path.join(save_path, 'final_model'))
