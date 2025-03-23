import os
import torch
import torch.nn as nn
from py_ext.alist import alist

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
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

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

class CustomCheckpointCallback(CheckpointCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        file = os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}.{extension}")
        # 删除文件
        if os.path.exists(file):
            os.remove(file)
        return file

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

            # 上传更新到alist
            ALIST_UPLOAD_FOLDER = 'rl_learning_process'
            client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
            upload_folder = f'/{ALIST_UPLOAD_FOLDER}/'
            client.mkdir(upload_folder)
            client.upload(model_path, upload_folder)
            log('upload done')

        return True

from dl_helper.rl.rl_env.snake.snake_env import SnakeEnv
from py_ext.tool import init_logger, log
import sys

model_type = 'cnn'
train_folder = train_title = f'20250323_snake_sb3' + f'_{model_type}'
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
        'model_type': model_type,
    }

    env = SnakeEnv(
        config=env_config,
    )

    # 设置保存参数
    save_path = os.path.join(train_folder, 'checkpoint')
    save_freq = 10000           # 每 10000 步保存一次

    # 创建回调实例
    checkpoint_callback = CustomCheckpointCallback(save_freq=save_freq, save_path=save_path)

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
