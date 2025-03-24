import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime

from py_ext.lzma import decompress, compress_folder
from py_ext.alist import alist

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from dl_helper.rl.rl_env.snake2.snake_env import SnakeEnv
from py_ext.tool import init_logger, log
import sys

model_type = 'cnn'
train_folder = train_title = f'20250324_snake_linyiLYi_sb3' + f'_{model_type}'
init_logger(f"{train_title}_{datetime.datetime.now().strftime('%Y%m%d')}", home=train_folder, timestamp=False)

# 自定义 CNN 特征提取器
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # 获取输入通道数，例如 (3, 10, 10) 的 RGB 图像，n_input_channels = 3
        n_input_channels = observation_space.shape[0]
        # 定义 CNN 架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
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

class CustomCheckpointCallback(BaseCallback):

    def __init__(self, save_freq, train_folder: str):
        super().__init__()

        self.save_freq = save_freq
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
        self.log(metrics_dict)

        self.metrics.append(metrics_dict)
        pd.DataFrame(self.metrics).to_csv(os.path.join(self.train_folder, "training_metrics.csv"), index=False)

    def log(self, metrics_dict):
        # 将数据保存到文件
        log("------------------------------------------\n")
        for key, value in metrics_dict.items():
            log(f"{key}: {value}")
        log("------------------------------------------\n\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 保存检查点
            model_path = os.path.join(self.checkpoint_path, f'checkpoint.zip')
            self.model.save(model_path)

            # 打包文训练件夹，并上传到alist
            zip_file = f'{train_folder}.7z'
            if os.path.exists(zip_file):
                os.remove(zip_file)
            compress_folder(train_folder, zip_file, 9, inplace=False)
            log('compress_folder done')
            # 上传更新到alist
            ALIST_UPLOAD_FOLDER = 'rl_learning_process'
            client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
            upload_folder = f'/{ALIST_UPLOAD_FOLDER}/'
            client.mkdir(upload_folder)
            client.upload(zip_file, upload_folder)
            log('upload done')

        return True

if __name__ == "__main__":
    env_config = {}

    env = SnakeEnv(
        config=env_config,
    )

    # 创建回调实例
    checkpoint_callback = CustomCheckpointCallback(save_freq=10000, train_folder=train_folder)

    # 配置 policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=24),
    )

    model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    log(model.policy)
    log(f'total parameters: {sum(p.numel() for p in model.policy.parameters())}')# total parameters: 475357

    # 测试
    from dl_helper.rl.rl_env.tool import ai_control
    ai_control(env_class=SnakeEnv, env_config=env_config, checkpoint_abs_path=r"C:\Users\lh\Downloads\checkpoint.zip", times=10, sb3_rl_model=model)
    sys.exit()

    model.learn(
        total_timesteps=int(100000000),
        callback=[checkpoint_callback]
    )
    env.close()

    # Save the final model
    model.save(os.path.join(train_folder, 'final_model'))
