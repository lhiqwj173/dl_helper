# pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import safe_mean
import pandas as pd
import torch
import torch.nn as nn
import pygame
import numpy as np
import random
import sys, os, shutil
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ['ALIST_USER'] = 'admin'
os.environ['ALIST_PWD'] = 'LHss6632673'

import gym
import pygame

from py_ext.lzma import decompress, compress_folder
from py_ext.alist import alist

class SnakeEnv(gym.Env):
    def __init__(self, config: dict):
        super(SnakeEnv, self).__init__()
        self.s = config.get('size', 10)  # 网格大小
        self.render_mode = config.get('render_mode', 'none')  # 渲染模式

        self.grid_size = (self.s, self.s)  # 网格大小
        self.orientation = 0  # 初始方向：0-上, 1-右, 2-下, 3-左
        self.actions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # 动作对应的移动

        self.last_turn = 0  # 最后转弯方向
        self.snake = None  # 蛇的位置列表
        self.apple = None  # 苹果的位置
        self.time = 0  # 步数
        self.score = 0  # 分数
        self.max_steps = self.s ** 2  # 最大步数
        self.reward = 0  # 奖励

        # 前一个苹果距离
        self.prev_apple_distance = None

        # 定义动作空间和观测空间
        self.action_space = gym.spaces.Discrete(3)  # 0-左转, 1-前进, 2-右转
        # 完整的网格数据
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, self.s,self.s), dtype=np.float32)

        # 初始化 Pygame
        if self.render_mode == 'human':
            pygame.init()
            self.cell_size = 60  # 每个单元格的像素大小
            self.window_size = (self.s * self.cell_size, self.s * self.cell_size)
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)  # 用于显示文本
            self.metadata = {'render_fps': 10}  # 渲染帧率

    def reset(self):
        self.snake = np.array([[self.s // 2, self.s // 2]])  # 蛇初始位置在网格中心
        self.orientation = 0
        self.last_turn = 0
        self.score = 0
        self.time = 0
        self.reward = 0
        self.reset_apple()
        self.prev_apple_distance = None
        return self.state()

    def _cal_apple_distance(self, pos, apple_pos):
        return abs(pos[0] - apple_pos[0]) + abs(pos[1] - apple_pos[1])

    def step(self, action):
        self.time += 1

        self.orientation = (self.orientation + action - 1) % 4  # 更新方向
        new_pos = self.snake[0] + self.actions[self.orientation]  # 计算新位置

        if action != 1:
            self.last_turn = action / 2  # 更新最后转弯方向

        no_wall_collision = (0 <= new_pos[0] < self.s) and (0 <= new_pos[1] < self.s)
        no_tail_collision = not any(np.array_equal(pos, new_pos) for pos in self.snake)

        if no_wall_collision and no_tail_collision and self.time < self.max_steps:
            self.snake = np.insert(self.snake, 0, new_pos, axis=0)

            if np.array_equal(new_pos, self.apple):
                self.score += 1
                if len(self.snake) < self.s ** 2:
                    self.reset_apple()
                    self.reward = len(self.snake)  # 吃苹果的奖励
                    terminated = False
                # else:
                #     self.reward = 1000000  # 填满网格的奖励
                #     terminated = True
            else:
                self.snake = np.delete(self.snake, -1, axis=0)
                # self.reward = -self.time  # 每步的负奖励

                # 距离苹果的附加奖励惩罚
                apple_distance = self._cal_apple_distance(new_pos, self.apple)
                self.reward = -(apple_distance / (self.s * 2))

                terminated = False
        else:
            # self.reward = -1000  # 撞墙或撞尾的惩罚
            terminated = True

        observation = self.state()
        truncated = False  # 可根据需要添加截断逻辑
        info = {}

        return observation, self.reward, terminated, info
    
    def reset_apple(self):
        possible_positions = [(i, j) for i in range(self.s) for j in range(self.s) 
                             if not any(np.array_equal([i, j], pos) for pos in self.snake)]
        self.apple = np.array(possible_positions[np.random.randint(len(possible_positions))])
        self.time = 0

    def state(self):
        # 创建三个 (10,10) 的通道，初始化为 0，数据类型为 float32
        head_channel = np.zeros(self.grid_size, dtype=np.float32)
        body_channel = np.zeros(self.grid_size, dtype=np.float32)
        apple_channel = np.zeros(self.grid_size, dtype=np.float32)
        
        # 设置蛇头位置
        if len(self.snake):  # 确保蛇列表不为空
            head_pos = self.snake[0]
            head_channel[head_pos[0], head_pos[1]] = 1.0
        
        # 设置蛇身位置（不包括蛇头）
        for pos in self.snake[1:]:
            body_channel[pos[0], pos[1]] = 1.0
        
        # 设置苹果位置
        apple_channel[self.apple[0], self.apple[1]] = 1.0
        
        # 沿第 -1 轴堆叠，得到 (3, 10,10) 的数组
        state = np.stack([head_channel, body_channel, apple_channel], axis=0)
        return state

    def _get_state(self):
        # 返回一个二维数组表示游戏状态
        state = np.zeros(self.grid_size, dtype=int)
        for idx, pos in enumerate(self.snake):
            if idx == 0:
                state[pos[0], pos[1]] = 2  # 蛇头
            else:
                state[pos[0], pos[1]] = 1  # 蛇身
        state[self.apple[0], self.apple[1]] = 3  # 苹果
        return state

    def render(self, mode='human'):
        if mode != 'human':
            return

        self.screen.fill((0, 0, 0))  # 清屏

        state = self._get_state()
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if state[y, x] == 1 or state[y, x] == 2:  # 蛇身或蛇头
                    pos_idx = next((i for i, p in enumerate(self.snake) if np.array_equal(p, [y, x])), None)
                    if pos_idx == 0:
                        color = (0, 255, 0, 255)  # 蛇头，绿色不透明
                    elif pos_idx == len(self.snake) - 1:
                        color = (255, 255, 0, 255)  # 蛇尾，黄色不透明
                    else:
                        m = len(self.snake) - 2  # 蛇身段数
                        if m > 1:
                            alpha = 255 - int(((pos_idx - 1) / (m - 1)) * (255 - 150))
                        else:
                            alpha = 255
                        color = (0, 255, 0, alpha)  # 蛇身，绿色渐变透明
                elif state[y, x] == 3:
                    color = (0, 0, 255, 255)  # 苹果，蓝色不透明
                else:
                    continue  # 跳过空白单元格

                # 使用带 alpha 的表面绘制
                cell_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                cell_surface.fill(color)
                self.screen.blit(cell_surface, (x * self.cell_size, y * self.cell_size))

        # 显示分数、剩余步数和奖励
        if mode == 'human':
            remaining_steps = max(0, self.max_steps - self.time)
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            steps_text = self.font.render(f"Remaining Steps: {remaining_steps}", True, (255, 255, 255))
            reward_text = self.font.render(f"Reward: {self.reward:.2f}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(steps_text, (10, 50))
            self.screen.blit(reward_text, (10, 90))

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        pygame.quit()

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
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # 获取输入通道数，例如 (3, 10, 10) 的 RGB 图像，n_input_channels = 3
        n_input_channels = observation_space.shape[0]
        # 定义 CNN 架构
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
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

        self.metrics.append(metrics_dict)

        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.train_folder, "training_metrics.csv"), index=False)
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
            axs[0].plot(df['rollout/ep_len_mean'], label='ep_len_mean', alpha=0.5)
            axs[0].plot(smooth_data(df['rollout/ep_len_mean']), label='smoothed', linewidth=2)
            axs[0].set_title('Episode Length Mean')
            axs[0].set_ylabel('Length')
            axs[0].legend()
            axs[0].grid(True)

        # 图 2: ep_rew_mean
        if 'rollout/ep_rew_mean' in df.columns:
            axs[1].plot(df['rollout/ep_rew_mean'], label='ep_rew_mean', alpha=0.5)
            axs[1].plot(smooth_data(df['rollout/ep_rew_mean']), label='smoothed', linewidth=2)
            axs[1].set_title('Episode Reward Mean')
            axs[1].set_ylabel('Reward')
            axs[1].legend()
            axs[1].grid(True)

        # 图 3: train/loss
        if 'train/loss' in df.columns:
            axs[2].plot(df['train/loss'], label='loss', alpha=0.5)
            axs[2].plot(smooth_data(df['train/loss']), label='smoothed', linewidth=2)
            axs[2].set_title('Training Loss')
            axs[2].set_ylabel('Loss')
            axs[2].legend()
            axs[2].grid(True)

        # 图 4: train/entropy_loss
        if 'train/entropy_loss' in df.columns:
            axs[3].plot(df['train/entropy_loss'], label='entropy_loss', alpha=0.5)
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

# 创建带 Monitor 的环境函数
def make_env():
    env = SnakeEnv({})
    return env

model_type = 'MlpPolicy'
model_type = 'CnnPolicy'
run_type = 'train'# 'train' or 'test'
run_type = 'test'# 'train' or 'test'
train_folder = f'simple_snake5_{model_type}'
os.makedirs(train_folder, exist_ok=True)

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

# 配置 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=32),
    net_arch = []
)

if run_type == 'train':

    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecMonitor(env)  # 添加监控器

    # 指定设备为 GPU (cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成模型实例    
    kaggle_keep_trained_file = rf'/kaggle/input/checkpoint/{train_folder}.checkpoint'
    if os.path.exists(kaggle_keep_trained_file):
        print(f'加载模型, 继续训练')
        # 从kaggle保存的模型文件中拷贝到当前目录
        shutil.copy(kaggle_keep_trained_file, f'./{train_folder}_checkpoint.zip')
        # load model
        model = PPO.load(f'./{train_folder}_checkpoint.zip', env, device=device, custom_objects= {"policy_kwargs": policy_kwargs} if model_type == 'CnnPolicy' else None)
    else:
        print(f'新训练 {model_type}')
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        model = PPO(
            model_type, 
            env, 
            verbose=1, 
            # learning_rate=5e-5,
            # clip_range=0.3,
            # ent_coef=0.1,
            # batch_size=256,
            # gae_lambda=0.98,
            # gamma=0.94,
            # device=device, 
            # clip_range=clip_range_schedule,
            policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
        )

    # 打印模型结构
    print("模型结构:")
    print(model.policy)
    print(f'参数量: {sum(p.numel() for p in model.policy.parameters())}')
    # sys.exit()

    # model.learn(total_timesteps=1000000)
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
    env = SnakeEnv({
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