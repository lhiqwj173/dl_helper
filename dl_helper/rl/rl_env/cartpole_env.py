import copy, os
import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch.nn as nn
import numpy as np

class cartpole_mlp(nn.Module):
    def __init__(self):
        super(cartpole_mlp, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

    @staticmethod
    def get_feature_size():
        return 64

class CartPoleEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        """
        CartPole 环境
        """
        self.env = gym.make('CartPole-v1', *args, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # 上传文件
        self.need_upload_file = ''
        self.env_bak = None

    def set_data_type(self, _type):
        file_bak_name = '-episode-0.mp4'
        if _type in ['val', 'test']:
            # 删除当前目录下的所有 mp4 文件
            for file in os.listdir('.'):
                if file.endswith('.mp4'):
                    os.remove(file)

            # 录制游戏
            # {name}-episode-0.mp4
            self.need_upload_file = f'{_type}_{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")}' + file_bak_name
            
            # 只在第一次调用时备份原始环境
            self.env_bak = self.env

            # 使用 RecordVideo 包装环境
            new_env = gym.make('CartPole-v1', render_mode='rgb_array')
            self.env = RecordVideo(
                new_env,
                video_folder='.',
                episode_trigger=lambda x: True,  # 录制所有回合
                name_prefix=self.need_upload_file.replace(file_bak_name, ''))
        else:
            if self.env_bak is not None:
                # 恢复原始环境
                self.env = self.env_bak
                # 重置变量
                self.env_bak = None
                self.need_upload_file = ''

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        info['period_done'] = truncated or done
        info['act_criteria'] = 0
        return state, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def no_need_track_info_item(self):
        return ['act_criteria', 'period_done']

    def need_wait_close(self):
        return False
