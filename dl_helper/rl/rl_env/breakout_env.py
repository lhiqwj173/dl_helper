import gymnasium as gym
from gymnasium.wrappers import TransformObservation

try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# 计算经过卷积后的特征图大小
def conv2d_size_out(size, kernel_size, stride):
    return ((size - (kernel_size - 1) - 1) // stride) + 1

class cnn_breakout(nn.Module):
    def __init__(self):
        super(cnn_breakout, self).__init__()
        
        # 修改卷积层参数以减小输出维度
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4)  # 增加stride到4
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)  # 增加stride到2

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def get_feature_size():
        # 使用新的卷积参数计算输出尺寸
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(161, 8, 4), 4, 4), 3, 2)
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(144, 8, 4), 4, 4), 3, 2)
        # 特征提取层输出维度
        features_dim = convw * convh * 64
        return features_dim

def crop_observation(obs):
    return obs[32:193, 8:152]

class BreakoutEnv(gym.Env):
    def __init__(self, capacity=4):
        """
        s: 游戏画面 shape: (161, 144)
        """
        self.env = gym.make('ALE/Breakout-v5', obs_type='grayscale')
        self.env = TransformObservation(self.env, crop_observation)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(161, 144), dtype=np.uint8)
        self.action_space = self.env.action_space

        # 累计最近的4个状态
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def reset(self):
        state, info = self.env.reset()
        self.buffer.append(state)
        
        while len(self.buffer) < self.capacity:
            state, reward, done, truncated, info = self.env.step(0)# NOOP
            self.buffer.append(state)

        return np.stack(self.buffer, axis=0) / 255, info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        info['period_done'] = truncated or done
        info['act_criteria'] = 0
        self.buffer.append(state)
        return np.stack(self.buffer, axis=0) / 255, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def no_need_track_info_item(self):
        return ['act_criteria', 'period_done']

    def need_wait_close(self):
        return False
