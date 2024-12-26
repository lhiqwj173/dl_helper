import gymnasium as gym
import ale_py
import collections
import numpy as np

gym.register_envs(ale_py)

def crop_observation(obs):
    return obs[32:193, 8:152]

class BreakoutEnv(gym.Env):
    def __init__(self, capacity=4):
        """
        s: 游戏画面 shape: (161, 144)
        """
        self.env = gym.make('ALE/Breakout-v5', render_mode='grayscale')
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

        return np.stack(self.buffer, axis=0), info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        info['period_done'] = truncated or done
        info['act_criteria'] = 0
        self.buffer.append(state)
        return np.stack(self.buffer, axis=0), reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def no_need_track_info_item(self):
        return ['act_criteria', 'period_done']

    def need_wait_close(self):
        return False
