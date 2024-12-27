import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

import ale_py
gym.register_envs(ale_py)

def record_game(video_name="my_breakout"):
    # 创建环境
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', obs_type='grayscale')
    
    # 使用 RecordVideo 包装环境
    env = RecordVideo(env, 
                     video_folder=".",  # 当前目录
                     name_prefix=video_name,  # 自定义文件名前缀
                     episode_trigger=lambda x: True)
    
    # 设置随机种子
    env.reset(seed=42)
    
    episodes = 1
    
    for episode in range(episodes):
        print(f'Episode {episode + 1}')
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
        print(f'Episode {episode + 1} finished with reward: {total_reward}')
    
    env.close()

if __name__ == '__main__':
    # 可以自定义视频名称
    record_game("val_game")