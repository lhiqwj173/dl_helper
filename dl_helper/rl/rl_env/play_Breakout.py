import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import pygame, time
import numpy as np

import ale_py
gym.register_envs(ale_py)

delay = 0.05  # 设置延迟时间（秒），可以调整这个值来改变速度

def rotate_observation(obs):
    return np.rot90(obs, k=1)

def process_frame(frame):
    """处理游戏画面以便显示"""
    # 调整大小以便更好地显示
    return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)

def main():
    # 创建环境
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    # 正确创建 TransformObservation
    env = TransformObservation(
        env,
        rotate_observation,
        observation_space=env.observation_space
    )
    observation, info = env.reset()
    
    # 初始化 Pygame
    pygame.init()
    # 由于 Atari 游戏画面是 210x160，我们放大两倍显示
    screen = pygame.display.set_mode((320, 420))
    pygame.display.set_caption('Atari Breakout')
    clock = pygame.time.Clock()
    
    # 游戏主循环
    done = False
    total_reward = 0

    while not done:
        # 显示游戏画面
        if observation is not None:
            # 处理和显示游戏画面
            processed_frame = process_frame(observation)
            surf = pygame.surfarray.make_surface(processed_frame)
            screen.blit(surf, (0, 0))
            
            # 显示得分
            font = pygame.font.Font(None, 36)
            score_text = font.render(f'Score: {total_reward}', True, (255, 255, 255))
            screen.blit(score_text, (10, 400))
            
            pygame.display.flip()
            clock.tick(60)

        time.sleep(delay)  # 添加延时
        action = 0  # NOOP 默认动作
        
        # 处理键盘输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 2  # LEFT
                elif event.key == pygame.K_RIGHT:
                    action = 3  # RIGHT
                elif event.key == pygame.K_SPACE:
                    action = 1  # FIRE/START
                elif event.key == pygame.K_q:
                    done = True
        
        # 执行动作
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # 如果获得奖励,显示奖励值
        if reward > 0:
            reward_font = pygame.font.Font(None, 48)
            reward_text = reward_font.render(f'+{reward}', True, (255, 255, 0))
            screen.blit(reward_text, (160, 210))  # 在屏幕中央显示
            pygame.display.flip()

        if info:
            print(info)
        
        time.sleep(0.5)  # 显示半秒钟
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()