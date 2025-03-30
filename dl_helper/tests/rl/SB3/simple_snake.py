from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch
import pygame
import numpy as np
import random
import sys

import gym
from gym import spaces

# since number of actions == 4 that means that the action value is from 0 to 3 inclusive
class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    difficulty = 20
    # Window size
    frame_size_x = 100
    frame_size_y = 100
    # Colors (R, G, B)
    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)
    # Action Constants
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    def __init__(self):
        super(SnakeEnv, self).__init__()

        self.pygame_inited = False
        pygame.init()

        # Game variables
        self.snake_pos = [50, 50]
        self.prev_snake_pos = [50, 50]
        self.snake_body = [[50, 50], [50 - 10, 50], [50 - (2 * 10), 50]]

        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True

        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.counter = 0
        self.score = 0
        self.game_over = False

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        number_of_actions = 4
        number_of_observations = 4
        self.action_space = spaces.Discrete(number_of_actions)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(number_of_observations,), dtype=np.float32)

    def _init_pygame(self):
        if not self.pygame_inited:
            self.pygame_inited = True
        else:
            return  
            
        # Initialise game window
        pygame.display.set_caption('Snake Eater')
        # FPS (frames per second) controller
        self.fps_controller = pygame.time.Clock()
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))

    def step(self, action):
        self.counter += 1
        if self.counter > 100:
            return np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]], dtype=np.float32), -100, True, {}
        if action == self.UP:
            self.change_to = 'UP'
        if action == self.DOWN:
            self.change_to = 'DOWN'
        if action == self.LEFT:
            self.change_to = 'LEFT'
        if action == self.RIGHT:
            self.change_to = 'RIGHT'
        # Making sure the snake cannot move in the opposite direction instantaneously
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Moving the snake
        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10
        # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.counter = 0
            self.score += 1
            self.food_spawn = False
        else:
            self.snake_body.pop()

        # Spawning food on the screen
        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True
        # Game Over conditions
        # Getting out of bounds
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - 10:
            self.game_over = True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - 10:
            self.game_over = True
        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                self.game_over = True
        reward = 0
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            reward = 100
        elif abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1]) > abs(self.prev_snake_pos[0] - self.food_pos[0]) + abs(self.prev_snake_pos[1] - self.food_pos[1]):
            reward = -1
        elif abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1]) < abs(self.prev_snake_pos[0] - self.food_pos[0]) + abs(self.prev_snake_pos[1] - self.food_pos[1]):
            reward = 1
        self.prev_snake_pos = self.snake_pos.copy()
        done = self.game_over
        info = {}
        return np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]], dtype=np.float32), reward, done, info
    
    def reset(self):
        # Game variables
        self.snake_pos = [50, 50]
        self.prev_snake_pos = [50, 50]
        self.snake_body = [[50, 50], [50 - 10, 50], [50 - (2 * 10), 50]]
        self.counter = 0
        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10,
                        random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True

        self.direction = 'RIGHT'
        self.change_to = self.direction

        self.score = 0
        self.game_over = False
        return np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]], dtype=np.float32)
    def render(self, mode='human'):
        self._init_pygame()
        # GFX
        self.game_window.fill(self.black)
        for pos in self.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(self.game_window, self.white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))
        self.show_score(1, self.white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        self.fps_controller.tick(self.difficulty)

    def close (self):
        pygame.quit()
        sys.exit()

    def show_score(self, choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (self.frame_size_x / 10, 15)
        else:
            score_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 1.25)
        self.game_window.blit(score_surface, score_rect)
        # pygame.display.flip()


# 创建带 Monitor 的环境函数
def make_env():
    env = SnakeEnv()
    return env

run_type = 'train'
# run_type = 'test'
if run_type == 'train':

    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecMonitor(env)  # 添加监控器

    # 指定设备为 GPU (cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO('MlpPolicy', env, verbose=1, device=device)

    # 打印模型结构
    print("模型结构:")
    print(model.policy)
    print(f'total params: {sum(p.numel() for p in model.policy.parameters())}')
    # # 可选：打印所有属性
    # print("\nAll Attributes:")
    # for key, value in model.__dict__.items():
    #     print(f"{key}: {value}")

    sys.exit()

    model.learn(total_timesteps=1000000)
    model.save(r"D:\code\dl_helper\dl_helper\tests\rl\SB3\simple_snake.zip")

    # for i in range(1000):
    #     model.learn(total_timesteps=1000000)
    #     model.save(f"simple_snake_{i}")
else:
    env = SnakeEnv()
    model = PPO.load(r"D:\code\dl_helper\dl_helper\tests\rl\SB3\simple_snake.zip")

    obs = env.reset()
    for i in range(2000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()