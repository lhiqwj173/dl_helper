import numpy as np
import os, torch
import pygame
import gymnasium as gym
from gymnasium import spaces
import time

def default_crash_reward(snake, food, grid_size, shared_data):
    return -1

def default_eat_reward(snake, food, grid_size, shared_data):
    return 1

def default_move_reward(snake, food, grid_size, shared_data):
    return 0

def default_truncated_reward(snake, food, grid_size, shared_data):
    return -1

class SnakeEnv(gym.Env):
    """贪吃蛇环境，用于强化学习，带有Pygame可视化功能"""
    
    REG_NAME = 'snake_simple'
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 10}
    
    def __init__(self, config: dict):
        super(SnakeEnv, self).__init__()

        # 激励函数
        self.crash_reward = config.get('crash_reward', default_crash_reward)
        self.eat_reward = config.get('eat_reward', default_eat_reward)
        self.move_reward = config.get('move_reward', default_move_reward)
        self.truncated_reward = config.get('truncated_reward', default_truncated_reward)
        
        self.grid_size = config.get('grid_size', (10, 10))

        self.render_mode = config.get('render_mode', 'none')

        # 最大步数
        self.max_steps = self.grid_size[0] * self.grid_size[1]
        self.steps = 0
        
        if self.render_mode == 'human':
            pygame.init()
            info = pygame.display.Info()
            screen_width, screen_height = info.current_w, info.current_h
            max_cell_size = min((screen_width - 300) // self.grid_size[0], 
                              (screen_height - 300) // self.grid_size[1])
            self.cell_size = max(10, max_cell_size)
            self.window_size = (self.grid_size[0] * self.cell_size, 
                              self.grid_size[1] * self.cell_size)
            self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
            pygame.display.set_caption('Snake Game')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        # 0.0 3个动作空间
        # 0 前进
        # 1 左转
        # 2 右转
        self.action_space = spaces.Discrete(3)

        # # 0.1 4个动作空间
        # # 0 前进
        # # 1 左转
        # # 2 右转
        # # 3 不动
        # self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, ), dtype=np.float32)
        
        # 前进方向
        self.direction = (0, -1) # 初始朝向向上

        # 共享数据
        self.shared_data = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 0.0 只初始化一个蛇头
        # self.snake = [(self.grid_size[0] // 2, self.grid_size[1] // 2)]
        # 0.1 初始化蛇头 + 2蛇身
        self.snake = [
            (self.grid_size[0] // 2, self.grid_size[1] // 2), 
            (self.grid_size[0] // 2, self.grid_size[1] // 2 - 1), 
            (self.grid_size[0] // 2, self.grid_size[1] // 2 - 2)
        ]

        self.food = self._generate_food()
        self.done = False
        self.score = 0
        if self.render_mode == 'human':
            self.pygame_start_time = pygame.time.get_ticks()  # 用于渲染显示
        observation = self._get_state()

        self.direction = (0, -1)
        
        # 重置步数
        self.steps = 0

        # 重置共享数据
        self.shared_data = {}

        info = {}
        return observation, info
    
    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.grid_size[0]), 
                    np.random.randint(0, self.grid_size[1]))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        state = np.array([
            self.snake[0][0], 
            self.snake[0][1], 
            self.food[0], 
            self.food[1]
        ], dtype=np.float32)
        return state
    
    def step(self, action):
        if self.done:
            raise Exception("游戏已结束，请先调用 reset()")
        
        self.steps += 1

        # 0.0 3个动作空间
        # 根据动作改变朝向
        # 0: 前进, 1: 左转, 2: 右转
        if action == 1:  # 左转
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:  # 右转
            self.direction = (-self.direction[1], self.direction[0])

        # # 0.1 4个动作空间
        # # 0 上
        # # 1 下
        # # 2 左
        # # 3 右
        # if action == 0:
        #     self.direction = (-1, 0)
        # elif action == 1:
        #     self.direction = (1, 0)
        # elif action == 2:
        #     self.direction = (0, -1)
        # elif action == 3:
        #     self.direction = (0, 1)

        # 计算新的蛇头位置
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        if (new_head[0] < 0 or new_head[0] >= self.grid_size[0] or
            new_head[1] < 0 or new_head[1] >= self.grid_size[1] or
            new_head in self.snake):
            # 检查是否撞击边界或自身
            self.done = True
            reward = self.crash_reward(self.snake, self.food, self.grid_size, self.shared_data)
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                # 吃到食物
                reward = self.eat_reward(self.snake, self.food, self.grid_size, self.shared_data)
                self.food = self._generate_food()
                self.score += 1
                # 奖励重置步数
                self.steps = 0
            else:
                # 正常移动
                self.snake.pop()
                reward = self.move_reward(self.snake, self.food, self.grid_size, self.shared_data)
        
        self.reward = reward

        # 达到最大步数，需要截断
        truncated = self.steps >= self.max_steps
        if truncated:
            reward = self.truncated_reward(self.snake, self.food, self.grid_size, self.shared_data)
        
        observation = self._get_state()

        terminated = self.done
        info = {}
        return observation, reward, terminated, truncated, info
    
    def _get_render_state(self):
        grid = np.zeros(self.grid_size, dtype=np.float32)
        for i, segment in enumerate(self.snake):
            if i == 0:
                grid[segment[1], segment[0]] = 2  # 蛇头
            else:
                grid[segment[1], segment[0]] = 1  # 蛇身
        grid[self.food[1], self.food[0]] = 3  # 食物

        return grid

    def render(self):
        if self.render_mode != 'human':
            return
        
        self.screen.fill((0, 0, 0))  # 清屏
        
        state = self._get_render_state()
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                if state[y, x] == 1 or state[y, x] == 2:  # 蛇身或蛇头
                    pos = self.snake.index((x, y))
                    if pos == 0:
                        color = (0, 255, 0, 255)  # 蛇头，绿色不透明
                    elif pos == len(self.snake) - 1:
                        color = (255, 255, 0, 255)  # 蛇尾，黄色不透明
                    else:
                        m = len(self.snake) - 2  # 蛇身段数
                        if m > 1:
                            alpha = 255 - int(((pos - 1) / (m - 1)) * (255 - 150))
                        else:
                            alpha = 255
                        color = (0, 255, 0, alpha)  # 蛇身，绿色渐变透明
                elif state[y, x] == 3:
                    color = (0, 0, 255, 255)  # 食物，蓝色不透明
                else:
                    continue  # 跳过空白单元格
                
                # 使用带 alpha 的表面绘制
                cell_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                cell_surface.fill(color)
                self.screen.blit(cell_surface, (x * self.cell_size, y * self.cell_size))
        
        # # 绘制网格线
        # for x in range(0, self.window_size[0], self.cell_size):
        #     pygame.draw.line(self.screen, (255, 255, 255), (x, 0), (x, self.window_size[1]))
        # for y in range(0, self.window_size[1], self.cell_size):
        #     pygame.draw.line(self.screen, (255, 255, 255), (0, y), (self.window_size[0], y))
        
        # 显示分数和剩余时间
        if self.render_mode == 'human':
            pygame_elapsed_time = (pygame.time.get_ticks() - self.pygame_start_time) / 1000
            remaining_steps = max(0, (self.max_steps) - self.steps)
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            steps_text = self.font.render(f"Remaining Steps: {remaining_steps}", True, (255, 255, 255))
            reward_text = self.font.render(f"Reward: {self.reward:.2f}", True, (255, 255, 255)) 
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(steps_text, (10, 50))
            self.screen.blit(reward_text, (10, 90))
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
    
    def close(self):
        if self.render_mode == 'human':
            pygame.quit()

if __name__ == "__main__":
    from dl_helper.rl.rl_env.tool import human_control
    human_control(
        env_class=SnakeEnv,
        control_map={
            pygame.K_UP: 0,
            pygame.K_LEFT: 1,
            pygame.K_RIGHT: 2,
        },
        default_action=0,
    )