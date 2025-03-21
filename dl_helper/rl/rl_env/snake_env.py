import numpy as np
import os, torch
import pygame
import gymnasium as gym
from gymnasium import spaces
import time
from ray.rllib.core.rl_module.rl_module import RLModule

def default_crash_reward(snake, food, grid_size):
    return -1

def default_eat_reward(snake, food, grid_size):
    return 1

def default_move_reward(snake, food, grid_size):
    return 0

class SnakeEnv(gym.Env):
    """贪吃蛇环境，用于强化学习，带有Pygame可视化功能"""
    
    REG_NAME = 'snake'
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 10}
    
    def __init__(self, config: dict, render_mode='none'):
        super(SnakeEnv, self).__init__()

        # 激励函数
        self.crash_reward = config.get('crash_reward', default_crash_reward)
        self.eat_reward = config.get('eat_reward', default_eat_reward)
        self.move_reward = config.get('move_reward', default_move_reward)
        
        self.grid_size = config.get('grid_size', (10, 10))
        self.need_flatten = config.get('need_flatten', False)

        self.render_mode = render_mode
        self.total_time = 300000  # 5分钟，单位毫秒
        
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
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, *self.grid_size), dtype=np.float32) if not self.need_flatten else spaces.Box(low=0, high=1, shape=(np.prod(self.grid_size), ), dtype=np.float32)
        
        self.reset()
    
    def _std_obs(self, obs):
        obs /= np.float32(3)
        if self.need_flatten:
            obs = obs.flatten()

        else:
            # obs = obs.unsqueeze(0) # 增加一个维度，变成(1, 10, 10)
            obs = np.expand_dims(obs, axis=0)  # 形状从 (10, 10) 变成 (1, 10, 10)
            
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(self.grid_size[0] // 2, self.grid_size[1] // 2)]
        self.food = self._generate_food()
        self.done = False
        self.score = 0
        self.start_time = time.time()  # 记录开始时间，适用于所有模式
        if self.render_mode == 'human':
            self.pygame_start_time = pygame.time.get_ticks()  # 用于渲染显示
        observation = self._get_state()
        
        # 标准化
        observation = self._std_obs(observation)

        info = {}
        return observation, info
    
    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.grid_size[0]), 
                    np.random.randint(0, self.grid_size[1]))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        grid = np.zeros(self.grid_size, dtype=np.float32)
        for i, segment in enumerate(self.snake):
            if i == 0:
                grid[segment[1], segment[0]] = 2  # 蛇头
            else:
                grid[segment[1], segment[0]] = 1  # 蛇身
        grid[self.food[1], self.food[0]] = 3  # 食物

        return grid
    
    def step(self, action):
        if self.done:
            raise Exception("游戏已结束，请先调用 reset()")
        
        direction = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        head = self.snake[0]
        new_head = (head[0] + direction[action][0], head[1] + direction[action][1])
        
        if (new_head[0] < 0 or new_head[0] >= self.grid_size[0] or
            new_head[1] < 0 or new_head[1] >= self.grid_size[1] or
            new_head in self.snake):
            # 检查是否撞击边界或自身
            self.done = True
            reward = self.crash_reward(self.snake, self.food, self.grid_size)
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                # 吃到食物
                reward = self.eat_reward(self.snake, self.food, self.grid_size)
                self.food = self._generate_food()
                self.score += 1
            else:
                # 正常移动
                self.snake.pop()
                reward = self.move_reward(self.snake, self.food, self.grid_size)
        
        self.reward = reward

        # 检查是否超时
        truncated = False
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.total_time / 1000:  # 转换为秒
            self.done = True
            truncated = True
        
        observation = self._get_state()

        # 标准化
        observation = self._std_obs(observation)

        terminated = self.done and not truncated
        info = {}
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human':
            return
        
        self.screen.fill((0, 0, 0))  # 清屏
        
        state = self._get_state()
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
            remaining_time = max(0, (self.total_time / 1000) - pygame_elapsed_time)
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            time_text = self.font.render(f"Time: {int(remaining_time)}s", True, (255, 255, 255))
            reward_text = self.font.render(f"Reward: {self.reward:.2f}", True, (255, 255, 255)) 
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(time_text, (10, 50))
            self.screen.blit(reward_text, (10, 100))
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
    
    def close(self):
        if self.render_mode == 'human':
            pygame.quit()

def human_control(env_config = {
    'grid_size': (10, 10),
}):
    # 测试环境 - 手动控制
    env = SnakeEnv(
        env_config,
        render_mode='human',
    )
    observation, info = env.reset()
    done = False
    action = 3  # 初始方向：右
    clock = pygame.time.Clock()

    time.sleep(3)
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
        
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        clock.tick(4)  # 控制游戏速度为10帧每秒
    
    env.close()

def ai_control(
    env_config = {
        'grid_size': (10, 10),
    },
    checkpoint_abs_path = '',
    times = 10,
):
    def model_action(obs, rl_module):
        obs = torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs
        results = rl_module.forward_inference({"obs":obs})
        action_logits = results['action_dist_inputs']  # 获取 logits 张量
        action = torch.argmax(action_logits).item()    # 取最大值索引并转为 Python 标量
        print(f"action: {action}")
        return action

    # 测试环境 - 模型控制
    env = SnakeEnv(
        env_config,
        render_mode='human',
    )

    # 模型
    rl_module_checkpoint_dir = os.path.join(checkpoint_abs_path,  "learner_group" , "learner" , "rl_module" , "default_policy")
    rl_module = RLModule.from_checkpoint(rl_module_checkpoint_dir)

    for _ in range(times):
        observation, info = env.reset()
        done = False
        clock = pygame.time.Clock()
        
        while not done:
            action = model_action(observation, rl_module)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
            clock.tick(4)  # 控制游戏速度为10帧每秒

    env.close()

def test_run():
    env = SnakeEnv(
        {
            'grid_size': (5, 5),
        },
    )
    observation, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()

if __name__ == "__main__":
    human_control() 
    # test_run()
