import numpy as np
import gym
import pygame

class SnakeEnv(gym.Env):
    def __init__(self, config: dict):
        super(SnakeEnv, self).__init__()
        self.s = config.get('size', 10)  # 网格大小
        self.render_mode = config.get('render_mode', 'none')  # 渲染模式
        
        self.orientation = 0  # 初始方向：0-上, 1-右, 2-下, 3-左
        self.actions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # 动作对应的移动

        self.last_turn = 0  # 最后转弯方向
        self.snake = None  # 蛇的位置列表
        self.apple = None  # 苹果的位置
        self.time = 0  # 步数
        self.score = 0  # 分数
        self.max_steps = self.s ** 2  # 最大步数
        self.reward = 0  # 奖励

        # 定义动作空间和观测空间
        self.action_space = gym.spaces.Discrete(3)  # 0-左转, 1-前进, 2-右转
        # self.observation_space = gym.spaces.Discrete(256 * 32 * 8 * 2 * 2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        # 初始化 Pygame
        if self.render_mode == 'human':
            pygame.init()
            self.cell_size = 60  # 每个单元格的像素大小
            self.grid_size = (self.s, self.s)  # 网格大小
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
        return self.state()

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
                    self.reward = len(self.snake) ** 2  # 吃苹果的奖励
                    terminated = False
                else:
                    self.reward = 1000000  # 填满网格的奖励
                    terminated = True
            else:
                self.snake = np.delete(self.snake, -1, axis=0)
                self.reward = -self.time  # 每步的负奖励
                terminated = False
        else:
            self.reward = -1000  # 撞墙或撞尾的惩罚
            terminated = True

        observation = self.state()
        truncated = False  # 可根据需要添加截断逻辑
        info = {}

        return observation, self.reward, terminated, info

    def state_0(self):
        x, y = self.snake[0, 0], self.snake[0, 1]
        ori = self.orientation

        surrounding = np.empty(3)
        for i in range(3):
            new_ori = (ori + i - 1) % 4
            new_pos = np.array([x, y]) + self.actions[new_ori]
            no_wall_collision = (0 <= new_pos[0] < self.s) and (0 <= new_pos[1] < self.s)
            no_tail_collision = not any(np.array_equal(pos, new_pos) for pos in self.snake)
            surrounding[i] = not (no_wall_collision and no_tail_collision)
        vis1 = int(surrounding[0] * 4 + surrounding[1] * 2 + surrounding[2])

        tail = np.zeros(4)
        for pos in self.snake[1:]:
            if pos[0] == x:
                if pos[1] > y: tail[(2 - ori) % 4] = 1
                elif pos[1] < y: tail[(0 - ori) % 4] = 1
            elif pos[1] == y:
                if pos[0] > x: tail[(3 - ori) % 4] = 1
                elif pos[0] < x: tail[(1 - ori) % 4] = 1
        vis2 = int(tail[0] * 4 + tail[1] * 2 + tail[2])

        ax, ay = self.apple[0], self.apple[1]
        if ax < x and ay == y: vis3 = (0 + 2 * ori) % 8
        elif ax < x and ay < y: vis3 = (1 + 2 * ori) % 8
        elif ax == x and ay < y: vis3 = (2 + 2 * ori) % 8
        elif ax > x and ay < y: vis3 = (3 + 2 * ori) % 8
        elif ax > x and ay == y: vis3 = (4 + 2 * ori) % 8
        elif ax > x and ay > y: vis3 = (5 + 2 * ori) % 8
        elif ax == x and ay > y: vis3 = (6 + 2 * ori) % 8
        elif ax < x and ay > y: vis3 = (7 + 2 * ori) % 8
        else: vis3 = 0

        vis4 = self.last_turn
        vis5 = int(len(self.snake) < self.s ** 2 / 3)

        return vis1 * 256 + vis2 * 32 + vis3 * 4 + vis4 * 2 + vis5

    def state(self):
        x, y = self.snake[0, 0], self.snake[0, 1]
        ori = self.orientation

        # Vision #1: 周围是否有墙或尾巴 (左、前、右)
        surrounding = np.empty(3)
        for i in range(3):
            new_ori = (ori + i - 1) % 4
            new_pos = np.array([x, y]) + self.actions[new_ori]
            no_wall_collision = (0 <= new_pos[0] < self.s) and (0 <= new_pos[1] < self.s)
            no_tail_collision = not any(np.array_equal(pos, new_pos) for pos in self.snake)
            surrounding[i] = not (no_wall_collision and no_tail_collision)
        vis1 = surrounding[0] * 4 + surrounding[1] * 2 + surrounding[2]  # 范围: 0-7

        # Vision #2: 尾巴在三个方向的存在 (左、前、右)
        tail = np.zeros(4)
        for pos in self.snake[1:]:
            if pos[0] == x:
                if pos[1] > y: tail[(2 - ori) % 4] = 1
                elif pos[1] < y: tail[(0 - ori) % 4] = 1
            elif pos[1] == y:
                if pos[0] > x: tail[(3 - ori) % 4] = 1
                elif pos[0] < x: tail[(1 - ori) % 4] = 1
        vis2 = tail[0] * 4 + tail[1] * 2 + tail[2]  # 范围: 0-7

        # Vision #3: 苹果相对位置 (8个象限)
        ax, ay = self.apple[0], self.apple[1]
        if ax < x and ay == y: vis3 = (0 + 2 * ori) % 8
        elif ax < x and ay < y: vis3 = (1 + 2 * ori) % 8
        elif ax == x and ay < y: vis3 = (2 + 2 * ori) % 8
        elif ax > x and ay < y: vis3 = (3 + 2 * ori) % 8
        elif ax > x and ay == y: vis3 = (4 + 2 * ori) % 8
        elif ax > x and ay > y: vis3 = (5 + 2 * ori) % 8
        elif ax == x and ay > y: vis3 = (6 + 2 * ori) % 8
        elif ax < x and ay > y: vis3 = (7 + 2 * ori) % 8
        else: vis3 = 0  # 范围: 0-7

        # Vision #4: 最后转弯方向
        vis4 = self.last_turn  # 范围: 0-1

        # Vision #5: 蛇的长度 (短或长)
        vis5 = int(len(self.snake) < self.s ** 2 / 3)  # 范围: 0-1

        # 返回一个包含所有特征的数组
        obs = np.array([vis1, vis2, vis3, vis4, vis5], dtype=np.float32)

        # 归一化，最大值np.array([7, 7, 7, 1, 1])
        obs = obs / np.array([7, 7, 7, 1, 1])

        return obs

    def reset_apple(self):
        possible_positions = [(i, j) for i in range(self.s) for j in range(self.s) 
                             if not any(np.array_equal([i, j], pos) for pos in self.snake)]
        self.apple = np.array(possible_positions[np.random.randint(len(possible_positions))])
        self.time = 0

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


if __name__ == "__main__":
    from dl_helper.rl.rl_env.tool import human_control, ai_control
    human_control(
        SnakeEnv, 
        {},         
        control_map={
            pygame.K_UP: 1,
            pygame.K_LEFT: 0,
            pygame.K_RIGHT: 2,
        },
        default_action=1,
    )
