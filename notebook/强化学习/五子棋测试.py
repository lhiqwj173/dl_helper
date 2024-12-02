import pygame, os
import numpy as np
import stable_baselines3 as sb3
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional, Tuple

# 颜色定义
BOARD_COLOR = (205, 170, 125)  # 棋盘底色
LINE_COLOR = (50, 50, 50)      # 棋盘线条颜色
WHITE_PIECE_COLOR = (255, 255, 255)  # 白棋颜色
BLACK_PIECE_COLOR = (0, 0, 0)        # 黑棋颜色
BACKGROUND_COLOR = (230, 220, 180)   # 背景颜色

class GomokuEnv(gym.Env):
    """
    自定义五子棋强化学习环境
    """
    def __init__(self, board_size: int = 15):
        super().__init__()
        
        # 定义动作和观察空间
        self.board_size = board_size
        self.action_space = gym.spaces.Discrete(board_size * board_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=2, 
            shape=(board_size, board_size), 
            dtype=np.int8
        )
        
        # 初始化游戏板
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        """
        重置环境，适配新的 Gymnasium 接口
        """
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        return self.board, {}  # 返回初始观察和额外信息
    
    def step(self, action: int):
        """
        执行动作并返回下一个状态、奖励、是否终止等信息
        """
        # 将1维动作转换为2维坐标
        row = action // self.board_size
        col = action % self.board_size
        
        # 检查动作是否有效
        if self.board[row, col] != 0:
            return self.board, -10, True, False, {}  # 无效动作的惩罚
        
        # 放置棋子
        self.board[row, col] = self.current_player
        
        # 检查游戏是否结束
        done = self._check_winner(row, col)
        
        # 切换玩家
        self.current_player = 3 - self.current_player
        
        # 计算奖励
        reward = 10 if done else 0
        
        return self.board, reward, done, False, {}
    
    def _check_winner(self, row: int, col: int) -> bool:
        """
        检查是否有玩家获胜
        """
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 对角线1
            (1, -1)   # 对角线2
        ]
        
        for dx, dy in directions:
            count = self._count_consecutive(row, col, dx, dy)
            if count >= 5:
                return True
        
        return False
    
    def _count_consecutive(self, row: int, col: int, dx: int, dy: int) -> int:
        """
        计算连续棋子数量
        """
        player = self.board[row, col]
        count = 1
        
        # 正方向计数
        x, y = row + dx, col + dy
        while (0 <= x < self.board_size and 0 <= y < self.board_size and
               self.board[x, y] == player):
            count += 1
            x += dx
            y += dy
        
        # 反方向计数
        x, y = row - dx, col - dy
        while (0 <= x < self.board_size and 0 <= y < self.board_size and
               self.board[x, y] == player):
            count += 1
            x -= dx
            y -= dy
        
        return count

def train_gomoku_agent():
    """
    训练五子棋强化学习代理
    """
    # 创建环境
    env = GomokuEnv()
    env = DummyVecEnv([lambda: env])

    # 选择算法：这里使用PPO
    if os.path.exists('gomoku_agent.zip'):
        model = sb3.PPO.load('gomoku_agent')
        model.set_env(env)
    else:
        model = sb3.PPO(
            policy='MlpPolicy', 
            env=env, 
            verbose=0,
            learning_rate=0.001,
            n_steps=2048,
            batch_size=64,
            n_epochs=10
        )

    # 训练前的性能
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Before training: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    
    # 训练模型
    model.learn(total_timesteps=1e8, progress_bar=True)
    
    # 训练后的性能
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    improvement = ((mean_reward - mean_reward_before) / abs(mean_reward_before)) * 100 if mean_reward_before != 0 else float('inf')
    print(f"训练后: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"性能提升: {improvement:.2f}%")
    
    # 保存模型
    model.save("gomoku_agent")
    
    return model

def print_board(board):
    """
    打印游戏板
    """
    # 确保board是NumPy数组
    if isinstance(board, tuple):
        board = board[0] if len(board) > 0 else board

    board_size = board.shape[0]
    
    # 打印列号，对齐
    print("   " + " ".join(f"{i:2d}" for i in range(board_size)))
    
    for i, row in enumerate(board):
        # 行号左对齐，棋子右对齐
        print(f"{i:2d} ", end="")
        for cell in row:
            if cell == 0:
                print(". ", end=" ")
            elif cell == 1:
                print("○", end=" ")
            else:
                print("●", end=" ")
        print()
def human_vs_ai():
    """
    人机对弈
    """
    # 加载训练好的模型
    model = sb3.PPO.load("gomoku_agent")
    
    # 初始化环境
    env = GomokuEnv()
    obs, _ = env.reset()
    
    # 游戏循环
    done = False
    human_player = 2  # 玩家使用黑棋（2）
    
    print("欢迎来到五子棋！")
    print("你是黑棋(●)，AI是白棋(○)")
    print("请输入棋子坐标，格式为 行 列，例如 7 7")
    
    while not done:
        print_board(obs)
        
        if env.current_player == human_player:
            # 人类玩家回合
            while True:
                try:
                    row, col = map(int, input("请输入你的落子坐标（行 列）: ").split())
                    action = row * env.board_size + col
                    
                    # 检查动作是否有效
                    if 0 <= row < env.board_size and 0 <= col < env.board_size and obs[row, col] == 0:
                        break
                    else:
                        print("无效的落子位置，请重新输入。")
                except (ValueError, IndexError):
                    print("输入格式错误，请重新输入。")
        else:
            # AI回合
            action, _ = model.predict(obs)
            print(f"AI落子：{action // env.board_size} {action % env.board_size}")
        
        # 执行动作
        obs, reward, done, _, _ = env.step(action)
        
        # 检查游戏结果
        if done:
            print_board(obs)
            if reward > 0:
                winner = "AI" if env.current_player == human_player else "玩家"
                print(f"{winner}获胜！")
            else:
                print("平局")
            break

class GomokuGame:
    def __init__(self, board_size=15, cell_size=40):
        pygame.init()
        
        self.board_size = board_size
        self.cell_size = cell_size
        self.board_width = board_size * cell_size
        self.board_height = board_size * cell_size
        
        # 额外的边距
        self.margin = cell_size
        
        # 窗口大小
        self.screen_width = self.board_width + 2 * self.margin
        self.screen_height = self.board_height + 2 * self.margin
        
        # 创建窗口
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("五子棋 - 人机对战")
        
        # 字体
        self.font = pygame.font.Font(None, 36)
        
        # 游戏环境和AI模型
        self.env = GomokuEnv(board_size)
        self.model = sb3.PPO.load("gomoku_agent")
        
        # 游戏状态
        self.board, _ = self.env.reset()
        self.game_over = False
        self.winner = None
        
    def draw_board(self):
        # 填充背景
        self.screen.fill(BACKGROUND_COLOR)
        
        # 绘制棋盘底色
        pygame.draw.rect(
            self.screen, 
            BOARD_COLOR, 
            (self.margin, self.margin, self.board_width, self.board_height)
        )
        
        # 绘制网格线
        for i in range(self.board_size + 1):
            # 垂直线
            pygame.draw.line(
                self.screen, 
                LINE_COLOR, 
                (self.margin + i * self.cell_size, self.margin),
                (self.margin + i * self.cell_size, self.margin + self.board_height)
            )
            # 水平线
            pygame.draw.line(
                self.screen, 
                LINE_COLOR, 
                (self.margin, self.margin + i * self.cell_size),
                (self.margin + self.board_width, self.margin + i * self.cell_size)
            )
        
        # 绘制棋子
        for row in range(self.board_size):
            for col in range(self.board_size):
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                
                if self.board[row, col] == 1:
                    # 白棋
                    pygame.draw.circle(
                        self.screen, 
                        WHITE_PIECE_COLOR, 
                        (x, y), 
                        self.cell_size // 2 - 2
                    )
                    pygame.draw.circle(
                        self.screen, 
                        LINE_COLOR, 
                        (x, y), 
                        self.cell_size // 2 - 2, 
                        2
                    )
                elif self.board[row, col] == 2:
                    # 黑棋
                    pygame.draw.circle(
                        self.screen, 
                        BLACK_PIECE_COLOR, 
                        (x, y), 
                        self.cell_size // 2 - 2
                    )
    
    def get_board_position(self, mouse_pos):
        """将鼠标坐标转换为棋盘坐标"""
        x, y = mouse_pos
        col = (x - self.margin + self.cell_size // 2) // self.cell_size
        row = (y - self.margin + self.cell_size // 2) // self.cell_size
        
        return int(row), int(col)
    
    def display_game_over(self):
        """显示游戏结束信息"""
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # 半透明黑色遮罩
        self.screen.blit(overlay, (0, 0))
        
        # 显示获胜信息
        if self.winner == 1:
            text = "AI WIN!"
        elif self.winner == 2:
            text = "YOU WIN!"
        else:
            text = "DRAW"
        
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.screen_width//2, self.screen_height//2))
        self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
        
        # 等待2秒后关闭
        pygame.time.wait(2000)
    
    def run(self):
        clock = pygame.time.Clock()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # 玩家回合
                if not self.game_over and self.env.current_player == 2:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        row, col = self.get_board_position(event.pos)
                        
                        # 检查落子是否有效
                        if (0 <= row < self.board_size and 
                            0 <= col < self.board_size and 
                            self.board[row, col] == 0):
                            
                            action = row * self.board_size + col
                            self.board, reward, done, _, _ = self.env.step(action)
                            
                            if done:
                                self.game_over = True
                                self.winner = 2  # 玩家获胜
            
            # AI 回合
            if not self.game_over and self.env.current_player == 1:
                action, _ = self.model.predict(self.board)
                self.board, reward, done, _, _ = self.env.step(action)
                
                if done:
                    self.game_over = True
                    self.winner = 1  # AI获胜
            
            # 绘制棋盘
            self.draw_board()
            
            # 如果游戏结束，显示结果
            if self.game_over:
                self.display_game_over()
                running = False
            
            # 更新显示
            pygame.display.flip()
            clock.tick(10)  # 控制帧率
        
        pygame.quit()

if __name__ == "__main__":
    train_gomoku_agent()
    # human_vs_ai()

    # game = GomokuGame()
    # game.run()