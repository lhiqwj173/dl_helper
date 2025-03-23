import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from collections import deque
import pygame

def default_reward_func(
        frame_size, 
        ball_x_0, ball_y_0, 
        ball_x_1, ball_y_1, paddle_x_1, paddle_y_1, 
        reward
    ):
    """
    默认奖励函数
    1. 计算球的落点 x,y
    2. 计算弱点与挡板之间的距离
    3. 根据距离进行惩罚
    """
    if ball_x is None:
        return reward
    return - abs(ball_x - paddle_x) + reward

class PongEnv:
    REG_NAME = 'pong'

    def __init__(self, config: dict = {}):
        """
        初始化 Pong-v5 环境的封装。
        
        参数:
            reward_func: 自定义奖励函数，接收原始奖励并返回处理后的奖励，默认直接返回原始值。
            stack_size: 堆叠的帧数，默认为 4。
            frame_size: 预处理后的帧大小，默认为 (84, 84)。
        """
        self.reward_func = config.get('reward_func', default_reward_func)
        self.stack_size = config.get('stack_size', 3)
        self.frame_size = config.get('frame_size', (84, 84))
        self.render_mode = config.get('render_mode', 'none')

        # obs
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.stack_size, *self.frame_size), dtype=np.float32)
        # action
        self.action_space = spaces.Discrete(3)

        # 创建 Pong-v5 环境
        self.env = gym.make("ALE/Pong-v5")

        # 初始化帧堆栈
        self.frames = deque(maxlen=self.stack_size)
        self._reset_frames()

        if self.render_mode == 'human':
            # 初始化 Pygame
            pygame.init()
            self.screen_width = self.frame_size[1] * 4  # 放大后的宽度
            self.screen_height = self.frame_size[0] * 4             # 放大后的高度
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Pong-v5 with Pygame")
            self.clock = pygame.time.Clock()

    def _find_ball_position(self, frame):
        """
        从单帧图像中找到球的中心位置。
        找值 217
        """
        # 查找值为 217 的所有位置
        positions = np.where(frame > 150)
        x_coords = positions[0]  # 行坐标
        y_coords = positions[1]  # 列坐标

        # 过滤 11 - 72 以外的
        bad_idxs = []
        for i, y in enumerate(y_coords):
            if y < 11 or y > 72:
                bad_idxs.append(i)
        x_coords = np.delete(x_coords, bad_idxs)
        y_coords = np.delete(y_coords, bad_idxs)
        
        if len(x_coords) == 0:
            return None, None
        elif len(x_coords) == 1:
            # 只有一个位置，返回其坐标
            return y_coords[0], x_coords[0], 
        else:
            # 有多个位置，返回 x 和 y 的均值
            mean_x = np.mean(x_coords)
            mean_y = np.mean(y_coords)
            return mean_y, mean_x

    def _find_paddle_position(self, frame):
        """
        获取挡板中心位置
        在 74 列找值 147
        """
        # 提取第 74 列
        column_74 = frame[:, 74]
        
        # 找到值为 147 的位置
        positions = np.where(column_74 == 147)[0]

        length = len(positions)
        assert length > 1

        mid_index = length // 2  # 整数除法，得到中间索引

        if length % 2 == 1:  # 奇数长度
            return (74, positions[mid_index])
        else:  # 偶数长度
            return (74, (positions[mid_index - 1] + positions[mid_index]) / 2)

    def _preprocess_frame(self, frame):
        """
        预处理单帧图像：灰度化、裁剪无关区域、缩放到指定大小。
        """
        # 转为灰度图
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 裁剪掉顶部和底部的无关区域（Pong 中通常是得分和边界）
        frame = frame[34:194]  # 裁剪到 160x160
        
        # 缩放到指定大小 (84x84)
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

        return frame

    def _reset_frames(self):
        """
        重置帧堆栈，填充初始帧。
        """
        self.frames.clear()
        # 用全零帧填充初始堆栈
        zero_frame = np.zeros(self.frame_size, dtype=np.float32)
        for _ in range(self.stack_size):
            self.frames.append(zero_frame)

    def _get_stacked_obs(self):
        """
        获取堆叠的观测值，形状为 (stack_size, height, width)。
        """
        return np.stack(self.frames, axis=0)

    def _stack_step(self, action):
        """
        堆叠步
        """
        total_reward = 0
        for i in range(self.stack_size):
            obs, reward, terminated, truncated, info = self.env.step(0 if i != 0 else action)# 只有第一步执行动作，其他都是 NOOP
            # obs, reward, terminated, truncated, info = self.env.step(action)# 所有步都执行动作
            total_reward += reward
            processed_frame = self._preprocess_frame(obs)
            self.frames.append(processed_frame)
            if terminated or truncated:
                break
        
        return total_reward, terminated, truncated, info

    def reset(self):
        """
        重置环境，返回初始堆叠观测值。
        """
        self.env.reset()
        # 跳过最开始的15个step
        for _ in range(15):
            self.env.step(0)
        self._reset_frames()

        # 累计最初的堆叠帧         
        reward, terminated, truncated, self.info = self._stack_step(0)
        # 归一化到 [0, 1]
        return self._get_stacked_obs() / 255.0, self.info

    def step(self, action):
        """
        执行一步动作，返回 (堆叠观测值, 奖励, 终止标志, 截断标志, 信息)。
        """
        # 校正动作
        if action > 0:
            action += 1

        # 新堆叠
        # reward, terminated, truncated, info = self._stack_step(action)
        # obs = self._get_stacked_obs()

        # 沿用旧堆叠
        obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.frames.append(self._preprocess_frame(obs))
        obs = self._get_stacked_obs()

        # 计算球和挡板的位置
        ball_x_1, ball_y_1 = self._find_ball_position(obs[-1])
        paddle_x_1, paddle_y_1 = self._find_paddle_position(obs[-1])
        ball_x_0, ball_y_0 = self._find_ball_position(obs[0])
        
        # 使用自定义或默认奖励函数处理奖励
        processed_reward = self.reward_func(
            self.frame_size, 
            ball_x_0, ball_y_0, 
            ball_x_1, ball_y_1, paddle_x_1, paddle_y_1, 
            reward
        )
        self.info['reward'] = processed_reward
        
        # 归一化到 [0, 1]
        return obs / 255.0, processed_reward, terminated, truncated, self.info

    def render(self):
        """
        使用 Pygame 渲染预处理的堆叠帧，并按 scale_factor 放大。
        """ 
        if self.render_mode != 'human':
            return
        
        # 获取当前堆叠帧
        stacked_obs = self._get_stacked_obs()  # 形状: (4, 84, 84)

        # 只使用最后一个
        display_frame = stacked_obs[-1].astype(np.uint8)  # 恢复到 0-255
        
        # 转置数组以匹配 Pygame 坐标系
        display_frame = np.transpose(display_frame)

        # 转换为 Pygame surface 并放大
        surface = pygame.surfarray.make_surface(display_frame)
        surface = pygame.transform.scale(surface, (self.screen_width, self.screen_height))
        
        # 渲染到屏幕
        self.screen.blit(surface, (0, 0))

        # 假设 self.info 是最新的信息字典，如果不是，请确保传入正确的 info
        if hasattr(self, 'info') and self.info:
            # 初始化字体（如果尚未初始化）
            if not hasattr(self, 'font'):
                pygame.font.init()
                self.font = pygame.font.SysFont('Arial', 24)  # 使用 Arial 字体，大小 24
            
            # 将 info 转换为字符串
            info_texts = [f'{k}: {v}' for k, v in self.info.items()]
            for idx, info_text in enumerate(info_texts):
                # 渲染文本
                text_surface = self.font.render(info_text, True, (255, 255, 255))  # 白色文本
                text_rect = text_surface.get_rect(topleft=(10, 10 + idx * 24))  # 左上角位置 (10, 10)
                
                # 将文本绘制到屏幕上
                self.screen.blit(text_surface, text_rect)
    
        pygame.display.flip()

    def close(self):
        """关闭环境和 Pygame"""
        self.env.close()
        pygame.quit()

# 测试代码
if __name__ == "__main__":
    from dl_helper.rl.rl_env.tool import human_control
    human_control(
        env_class=PongEnv,
        control_map={
            pygame.K_UP: 1, 
            pygame.K_DOWN: 2,
            pygame.K_SPACE: 0 
        },
        default_action=0,
        game_speed=8,
    )
