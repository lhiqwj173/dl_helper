import os
import time
import pygame
import torch
from ray.rllib.core.rl_module.rl_module import RLModule

def human_control(
        env_class,
        env_config={},
        control_map={
            pygame.K_UP: 0,
            pygame.K_DOWN: 1,
            pygame.K_LEFT: 2,
            pygame.K_RIGHT: 3,
        },
        default_action=None,
        game_speed=4,
    ):
    # 测试环境 - 手动控制
    if 'render_mode' not in env_config:
        env_config['render_mode'] = 'human'
    env = env_class(
        env_config,
    )
    observation, info = env.reset()
    done = False
    action = 3  # 初始方向：右
    clock = pygame.time.Clock()

    time.sleep(1)
    
    while not done:
        action = action if default_action is None else default_action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key in control_map:
                    action = control_map[event.key]
        
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        clock.tick(game_speed)  # 控制游戏速度
    
    env.close()

def ai_control(
    env_class,
    env_config={},
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
    if 'render_mode' not in env_config:
        env_config['render_mode'] = 'human'
    env = env_class(
        env_config,
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
