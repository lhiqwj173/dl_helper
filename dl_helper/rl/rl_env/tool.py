import os
import time
import pygame
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium
import gym

ENV_TYPE_NAMES = ['GYM', 'GYMNAIUM']
GYM, GYMNAIUM = range(2)

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

    # 判断是什么环境类型
    if isinstance(env, gymnasium.Env):
        env_type = GYMNAIUM
    elif isinstance(env, gym.Env):
        env_type = GYM
    print(f'环境类型: {ENV_TYPE_NAMES[env_type]}')

    if env_type == GYMNAIUM:
        observation, info = env.reset()
    else:
        observation = env.reset()

    done = False
    action = 3  # 初始方向：右
    clock = pygame.time.Clock()

    pygame.time.wait(3000)

    while not done:
        action = action if default_action is None else default_action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key in control_map:
                    action = control_map[event.key]
        
        if env_type == GYMNAIUM:
            observation, reward, terminated, truncated, info = env.step(action)
        else:
            observation, reward, terminated, info = env.step(action)
            truncated = False
        done = terminated or truncated
        env.render()
        clock.tick(game_speed)  # 控制游戏速度
    
    env.close()

def ai_control(
    env_class,
    env_config={},
    checkpoint_abs_path = '',
    times = 10,

    #sb3
    sb3_rl_model = None,
):
    def model_action(obs, rl_module, rllib=True):
        obs = torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs
        if rllib:
            # 增加batch维度
            obs = obs.unsqueeze(0)  # 从 (C,H,W) 变为 (B,C,H,W)
            results = rl_module.forward_inference({"obs":obs})
            action_logits = results['action_dist_inputs']  # 获取 logits 张量
            action = torch.argmax(action_logits).item()    # 取最大值索引并转为 Python 标量
        else:
            actions, states = rl_module.predict(obs)
            action = actions.item()
        print(f"action: {action}")
        return action

    # 测试环境 - 模型控制
    if 'render_mode' not in env_config:
        env_config['render_mode'] = 'human'
    env = env_class(
        env_config,
    )

    # 判断是什么环境类型
    if isinstance(env, gymnasium.Env):
        env_type = GYMNAIUM
    elif isinstance(env, gym.Env):
        env_type = GYM
    print(f'环境类型: {ENV_TYPE_NAMES[env_type]}')

    rllib = True if sb3_rl_model is None else False

    if rllib:
        # rllib
        rl_module_checkpoint_dir = os.path.join(checkpoint_abs_path,  "learner_group" , "learner" , "rl_module" , "default_policy")
        rl_module = RLModule.from_checkpoint(rl_module_checkpoint_dir)

    else:
        # sb3
        rllib = False
        sb3_rl_model.set_parameters(checkpoint_abs_path)
        rl_module = sb3_rl_model

    for _ in range(times):
        if env_type == GYMNAIUM:
            observation, info = env.reset()
        else:
            observation = env.reset()
        done = False
        clock = pygame.time.Clock()
        
        while not done:
            action = model_action(observation, rl_module, rllib)
            if env_type == GYMNAIUM:
                observation, reward, terminated, truncated, info = env.step(action)
            else:
                observation, reward, terminated, info = env.step(action)
                truncated = False
            done = terminated or truncated
            env.render()
            clock.tick(10)  # 控制游戏速度为10帧每秒

    env.close()