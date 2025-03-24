"""
配对交易强化学习
"""
import sys, os, time, shutil, pickle
import numpy as np
from dl_helper.tool import in_windows
# os.environ["RAY_DEDUP_LOGS"] = "0"
import matplotlib.pyplot as plt
from ray.tune.registry import get_trainable_cls, register_env

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import (
    ICM_MODULE_ID,
)
from ray.rllib.examples.rl_modules.classes.intrinsic_curiosity_model_rlm import (
    IntrinsicCuriosityModel,
)

from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics, stop
from dl_helper.rl.socket_base import request_need_val
from py_ext.tool import init_logger, log
from py_ext.datetime import beijing_time
from dl_helper.train_folder_manager import TrainFolderManager

from dl_helper.rl.costum_rllib_module.snake.mlp import MLPPPOCatalog
from dl_helper.rl.costum_rllib_module.snake.cnn import CNNPPOCatalog
from dl_helper.rl.rl_env.snake.snake_env import SnakeEnv
from dl_helper.rl.rl_env.tool import human_control, ai_control

model_type = 'mlp'
for arg in sys.argv:
    if arg == 'cnn':
        model_type = 'cnn'

train_folder = train_title = f'20250322_snake_build_in' + f'_{model_type}'
init_logger(train_title, home=train_folder, timestamp=False)

# 吃到食物标准奖励
STD_EAT_FOOD_REWARD = 100

# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_EAT_FOOD_REWARD / 2

"""
激励函数

# 最大吃食物数量
MAX_EAT_FOOD_NUM = 10 *10 - 1

# 奖励函数
shaping = -(距离²/(10² + 10²)) * STD_MOVE_REWARD - (MAX_EAT_FOOD_NUM - 吃到食物数量) * STD_EAT_FOOD_REWARD
"""

def crash_reward(snake, food, grid_size):
    # 10 * 10 的网格, 最大惩罚: -10000
    MAX_EAT_FOOD_NUM = grid_size[0] * grid_size[1] - 1
    return -(MAX_EAT_FOOD_NUM + 1) * STD_EAT_FOOD_REWARD

def keep_alive_reward(snake, food, grid_size):
    MAX_EAT_FOOD_NUM = grid_size[0] * grid_size[1] - 1
    eat_food_num = len(snake) - 1
    distance_sqrt = (snake[0][0] - food[0])**2 + (snake[0][1] - food[1])**2
    return -(distance_sqrt/(grid_size[0]**2 + grid_size[1]**2)) * STD_MOVE_REWARD - (MAX_EAT_FOOD_NUM - eat_food_num) * STD_EAT_FOOD_REWARD

def eat_reward(snake, food, grid_size):
    return 1

def move_reward(snake, food, grid_size):
    return 0

if __name__ == "__main__":

    env_config = {
        'grid_size': (10, 10),
        'crash_reward': crash_reward,
        'eat_reward': keep_alive_reward,
        'move_reward': keep_alive_reward,
        'model_type': model_type,
    }

    # 人控制
    # human_control(env_config)
    # import sys
    # sys.exit()

    # # 模型控制
    # ai_control(env_config, checkpoint_abs_path=r'C:\Users\lh\Desktop\temp\checkpoint')
    # import sys
    # sys.exit()

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes() if not in_windows() else 0
    log(f"num_learners: {num_learners}")

    if model_type == 'mlp':
        # # 模型参数量: 33048
        # model_config = {
        #     'input_dims': (10, 10),
        #     'hidden_sizes': [128, 128],
        #     'output_dims': 24,
        # } 
        model_config = DefaultModelConfig(
            fcnet_hiddens=[128, 128, 24],
        )

    elif model_type == 'cnn':
        # # 模型参数量: 25176
        # model_config = {
        #     'input_dims': (1, 10, 10),
        #     'hidden_sizes': [32, 64],
        #     'output_dims': 24,
        # }
        model_config = DefaultModelConfig(
            # Use a DreamerV3-style CNN stack for 64x64 images.
            conv_filters=[
                [32, 4, 2],  # 1st CNN layer: num_filters, kernel, stride(, padding)?
                [64, 4, 2],  # 2nd CNN layer
            ],
            # After the last CNN, the default model flattens, then adds an optional MLP.
        )

    # 验证配置
    eval_config = {
        'evaluation_interval': 30,
        'evaluation_duration': 4000,
        'evaluation_duration_unit': 'timesteps',
        'evaluation_sample_timeout_s': 24*60*60,
        'evaluation_force_reset_envs_before_iteration': True,
    }

    # 训练配置
    config = (
        ClientPPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            "snake",
            # 环境配置
            env_config=env_config
        )
        .env_runners(
            sample_timeout_s=24*60*60,
            num_env_runners=int(os.cpu_count() - num_learners) if not in_windows() else 0,# 设置成核心数减去gpu数, win下不使用
        )
        .debugging(log_level='DEBUG')
        .learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        .evaluation(**eval_config)
    )

    config = config.rl_module(
        model_config=model_config,
    )

    # 构建算法
    algo = config.build()
    log(algo.get_module())
    log(f'total params: {sum(p.numel() for p in algo.get_module().parameters())}')

    sys.exit()

    # 训练文件夹管理
    if not in_windows():
        train_folder_manager = TrainFolderManager(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(algo)

    # 训练循环
    begin_time = time.time()
    rounds = 50000
    # rounds = 30
    for i in range(rounds):
        log(f"Training iteration {i+1}/{rounds}")
        result = algo.train()

        # # 保存result
        # result_file = os.path.join(train_folder, f'result_{i}.pkl')
        # with open(result_file, 'wb') as f:
        #     pickle.dump(result, f)

        out_file = os.path.join(train_folder, f'out_{beijing_time().strftime("%Y%m%d")}.csv')
        simplify_rllib_metrics(result, out_func=log, out_file=out_file)

        if i>0 and (i % 10 == 0 or i == rounds - 1):
            if not in_windows():
                # 保存检查点
                checkpoint_dir = algo.save_to_path(train_folder_manager.checkpoint_folder)
                log(f"Checkpoint saved in directory {checkpoint_dir}")
            # 绘制训练曲线
            plot_training_curve(train_title, train_folder, out_file, time.time() - begin_time)
            if not in_windows():
                # 压缩并上传
                train_folder_manager.push()

    # 停止算法
    algo.stop()
    log(f"algo.stop done")