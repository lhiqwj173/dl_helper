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

from dl_helper.rl.custom_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.custom_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.custom_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics, stop
from dl_helper.rl.socket_base import request_need_val
from py_ext.tool import init_logger, log
from py_ext.datetime import beijing_time
from dl_helper.train_folder_manager import TrainFolderManager

from dl_helper.rl.custom_rllib_module.snake.mlp import MLPPPOCatalog
from dl_helper.rl.custom_rllib_module.snake.cnn import CNNPPOCatalog
from dl_helper.rl.rl_env.snake.snake_env import SnakeEnv
from dl_helper.rl.rl_env.tool import human_control, ai_control

use_intrinsic_curiosity = False
new_lr = 0.0
model_type = 'mlp'
for arg in sys.argv:
    if arg == 'ICM':
        use_intrinsic_curiosity = True
    elif arg == 'cnn':
        model_type = 'cnn'
    elif arg.startswith('lr='):
        new_lr = float(arg.split('=')[1])

train_folder = train_title = f'20250326_2_snake' + ("" if not use_intrinsic_curiosity else '_ICM') + f'_{model_type}'
init_logger(train_title, home=train_folder, timestamp=False)

# 吃到食物标准奖励
STD_REWARD = 100

# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_REWARD / 100

"""
激励函数

# 最大吃食物数量
MAX_EAT_FOOD_NUM = 10 *10 - 1

# 20250321 ####################################
# 吃到食物标准奖励
STD_EAT_FOOD_REWARD = 100
# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_EAT_FOOD_REWARD / 2
# 奖励函数
shaping = -(距离²/(10² + 10²)) * STD_MOVE_REWARD - (MAX_EAT_FOOD_NUM - 吃到食物数量) * STD_EAT_FOOD_REWARD
撞击惩罚 = -(MAX_EAT_FOOD_NUM + 1) * STD_EAT_FOOD_REWARD # 10 * 10 的网格, 最大惩罚: -10000

分析:
    每一步都有巨大的惩罚, 而撞击的惩罚不够大
    导致模型自杀来避免每步的累计惩罚


# 20250324 ####################################
# 吃到食物标准奖励
STD_EAT_FOOD_REWARD = 100
# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_EAT_FOOD_REWARD / 10
shaping = -(距离²/(10² + 10²)) * STD_MOVE_REWARD + STD_EAT_FOOD_REWARD * (距离 == 0)
撞击惩罚 = -(MAX_EAT_FOOD_NUM + 1) * STD_EAT_FOOD_REWARD # 10 * 10 的网格, 最大惩罚: -10000, 是游戏中可能的最大惩罚，持续移动的惩罚需要很久才能抵消吃到食物的奖励
模型应该会尽可能少的移动，来获取尽可能多的食物，同时避免撞击(自杀会获得最大的惩罚)


# 20250324 2 ####################################
# 吃到食物标准奖励
STD_REWARD = 100
# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_REWARD / 100
shaping = -(距离/(10 + 10)) * STD_MOVE_REWARD + STD_REWARD * (距离 == 0)
撞击惩罚 = -STD_REWARD # 避免惩罚过大，倾向于避免任何风险行为
模型应该会尽可能少的移动，来获取尽可能多的食物，同时避免撞击(自杀会获得最大的惩罚)

# 20250325 ####################################
# 吃到食物标准奖励
STD_REWARD = 100
# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_REWARD / 100
shaping = (1 - 距离/(10 + 10)) * STD_MOVE_REWARD + STD_REWARD * (距离 == 0)
撞击惩罚 = -STD_REWARD # 避免惩罚过大，倾向于避免任何风险行为
模型应该会尽可能靠近，并获取尽可能多的食物，同时避免撞击(自杀会获得最大的惩罚)
距离的距离变成正的，查看与负距离的区别


# 20250326 ####################################
# 吃到食物标准奖励
STD_REWARD = 100
# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_REWARD / 100
shaping = (1 - 距离/(10 + 10)) * STD_MOVE_REWARD + STD_REWARD * (距离 == 0)
撞击惩罚 = -STD_REWARD / 100 # 减小撞击惩罚，鼓励探索
模型应该会尽可能靠近，并获取尽可能多的食物，同时避免撞击(每一步都是正奖励, 不会自杀)

# 20250326 2 ####################################
# 吃到食物标准奖励与身体长度成正比
STD_REWARD = 100 * 身体长度
# 移动到实物的标准奖励
STD_MOVE_REWARD = STD_REWARD / 100
# 使用距离更远还是更近来给与奖励
shaping = (距离是否缩小) * STD_MOVE_REWARD - (距离是否增大) * STD_MOVE_REWARD + STD_REWARD * (距离 == 0) * (身体长度 - 1)
撞击惩罚/最大步数终止惩罚 = -STD_REWARD / 1000 # 减小撞击惩罚，鼓励探索
模型应该会尽可能靠近，并获取尽可能多的食物，同时避免撞击(每一步都是正奖励, 不会自杀)
"""

def stop_reward(snake, food, grid_size, shared_data):
    return -STD_REWARD / 1000

def keep_alive_reward(snake, food, grid_size, shared_data):
    # 计算当前曼哈顿距离
    distance = abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1])
    
    # 初始化前距离
    if 'last_distance' not in shared_data:
        shared_data['last_distance'] = distance

    # 计算距离变化
    distance_change = distance - shared_data['last_distance']
    shared_data['last_distance'] = distance

    # 转为 -1 / 0 / 1
    distance_change = 1 if distance_change > 0 else 0 if distance_change == 0 else -1

    return distance_change * STD_MOVE_REWARD + STD_REWARD * int(distance == 0) * (len(snake) - 1)

if __name__ == "__main__":

    env_config = {
        'grid_size': (10, 10),
        'crash_reward': stop_reward,
        'eat_reward': keep_alive_reward,
        'move_reward': keep_alive_reward,
        'truncated_reward': stop_reward,
        'model_type': model_type,
    }

    # # 人控制
    # human_control(
    #     env_class=SnakeEnv,
    #     env_config=env_config,
    # )
    # sys.exit()

    # 模型控制
    ai_control(SnakeEnv, env_config, checkpoint_abs_path=r"C:\Users\lh\Desktop\temp\checkpoint")
    sys.exit()

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes() if not in_windows() else 0
    log(f"num_learners: {num_learners}")

    if model_type == 'mlp':
        # total params: 34276
        model_config = {
            'input_dims': (10, 10),
            'hidden_sizes': [128, 128],
            'output_dims': 32,
        } 
    elif model_type == 'cnn':
        # total params: 34292
        model_config = {
            'input_dims': (1, 10, 10),
            'hidden_sizes': [8, 16], 
            'output_dims': 32,
        }

    # 验证配置
    eval_config = {
        'evaluation_interval': 100,
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

    catalog_class = MLPPPOCatalog if model_type == 'mlp' else CNNPPOCatalog
    if use_intrinsic_curiosity:
        config = config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    # The "main" RLModule (policy) to be trained by our algo.
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        catalog_class=catalog_class,
                        model_config=model_config,
                    ),
                    # The intrinsic curiosity model.
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=IntrinsicCuriosityModel,
                        # Only create the ICM on the Learner workers, NOT on the
                        # EnvRunners.
                        learner_only=True,
                        # Configure the architecture of the ICM here.
                        model_config={
                            "feature_dim": 64,
                        },
                    ),
                }
            ),
            # # Use a different learning rate for training the ICM.
            # algorithm_config_overrides_per_module={
            #     ICM_MODULE_ID: AlgorithmConfig.overrides(lr=0.0005)
            # },
        )

        config = config.training(
            lr=3e-4 if new_lr == 0.0 else new_lr,
            learner_config_dict={
                # Intrinsic reward coefficient.
                "intrinsic_reward_coeff": 0.05,
                # Forward loss weight (vs inverse dynamics loss). Total ICM loss is:
                # L(total ICM) = (
                #     `forward_loss_weight` * L(forward)
                #     + (1.0 - `forward_loss_weight`) * L(inverse_dyn)
                # )
                "forward_loss_weight": 0.2,
            }
        )
    else:
        config = config.rl_module(
            rl_module_spec=RLModuleSpec(catalog_class=catalog_class),# 使用自定义配置
            model_config=model_config,
        )

        config = config.training(
            lr=5e-5 if new_lr == 0.0 else new_lr,
        )

    # 构建算法
    algo = config.build()
    log(algo.get_module())
    log(f'total params: {sum(p.numel() for p in algo.get_module().parameters())}')
    # sys.exit()

    # 训练文件夹管理
    if not in_windows():
        train_folder_manager = TrainFolderManager(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(algo, only_params=new_lr != 0.0)# 修改了学习率，只需要加载模型参数

    # 训练循环
    begin_time = time.time()
    rounds = 500000000
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

        if i>0 and (i % 100 == 0 or i == rounds - 1):
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