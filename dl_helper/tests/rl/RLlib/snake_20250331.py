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

from dl_helper.rl.custom_rllib_module.ppoconfig import ClientPPOConfig, PPOConfig
from dl_helper.rl.custom_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.custom_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes, get_gpu_info
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics, stop
from dl_helper.rl.socket_base import request_need_val
from py_ext.tool import init_logger, log
from py_ext.datetime import beijing_time
from py_ext.lzma import decompress, compress_folder
from dl_helper.train_folder_manager import TrainFolderManager

from dl_helper.rl.custom_rllib_module.snake.mlp import MLPPPOCatalog
from dl_helper.rl.custom_rllib_module.snake.cnn import CNNPPOCatalog
from dl_helper.rl.rl_env.snake.snake_env_simple import SnakeEnv
from dl_helper.rl.rl_env.tool import human_control, ai_control

use_intrinsic_curiosity = False
use_alist = False
new_lr = 0.0
model_type = 'mlp'
for arg in sys.argv:
    if arg == 'ICM':
        use_intrinsic_curiosity = True
    elif arg.startswith('lr='):
        new_lr = float(arg.split('=')[1])

train_folder = train_title = f'20250331_snake' + ("" if not use_intrinsic_curiosity else '_ICM') + f'_{model_type}'
init_logger(train_title, home=train_folder, timestamp=False)

# 吃到食物标准奖励
STD_REWARD = 100

"""
激励函数

# 20250329 ####################################
# 吃到食物标准奖励
# 使用距离更远还是更近来给与奖励
# 更近 1 分，更远 -1 分，吃到食物 100 分
"""

def stop_reward(snake, food, grid_size, shared_data):
    return 0

def keep_alive_reward(snake, food, grid_size, shared_data):
    # 计算当前曼哈顿距离
    distance = abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1])

    if 'prev_distance' not in shared_data:
        shared_data['prev_distance'] = distance

    better = distance < shared_data['prev_distance']
    worse = distance > shared_data['prev_distance']
    shared_data['prev_distance'] = distance

    return (1 if better else -1 if worse else 0) + STD_REWARD * int(distance == 0)

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
    # import pygame
    # human_control(
    #     env_class=SnakeEnv,
    #     env_config=env_config,
    #     control_map={
    #         pygame.K_UP: 0,
    #         pygame.K_LEFT: 1,
    #         pygame.K_RIGHT: 2,
    #     },
    #     default_action=0,
    # )
    # sys.exit()

    # 模型控制
    ai_control(
        SnakeEnv, 
        env_config, 
        checkpoint_abs_path=r"C:\Users\lh\Desktop\temp\iter_299",
    ) 
    sys.exit()

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes() if not in_windows() else 0
    log(f"num_learners: {num_learners}")

    if model_type == 'mlp':
        # total params: 9347
        model_config = {
            'input_dims': (4,),
            'hidden_sizes': [64],
            'need_layer_norm': False,
            'active_func': 'tanh',
            'output_dims': 64,
        } 
    elif model_type == 'cnn':
        # total params: 8987
        model_config = {
            'input_dims': (1, 10, 10),
            'hidden_sizes': [6], 
            'need_layer_norm': False,
            'active_func': 'relu',
            'output_dims': 8,
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
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            "snake_simple",
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
            num_gpus_per_learner=1 if get_gpu_info() != 'CPU' else 0,
        )
        .evaluation(**eval_config)
    )

    training_config = {
        ##################
        # 通用参数
        ##################
        'lr': 3e-4 if new_lr == 0.0 else new_lr,
        'gamma': 0.99,
        'train_batch_size_per_learner': 2048,
        'minibatch_size': 64,
        'num_epochs': 10,

        ##################
        # ppo 参数
        ##################
        'use_critic': True,
        'use_gae': True,
        'lambda_': 0.95,

        # SB3中没有使用 > 关闭
        'use_kl_loss': False,
        'kl_coeff': 0.2,
        'kl_target': 0.01,

        'vf_loss_coeff': 0.5,
        'entropy_coeff': 0.0,
        'clip_param':0.2,
        'vf_clip_param':float("inf"),
        'grad_clip':0.5,
    }

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

        training_config['learner_config_dict'] = {
            # Intrinsic reward coefficient.
            "intrinsic_reward_coeff": 0.05,
            # Forward loss weight (vs inverse dynamics loss). Total ICM loss is:
            # L(total ICM) = (
            #     `forward_loss_weight` * L(forward)
            #     + (1.0 - `forward_loss_weight`) * L(inverse_dyn)
            # )
            "forward_loss_weight": 0.2,
        }
    else:
        config = config.rl_module(
            rl_module_spec=RLModuleSpec(catalog_class=catalog_class),# 使用自定义配置
            model_config=model_config,
        )

    config = config.training(
        **training_config
    )

    # 构建算法
    algo = config.build()
    if in_windows():
        params = algo.learner_group._learner._module._rl_modules['default_policy']
        log(params)
        log(f'total params: {sum(p.numel() for p in params.parameters())}')
        # sys.exit()

    # 训练文件夹管理
    if not in_windows() and use_alist:
        train_folder_manager = TrainFolderManager(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(algo, only_params=new_lr != 0.0)# 修改了学习率，只需要加载模型参数

    # 训练循环
    begin_time = time.time()
    rounds = 500000000
    rounds = 100
    rounds = 300
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
            if not in_windows() and use_alist:
                # 保存检查点
                checkpoint_dir = algo.save_to_path(os.path.join(train_folder_manager.checkpoint_folder, f'iter_{i}'))
                log(f"Checkpoint saved in directory {checkpoint_dir}")
            else:
                # 保存检查点
                checkpoint_dir = algo.save_to_path(os.path.join(os.path.abspath(train_folder), f'iter_{i}'))
                log(f"Checkpoint saved in directory {checkpoint_dir}")
                zip_file = f'{train_title}.7z'
                if os.path.exists(zip_file):
                    os.remove(zip_file)
                compress_folder(train_folder, zip_file, 9, inplace=False)
                log('compress_folder done')

            # 绘制训练曲线
            plot_training_curve(train_title, train_folder, out_file, time.time() - begin_time)
            if not in_windows() and use_alist:
                # 压缩并上传
                train_folder_manager.push()

    # 停止算法
    algo.stop()
    log(f"algo.stop done")