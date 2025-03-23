import sys, os, time

from dl_helper.tool import in_windows

from ray.rllib.utils.from_config import NotProvided, from_config

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import ICM_MODULE_ID
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.costum_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.costum_rllib_module.lob.lob_model import LobCallbacks, LobPlotter
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics
from dl_helper.rl.socket_base import request_need_val
from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.train_folder_manager import TrainFolderManager

from py_ext.tool import log
from py_ext.datetime import beijing_time

def keep_only_latest_files(folder, num=50):
    """
    只保留文件夹中的最新修改的50个文件
    """
    files = os.listdir(folder)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    for file in files[:-num]:
        os.remove(os.path.join(folder, file))

def remove_old_env_output_files(save_folder, num=50):
    for folder in os.listdir(save_folder):
        folder = os.path.join(save_folder, folder)
        if os.path.isdir(folder):
            keep_only_latest_files(folder, num)

def run(
        train_folder, 
        train_title,
        catalog_class,
        model_config,
        env_config = {},
        intrinsic_curiosity_model_class = None,
        intrinsic_curiosity_model_config = {},
    ):
    run_type = 'self'
    new_lr = 0

    # 获取参数
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "test":
                run_type = 'test'
            elif arg.startswith('lr='):
                new_lr = float(arg.split('=')[1])

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes() if not in_windows() else 0
    log(f"num_learners: {num_learners}")

    # 训练配置
    defualt_env_config={
        'data_type': 'train',# 训练/测试
        'his_len': 10,# 每个样本的 历史数据长度
        'file_num': 10,# 数据生产器 限制最多使用的文件（日期）数量
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        'use_symbols': ['513050'],
    }
    for k, v in defualt_env_config.items():
        if k not in env_config:
            env_config[k] = v
    env_config['train_folder'] = train_folder
    env_config['train_title'] = train_title
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            "lob",
            # 环境配置
            env_config=env_config
        )
        .env_runners(
            sample_timeout_s=24*60*60,
            num_env_runners=int(os.cpu_count() - num_learners) if not in_windows() else 0,# 设置成核心数减去gpu数, win下不使用
        )
        .callbacks(LobCallbacks)
        .debugging(log_level='DEBUG')
        .training(
            lr=new_lr if new_lr > 0 else NotProvided,
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
    )

    if None is intrinsic_curiosity_model_class:
        config = config.rl_module(
            rl_module_spec=RLModuleSpec(catalog_class=catalog_class),# 使用自定义配置
            model_config=model_config,
        )
    else:
        # 添加 好奇心模块
        config = config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        catalog_class=catalog_class,
                        model_config=model_config,
                    ),
                    # The intrinsic curiosity model.
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=intrinsic_curiosity_model_class,
                        # Only create the ICM on the Learner workers, NOT on the
                        # EnvRunners.
                        learner_only=True,
                        # Configure the architecture of the ICM here.
                        model_config=intrinsic_curiosity_model_config,
                    ),
                }

            ),
            # # Use a different learning rate for training the ICM.
            # algorithm_config_overrides_per_module={
            #     ICM_MODULE_ID: AlgorithmConfig.overrides(lr=0.0005)
            # },
        )

    # 验证配置
    eval_config = {
        'evaluation_interval': 30,
        'evaluation_duration': 4000,
        'evaluation_duration_unit': 'timesteps',
        'evaluation_sample_timeout_s': 24*60*60,
        'evaluation_force_reset_envs_before_iteration': True,
    }

    if run_type == 'test':
        # 单机测试
        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        config = config.evaluation(**eval_config)

        # 构建算法
        algo = config.build()

        begin_time = time.time()
        rounds = 30
        for i in range(rounds):
            log(f"Training iteration {i+1}/{rounds}")
            result = algo.train()

            out_file = os.path.join(train_folder, f'out_{beijing_time().strftime("%Y%m%d")}.csv')
            simplify_rllib_metrics(result, out_func=log, out_file=out_file)

            # 删除旧的文件
            remove_old_env_output_files(os.path.join(train_folder, 'env_output'), num=5)

        # 绘制训练曲线
        plot_training_curve(train_title, train_folder, out_file, time.time() - begin_time, custom_plotter=LobPlotter(), y_axis_max=10000)
        # 停止算法
        algo.stop()
        log(f"algo.stop done")

    else:
        # 单机运行
        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        # # FOR DEBUG
        # eval_config['evaluation_interval'] = 1
        config = config.evaluation(**eval_config)

        # 构建算法
        algo = config.build()
        # print(algo.learner_group._learner.module._rl_modules['default_policy'])

        # 训练文件夹管理
        if not in_windows():
            train_folder_manager = TrainFolderManager(train_folder)
            if train_folder_manager.exists():
                log(f"restore from {train_folder_manager.checkpoint_folder}")
                train_folder_manager.load_checkpoint(algo, only_params=new_lr>0)

        begin_time = time.time()
        rounds = 5000
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

            # 删除旧的文件
            remove_old_env_output_files(os.path.join(train_folder, 'env_output'), num=5)

            # # FOR DEBUG
            # break

            if i>0 and (i % 10 == 0 or i == rounds - 1):
                if not in_windows():
                    # 保存检查点
                    checkpoint_dir = algo.save_to_path(train_folder_manager.checkpoint_folder)
                    log(f"Checkpoint saved in directory {checkpoint_dir}")
                # 绘制训练曲线
                plot_training_curve(train_title, train_folder, out_file, time.time() - begin_time, custom_plotter=LobPlotter())
                if not in_windows():
                    # 压缩并上传
                    train_folder_manager.push()
                # 迭代完成
                LOB_trade_env.iteration_done()

        # 停止算法
        algo.stop()
        log(f"algo.stop done")

