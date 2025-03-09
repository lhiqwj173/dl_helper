import sys, os, time
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.costum_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.costum_rllib_module.lob.lob_model import LobCallbacks, LobPlotter
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics
from dl_helper.rl.socket_base import request_need_val
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

def remove_old_env_output_files(save_folder):
    for folder in os.listdir(save_folder):
        folder = os.path.join(save_folder, folder)
        if os.path.isdir(folder):
            keep_only_latest_files(folder)

def run(
        train_folder, 
        train_title,
        catalog_class,
        model_config,
        env_config={
            'data_type': 'train',# 训练/测试
            'his_len': 10,# 每个样本的 历史数据长度
            'file_num': 10,# 数据生产器 限制最多使用的文件（日期）数量
            'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
            'use_symbols': ['513050'],
        }
    ):
    run_type = 'self'

    # 获取参数
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "server":
                run_type = 'server'
            elif arg == "client":
                run_type = 'client'
            elif arg == "test":
                run_type = 'test'

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes()
    log(f"num_learners: {num_learners}")

    # 训练配置
    env_config['train_folder'] = train_folder
    env_config['train_title'] = train_title
    config = (
        ClientPPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            "lob",
            # 环境配置
            env_config=env_config
        )# 直接使用
        .env_runners(
            sample_timeout_s=24*60*60,
            num_env_runners=int(os.cpu_count() - num_learners),# 设置成核心数减去gpu数
        )
        # 自定义模型
        .rl_module(
            rl_module_spec=RLModuleSpec(catalog_class=catalog_class),# 使用自定义配置
            model_config=model_config,
        )
        .callbacks(LobCallbacks)
    )

    if run_type == 'server':
        config = config.extra_config(
            learner_group_class=ClientLearnerGroup,
            learner_group_kwargs={
                'train_folder': train_folder,
                "train_title": train_title,
            },
        )

        # dump config
        add_train_title_item(train_title, config)

    elif run_type == 'client':
        # 客户端配置
        config = config.training(
            learner_class=ClientPPOTorchLearner,# 分布式客户端 学习者
        )

        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )

        config = config.extra_config(
            learner_group_class=ClientLearnerGroup,
            learner_group_kwargs={
                'train_folder': train_folder,
                "train_title": train_title,
            },
        )

        # 询问服务器，本机是否需要验证环节
        need_val = request_need_val(train_title)
        log(f"need_val: {need_val}")
        if need_val:
            config = config.evaluation(
                evaluation_interval=15,
                evaluation_duration=3,
            )

        # 客户端运行
        # 构建算法
        algo = config.build()

        begin_time = time.time()
        # 训练循环
        # rounds = 2000
        # rounds = 100
        rounds = 30
        for i in range(rounds):
            log(f"\nTraining iteration {i+1}/{rounds}")
            result = algo.train()
            simplify_rllib_metrics(result, out_func=log)

            # 删除旧的文件
            remove_old_env_output_files(os.path.join(train_folder, 'env_output'))
        
        # 停止学习者额外的事件进程
        algo.learner_group.stop_extra_process()
        log(f"learner_group.stop_extra_process done")

        if need_val:
            # 绘制训练曲线
            plot_training_curve(train_folder, time.time() - begin_time, y_axis_max=30)

        # 停止算法
        algo.stop()
        log(f"algo.stop done")

    elif run_type == 'test':
        # 单机测试
        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        config = config.evaluation(
            evaluation_interval=15,
            evaluation_duration=3,
        )

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
            remove_old_env_output_files(os.path.join(train_folder, 'env_output'))

        # 绘制训练曲线
        plot_training_curve(train_folder, out_file, time.time() - begin_time, custom_plotter=LobPlotter())
        # 停止算法
        algo.stop()
        log(f"algo.stop done")

    else:
        # 单机运行
        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        config = config.evaluation(
            evaluation_interval=15,
            evaluation_duration=3,
        )

        # 构建算法
        algo = config.build()
        # print(algo.learner_group._learner.module._rl_modules['default_policy'])

        # 训练文件夹管理
        train_folder_manager = TrainFolderManager(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            algo.restore_from_path(train_folder_manager.checkpoint_folder)

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
            remove_old_env_output_files(os.path.join(train_folder, 'env_output'))

            if i>0 and (i % 10 == 0 or i == rounds - 1):
                # 保存检查点
                checkpoint_dir = algo.save_to_path(train_folder_manager.checkpoint_folder)
                log(f"Checkpoint saved in directory {checkpoint_dir}")
                # 绘制训练曲线
                plot_training_curve(train_folder, out_file, time.time() - begin_time, custom_plotter=LobPlotter())
                # 压缩并上传
                train_folder_manager.push()

        # 停止算法
        algo.stop()
        log(f"algo.stop done")

