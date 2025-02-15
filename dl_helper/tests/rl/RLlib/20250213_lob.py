import sys, os, time
# os.environ["RAY_DEDUP_LOGS"] = "0"
import matplotlib.pyplot as plt
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.costum_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.costum_rllib_module.lob_model import lob_PPOCatalog, LobCallbacks
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics
from dl_helper.rl.socket_base import request_need_val
from py_ext.tool import init_logger, log

from dl_helper.train_folder_manager import TrainFolderManager

train_folder = 'lob'
init_logger('20250213_lob', home=train_folder, timestamp=False)

if __name__ == "__main__":
    run_type = 'self'

    # 获取参数
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "server":
                run_type = 'server'
            elif arg == "client":
                run_type = 'client'

    train_title = f'20250213_lob'

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes()

    # 训练配置
    config = (
        ClientPPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            "lob",
            # 环境配置
            env_config={
                'data_type': 'train',# 训练/测试
                'his_len': 10,# 每个样本的 历史数据长度
                'file_num': 5,# 数据生产器 限制最多使用的文件（日期）数量
                'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
            }
        )# 直接使用
        .env_runners(num_env_runners=1)# 4核cpu，暂时选择1个环境运行器
        # 自定义模型
        .rl_module(
            rl_module_spec=RLModuleSpec(catalog_class=lob_PPOCatalog),# 使用自定义配置
            model_config={
                # 自定义编码器参数
                'input_dims' : (20, 10),
                'extra_input_dims' : 4,
                'ds' : (20, 40, 40, 3),
                'ts' : (10, 6, 3, 1),
            },
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
                evaluation_interval=10,
                evaluation_duration=3,
            )

        # 客户端运行
        # 构建算法
        algo = config.build()

        begin_time = time.time()
        # 训练循环
        # rounds = 2000
        rounds = 100
        # rounds = 3
        for i in range(rounds):
            log(f"\nTraining iteration {i+1}/{rounds}")
            result = algo.train()
            simplify_rllib_metrics(result, out_func=log)
        
        # 停止学习者额外的事件进程
        algo.learner_group.stop_extra_process()
        log(f"learner_group.stop_extra_process done")

        if need_val:
            # 绘制训练曲线
            plot_training_curve(train_folder, time.time() - begin_time, y_axis_max=30)

    else:
        # 单机运行
        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        config = config.evaluation(
            evaluation_interval=5,
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
        # 训练循环 TODO 拉取参数/同步参数/同步训练记录/日志
        rounds = 2000
        # rounds = 100
        # rounds = 30
        for i in range(rounds):
            log(f"\nTraining iteration {i+1}/{rounds}")
            result = algo.train()
            simplify_rllib_metrics(result, out_func=log)

            if i % 10 == 0 or i == rounds - 1:
                # 保存检查点
                checkpoint_dir = algo.save_to_path(train_folder_manager.checkpoint_folder)
                log(f"Checkpoint saved in directory {checkpoint_dir}")
                # 绘制训练曲线
                plot_training_curve(train_folder, time.time() - begin_time)
                # 压缩并上传
                train_folder_manager.push()

