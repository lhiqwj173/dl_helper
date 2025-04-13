import sys, os, time, shutil, pickle
# os.environ["RAY_DEDUP_LOGS"] = "0"
import matplotlib.pyplot as plt
from ray.tune.registry import get_trainable_cls, register_env
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

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import ICM_MODULE_ID
from ray.rllib.examples.rl_modules.classes.intrinsic_curiosity_model_rlm import (
    IntrinsicCuriosityModel,
)

train_folder = train_title = f'20250130_cartpole'
init_logger(train_title, home=train_folder, timestamp=False)

if __name__ == "__main__":
    run_type = 'self'

    # 获取参数
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "test":
                run_type = 'test'

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes()
    log(f"num_learners: {num_learners}")

    # 训练配置
    config = (
        ClientPPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("CartPole-v1")
        .env_runners(num_env_runners=1)# 4核cpu，暂时选择1个环境运行器
    )

    # 单机运行
    config = config.learners(    
        num_learners=num_learners,
        num_gpus_per_learner=1,
    )
    config = config.evaluation(
        evaluation_interval=5,
        evaluation_duration=3,
    )

    if run_type == 'test':
        # 添加 好奇心模块
        config = config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    # The "main" RLModule (policy) to be trained by our algo.
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        **(
                            {"model_config": {"vf_share_layers": True}}
                        ),
                    ),
                    # The intrinsic curiosity model.
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=IntrinsicCuriosityModel,
                        # Only create the ICM on the Learner workers, NOT on the
                        # EnvRunners.
                        learner_only=True,
                        # Configure the architecture of the ICM here.
                        model_config={
                            "feature_dim": 4,
                            "feature_net_hiddens": (8, 8),
                            "feature_net_activation": "relu",
                            "inverse_net_hiddens": (8, 8),
                            "inverse_net_activation": "relu",
                            "forward_net_hiddens": (8, 8),
                            "forward_net_activation": "relu",
                        },
                    ),
                }
            ),
            # # Use a different learning rate for training the ICM.
            # algorithm_config_overrides_per_module={
            #     ICM_MODULE_ID: AlgorithmConfig.overrides(lr=0.0005)
            # },
        )


    # 构建算法
    algo = config.build()
    # print(algo.learner_group._learner.module._rl_modules['default_policy'])

    # # 训练文件夹管理
    # train_folder_manager = TrainFolderManager(train_folder)
    # if train_folder_manager.exists():
    #     log(f"restore from {train_folder_manager.checkpoint_folder}")
    #     train_folder_manager.load_checkpoint(algo, only_params=True)

    begin_time = time.time()
    # 训练循环
    rounds = 100
    out_file = os.path.join(train_folder, f'out_{beijing_time().strftime("%Y%m%d")}.csv')
    for i in range(rounds):
        log(f"\nTraining iteration {i+1}/{rounds}")
        result = algo.train()
        simplify_rllib_metrics(result, out_func=log, out_file=out_file)

    # # 保存检查点
    # checkpoint_dir = algo.save_to_path(train_folder_manager.checkpoint_folder)
    # log(f"Checkpoint saved in directory {checkpoint_dir}")
    # 绘制训练曲线
    log(f"plot_training_curve done")
    plot_training_curve(train_title, train_folder, out_file, time.time() - begin_time, y_axis_max=800)
    # # 压缩并上传
    # train_folder_manager.push()
    
    # 停止算法
    algo.stop()
    log(f"algo.stop done")
