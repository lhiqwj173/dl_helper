import sys, os, time
# os.environ["RAY_DEDUP_LOGS"] = "0"
import matplotlib.pyplot as plt
from ray.tune.registry import get_trainable_cls, register_env
from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.costum_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics
from dl_helper.rl.socket_base import request_need_val
from py_ext.tool import init_logger, log
from py_ext.wechat import wx
from py_ext.alist import alist

train_folder = 'breakout'
init_logger('20250108_breakout', home=train_folder, timestamp=False)

if __name__ == "__main__":
    is_server = False
    # 获取参数
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "server":
                is_server = True

    train_title = f'20250110_breakout'

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes()

    # 训练配置
    config = (
        ClientPPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("breakout")# 直接使用
        .env_runners(num_env_runners=1)# 4核cpu，暂时选择1个环境运行器
        .rl_module(
            model_config={
                "conv_filters": [
                    [32, [8, 8], 4],  # [输出通道数, [kernel_size_h, kernel_size_w], stride]
                    [64, [4, 4], 2],  # [64个通道, 4x4卷积核, stride=2]
                    [64, [3, 3], 1],  # [64个通道, 3x3卷积核, stride=1] 
                ],
            },
        )
        .extra_config(
            learner_group_class=ClientLearnerGroup,
            learner_group_kwargs={
                'train_folder': train_folder,
                "train_title": train_title,
            },
        )
    )

    if is_server:
        # dump config
        add_train_title_item(train_title, config)

    else:
        # 客户端配置
        config = config.training(
            learner_class=ClientPPOTorchLearner,# 分布式客户端 学习者
        )

        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
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