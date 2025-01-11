import sys, os
from ray.tune.registry import get_trainable_cls, register_env
from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.costum_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item
from py_ext.tool import init_logger, log

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
        .env_runners(num_env_runners=2)# 4核cpu，暂时选择2个环境运行器
        # 暂时不支持，计划将eval放在服务端
        .evaluation(
            evaluation_interval=2,
            evaluation_duration=2,
        )
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
                "train_title": train_title,
            },
        )
        .learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        # for debug
        # .learners(    
        #     num_learners=2,
        #     num_gpus_per_learner=0,
        #     num_cpus_per_learner=0.3,
        # )
    )

    if is_server:
        # dump config
        add_train_title_item(train_title, config)

    else:
        # 客户端配置
        config = config.training(
            learner_class=ClientPPOTorchLearner,# 分布式客户端 学习者
        )

        # 客户端运行
        # 构建算法
        algo = config.build()

        # 创建检查点保存目录
        checkpoint_base_dir = os.path.join(train_folder, 'checkpoints')
        os.makedirs(checkpoint_base_dir, exist_ok=True)

        # 训练循环
        rounds = 2000
        for i in range(rounds):
            log(f"\nTraining iteration {i+1}/{rounds}")
            result = algo.train()
            simplify_rllib_metrics(result, out_func=log)
            
            if (i + 1) % 10 == 0:
                checkpoint_dir = algo.save_to_path(
                    os.path.join(os.path.abspath(checkpoint_base_dir), f"checkpoint_{i+1}")
                )
                log(f"Checkpoint saved in directory {checkpoint_dir}")

        # 保存最终模型
        final_checkpoint_dir = algo.save_to_path(
            os.path.join(os.path.abspath(checkpoint_base_dir), "final_model")
        )
        log(f"Final model saved in directory {final_checkpoint_dir}")