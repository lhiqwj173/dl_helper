import sys, os
import matplotlib.pyplot as plt
from ray.tune.registry import get_trainable_cls, register_env
from dl_helper.rl.rl_env.breakout_env import BreakoutEnv# 自定义环境
from dl_helper.rl.easy_helper import *
from dl_helper.rl.rl_utils import plot_training_curve
from dl_helper.train_param import match_num_processes
from py_ext.tool import init_logger, log, get_log_file

train_folder = 'breakout'
init_logger('20250108_breakout', home='breakout', timestamp=False)

if __name__ == "__main__":
    # algo = "Rainbow_DQN"
    algo = "PPO"
    if len(sys.argv) > 1:
        algo = sys.argv[1]

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes()

    # 注册环境
    register_env("breakout", lambda config: BreakoutEnv())

    # 实例化简单算法配置类
    simple_algo = globals()[algo]()

    # 训练配置
    config = (
        get_trainable_cls(simple_algo.algo)
        .get_default_config()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("breakout")# 
        .env_runners(num_env_runners=1)# 4核cpu，暂时选择1个环境运行器
        .evaluation(
            evaluation_interval=30,
            evaluation_duration=5,
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
        .training(**simple_algo.training_kwargs)
        .learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
            
            # # for debug
            # num_learners=2,
            # num_gpus_per_learner=0,
            # num_cpus_per_learner=0.3,
        )
    )

    # 构建算法
    algo = config.build()

    # 创建检查点保存目录
    checkpoint_base_dir = os.path.join(train_folder, 'checkpoints')
    os.makedirs(checkpoint_base_dir, exist_ok=True)

    # 训练循环
    # rounds = 2000
    rounds = 100
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

    # 绘制训练曲线
    plot_training_curve(train_folder)
    