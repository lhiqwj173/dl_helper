"""
配对交易强化学习

最终目标 2 : 超额净值作为 shaping, reward = shaping_t - shaping_t-1
    设定0.0003为最大超额收益
    shaping = max(min((策略净值 - 基准净值) / 0.0003, 1), -1) - 1, shaping范围: [-2, 0]
    中间奖励1 = shaping_t
        1. 偏移-1, 所有的中间奖励都为负数
            + 鼓励探索
            + 不贪恋中间奖励, 鼓励尽快达成目标
        2. 偏移一个很小的负值, 超额收益=0也为负
            + 超额收益较小/为0/负值时 鼓励探索
            + 鼓励尽可能获取较大的超额收益
        3. 无偏移，超额收益=0也无惩罚
            - 超额收益=0时, 中性
    中间奖励2 = shaping_t - shaping_t-1
    最终奖励 = max(min((策略净值 - 基准净值) / 0.0003, 1), -1), 范围: [-1, 1]

MAX_EXCESS_RETURN = 0.0003
shaping = max(min((acc_return - bm_return) / MAX_EXCESS_RETURN, 1), -1)
    
中间奖励 = shaping_t - MAX_EXCESS_RETURN*0.01 #偏移一个很小的负值
最终奖励 = shaping_t, 范围: [-1, 1]
"""
import sys, os, time, shutil, pickle
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

from dl_helper.tool import remove_old_env_output_files
from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.costum_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics, stop
from dl_helper.rl.socket_base import request_need_val
from py_ext.tool import init_logger, log
from py_ext.datetime import beijing_time
from dl_helper.train_folder_manager import TrainFolderManager

from dl_helper.rl.rl_env.match_trade.match_env_reward import EndPositionRewardStrategy, ClosePositionRewardStrategy, HoldPositionRewardStrategy, BalanceRewardStrategy, RewardStrategy
from dl_helper.rl.costum_rllib_module.match_model import MatchCallbacks, MatchPlotter
from dl_helper.rl.rl_env.match_trade.match_env import MATCH_trade_env

use_intrinsic_curiosity = False
if len(sys.argv) > 1 and sys.argv[1] == 'ICM':
    use_intrinsic_curiosity = True

train_folder = train_title = f'20250321_match_trade' + ("" if not use_intrinsic_curiosity else '_ICM')
init_logger(train_title, home=train_folder, timestamp=False)

class ExcessReturnRewardStrategy(RewardStrategy):
    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        """根据超额回报计算奖励"""
        MAX_EXCESS_RETURN = 0.0003
        excess_return = res['excess_return']
        shaping = max(min(excess_return / MAX_EXCESS_RETURN, 1), -1) - MAX_EXCESS_RETURN*0.01 #偏移一个很小的负值
        return shaping, False

class EndExcessReturnRewardStrategy(RewardStrategy):
    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        """根据超额回报计算奖励"""
        MAX_EXCESS_RETURN = 0.0003
        excess_return = res['excess_return']
        shaping = max(min(excess_return / MAX_EXCESS_RETURN, 1), -1) #无偏移
        return shaping, False
    
if __name__ == "__main__":
    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes() if not in_windows() else 0
    log(f"num_learners: {num_learners}")

    env_config = {
        # 用于实例化 数据生产器
        'data_type': 'train',# 训练/测试
        'his_daily_len': 5,# 每个样本的 历史日的价差长度
        'his_tick_len': 10,# 每个样本的 历史tick的价差长度

        # 终止游戏的超额亏损阈值
        'loss_threshold': -0.005,# 最大超额亏损阈值

        # 奖励策略
        'reward_strategy_class_dict': {
            'end_position': EndExcessReturnRewardStrategy,
            'close_position': ExcessReturnRewardStrategy,
            'hold_position': ExcessReturnRewardStrategy,
            'balance': ExcessReturnRewardStrategy
        },

        'train_folder': train_folder,
        'train_title': train_title,
    }

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
            "match",
            # 环境配置
            env_config=env_config
        )
        .env_runners(
            sample_timeout_s=24*60*60,
            num_env_runners=int(os.cpu_count() - num_learners) if not in_windows() else 0,# 设置成核心数减去gpu数, win下不使用
        )
        .callbacks(MatchCallbacks)
        .debugging(log_level='DEBUG')
        .learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        .evaluation(**eval_config)
    )

    if use_intrinsic_curiosity:
        config = config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    # The "main" RLModule (policy) to be trained by our algo.
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        **({}),
                    ),
                    # The intrinsic curiosity model.
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=IntrinsicCuriosityModel,
                        # Only create the ICM on the Learner workers, NOT on the
                        # EnvRunners.
                        learner_only=True,
                        # Configure the architecture of the ICM here.
                        model_config={
                            "feature_dim": 12,
                            "feature_net_hiddens": (24, 24),
                            "feature_net_activation": "relu",
                            "inverse_net_hiddens": (24, 24),
                            "inverse_net_activation": "relu",
                            "forward_net_hiddens": (24, 24),
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

        config = config.training(
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

    # 构建算法
    algo = config.build()
    # print(algo.learner_group._learner.module._rl_modules['default_policy'])

    # 训练文件夹管理
    if not in_windows():
        train_folder_manager = TrainFolderManager(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(algo)

    # 训练循环
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

        if i>0 and (i % 10 == 0 or i == rounds - 1):
            if not in_windows():
                # 保存检查点
                checkpoint_dir = algo.save_to_path(train_folder_manager.checkpoint_folder)
                log(f"Checkpoint saved in directory {checkpoint_dir}")
            # 绘制训练曲线
            plot_training_curve(train_title, train_folder, out_file, time.time() - begin_time, custom_plotter=MatchPlotter())
            if not in_windows():
                # 压缩并上传
                train_folder_manager.push()
            # 迭代完成
            MATCH_trade_env.iteration_done()

    # 停止算法
    algo.stop()
    log(f"algo.stop done")