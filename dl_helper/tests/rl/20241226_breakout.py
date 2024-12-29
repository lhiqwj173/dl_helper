import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pstats
from datetime import datetime

from py_ext.tool import log

from dl_helper.rl.dqn.c51 import C51 
from dl_helper.rl.dqn.dqn import DQN
from dl_helper.rl.rl_env.breakout_env import cnn_breakout
from dl_helper.rl.rl_utils import ReplayBuffer, PrioritizedReplayBuffer, LRTrainParams, Profiler, init_logger_by_ip
from dl_helper.rl.run import run_client_learning_device_breakout
from dl_helper.train_param import in_kaggle

if __name__ == '__main__':
    # 初始化参数
    params = LRTrainParams(
        train_title='20241226_breakout',
        run_client_learning_func=run_client_learning_device_breakout,

        ####################
        # 算法
        agent_class = C51,
        # agent_class = DQN,# 普通DQN算法
        ####################

        need_reshape=None,
        features_dim=cnn_breakout.get_feature_size(), 
        action_dim=4,
        net_arch=[512, 4],

        ####################
        # 回放池
        train_buffer_class=PrioritizedReplayBuffer,
        # train_buffer_class=ReplayBuffer,
        ####################

        features_extractor_class=cnn_breakout,
        features_extractor_kwargs={},
    )

    params.local = True
    params.num_episodes = float('inf')
    params.buffer_size = 1000

    # 使用命令行 更新参数
    if len(sys.argv) > 1:
        params.update_from_args(sys.argv[1:])

    # 初始化日志
    init_logger_by_ip(params.train_title)

    # 性能分析
    p = Profiler(params)
    p.before_train()

    # 运行
    params.run()

    # 性能分析输出，若有
    p.after_train()
