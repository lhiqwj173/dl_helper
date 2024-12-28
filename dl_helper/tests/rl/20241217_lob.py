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
from dl_helper.models.binctabl import m_bin_ctabl_fix_shape
from dl_helper.rl.rl_utils import ReplayBufferWaitClose, PrioritizedReplayBufferWaitClose, PrioritizedReplayBuffer, LRTrainParams, Profiler, init_logger_by_ip
from dl_helper.rl.run import run_client_learning_device_lob
from dl_helper.train_param import in_kaggle

action_dim=3
t1, t2, t3, t4 = [100, 30, 10, 1]
d1, d2, d3, d4 = [130, 60, 30, 7]
features_extractor_kwargs = {'d2': d2, 'd1': d1, 't1': t1, 't2': t2, 'd3': d3, 't3': t3, 'd4': d4, 't4': t4}

if __name__ == '__main__':
    # 初始化参数
    params = LRTrainParams(
        train_title='20241217_lob',
        run_client_learning_func=run_client_learning_device_lob,

        ####################
        # 算法
        agent_class = C51,
        # agent_class = DQN,# 普通DQN算法
        ####################

        need_reshape=(100, 130),
        features_dim=d4+4,
        action_dim=action_dim,
        net_arch=[6, action_dim],

        ####################
        # 回放池
        train_buffer_class=PrioritizedReplayBufferWaitClose,# PER 回放池
        # train_buffer_class=ReplayBufferWaitClose, # 普通回放池
        ####################

        features_extractor_class=m_bin_ctabl_fix_shape,
        features_extractor_kwargs=features_extractor_kwargs,
    )

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
