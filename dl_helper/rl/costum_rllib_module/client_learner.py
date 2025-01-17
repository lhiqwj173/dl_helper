import ray
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.algorithms.impala.impala import IMPALA
from ray.rllib.algorithms.appo.appo import APPO
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_MULTI_RL_MODULE_SPEC,
    COMPONENT_RL_MODULE,
)

import numpy as np
import time
import copy, pickle
from typing import Dict, Any
import requests
import threading

from py_ext.tool import safe_share_memory, log, Event

from dl_helper.rl.param_keeper import AsyncRLParameterServer
from dl_helper.rl.socket_base import get_server_weights, send_gradients, request_client_id
from dl_helper.rl.rl_utils import GradientCompressor, ParamCompressor, GradientAccumulator

"""
# 原始PPO训练流程

_update_from_batch_or_episodes()
    for tensor_minibatch in batch_iter:
        # tensor_minibatch 的迭代
        fwd_out, loss_per_module, tensor_metrics = self._update(tensor_minibatch.policy_batches)


# 分布式训练流程

# 0. 训练前
# 从参数服务器拉取最新参数，确保所有训练者从相同的起点开始
pull_params_from_server()

# 1. 数据收集阶段
train_batch = synchronous_parallel_sample(max_agent_steps=batch_size)
# 此时不需要同步，因为每个训练者都在用自己的策略独立采样

# 2. 训练循环
for epoch in range(num_epochs):
    for minibatch in minibatches:
        # 计算梯度
        gradients = compute_gradients(minibatch)
        
        # 3. 梯度同步（异步方式）
        # 方案A: 直接将梯度发送给参数服务器
        async_push_gradients_to_server(gradients)
        # 方案B: 累积一定步数的梯度后再同步
        if steps % gradient_sync_frequency == 0:
            async_push_gradients_to_server(accumulated_gradients)
        
        # 4. 定期从服务器拉取最新参数（异步方式）
        if steps % param_sync_frequency == 0:
            async_pull_params_from_server()


"""

class SharedParam:
    """
    共享参数类，用于多进程间的同步
    """
    def __init__(self, params_dict, create=True):
        """
        params_dict: 参数字典, 键为参数名, 值为numpy.ndarray
        create: 是否创建共享内存
        """
        self.create = create
        self._params_dict = {}
        self.weights = {COMPONENT_RL_MODULE: {'default_policy': self._params_dict}}
        self.ssms = []

        # 共享事件
        self.event = Event(name=f'_SharedParam_event')
        if create:
            self.event.clear()

        # 共享内存
        for k, v in params_dict.items():
            # v 是numpy.ndarray
            ssm = safe_share_memory(name=f'_SharedParam_{k}', size=v.nbytes)
            shm = ssm.get()
            _v = np.ndarray(v.shape, dtype=v.dtype, buffer=shm.buf)
            self._params_dict[k] = _v
            self.ssms.append(ssm)

    def get_weights(self):
        return self.weights

    def set_param(self, params_dict):
        for k, v in params_dict.items():
            self._params_dict[k][:] = v[:]

class ClientLearnerGroup(LearnerGroup):
    """
    客户端的learner组
    - 若存在多个 learner，则需要选择设置是否与参数服务器通信
    - 需要在每次 update_from_batch 后，获取communicate_learner的参数，并更新到其他learner
    """
    def __init__(self, *args, train_title='', **kwargs):
        super().__init__(*args, **kwargs)
        assert not isinstance(self.config.algo_class, (IMPALA, APPO)), "暂不支持异步算法 IMPALA/APPO"

        # 训练标题
        assert train_title != '', "train_title 不能为空"
        self.train_title = train_title
        
        # 参数压缩器
        self.param_compressor = ParamCompressor()

        # 共享参数
        self.shared_param = None

        # 初始化客户端learner
        self._init_client_learner()

    def _init_client_learner(self):
        """初始化客户端learner"""
        # 设置每个learner的train_title
        log(f"init_client_learner")
        # !!! 字符串需要通过 ray.put 传递
        # res = self.foreach_learner(lambda learner: learner.set_train_title('20250108_breakout'))
        # res = self.foreach_learner(lambda learner: learner.set_train_title(1))
        state_ref = ray.put(self.train_title)
        res = self.foreach_learner(
            lambda _learner, _ref=state_ref: _learner.set_train_title(ray.get(_ref))
        )
        log(f"set train_title to all learners, res: {res}")

        # 设置 除第一个外 learner的 client_id > 不与参数服务器通信
        remote_actor_ids = self._worker_manager.actor_ids()[1:]
        res = self.foreach_learner(lambda learner: learner.set_client_id(-1), remote_actor_ids = remote_actor_ids)

        # # 或 请求client_id
        # res = self.foreach_learner(lambda learner: learner.request_client_id())
        log(f"set client_id to all learners, res: {res}")

        # 初始化参数 使用服务器的最新参数
        self._sync_learner_weights()

        # 初始化
        self.foreach_learner(lambda learner: learner.init_net_thread())

    def _sync_learner_weights(self):
        # 获取服务器的参数，并更新到其他learner
        log('request server weights')
        params_list, info, version = get_server_weights(self.train_title)
        # 解压参数
        params_dict = self.param_compressor.decompress_params_dict(params_list, info)
        weights = {'default_policy': params_dict}
        # 更新到所有learner
        self.set_weights(weights)
        log(f"set weights to all learners, version: {version}")
        res = self.foreach_learner(lambda learner: learner.set_weights_version(version))
        log(f"set weights to all learners, res: {res}")

        # 创建共享参数
        self.shared_param = SharedParam(params_dict, create=True)
        log(f"SharedParam init Done")

        # 初始化learner的共享参数
        log(f"foreach_learner: init shared param")
        self.foreach_learner(lambda learner: learner.init_shared_param())

class ClientPPOTorchLearner(PPOTorchLearner):
    """
    每个客户端只需要有一个与参数服务器通信的learner
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 梯度压缩器
        self.gradient_compressor = GradientCompressor()

        # 参数压缩器
        self.param_compressor = ParamCompressor()

        # 梯度累积器
        self.gradient_accumulator = GradientAccumulator()
        
        # 梯度同步频率 8
        self.gradient_sync_frequency = 1

        # 共享参数
        self.shared_param = None

        # 客户端 id, 0表示与参数服务器通信
        self.client_id = 0

        # 更新计数
        self.update_count = 0

        # 网络传输线程， 只有主 learner 需要
        self.thread1 = None
        self.net_event = None

    def init_shared_param(self):
        log(f"[{self.client_id}] init_shared_param")
        # 获取参数字典
        params_dict = self.get_state(components=COMPONENT_RL_MODULE)['rl_module']['default_policy']
        # 获取共享参数
        self.shared_param = SharedParam(params_dict, create=False)

    def init_net_thread(self):
        if self.client_id == 0:
            # 主learner
            self.net_event = threading.Event()
            self.thread1 = threading.Thread(target=self.net_thread)
            self.thread1.start()

    def net_thread(self):
        log(f"[{self.client_id}] net_thread start")
        while True:
            self.net_event.wait()
            self.net_event.clear()

            log(f"[{self.client_id}] request server weights")
            params_list, info, self.version = get_server_weights(self.train_title)
            # 解压参数
            params_dict = self.param_compressor.decompress_params_dict(params_list, info)
            # 更新共享参数
            self.shared_param.set_param(params_dict)
            # 触发共享参数更新事件
            self.shared_param.event.set()
            log(f"[{self.client_id}] update latest server weights done")

    @staticmethod
    def merge_gradients(gradient_list):
        # 简单平均多个batch的梯度
        merged = []
        length = len(gradient_list)
        for i in range(len(gradient_list[0])):
            merged.append(sum(g[i] for g in gradient_list) / length)
        return merged
    
    # BENCHMARK 100 iter about 0.6H
    # compress data all use 100 iter about 4.35H -35%
    # all use 100 iter about 6.73H 
    # nouse5 100 iter about H
    # nouse4 100 iter about H
    def compute_gradients(self, *args, **kwargs):
        if self.client_id == 0:
            # 主learner 触发请求服务器参数事件
            self.net_event.set()

        # 计算梯度
        # log('self._params:')
        # for idx, (k, v) in enumerate(self._params.items()):
        #     log(f'{idx} {k.shape} {v.shape}')
        # log('compute gradients:')
        gradients_dict = super().compute_gradients(*args, **kwargs)
        # for idx, (k, v) in enumerate(gradients_dict.items()):
        #     log(f'{idx} {k.shape} {v.shape}')
        # log(f'compute gradients done')

        self.update_count += 1

        # nouse3 100 iter about 0.695H -89.66%
        if self.client_id == 0:
            # 主learner
            cpu_gradients = [v.cpu() for _, v in gradients_dict.items()]
            self.gradient_accumulator.add_gradients(cpu_gradients)
            # log(f'add gradients to gradient_accumulator')
            if self.update_count % self.gradient_sync_frequency == 0:
                # 汇总梯度
                # log(f'merge gradients')
                # pickle.dump(self.gradient_accumulator, open(f'gradient_accumulator.pkl', 'wb'))
                merged_gradients = self.gradient_accumulator.merge_gradients()
                # 压缩梯度
                # log(f'compress gradients')
                compressed_grads, compress_info = self.gradient_compressor.compress(merged_gradients)
                # nouse2 100 iter about 0.706H -89.51%
                # 发送梯度
                # log(f'send gradients')
                send_gradients(self.train_title, compressed_grads, compress_info, self.version)
                # nouse2
        # nouse3

        return gradients_dict
    # nouse4
    
    def apply_gradients(self, *args, **kwargs):
        # compress nouse1 100 iter about 3.39H -49%
        # nouse1 100 iter about 3.63H -46%
        # 拉取模型 并同步到所有learner上
        if self.update_count % self.gradient_sync_frequency == 0:
            # 等待参数更新
            self.shared_param.event.wait()
            self.shared_param.event.clear()
            # 获取参数
            weights = self.shared_param.get_weights()
            # 应用到learner
            self.set_state(weights)
            # nouse0
        # nouse1

        # 修改梯度 TODO

        # 不要影响原apply_gradients更新
        return super().apply_gradients(*args, **kwargs)

    def after_gradient_based_update(self, *args, **kwargs):
        self.update_count = 0
        return super().after_gradient_based_update(*args, **kwargs)
    # nouse5

    def set_client_id(self, client_id):
        log(f"[{id(self)}] set_client_id: {client_id}")
        self.client_id = client_id

    def set_train_title(self, train_title):
        log(f"[{id(self)}] set_train_title: {train_title}")
        self.train_title = train_title

    def request_client_id(self):
        # 获取客户端 id
        log(f"[{id(self)}] request_client_id")
        self.client_id = request_client_id(self.train_title)
        log(f"[{id(self)}] client_id: {self.client_id}")
        return self.client_id

    def set_weights_version(self, version):
        log(f"[{id(self)}] set_version: {version}")
        self.version = version
        return self.version


if __name__ == '__main__':
    gradient_buffer_file = r"C:\Users\lh\Downloads\gradient_buffer_0.pkl"
    gradient_buffer = pickle.load(open(gradient_buffer_file, 'rb'))
    log(gradient_buffer)

    merged_gradients = ClientPPOTorchLearner.merge_gradients(gradient_buffer)
    log(merged_gradients)
