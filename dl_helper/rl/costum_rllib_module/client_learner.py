import ray
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.algorithms.impala.impala import IMPALA
from ray.rllib.algorithms.appo.appo import APPO

from typing import Dict, Any
import requests

from dl_helper.rl.param_keeper import AsyncRLParameterServer
from dl_helper.rl.socket_base import get_server_weights, send_gradients, request_client_id

"""
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

        # 初始化客户端learner
        self._init_client_learner()

    def _init_client_learner(self):
        """初始化客户端learner"""
        # 设置每个learner的train_title
        print(f"init_client_learner")
        # !!! 字符串需要通过 ray.put 传递
        # res = self.foreach_learner(lambda learner: learner.set_train_title('20250108_breakout'))
        # res = self.foreach_learner(lambda learner: learner.set_train_title(1))
        state_ref = ray.put(self.train_title)
        res = self.foreach_learner(
            lambda _learner, _ref=state_ref: _learner.set_train_title(ray.get(_ref))
        )
        print(f"set train_title to all learners, res: {res}")

        # 设置 除第一个外 learner的 client_id > 不与参数服务器通信
        remote_actor_ids = self._worker_manager.actor_ids()[1:]
        res = self.foreach_learner(lambda learner: learner.set_client_id(-1), remote_actor_ids = remote_actor_ids)

        # # 或 请求client_id
        # res = self.foreach_learner(lambda learner: learner.request_client_id())
        print(f"set client_id to all learners, res: {res}")

        # 初始化参数 使用服务器的最新参数
        self._sync_learner_weights()

    def _sync_learner_weights(self):
        # 获取服务器的参数，并更新到其他learner
        print('request server weights')
        state, version = get_server_weights(self.train_title)
        # print('state:', state)
        # print('version:', version)
        weights = {'default_policy': state['rl_module']['default_policy']}
        # 更新到所有learner
        self.set_weights(weights)
        print(f"set weights to all learners, version: {version}")
        res = self.foreach_learner(lambda learner: learner.set_weights_version(version))
        print(f"set weights to all learners, res: {res}")

class ClientPPOTorchLearner(PPOTorchLearner):
    """
    每个客户端只需要有一个与参数服务器通信的learner
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 客户端 id, 0表示与参数服务器通信
        self.client_id = 0

        # 版本号
        self.version = 0

        # 梯度同步频率
        self.gradient_sync_frequency = 8
        self.gradient_buffer = []
        
    def _merge_gradients(self, gradient_list):
        # 简单平均多个batch的梯度
        merged = {}
        for key in gradient_list[0].keys():
            merged[key] = sum(g[key] for g in gradient_list) / len(gradient_list)
        return merged
    
    def apply_gradients(self, gradients_dict) -> None:
        if self.client_id == 0:
            self.gradient_buffer.append(gradients_dict)
            if len(self.gradient_buffer) >= self.gradient_sync_frequency:
                # 发送梯度
                merged_gradients = self._merge_gradients(self.gradient_buffer)
                print(f"[{self.client_id}] send_gradients")
                send_gradients(self.train_title, merged_gradients, self.version)
                self.gradient_buffer = []
        super().apply_gradients(gradients_dict)

    def set_client_id(self, client_id):
        print(f"[{id(self)}] set_client_id: {client_id}")
        self.client_id = client_id

    def set_train_title(self, train_title):
        print(f"[{id(self)}] set_train_title: {train_title}")
        self.train_title = train_title

    def request_client_id(self):
        # 获取客户端 id
        print(f"[{id(self)}] request_client_id")
        self.client_id = request_client_id(self.train_title)
        print(f"[{id(self)}] client_id: {self.client_id}")
        return self.client_id

    def set_weights_version(self, version):
        print(f"[{id(self)}] set_version: {version}")
        self.version = version
        return self.version
