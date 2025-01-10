import ray
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.algorithms.impala.impala import IMPALA
from ray.rllib.algorithms.appo.appo import APPO

from typing import Dict, Any
import requests

from dl_helper.rl.param_keeper import AsyncRLParameterServer
from dl_helper.rl.socket_base import get_server_weights, send_gradients

class ClientLearnerGroup(LearnerGroup):
    """
    客户端的learner组
    - 若存在多个 learner，则需要选择设置是否与参数服务器通信
    - 需要在每次 update_from_batch 后，获取communicate_learner的参数，并更新到其他learner
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not isinstance(self.config.algo_class, (IMPALA, APPO)), "暂不支持异步算法 IMPALA/APPO"

        # 获取learner列表 self.config.num_learners
        actors = self._worker_manager.actors()
        self.communicate_learner = actors[0]
        communicate_learner_id = id(self.communicate_learner)
        print(f"communicate_learner_id: {communicate_learner_id}")   
        # 选择第一个若 数量>1 
        # 数量 == 1, 默认会进行与参数服务器通信
        if self.config.num_learners > 1:
            # 广播id，非id的learner将不会进行与参数服务器通信
            self.foreach_learner(
                func = lambda _learner: _learner.set_communicate_with_param_server(communicate_learner_id)
            )
        
        # 从服务器初始化参数
        print(f"init weights from param server")
        self.foreach_learner(
            lambda _learner: _learner.init_weights()
        )

        # 同步参数
        self._sync_learner_weights()

    def _sync_learner_weights(self):
        """广播 communicate_learner 的参数到其他learner"""
        if self.config.num_learners > 1:
            print(f"set weights to all learners")
            # 获取id==communicate_learner_id的learner的参数，并更新到其他learner
            state_future = self.communicate_learner.get_state.remote()
            # 等待权重获取完成
            state = ray.get(state_future)
            weights = {'default_policy': state['rl_module']['default_policy']}
            # 更新到所有learner
            self.set_weights(weights)

    def update_from_batch(
        self,
        batch,
        *,
        timesteps = None,
        async_update = False,
        return_state = False,
        num_epochs = 1,
        minibatch_size = None,
        shuffle_batch_per_epoch = False,
        # User kwargs.
        **kwargs,
    ):
        res = super().update_from_batch(
            batch, 
            timesteps=timesteps, 
            async_update=async_update, 
            return_state=return_state, 
            num_epochs=num_epochs, 
            minibatch_size=minibatch_size, 
            shuffle_batch_per_epoch=shuffle_batch_per_epoch, 
            **kwargs)
        # 同步参数
        self._sync_learner_weights()
        return res

class ClientPPOTorchLearner(PPOTorchLearner):
    """
    每个客户端只需要有一个与参数服务器通信的learner
    """
    def __init__(self, train_title, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 训练标题
        self.train_title = train_title

        # 是否与参数服务器通信（默认为 True）
        # 通过 set_communicate_with_param_server 更改设置
        self.communicate_with_param_server = True

        # 版本号
        self.version = 0

        # obj id
        self._id = id(self)

    def apply_gradients(self, gradients_dict) -> None:
        if self.communicate_with_param_server:
            # 发送梯度
            print(f"[{self._id}] send_gradients")
            send_gradients(self.train_title, gradients_dict, self.version)
            # 获取参数服务器权重
            self.init_weights()
        # 其他learner什么也不做

    @ray.method(num_returns=0)
    def set_communicate_with_param_server(self, need_communicate_id: int):
        """
        设置是否与参数服务器通信
        """
        self.communicate_with_param_server = id(self) == need_communicate_id

    @ray.method(num_returns=0)
    def init_weights(self):
        """
        从参数服务器获取初始化权重
        """
        if self.communicate_with_param_server:
            print(f"[{self._id}] set weights from server, version: {self.version}")
            rl_module_only_state, self.version = get_server_weights(self.train_title)
            self.module.set_state(rl_module_only_state)


