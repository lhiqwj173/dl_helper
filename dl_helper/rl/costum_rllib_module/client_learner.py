import ray
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.algorithms.impala.impala import IMPALA
from ray.rllib.algorithms.appo.appo import APPO

from typing import Dict, Any
import requests

from dl_helper.rl.param_keeper import AsyncRLParameterServer
from dl_helper.rl.socket_base import get_server_weights, send_gradients, request_client_id

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

    def init_client_learner(self):
        """初始化客户端learner"""
        # 设置每个learner的train_title
        print(f"init_client_learner")
        res = self.foreach_learner(lambda learner: learner.set_train_title(self.train_title))
        print(f"set train_title to all learners, res: {res}")

        # 同步参数
        self._sync_learner_weights()

    def _sync_learner_weights(self):
        """广播 communicate_learner 的参数到其他learner"""
        # 获取服务器的参数，并更新到其他learner
        print('request server weights')
        state, version = get_server_weights(self.train_title)
        print('state:', state)
        print('version:', version)
        weights = {'default_policy': state['rl_module']['default_policy']}
        # 更新到所有learner
        self.set_weights(weights)
        print(f"set weights to all learners, version: {version}")
        res = self.foreach_learner(lambda learner: learner.set_weights_version(version))
        print(f"set weights to all learners, version: {res}")

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 版本号
        self.version = 0

    def apply_gradients(self, gradients_dict) -> None:
        if self.client_id == 0:
            # 发送梯度
            print(f"[{self.client_id}] send_gradients")
            send_gradients(self.train_title, gradients_dict, self.version)
        # 其他learner什么也不做

    def set_train_title(self, train_title):
        print(f"[{id(self)}] set_train_title: {train_title}")
        return 1
    
        print(f"[{id(self)}] set_train_title: {train_title}")
        self.train_title = train_title
        # 获取客户端 id
        print(f"[{id(self)}] request_client_id")
        self.client_id = request_client_id(self.train_title)
        print(f"[{id(self)}] client_id: {self.client_id}")
        return self.client_id

    def set_weights_version(self, version):
        print(f"[{id(self)}] set_version: {version}")
        self.version = version
        return self.version
