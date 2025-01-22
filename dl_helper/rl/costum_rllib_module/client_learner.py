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
import torch
import queue
import time
import copy, pickle
from typing import Dict, Any
import requests
import threading

from py_ext.tool import safe_share_memory, share_tensor, log, Event

from dl_helper.rl.param_keeper import AsyncRLParameterServer
from dl_helper.rl.socket_base import get_server_weights, send_gradients, request_client_id
from dl_helper.rl.rl_utils import GradientCompressor, ParamCompressor, GradientAccumulator

"""
# 分布式训练流程
# n step(minibatch) 同步

# 0. 训练前
# 从参数服务器拉取最新参数，确保所有训练者从相同的起点开始
pull_params_from_server()

# 1. 数据收集阶段
train_batch = synchronous_parallel_sample(max_agent_steps=batch_size)
# 此时不需要同步，因为每个训练者都在用自己的策略独立采样

# 2. 训练循环
for epoch in range(num_epochs):
    step_count = 0
    for minibatch in minibatches:
        step_count += 1

        # 正向传播

        # 网络线程 开始拉取服务器参数, 完成后会更新到共享参数中，等待使用
        if step_count % n == 0:
            async_pull_params_from_server_to_shared_param()

        # 计算梯度
        gradients = compute_gradients(minibatch)
        if step_count % n == 0:
            # 汇总梯度并推送
            async_push_gradients_to_server(accumulated_gradients)
            # 拷贝汇总梯度到共享梯度
            copy_to_shared_memory(accumulated_gradients)
        
        # 参数/梯度应用
        if step_count % n == 0:
            # 应用共享梯度
            apply_shared_gradients()
            # 应用共享参数
            apply_shared_params()
            # 梯度更新, 更新完成后参数与服务器参数一致
            self.apply_gradients(postprocessed_gradients)
"""

class SharedParam:
    """
    共享参数类，用于多进程间的同步
    """
    def __init__(self, params_dict, grad_params_dict, create=True):
        """
        params_dict: 参数字典, 键为参数名, 值为 torch.Tensor
        grad_params_dict: 梯度参数字典, 键为参数名, 值为 torch.Tensor
        create: 是否创建共享内存
        """
        self.create = create
        self._params_dict_np = {}
        self._params_dict = {}
        self._grad_params_list_np = []
        self._grad_params_list = []
        self.ssms = []

        # 共享事件 - 参数更新完毕
        self.param_event = Event(name=f'_SharedParam_event')
        if create:
            self.param_event.clear()
        
        # 共享事件 - 汇聚梯度准备完毕
        self.grad_event = Event(name=f'_SharedParam_grad_event')
        if create:
            self.grad_event.clear()

        # 共享参数内存 储存解压后的服务器参数 float32
        for k, v in params_dict.items():
            # v 是 torch.Tensor
            ssm = safe_share_memory(name=f'_SharedParam_{k}', size=v.numel() * v.element_size())
            shm = ssm.get()
            _v = np.ndarray(v.shape, dtype=np.float32, buffer=shm.buf)
            self._params_dict_np[k] = _v
            self._params_dict[k] = torch.from_numpy(_v)
            self.ssms.append(ssm)

        # 共享梯度内存 储存主learner的聚合梯度 float32
        for idx, v in enumerate(grad_params_dict.values()):
            # v 是torch.Tensor
            ssm = safe_share_memory(name=f'_SharedParam_grad_{idx}', size=v.numel() * v.element_size())
            shm = ssm.get()
            _v = np.ndarray(v.shape, dtype=np.float32, buffer=shm.buf)
            self._grad_params_list_np.append(_v)
            self._grad_params_list.append(torch.from_numpy(_v))
            self.ssms.append(ssm)

    def get_weights(self):
        """
        返回共享参数字典
        {
            'fc1.weight': torch.Tensor,
            'fc1.bias': torch.Tensor,
            ...
        }
        """
        # 不需要拷贝数据，只读
        return self._params_dict

    def set_param(self, params_dict):
        """
        参数:
        params_dict: 参数字典, 键为参数名, 值为torch.Tensor
        """
        for k, v in params_dict.items():
            self._params_dict[k][:] = v[:]

    def set_grad(self, grad_params_list):
        """
        参数:
        grad_params_list: 梯度参数字典, 键为参数名, 值为torch.Tensor
        """
        for i, v in enumerate(grad_params_list):
            self._grad_params_list[i][:] = v[:]

    def apply_grad_to_local(self, learner):
        """
        将共享梯度应用到本地learner
        参数:
        learner: 本地learner
        """
        # 不需要拷贝数据，只读
        params = learner._params
        for idx, k in enumerate(params.keys()):
            params[k].grad = self._grad_params_list[idx].to(learner._device)

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
        self.foreach_learner(lambda learner: learner.init_param_thread())

    def _sync_learner_weights(self):
        # 获取服务器的参数，并更新到其他learner
        log('request server weights')
        params_list, info, version = get_server_weights(self.train_title)
        # 解压参数
        params_dict = self.param_compressor.decompress_params_dict(params_list, info)
        # 更新参数到所有learner
        if self.is_local:
            self._learner.module._rl_modules['default_policy'].load_state_dict(params_dict)
        else:
            state_ref = ray.put(params_dict)
            self.foreach_learner(
                lambda _learner, _ref=state_ref: _learner.module._rl_modules['default_policy'].load_state_dict(ray.get(_ref))
            )
        log(f"set weights to all learners, version: {version}")
        res = self.foreach_learner(lambda learner: learner.set_weights_version(version))
        # log(f"set weights to all learners, res: {res}")

        # 获取一个learner的梯度字典
        if self.is_local:
            grad_params_dict = self._learner._params
        else:
            worker = self._worker_manager.healthy_actor_ids()[0]
            results = self._worker_manager.foreach_actor(
                lambda w: w._params,
                remote_actor_ids=[worker],
            )
            grad_params_dict = self._get_results(results)[0]

        # 创建共享参数
        self.shared_param = SharedParam(params_dict, grad_params_dict, create=True)
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
        
        # 梯度推送频率
        self.gradient_sync_frequency = 1

        # 限制最小参数同步间隔
        # 若频率更高, 会增加服务器压力
        self.min_param_sync_interval = 4 

        # 拉取应用参数后，是否需要应用最后的梯度
        self.apply_last_grad = False

        # 共享参数
        self.shared_param = None

        # 客户端 id, 0表示与参数服务器通信
        self.client_id = 0

        # 更新计数
        self.update_count = 0

        # 参数传输线程， 只有主 learner 需要
        self.thread0 = None
        self.param_q = None
        self.param_done_q = None

        # 梯度传输线程， 只有主 learner 需要
        self.thread1 = None
        self.grad_q = None

        # learner 之间的同步 
        self.load_param_event = Event(name=f'load_param_event')# 等待主learner同步
        self.main_learner_ready_event = Event(name=f'main_learner_ready_event')# 被设置，说明需要更新参数
        
    def init_shared_param(self):
        log(f"[{self.client_id}] init_shared_param")
        # 获取参数字典
        params_dict = self.get_state(components=COMPONENT_RL_MODULE)['rl_module']['default_policy']
        # 获取梯度字典
        grad_params_dict = self._params
        # 获取共享参数
        self.shared_param = SharedParam(params_dict, grad_params_dict, create=False)


    def init_param_thread(self):
        if self.client_id == 0:
            # 主learner
            self.param_q = queue.Queue()
            self.param_done_q = queue.Queue()
            self.thread0 = threading.Thread(target=self.param_thread)
            self.thread0.start()

            self.grad_q = queue.Queue()
            self.thread1 = threading.Thread(target=self.grad_thread)
            self.thread1.start()

    def param_thread(self):
        """
        获取服务器参数
        """
        log(f"[{self.client_id}] param_thread start")
        last_ask_update_count = 0
        while True:
            # 请求参数的轮次
            ask_update_count = self.param_q.get()
            if self.min_param_sync_interval > 1 and ask_update_count - last_ask_update_count < self.min_param_sync_interval:
                # 等待最小同步间隔
                continue

            last_ask_update_count = ask_update_count

            log(f"[{self.client_id}] request server weights")
            # 获取参数
            params_list, info, self.version = get_server_weights(self.train_title, self.version)
            # 解压参数
            params_dict = self.param_compressor.decompress_params_dict(params_list, info)
            # 更新共享参数
            self.shared_param.set_param(params_dict)
            # 触发共享参数更新事件
            self.shared_param.param_event.set()
            log(f"[{self.client_id}] update latest server weights done")

    def grad_thread(self):
        """
        推送本地梯度
        """
        log(f"[{self.client_id}] grad_thread start")
        while True:
            # 获取汇总梯度
            merged_gradients = self.grad_q.get()

            # 压缩梯度
            compressed_grads, compress_info = self.gradient_compressor.compress(merged_gradients)

            # 发送梯度
            send_gradients(self.train_title, compressed_grads, compress_info, self.version)

    # BENCHMARK 100 iter about 0.6H
    # compress data all use 100 iter about 4.35H -35%
    # all use 100 iter about 6.73H 
    # nouse5 100 iter about H
    # nouse4 100 iter about H
    def compute_gradients(self, *args, **kwargs):
        self.update_count += 1

        if self.client_id == 0:
            # 每次都请求，使用最大同步间隔(与每次同步耗时共同影响)来控制同步频率
            # 主learner 触发请求服务器参数事件
            # 参数线程开始获取服务器参数 》 暂时默认本次update应用拉取的参数  TODO
            #    本次update应用:    拉取的参数少了本次推送的梯度, apply_gradients函数中覆盖本地参数后需要应用本次的推送梯度
            #    n次update后应用:   已默认允许延迟，故不再应用梯度
            self.param_q.put(self.update_count)

            # 清空learner之间同步的事件
            self.main_learner_ready_event.clear()
            self.load_param_event.clear()

        # 计算梯度
        # log('self._params:')
        # for idx, (k, v) in enumerate(self._params.items()):
        #     log(f'{idx} {k.shape} {v.shape}')
        # log('compute gradients:')
        gradients_dict = super().compute_gradients(*args, **kwargs)
        # for idx, (k, v) in enumerate(gradients_dict.items()):
        #     log(f'{idx} {k.shape} {v.shape}')
        # log(f'compute gradients done')

        # nouse3 100 iter about 0.695H -89.66%
        if self.client_id == 0:
            # 主learner
            cpu_gradients = [v.cpu() for _, v in gradients_dict.items()]

            if self.gradient_sync_frequency > 1:
                # 累积梯度
                self.gradient_accumulator.add_gradients(cpu_gradients)
                log(f'add gradients to gradient_accumulator')
                if self.update_count % self.gradient_sync_frequency == 0:
                    # 汇总梯度
                    log(f'merge gradients')
                    gradients_to_send = self.gradient_accumulator.merge_gradients()
                    # 加入推送队列
                    self.grad_q.put(gradients_to_send)

                    if self.apply_last_grad:
                        # 需要替换梯度
                        # 1. 拷贝到共享梯度
                        self.shared_param.set_grad(gradients_to_send)
                        # 2. 触发梯度更新事件
                        self.shared_param.grad_event.set()
            else:
                # 加入推送队列
                self.grad_q.put(cpu_gradients)

        # nouse3
        # 返回空
        return {}
    # nouse4

    def postprocess_gradients(self, gradients_dict):
        # 优化: 做一次过滤
        if not gradients_dict:
            return {}
        return super().postprocess_gradients(gradients_dict)
    
    def apply_gradients(self, *args, **kwargs):
        # compress nouse1 100 iter about 3.39H -49%
        # nouse1 100 iter about 3.63H -46%
        # 拉取模型 并同步到所有learner上
        # 检查参数是否拉取完毕
        if self.client_id != 0:
            # 等待主learner同步
            log(f'[not main learner] wait main learner')
            self.main_learner_ready_event.wait()
        else:
            # 主learner检查是否是否有参数准备好
            if self.shared_param.param_event.is_set():
                # 其他learner需要load参数
                log(f'[main learner] set load param event')
                self.load_param_event.set()
            # 触发其他learner继续向后执行
            log(f'[main learner] all go on')
            self.main_learner_ready_event.set()

        if self.load_param_event.is_set():
            if self.gradient_sync_frequency > 1 and self.apply_last_grad:
                # 将共享梯度应用到本地参数
                log(f'[{self.client_id}] wait shared grad set')
                self.shared_param.grad_event.wait()
                self.shared_param.grad_event.clear()
                # 替换应用梯度
                self.shared_param.apply_grad_to_local(self)
            else:
                # 已经存在正确的梯度
                pass

            # 获取参数覆盖本地参数
            log(f'[{self.client_id}] apply shared param')
            p = self.shared_param.get_weights()
            self.module._rl_modules['default_policy'].load_state_dict(p)

            if self.apply_last_grad:
                # 使用梯度更新一次参数 > 更新完成后参数与服务器参数一致
                super().apply_gradients(*args, **kwargs)
            # nouse0
        # nouse1
        # 非同步模型，则不需要应用梯度

    def after_gradient_based_update(self, *args, **kwargs):
        # 重置
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
