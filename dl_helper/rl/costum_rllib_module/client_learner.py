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

from collections import OrderedDict
import numpy as np
import torch
import queue
import time
import asyncio
from asyncio import Queue as AsyncQueue
import copy, pickle
from typing import Dict, Any
import requests
import threading
import multiprocessing
import socket

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from py_ext.tool import safe_share_memory, share_tensor, log, Event, get_exception_msg

from dl_helper.rl.param_keeper import AsyncRLParameterServer
from dl_helper.rl.socket_base import get_server_weights, send_gradients, request_client_id
from dl_helper.rl.socket_base import HOST, PORT, send_msg, recv_msg, CODE, _get_server_weights, _send_gradients
from dl_helper.rl.socket_base import async_send_msg, async_recv_msg, _async_send_gradients, _async_get_server_weights

from dl_helper.rl.rl_utils import ParamCompressor, GradientAccumulator
from dl_helper.deep_gradient_compression import DeepGradientCompression
from dl_helper.param_compression import IncrementalCompressor
from dl_helper.tool import report_memory_usage, AsyncProcessQueueReader, Empty

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
        # param_keys = list(self.get_weights()['default_policy'].keys())
        # self.param_compressor = ParamCompressor(param_keys)
        self.param_compressor = IncrementalCompressor()

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
        params_list, info, version, need_warn_up = get_server_weights(self.train_title)

        # 解压参数
        # _params_dict = self.param_compressor.decompress_params_dict(params_list, info)
        _params_dict_np = self.get_weights()['default_policy']
        _params_dict = OrderedDict()
        for k, v in _params_dict_np.items():
            _params_dict[f'module.{k}'] = torch.from_numpy(v)
        self.param_compressor.decompress(params_list, info, _params_dict)
        
        # 更新参数到所有learner
        if self.is_local:
            self._learner.module._rl_modules['default_policy'].load_state_dict(_params_dict)
        else:
            state_ref = ray.put(_params_dict)
            self.foreach_learner(
                lambda _learner, _ref=state_ref: _learner.module._rl_modules['default_policy'].load_state_dict(ray.get(_ref))
            )
        log(f"set weights to all learners, version: {version}")

        res = self.foreach_learner(lambda learner: learner.set_weights_version(version))
        log(f"set version to all learners, res: {res}")

        res = self.foreach_learner(lambda learner: learner.set_weights_version(int(need_warn_up)))
        log(f"set need_warn_up to all learners, res: {res}")

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
        self.shared_param = SharedParam(_params_dict, grad_params_dict, create=True)
        log(f"SharedParam init Done")

        # 初始化learner的共享参数
        log(f"foreach_learner: init shared param")
        self.foreach_learner(lambda learner: learner.init_shared_param())

class AsyncProcessQueueReader_grad_param(AsyncProcessQueueReader):
    """
    异步进程队列读取器
    用于转发 进程任务 到 事件循环
    """
    def __init__(self, queue, param_q, grad_q, start: bool = True):
        self.queue = queue
        self._loop = None
        self._thread = None
        self._running = False
        self._stop = False

        # 使用传入的队列
        self.param_q = param_q
        self.grad_q = grad_q

        # 启动
        if start:
            self._start()

    def _reader_thread(self):
        """后台读取线程"""
        while not self._stop:
            try:
                # 使用较短的超时以便能够响应停止信号
                item = self.queue.get(timeout=0.1)

                # 分发事件
                # 使用线程安全的方式将任务加入事件循环
                if isinstance(item, int):
                    # 参数事件
                    asyncio.run_coroutine_threadsafe(
                        self.param_q.put(item), 
                        self._loop
                    )
                
                else:
                    # 梯度事件
                    asyncio.run_coroutine_threadsafe(
                        self.grad_q.put(item), 
                        self._loop
                    )


            except Empty:
                continue
            except Exception as e:
                log(f"Reader thread error: {e}")
                # 出错时短暂等待后继续
                time.sleep(0.1)

class ClientPPOTorchLearner(PPOTorchLearner):
    """
    每个客户端只需要有一个与参数服务器通信的learner
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 参数压缩器
        self.param_compressor = None

        # 梯度累积器
        self.gradient_accumulator = GradientAccumulator()
        
        # 梯度推送频率
        self.gradient_sync_frequency = 1

        # 限制最小参数同步间隔
        # 若频率更高, 会增加服务器压力
        self.min_param_sync_interval = 4

        # 拉取应用参数后，是否需要应用最后的梯度
        self.apply_last_grad = True

        # 共享参数
        self.shared_param = None

        # 客户端 id, 0表示与参数服务器通信
        self.client_id = 0

        # 更新计数
        self.update_count = 0

        self.version = 0
        self.need_warn_up = False

        ####################
        # 只有主 learner 需要初始化
        ####################
        # 时间队列(与协程进程通讯)
        self.task_queue = None
        # 用于初始化共享参数
        self.params_dict = None
        self.grad_params_dict = None
        ####################
        # 只有主 learner 需要初始化
        ####################

    def init_shared_param(self):
        log(f"[{self.client_id}] init_shared_param")
        # 获取参数字典
        params_dict = self.module._rl_modules['default_policy'].state_dict()
        # 获取梯度字典
        grad_params_dict = self._params
        if self.client_id == 0:
            self.params_dict = params_dict
            self.grad_params_dict = grad_params_dict
        # 获取共享参数
        self.shared_param = SharedParam(params_dict, grad_params_dict, create=False)

    def init_param_thread(self):
        if self.client_id == 0:
            # 主learner
            self.task_queue = multiprocessing.Queue()

            # 在新进程中运行事件循环
            self.event_loop_process = multiprocessing.Process(
                target=self._run_event_loop_process, 
                args=(
                    self.task_queue, self.train_title, self.client_id, self.params_dict, self.grad_params_dict,
                    self.version, self.need_warn_up
                )
            )
            self.event_loop_process.start()

    @staticmethod
    def _run_event_loop_process(task_queue, train_title, client_id, params_dict, grad_params_dict, version, need_warn_up):
        """在新进程中运行事件循环"""
        # 事件循环
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 如果当前线程没有事件循环，则创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        class share_info:
            """
            共享信息类
            """
            def __init__(self, train_title, client_id, version, need_warn_up):
                self.train_title = train_title
                self.client_id = client_id
                self.version = version
                self.need_warn_up = need_warn_up
        
        info_data = share_info(train_title, client_id, version, need_warn_up)
        
        # 梯度事件队列
        grad_q = AsyncQueue(maxsize=30)
        # 参数事件队列
        param_q = AsyncQueue()

        # 独立线程转发 进程任务
        apqr = AsyncProcessQueueReader_grad_param(task_queue, param_q, grad_q)

        # 进程池,用于压缩/加压等计算密集任务计算
        process_pool = ProcessPoolExecutor(max_workers=3)  # 可以根据需要调整进程数
    
        # 启动协程任务
        tasks = []
        for _idx in range(10):
            task = loop.create_task(ClientPPOTorchLearner.grad_coroutine(_idx, info_data, process_pool, grad_q))
            tasks.append(task)

        # 启动参数协程
        task = loop.create_task(ClientPPOTorchLearner.param_coroutine(info_data, process_pool, param_q, params_dict, grad_params_dict)) 
        tasks.append(task)

        # 运行事件循环
        loop.run_forever()

    @staticmethod
    async def grad_coroutine(idx, info_data, process_pool, grad_q):
        """
        梯度协程
        """
        log(f"[{info_data.client_id}] grad_coroutine {idx} start")

        # 梯度压缩器
        gradient_compressor = DeepGradientCompression()

        while True:
            try:
                # 创建异步socket连接
                reader, writer = await asyncio.open_connection(HOST, PORT)
                # 发送连接类型
                await async_send_msg(writer, f'{CODE}_long')

                send_count = 0
                while True:
                    # 获取汇总梯度 TODO 使用共享内存替代
                    merged_gradients = await grad_q.get()
                    # log(f"[{idx}] queue size: {grad_q.qsize()}")
                    
                    # 在进程池中执行压缩操作
                    loop = asyncio.get_event_loop()
                    compressed_result = await loop.run_in_executor(
                        process_pool,
                        partial(gradient_compressor.compress, merged_gradients, info_data.need_warn_up)
                    )
                    compressed_grads, compress_info = compressed_result

                    # 发送梯度
                    await _async_send_gradients(writer, info_data.train_title, compressed_grads, compress_info, info_data.version)
                    # log(f"[{idx}][{send_count}] send gradients done")
                    send_count += 1

                    # 每10次接收一次响应
                    if send_count % 10 == 0:
                        # log(f"[{idx}] wait response")
                        await async_recv_msg(reader)
                        # log(f"[{idx}] recv response done")

            except Exception as e:
                log(f"[{idx}] 连接服务器失败: \n{get_exception_msg()}")
                # 关闭连接
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    @staticmethod
    async def param_coroutine(info_data, process_pool, param_q, params_dict, grad_params_dict):
        """
        获取服务器参数
        """
        log(f"[{info_data.client_id}] param_coroutine start")

        # 参数解压器
        # params_keys = list(params_dict.keys())
        # param_compressor = ParamCompressor(params_keys)
        param_compressor = IncrementalCompressor()
        sync_params_dict = copy.deepcopy(params_dict)

        # 共享参数
        shared_param = SharedParam(params_dict, grad_params_dict, create=False)

        # 统计拉取参数耗时
        total_cost_time = 0
        total_count = 0

        log(f"param_coroutine init done")
        while True:
            try:
                # 创建异步socket连接
                reader, writer = await asyncio.open_connection(HOST, PORT)
                # 发送连接类型
                await async_send_msg(writer, f'{CODE}_long')
                log(f"param_coroutine connect to server")

                send_count = 0
                while True:
                    # 请求参数的轮次
                    ask_update_count = await param_q.get()

                    t0 = time.time()

                    log(f"[{ask_update_count}] request server weights")
                    send_count += 1

                    # 获取参数
                    params_list, info, info_data.version, info_data.need_warn_up = await _async_get_server_weights(writer, reader, info_data.train_title, info_data.version)
                    log(f"[{ask_update_count}] recv params data")
                    
                    # 在进程池中执行解压操作
                    # loop = asyncio.get_event_loop()
                    # decompressed_result = await loop.run_in_executor(
                    #     process_pool,
                    #     partial(param_compressor.decompress_params_dict, params_list, info)
                    # )
                    # # 更新共享参数
                    # shared_param.set_param(decompressed_result)

                    # 增量解压操作
                    log(f"[{ask_update_count}] decompress params data")
                    param_compressor.decompress(params_list, info, sync_params_dict)
                    # 更新共享参数
                    shared_param.set_param(sync_params_dict)

                    log(f"[{ask_update_count}] set params to shared param")
                    # 触发共享参数更新事件
                    shared_param.param_event.set()
                    log(f"[{ask_update_count}] update latest server weights done")

                    # 统计耗时
                    total_cost_time += time.time() - t0
                    total_count += 1

                    # 每10次接收一次响应
                    if send_count % 10 == 0:
                        # log(f"wait response")
                        await async_recv_msg(reader)
                        # log(f"recv response done")

                    if total_count % 30 == 0:
                        log(f"[{ask_update_count}] avg cost time: {int((total_cost_time / total_count) * 1000)}ms")

            except Exception as e:
                log(f"连接服务器失败: \n{get_exception_msg()}")
                # 关闭连接
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    # BENCHMARK 100 iter about 0.6H
    # compress data all use 100 iter about 4.35H -35%
    # all use 100 iter about 6.73H 
    # nouse5 100 iter about H
    # nouse4 100 iter about H
    def compute_gradients(self, *args, **kwargs):
        self.update_count += 1
        # report_memory_usage(f'[{self.update_count}][0]')

        if self.client_id == 0:
            # 按照梯度同步频率请求服务器参数
            if self.update_count % self.min_param_sync_interval == 0:
                log(f'[{self.update_count}] request param and reset event')
                self.task_queue.put(self.update_count)
                # 重置event
                self.shared_param.param_event.clear()

            # 清空梯度事件
            self.shared_param.grad_event.clear()

        # 计算梯度
        gradients_dict = super().compute_gradients(*args, **kwargs)

        # nouse3 100 iter about 0.695H -89.66%
        if self.client_id == 0:
            # 主learner
            cpu_gradients = [v.cpu() for _, v in gradients_dict.items()]

            if self.gradient_sync_frequency > 1:
                # 累积梯度
                self.gradient_accumulator.add_gradients(cpu_gradients)
                # log(f'add gradients to gradient_accumulator')
                if self.update_count % self.gradient_sync_frequency == 0:
                    # 汇总梯度
                    # log(f'merge gradients')
                    gradients_to_send = self.gradient_accumulator.merge_gradients()
                    # 加入推送队列
                    self.task_queue.put(gradients_to_send)

                    if self.apply_last_grad:
                        # 需要替换梯度
                        # 1. 拷贝到共享梯度
                        self.shared_param.set_grad(gradients_to_send)
                        # 2. 触发梯度更新事件
                        self.shared_param.grad_event.set()
            else:
                # 加入推送队列
                self.task_queue.put(cpu_gradients)
        
            log(f'[{self.update_count}] compute_gradients done')
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
        # 是否需要等待新参数就绪并应用, (可选是否应用最近的一次梯度)
        if self.update_count % self.min_param_sync_interval == 0:
            # 正确处理最后一次的梯度
            if self.gradient_sync_frequency > 1 and self.apply_last_grad:
                # 将共享梯度应用到本地参数
                if self.client_id == 0:
                    log(f'[{self.update_count}] wait shared grad set')
                self.shared_param.grad_event.wait()
                # 替换应用梯度
                self.shared_param.apply_grad_to_local(self)
            else:
                # 已经存在正确的梯度
                pass

            # 等待并应用新的参数
            # 等待参数就绪
            if self.client_id == 0:
                log(f'[{self.client_id}] wait param ready')
            self.shared_param.param_event.wait()
            # 获取参数覆盖本地参数
            # log(f'[{self.client_id}] apply shared param')
            p = self.shared_param.get_weights()
            self.module._rl_modules['default_policy'].load_state_dict(p)

            # 使用梯度更新一次参数 > 更新完成后参数与服务器参数一致
            if self.apply_last_grad:
                super().apply_gradients(*args, **kwargs)

        if self.client_id == 0:
            log(f'[{self.update_count}] apply_gradients done')

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

    def set_need_warn_up(self, need_warn_up):
        log(f"[{id(self)}] set_need_warn_up: {need_warn_up}")
        self.need_warn_up = bool(need_warn_up)

if __name__ == '__main__':
    gradient_buffer_file = r"C:\Users\lh\Downloads\gradient_buffer_0.pkl"
    gradient_buffer = pickle.load(open(gradient_buffer_file, 'rb'))
    log(gradient_buffer)

    merged_gradients = ClientPPOTorchLearner.merge_gradients(gradient_buffer)
    log(merged_gradients)
