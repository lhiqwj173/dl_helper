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

from py_ext.tool import safe_share_memory, share_tensor, log, Event, get_exception_msg, get_log_folder, init_logger

from dl_helper.rl.param_keeper import AsyncRLParameterServer
from dl_helper.rl.socket_base import get_server_weights
from dl_helper.rl.socket_base import HOST, PORT, CODE, GRAD_BATCH_SIZE, CHUNK_SIZE, connect_and_tune
from dl_helper.rl.socket_base import async_send_msg, async_recv_msg, _async_wait_server_weights, ack, wait_ack

from dl_helper.rl.rl_utils import ParamCompressor, GradientAccumulator
from dl_helper.deep_gradient_compression import DeepGradientCompression
from dl_helper.param_compression import IncrementalCompressor
from dl_helper.tool import report_memory_usage, AsyncProcessQueueReader, Empty
from dl_helper.train_param import match_num_processes

train_folder = 'cartpole'
init_logger('20250130_cartpole', home=train_folder, timestamp=False)

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

def get_results(RemoteCallResults):
    res = []
    for r in RemoteCallResults:
        res.append(r.get())
    return res

class ClientLearnerGroup(LearnerGroup):
    """
    客户端的learner组
    - 若存在多个 learner，则需要选择设置是否与参数服务器通信
    - 需要在每次 update_from_batch 后，获取communicate_learner的参数，并更新到其他learner
    """
    def __init__(self, *args, train_title='',train_folder='', **kwargs):
        super().__init__(*args, **kwargs)
        assert not isinstance(self.config.algo_class, (IMPALA, APPO)), "暂不支持异步算法 IMPALA/APPO"

        # 训练标题
        assert train_title != '', "train_title 不能为空"
        assert train_folder != '', "train_folder 不能为空"
        self.train_title = train_title
        self.train_folder = train_folder

        # 初始化日志
        init_logger(train_title, home=train_folder, timestamp=False)

        # 参数压缩器
        # param_keys = list(self.get_weights()['default_policy'].keys())
        # self.param_compressor = ParamCompressor(param_keys)
        self.param_compressor = IncrementalCompressor()

        # 共享参数
        self.shared_param = None

        # 初始化客户端learner
        self._init_client_learner()

        log(f"ClientLearnerGroup init done")

    def _init_client_learner(self):
        """初始化客户端learner"""
        # 设置每个learner的train_title
        log(f"init_client_learner")

        # 设置 除第一个外 learner的 client_id > 不与参数服务器通信
        remote_actor_ids = self._worker_manager.actor_ids()[1:]
        res = self.foreach_learner(lambda learner: learner.set_client_id(-1), remote_actor_ids = remote_actor_ids)
        log(f"set set_client_id to not main learners, res: {get_results(res)}")

        # 设置各个learner的 train_title 和 train_folder
        # !!! 字符串需要通过 ray.put 传递
        # res = self.foreach_learner(lambda learner: learner.set_train_title('20250108_breakout'))
        # res = self.foreach_learner(lambda learner: learner.set_train_title(1))
        state_ref = ray.put(self.train_title)
        res = self.foreach_learner(
            lambda _learner, _ref=state_ref: _learner.set_train_title(ray.get(_ref))
        )
        log(f"set train_title to all learners, res: {get_results(res)}")
        state_ref = ray.put(self.train_folder)
        res = self.foreach_learner(
            lambda _learner, _ref=state_ref: _learner.set_train_folder(ray.get(_ref))
        )
        log(f"set train_title to all learners, res: {get_results(res)}")

        # 初始化各个learner的日志
        res = self.foreach_learner(lambda learner: learner.init_logger())
        log(f"foreach_learner: init_logger, res: {get_results(res)}")

        # 初始化参数 使用服务器的最新参数
        self._sync_learner_weights()

        # 初始化
        res = self.foreach_learner(lambda learner: learner.init_param_thread())
        log(f"foreach_learner: init_param_thread, res: {get_results(res)}")

    def _sync_learner_weights(self):
        # 获取服务器的参数，并更新到其他learner
        log('request server weights')
        params_list, info, version, need_warn_up = get_server_weights(self.train_title)

        # 解压参数
        # _params_dict = self.param_compressor.decompress_params_dict(params_list, info)
        _params_dict_np = self.get_weights()['default_policy']
        _params_dict = OrderedDict()
        for k, v in _params_dict_np.items():
            _params_dict[f'module.{k}'] = torch.from_numpy(v.copy())
        self.param_compressor.decompress(params_list, info, _params_dict)
        
        # 更新参数到所有learner
        log(f"set weights to all learners, version: {version}")
        if self.is_local:
            self._learner.module._rl_modules['default_policy'].load_state_dict(_params_dict)
        else:
            state_ref = ray.put(_params_dict)
            res = self.foreach_learner(
                lambda _learner, _ref=state_ref: _learner.module._rl_modules['default_policy'].load_state_dict(ray.get(_ref))
            )

        res = self.foreach_learner(lambda learner: learner.set_weights_version(version))
        log(f"set version to all learners, res: {get_results(res)}")

        res = self.foreach_learner(lambda learner: learner.set_need_warn_up(int(need_warn_up)))
        log(f"set need_warn_up to all learners, res: {get_results(res)}")

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
        self.foreach_learner(lambda learner: learner.init_shared_param())
        log(f"foreach_learner: init shared param, res: {get_results(res)}")

    def stop_extra_process(self):
        """停止额外进程"""
        res = self.foreach_learner(lambda learner: learner.stop_param_thread())
        log(f"foreach_learner: stop_param_thread, res: {get_results(res)}")

class AsyncProcessQueueReader_grad_param(AsyncProcessQueueReader):
    """
    异步进程队列读取器
    用于转发 进程任务 到 事件循环
    """
    def __init__(self, queue, grad_q):

        self.queue = queue
        self._loop = None
        self._thread = None
        self._running = False
        self._stop = False

        # 使用传入的队列
        self.grad_q = grad_q

    def _reader_thread(self):
        """后台读取线程"""
        while not self._stop:
            try:
                # 使用较短的超时以便能够响应停止信号
                item = self.queue.get(timeout=0.1)

                # 推送到队列
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

        # 共享参数
        self.shared_param = None

        # 客户端 id, 0表示与参数服务器通信
        self.client_id = 0

        # 更新计数
        self.update_count = 0

        self.version = 0
        self.need_warn_up = False

        # learner 之间的同步
        self.sync_learner_event = Event(name=f'_sync_learner_event')
        self.sync_learner_param_event = Event(name=f'_sync_learner_param_event')

        # learner 的个数(gpu数)
        self.num_learners = match_num_processes()

        # 跳过第一轮的等待参数，加速训练
        self.skiped = False 

        ####################
        # 只有主 learner 需要初始化
        ####################
        # 时间队列(与协程进程通讯)
        self.task_queue = None
        self.grads_count = 0
        self.update_param_count = 0
        # 用于初始化共享参数
        self.params_dict = None
        self.grad_params_value_shape = None
        # 是否处于训练阶段
        self.is_training_event = None
        # stop event
        self.stop_event = None
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
            self.params_dict = OrderedDict()
            for k,v in params_dict.items():
                self.params_dict[k] = v.cpu()
            self.grad_params_value_shape = [i.shape for i in grad_params_dict.values()]
        # 获取共享参数
        self.shared_param = SharedParam(params_dict, grad_params_dict, create=False)
        log(f"[{self.client_id}] init_shared_param done")
        return True

    def init_param_thread(self):
        if self.client_id == 0:
            # 主learner
            self.task_queue = multiprocessing.Queue(maxsize=GRAD_BATCH_SIZE)

            # 是否处于训练阶段
            self.is_training_event = Event(name=f'_is_training_event')

            # stop event
            self.stop_event = Event(name=f'_stop_loop_event')

            # 在新进程中运行事件循环
            self.event_loop_process = multiprocessing.Process(
                target=self._run_event_loop_process, 
                args=(
                    self.task_queue, self.train_title, self.train_folder, self.client_id, self.params_dict, self.grad_params_value_shape,
                    self.version, self.need_warn_up
                )
            )
            self.event_loop_process.start()
        log(f"[{self.client_id}] init_param_thread done")
        return True

    def stop_param_thread(self):
        if self.client_id == 0:
            # 停止事件循环
            log(f"[{self.client_id}] stop_param_thread")
            self.stop_event.set()
            # 等待子进程退出
            self.event_loop_process.join(timeout=5)  # 给子进程一些时间来退出
            if self.event_loop_process.is_alive():
                log(f"[{self.client_id}] Force terminating the process...")
                self.event_loop_process.terminate()

            self.event_loop_process.join()  # 确保资源被释放
        return True

    @staticmethod
    def _run_event_loop_process(task_queue, train_title, train_folder, client_id, params_dict, grad_params_value_shape, version, need_warn_up):
        """在新进程中运行事件循环"""
        # 初始化日志
        init_logger(train_title, home=train_folder, timestamp=False)

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

        grad_coroutine_num = 1
        
        # 梯度事件队列
        grad_q = AsyncQueue(maxsize=task_queue._maxsize)
        log(f'forwarding queue: {grad_q._maxsize}')

        # 独立线程转发 进程任务
        apqr = AsyncProcessQueueReader_grad_param(task_queue, grad_q)
        apqr.start(loop)

        # 进程池,用于压缩/加压等计算密集任务计算
        process_pool = ProcessPoolExecutor(max_workers=GRAD_BATCH_SIZE)  # 可以根据需要调整进程数
    
        # 启动协程任务
        tasks = []
        for _idx in range(grad_coroutine_num):
            task = loop.create_task(ClientPPOTorchLearner.grad_coroutine(_idx, info_data, process_pool, grad_q))
            tasks.append(task)

        # 启动参数协程
        task = loop.create_task(ClientPPOTorchLearner.param_coroutine(info_data, process_pool, params_dict, grad_params_value_shape)) 
        tasks.append(task)

        # 启动停止事件循环协程
        task = loop.create_task(ClientPPOTorchLearner.stop_loop_event())
        tasks.append(task)

        # # debug
        # loop.set_debug(True)

        # 运行事件循环
        loop.run_forever()

    @staticmethod
    async def stop_loop_event():
        """停止事件循环"""
        stop_event = Event(name=f'_stop_loop_event')
        while True:
            if stop_event.is_set():
                break
            await asyncio.sleep(1)

        # 停止事件循环
        loop = asyncio.get_event_loop()
        loop.stop()

    @staticmethod
    async def grad_coroutine(idx, info_data, process_pool, grad_q):

        """
        梯度协程
        """
        log(f"[{info_data.client_id}] grad_coroutine {idx} start")

        # 梯度压缩器
        gradient_compressor = DeepGradientCompression()

        # 统计耗时
        all_begin_time = 0
        total_count = 0
        total_wait_time = 0
        total_handle_time = 0

        # 存储一个batch的压缩梯度，一起推送
        batch_compressed_results = []
        while True:
            try:
                # 创建异步socket连接
                # reader, writer = await asyncio.open_connection(HOST, PORT)
                reader, writer = await connect_and_tune(HOST, PORT)
                # 发送连接验证
                await async_send_msg(writer, f'{CODE}')

                # 发送指令类型
                await async_send_msg(writer, f'{info_data.train_title}:update_gradients')

                send_count = 0
                while True:
                    # 获取汇总梯度 TODO 使用共享内存替代
                    grads = await grad_q.get()# 获取到1个梯度
                    begin_time = time.time()
                    if all_begin_time == 0:
                        all_begin_time = begin_time
                    q_size = grad_q.qsize()
                    log(f"[{idx}][{send_count}] grad handler begin, queue size: {q_size}")

                    # 在进程池中执行压缩操作
                    loop = asyncio.get_event_loop()
                    compressed_result = await loop.run_in_executor(
                        process_pool,
                        partial(gradient_compressor.compress, grads, info_data.need_warn_up)
                    )

                    compressed_grads, compress_info = compressed_result
                    batch_compressed_results.append((compressed_grads, compress_info))
                    log(f"[{idx}][{send_count}] compress gradients done, cost time: {int((time.time() - begin_time) * 1000)}ms")

                    if len(batch_compressed_results) == GRAD_BATCH_SIZE:
                        # 达到GRAD_BATCH_SIZE个梯度，发送梯度
                        data = pickle.dumps((batch_compressed_results, info_data.version))

                        send_begin_time = time.time()
                        await async_send_msg(writer, data)
                        log(f"[{idx}][{send_count}] send grads done({len(data)}), cost time: {int((time.time() - begin_time) * 1000)}ms")

                        # 等待回复
                        await wait_ack(reader)
                        wait_time = time.time() - send_begin_time
                        total_wait_time += wait_time
                        log(f"[{idx}][{send_count}] recv response done, cost time: {int((time.time() - begin_time) * 1000)}ms, wait time: {int(wait_time * 1000)}ms")

                        send_count += 1
                        total_count += 1
                        log(f"[{idx}][{send_count}] grad handler done, cost time: {int(time.time() - begin_time * 1000)}ms")

                        if total_count % 10 == 0:
                            # 每次发送梯度耗时(avg grad send time): 本机处理耗时(avg handle time) + 等待耗时(发送，确认返回, avg wait time) + 等待梯度耗时
                            # 网络传输耗时: 等待耗时(发送，确认返回, avg wait time) - 服务端处理耗时(服务端统计)
                            # avg grad send time: 925ms, avg wait time: 523ms, avg handle time: 171ms
                            # 优化空间:
                            #     平均等待梯度时间 = 925 -523-171 = 231ms > 取消强制参数同步的等待, 不断计算梯度，消除 平均等待梯度时间(只受限于梯度被处理的速度, 队列大小: GRAD_BATCH_SIZE)
                            #     网络传输耗时 = 523 - 15 = 508ms
                            #     压缩处理耗时 = 171ms
                            log(f"[{idx}] avg grad send time: {int(((time.time() - all_begin_time) / total_count) * 1000)}ms, avg wait time: {int(total_wait_time / total_count * 1000)}ms, avg handle time: {int((total_handle_time - total_wait_time) / total_count * 1000)}ms")
                            
                        # 清空
                        batch_compressed_results.clear()

                    # 统计耗时
                    handle_cost_time = time.time() - begin_time
                    total_handle_time += handle_cost_time

            except Exception as e:
                log(f"[{idx}] connect to server failed: \n{get_exception_msg()}")

                # 关闭连接
                try:

                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    @staticmethod
    async def param_coroutine(info_data, process_pool, params_dict, grad_params_value_shape):
        """
        获取服务器参数
        """
        log(f"[{info_data.client_id}] param_coroutine start")

        # 参数解压器
        # params_keys = list(params_dict.keys())
        # param_compressor = ParamCompressor(params_keys)
        param_compressor = IncrementalCompressor()

        sync_params_dict = copy.deepcopy(params_dict)

        # 是否处于训练阶段
        is_training_event = Event(name=f'_is_training_event')

        # 共享参数
        grad_params_dict = OrderedDict()
        for idx, shape in enumerate(grad_params_value_shape):
            grad_params_dict[f'grad_{idx}'] = torch.zeros(shape)
        shared_param = SharedParam(params_dict, grad_params_dict, create=False)

        # 统计拉取参数耗时
        begin_time = 0
        total_count = 0
        total_handle_time = 0

        log(f"param_coroutine init done")
        while True:
            try:
                # 创建异步socket连接
                # reader, writer = await asyncio.open_connection(HOST, PORT)
                reader, writer = await connect_and_tune(HOST, PORT)
                # 发送连接验证
                await async_send_msg(writer, f'{CODE}')
                log(f"param_coroutine connect to server")
                # 发送指令
                await async_send_msg(writer, f'{info_data.train_title}:wait_params'.encode())

                while True:
                    # 被动获取参数
                    # log(f"[{total_count}] wait params")
                    params_list, info, info_data.version, info_data.need_warn_up = await _async_wait_server_weights(reader)
                    total_count += 1
                    t = time.time()
                    if begin_time == 0:
                        begin_time = t
                    log(f"[{total_count}] recv params push")

                    # 当前是否处于训练阶段
                    if is_training_event.is_set():
                        # 增量解压操作
                        # log(f"[{total_count}] decompress params data")
                        param_compressor.decompress(params_list, info, sync_params_dict)
                        log(f"[{total_count}] decompress params done, cost: {int(1000*(time.time() - t))}ms")

                        # 更新共享参数
                        shared_param.set_param(sync_params_dict)
                        log(f"[{total_count}] set params to shared param, cost: {int(1000*(time.time() - t))}ms")

                        # log(f"[{total_count}] set params to shared param, sem_value: {shared_param.param_event.sem.value}")
                        # 触发共享参数更新事件
                        shared_param.param_event.clear_reset(1)
                        log(f"[{total_count}] update latest server weights done,  sem_value: {shared_param.param_event.sem.value}, cost: {int(1000*(time.time() - t))}ms")

                    # 处理完成回复
                    await ack(writer)

                    # 统计耗时
                    handle_cost_time = time.time() - t
                    total_handle_time += handle_cost_time

                    # 统计耗时
                    if total_count % 30 == 0:
                        # 本机接收后处理耗时(avg handle time)
                        # avg param push time: 928ms, avg handle time: 0ms
                        log(f"[{total_count}] avg param push time: {int(((time.time() - begin_time) / total_count) * 1000)}ms, avg handle time: {int(total_handle_time / total_count * 1000)}ms")

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
            # 清空梯度事件
            self.shared_param.grad_event.clear()
            # 清空learner参数同步事件
            self.sync_learner_param_event.clear()
            # 清空learner同步事件
            self.sync_learner_event.clear()

        # 计算梯度
        gradients_dict = super().compute_gradients(*args, **kwargs)
        # log(f'[{self.client_id}][{self.update_count}] gradients_dict ready')

        # nouse3 100 iter about 0.695H -89.66%
        if self.client_id == 0:
            # 主learner
            cpu_gradients = [v.cpu() for _, v in gradients_dict.items()]
            # log(f'[{self.client_id}][{self.update_count}] cpu_gradients ready')

            if self.gradient_sync_frequency > 1:
                # 累积梯度
                self.gradient_accumulator.add_gradients(cpu_gradients)
                # log(f'add gradients to gradient_accumulator')
                if self.update_count % self.gradient_sync_frequency == 0:
                    # 汇总梯度
                    # log(f'merge gradients')
                    gradients_to_send = self.gradient_accumulator.merge_gradients()
                    # 加入队列
                    self.task_queue.put(gradients_to_send)
            else:
                # 加入队列
                self.task_queue.put(cpu_gradients)

            self.grads_count += 1

            # 加入推送队列
            need_new_params = False
            if self.grads_count % GRAD_BATCH_SIZE == 0:
                need_new_params = True

            log(f'[{self.client_id}][{self.update_count}] task_queue: {self.task_queue.qsize()} / {self.task_queue._maxsize}')
            # log(f'[{self.client_id}][{self.update_count}] sync_learner_event: {self.sync_learner_event.is_set()}')
            # log(f'[{self.client_id}][{self.update_count}] sync_learner_param_event: {self.sync_learner_param_event.is_set()}')

            need_check_if_param_ready = True
            # need_check_if_param_ready = False
            # # 累计GRAD_BATCH_SIZE个梯度后，需要强制等待新的参数就位
            # if need_new_params:
            #     if not self.skiped:
            #         # 跳过第一个参数更新等待
            #         log(f'[{self.client_id}][{self.update_count}] force sync step, skiped first param update')
            #         self.skiped = True
            #     else:
            #         # 等待新的参数就位
            #         should_update_num = self.grads_count / GRAD_BATCH_SIZE - 1#跳过一个
            #         if self.update_param_count < should_update_num:
            #             t = time.time()
            #             log(f'[{self.client_id}][{self.update_count}] force sync step, wait new params ready, should_update_num: {should_update_num}, update_param_count: {self.update_param_count}')
            #             self.shared_param.param_event.wait()
            #             # 触发主learner的参数更新事件
            #             log(f'[{self.client_id}][{self.update_count}] force sync step, wait new params ready, cost: {int(1000*(time.time() - t))}ms')
            #             self.sync_learner_param_event.set(1)
            #             self.update_param_count += 1
            #         else:
            #             # 已经满足 should_update_num，无需强制等待
            #             log(f'[{self.client_id}][{self.update_count}] force sync step, should_update_num satisfied')
            #             need_check_if_param_ready = True

            # else:
            #     # 不需要强制同步参数的step, 
            #     log(f'[{self.client_id}][{self.update_count}] not force sync step')
            #     need_check_if_param_ready = True

            if need_check_if_param_ready:
                # 只检查是否有准备好的参数，而不强制等待
                log(f'[{self.client_id}][{self.update_count}] check if param ready: {self.shared_param.param_event.is_set()}')
                if self.shared_param.param_event.is_set():
                    # 触发主learner的参数更新事件
                    self.sync_learner_param_event.set(1)
                    self.update_param_count += 1

            # 触发主learner的梯度更新事件
            self.sync_learner_event.set(self.num_learners - 1)
        else:
            # 非主learner, 等待主learner的梯度更新事件
            # log(f'[{self.client_id}][{self.update_count}] wait sync_learner_event: {self.sync_learner_event.is_set()}')
            self.sync_learner_event.wait()

        # if self.client_id == 0:
        #     log(f'[{self.client_id}][{self.update_count}] compute_gradients done')
        # log(f'[{self.client_id}][{self.update_count}] compute_gradients done')

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

        if self.sync_learner_param_event.is_set():
            if self.client_id == 0:
                log(f'[{self.update_count}] apply new param to local')

            # 获取参数覆盖本地参数
            p = self.shared_param.get_weights()
            self.module._rl_modules['default_policy'].load_state_dict(p)

        # if self.client_id == 0:
        #     log(f'[{self.update_count}] apply_gradients done')

    def after_gradient_based_update(self, *args, **kwargs):
        # 重置
        self.update_count = 0
        if self.client_id == 0:
            # 训练结束, 清空参数事件
            self.shared_param.param_event.clear()
            # 训练结束, 参数协程不在处理新参数
            self.is_training_event.clear()
        return super().after_gradient_based_update(*args, **kwargs)
    
    def before_gradient_based_update(self, *args, **kwargs):
        if self.client_id == 0:
            # 训练开始，参数协程开始处理新参数
            self.is_training_event.set(1)
        return super().before_gradient_based_update(*args, **kwargs)

    def set_client_id(self, client_id):
        self.client_id = client_id
        return True

    def set_train_title(self, train_title):
        self.train_title = train_title
        return True

    def set_train_folder(self, train_folder):
        self.train_folder = train_folder
        return True

    def init_logger(self):
        init_logger(self.train_title, home=self.train_folder, timestamp=False)
        log(f"[{self.client_id}] init_logger done: {get_log_folder()}")
        return True
    
    def set_weights_version(self, version):
        log(f"[{self.client_id}] set_version: {version}")
        self.version = version
        return self.version

    def set_need_warn_up(self, need_warn_up):
        log(f"[{self.client_id}] set_need_warn_up: {need_warn_up}")
        self.need_warn_up = bool(need_warn_up)
        return True

if __name__ == '__main__':
    gradient_buffer_file = r"C:\Users\lh\Downloads\gradient_buffer_0.pkl"
    gradient_buffer = pickle.load(open(gradient_buffer_file, 'rb'))
    log(gradient_buffer)

    merged_gradients = ClientPPOTorchLearner.merge_gradients(gradient_buffer)
    log(merged_gradients)
