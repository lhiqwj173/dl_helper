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
import psutil
import torch
import queue, os, sys
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

from py_ext.tool import safe_share_memory, share_tensor, log, Event, get_exception_msg, get_log_folder, init_logger, Lock, safe_share_memory_queue

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
    def __init__(self, name, params_dict, create=True):
        """
        params_dict: 参数字典, 键为参数名, 值为 torch.Tensor
        create: 是否创建共享内存
        """
        self.create = create
        self._params_dict_np = {}
        self._params_dict = {}
        self.ssms = []
        self.name = name

        # 锁
        self.lock = Lock(name=f'_{self.name}_SharedParam_lock')

        # 共享事件 - 参数更新完毕
        self.param_event = Event(name=f'_{self.name}_SharedParam_event')
        if create:
            self.param_event.clear()

        # 共享参数内存 储存解压后的服务器参数 float32
        for k, v in params_dict.items():
            # v 是 torch.Tensor
            ssm = safe_share_memory(name=f'_{self.name}_SharedParam_{k}', size=v.numel() * v.element_size())
            shm = ssm.get()
            _v = np.ndarray(v.shape, dtype=np.float32, buffer=shm.buf)
            self._params_dict_np[k] = _v
            self._params_dict[k] = torch.from_numpy(_v)
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
        with self.lock:
            # 拷贝返回
            return copy.deepcopy(self._params_dict)

    def set_param(self, params_dict):
        """
        参数:
        params_dict: 参数字典, 键为参数名, 值为torch.Tensor
        """
        with self.lock:
            for k, v in params_dict.items():
                self._params_dict[k][:] = v.to(self._params_dict[k].device)[:]

class share_info:
    """
    共享信息类
    """
    def __init__(self, client_id=None, version=None, need_warn_up=None):
        self.CLIENT_ID, self.VERSION, self.NEED_WARN_UP = range(3)
        self._data = share_tensor(name=f'_share_info_data', shape=(3,), dtype='int64')
        self._lock = Lock(name=f'_share_info_lock_event')

        self.set(client_id, version, need_warn_up)

    def _get(self, idx):
        with self._lock:
            return self._data.data_np[idx]

    def set(self, client_id=None, version=None, need_warn_up=None):
        if client_id is not None:
            with self._lock:
                self._data.data_np[self.CLIENT_ID] = client_id
        if version is not None:
            with self._lock:
                self._data.data_np[self.VERSION] = version
        if need_warn_up is not None:
            with self._lock:
                self._data.data_np[self.NEED_WARN_UP] = need_warn_up

    @ property
    def client_id(self):
        return self._get(self.CLIENT_ID)

    @ property
    def version(self):
        return self._get(self.VERSION)

    @ property
    def need_warn_up(self):
        return self._get(self.NEED_WARN_UP)

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

        # 初始化客户端learner
        self._init_client_learner()

        log(f"ClientLearnerGroup init done")

    def _init_client_learner(self):
        """初始化客户端learner"""
        # 设置每个learner的train_title
        log(f"init_client_learner")

        # 设置 除第一个外 learner的 client_id > 不与参数服务器通信
        # 若有大于1个learner，则第二个learner分配解压参数，准备参数，否则仍由主learner解压
        second_learner_seted = False
        if self._worker_manager is not None:
            all_ids = self._worker_manager.actor_ids()
            if len(all_ids) > 1:
                remote_actor_ids = all_ids[1:]
                res = self.foreach_learner(lambda learner: learner.set_client_id(-1), remote_actor_ids = remote_actor_ids)
                log(f"set set_client_id to not main learners, res: {get_results(res)}")

                # 第二个learner分配解压参数，准备参数
                res = self.foreach_learner(lambda learner: learner.set_params_job(), remote_actor_ids = all_ids[1:2])
                log(f"set set_params_job to second learner, res: {get_results(res)}")
                second_learner_seted = True

        if not second_learner_seted:
            # 只有一个learner
            res = self.foreach_learner(lambda learner: learner.set_params_job())
            log(f"set set_params_job to main learner, res: {get_results(res)}")

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
        _params_dict_np = self.get_weights()['default_policy']
        _params_dict = OrderedDict()
        for k, v in _params_dict_np.items():
            _params_dict[f'module.{k}'] = torch.from_numpy(v.copy())
        IncrementalCompressor.decompress(params_list, info, _params_dict)
        
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

        # 创建共享参数
        self.shared_param_between_learner = SharedParam("learner", _params_dict, create=True)
        log(f"SharedParam init Done")

        # 初始化learner的共享参数
        res = self.foreach_learner(lambda learner: learner.init_shared_param())
        log(f"foreach_learner: init shared param, res: {get_results(res)}")

    def stop_extra_process(self):
        """停止额外进程"""
        res = self.foreach_learner(lambda learner: learner.stop_param_thread())
        log(f"foreach_learner: stop_param_thread, res: {get_results(res)}")

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

        # 共享参数
        self.shared_param_between_learner = None

        # 客户端 id, 0表示与参数服务器通信
        self.client_id = 0

        # 是否再本learner中进行参数解压准备
        self.ready_params_job = False

        # 更新计数
        self.update_count = 0

        self.version = 0
        self.need_warn_up = False

        # learner 之间的同步
        self.sync_learner_event = Event(name=f'_sync_learner_event')
        self.sync_learner_param_event = Event(name=f'_sync_learner_param_event')

        # learner 的个数(gpu数)
        self.num_learners = match_num_processes()

        # 梯度计数
        self.grads_count = 0

        # 用于解压参数的临时参数字典 / 初始化共享dump队列
        self.params_dict = None

        self.info_data = None

        ####################
        # 只有主 learner 需要初始化
        ####################
        # 时间队列(与协程进程通讯)
        self.task_queue = None
        self.grad_params_list = None
        self.grad_params_value_shape = None
        # stop event
        self.stop_event = None
        # 梯度压缩器
        self.gradient_compressor = None
        self.tatal_compress_cost = 0
        # 共享dump参数队列
        self.params_dump_q = None
        ####################
        # 只有主 learner 需要初始化
        ####################

        ####################
        # 只有次 learner 需要初始化，
        # 若只有一个learner，则次learner与主learner相同
        ####################
        self.update_param_count = 0
        ####################
        # 只有次 learner 需要初始化，
        # 若只有一个learner，则次learner与主learner相同
        ####################
    def init_shared_param(self):
        log(f"[{self.client_id}] init_shared_param")
        # 获取参数字典
        params_dict = self.module._rl_modules['default_policy'].state_dict()
        # 解压用的临时参数字典 / 初始化共享dump队列
        self.params_dict = OrderedDict()
        for k,v in params_dict.items():
            self.params_dict[k] = v.clone().detach().cpu()

        # 获取梯度字典
        grad_params_dict = self._params
        if self.client_id == 0:
            self.grad_params_list = []
            for k,v in grad_params_dict.items():
                self.grad_params_list.append(v.clone().detach().cpu())
            self.grad_params_value_shape = [i.shape for i in grad_params_dict.values()]

        if self.ready_params_job:
            _temp_dump_data = pickle.dumps(([v for _, v in params_dict.items()], {'full': True}, np.int64(0), np.int64(0)))
            log(f"[{self.client_id}] init params_dump_q, buffer size: {len(_temp_dump_data)}")
            self.params_dump_q = safe_share_memory_queue('param_coroutine_dump_q', len(_temp_dump_data), 4)
            self.params_dump_q.clear()

        self.shared_param_between_learner = SharedParam("learner",params_dict, create=False)
        log(f"[{self.client_id}] init_shared_param done")
        return True

    def init_param_thread(self):
        # 共享信息
        self.info_data = share_info()

        if self.client_id == 0:
            # 主learner
            # 梯度压缩器
            self.gradient_compressor = DeepGradientCompression()

            # 获取一个梯度不压缩数据, 作为队列大小
            _g, _info = self.gradient_compressor.compress(self.grad_params_list, True)
            _size = len(pickle.dumps((_g, _info)))
            self.gradient_compressor.clear()# 清理
            log(f"[{self.client_id}] init grad_q, buffer size: {_size}")
            self.task_queue = safe_share_memory_queue('grad_data_info_q', _size, 4, len(pickle.dumps(np.int64(0))))# 额外一个 np.int64 用于保存梯度版本
            self.task_queue.clear()

            # stop event
            self.stop_event = Event(name=f'_stop_loop_event')

            # 在新进程中运行事件循环
            self.event_loop_process = multiprocessing.Process(
                target=self._run_event_loop_process, 
                args=(
                    self.train_title, self.train_folder, self.client_id, self.params_dict, self.grad_params_value_shape,
                    self.version, self.need_warn_up,
                    _size,
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
            self.event_loop_process.join(timeout=10)  # 给子进程一些时间来退出
            if self.event_loop_process.is_alive():
                log(f"[{self.client_id}] Force terminating the process...")
                self.event_loop_process.terminate()
                self.event_loop_process.join()  # 确保资源被释放
        return True

    @staticmethod
    def _run_event_loop_process(train_title, train_folder, client_id, params_dict, grad_params_value_shape, version, need_warn_up, grad_q_size):
        """在新进程中运行事件循环"""
        log(f'_run_event_loop_process')

        # 初始化日志
        init_logger(train_title, home=train_folder, timestamp=False)

        # 共享梯度队列
        log(f"[{client_id}] init grad_q, buffer size: {grad_q_size}")
        grad_q = safe_share_memory_queue('grad_data_info_q', grad_q_size, 4, len(pickle.dumps(np.int64(0))))

        # 事件循环
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 如果当前线程没有事件循环，则创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    
        info_data = share_info(client_id, version, need_warn_up)

        async_share_data = {
            'stop': False,
        }

        grad_coroutine_num = 1
        assert grad_coroutine_num == 1, f'grad_coroutine_num: {grad_coroutine_num} must be 1 for now'
        
        # 获取设备ip
        _ip = requests.get('https://api.ipify.org').text

        # 启动协程任务
        tasks = []
        for _idx in range(grad_coroutine_num):
            task = loop.create_task(ClientPPOTorchLearner.grad_coroutine(async_share_data, _ip, train_title, _idx, info_data, grad_q))
            tasks.append(task)

        # 启动参数协程
        task = loop.create_task(ClientPPOTorchLearner.param_coroutine(async_share_data, _ip, train_title, info_data, params_dict, grad_params_value_shape)) 
        tasks.append(task)

        # 启动停止事件循环协程
        task = loop.create_task(ClientPPOTorchLearner.stop_loop_event(async_share_data, ))
        tasks.append(task)

        try:
            # 等待所有任务完成或者被取消
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except Exception as e:
            log(f"Event loop error: {str(e)}")
        finally:
            # 关闭事件循环
            loop.close()
            log("Event loop closed")

        log(f"event loop process done")

    @staticmethod
    async def stop_loop_event(async_share_data):
        """负责监控停止事件"""
        stop_event = Event(name=f'_stop_loop_event')
        while True:
            if stop_event.is_set():
                break
            await asyncio.sleep(1)

        async_share_data['stop'] = True
        log(f"stop_loop_event done")

    @staticmethod
    async def grad_coroutine(async_share_data, ip, train_title, idx, info_data, grad_q):
        """
        梯度协程
        """
        log(f"[{info_data.client_id}] grad_coroutine {idx} start")

        # 统计耗时
        all_begin_time = 0
        total_count = 0
        total_wait_time = 0
        total_handle_time = 0
        mean_send_size = 0
        total_version_diff = 0

        # 存储一个batch的压缩梯度，一起推送
        batch_compressed_results = []
        try:
            # 创建异步socket连接
            log(f'[{idx}] grad_coroutine connect to server')
            # reader, writer = await asyncio.open_connection(HOST, PORT)
            reader, writer = await connect_and_tune(HOST, PORT)
            log(f'[{idx}] grad_coroutine connect to server done')
            # 发送连接验证
            await async_send_msg(writer, f'{CODE}_{ip}')
            log(f'[{idx}] grad_coroutine send CODE_IP done')
            # 发送指令类型
            await async_send_msg(writer, f'{train_title}:update_gradients')
            log(f'[{idx}] grad_coroutine send CMD done')

            send_count = 0
            last_None = False
            while True:
                # 停止事件
                if async_share_data['stop']:
                    break

                # 获取到1个发送数据
                # (dump(bytes), extra_data(int64))
                try:
                    send_data = grad_q.get(block=False)
                except Empty:
                    if not last_None:
                        log(f'[{idx}] grad_coroutine get None from queue')
                        last_None = True
                    await asyncio.sleep(0.001)
                    continue

                if last_None:
                    last_None = False

                begin_time = time.time()
                if all_begin_time == 0:
                    all_begin_time = begin_time
                q_size = grad_q.qsize()
                batch_compressed_results.append(send_data)
                _batch_size = len(batch_compressed_results)
                log(f"[{idx}][{send_count}] add send data to batch, batch size: {_batch_size}, queue size: {q_size}")

                if _batch_size == GRAD_BATCH_SIZE:
                    # 达到GRAD_BATCH_SIZE个梯度，发送梯度

                    # 统计版本diff info_data.version
                    current_version = info_data.version
                    if GRAD_BATCH_SIZE > 1:
                        diff = [current_version - i[1] for i in batch_compressed_results]
                        diff = sum(diff)
                        data = pickle.dumps(batch_compressed_results)
                    else:
                        data = pickle.dumps(batch_compressed_results[0])
                        diff = current_version - batch_compressed_results[0][1]
                    total_version_diff += diff

                    # 发送梯度
                    send_begin_time = time.time()
                    await async_send_msg(writer, data)
                    send_size = len(data)
                    mean_send_size = (mean_send_size * send_count + send_size) / (send_count + 1)
                    log(f"[{idx}][{send_count}] send data done({send_size}), cost time: {int((time.time() - begin_time) * 1000)}ms")

                    # # 等待回复
                    # await wait_ack(reader)
                    # log(f"[{idx}][{send_count}] recv response done, cost time: {int((time.time() - begin_time) * 1000)}ms, wait time: {int(wait_time * 1000)}ms")

                    wait_time = time.time() - send_begin_time
                    total_wait_time += wait_time

                    send_count += 1
                    total_count += 1

                    if total_count % 10 == 0:
                        # 每次发送梯度耗时(avg grad send time): 本机处理耗时(avg handle time) + 等待耗时(发送，确认返回, avg wait time) + 等待梯度耗时
                        # 网络传输耗时: 等待耗时(发送，确认返回, avg wait time) - 服务端处理耗时(服务端统计)

                        # ROUND 0
                        # avg grad send time: 925ms, avg wait time: 523ms, avg handle time: 171ms
                        # 优化空间:
                        #     平均等待梯度时间 = 925 -523-171 = 231ms > 取消强制参数同步的等待, 不断计算梯度，消除 平均等待梯度时间(只受限于梯度被处理的速度, 队列大小: GRAD_BATCH_SIZE)
                        #     网络传输耗时 = 523 - 15 = 508ms
                        #     压缩处理耗时 = 171ms

                        # ROUND 4 GRAD_BATCH_SIZE=1
                        # avg grad send time: 43ms, avg wait time: 0ms, avg handle time: 0ms, mean send size: 40511
                        # 优化空间:
                        #     平均等待梯度时间 = 43 - 0 - 0 = 43ms  > 目标达成
                        #     网络传输耗时 = 0ms                    > 目标达成
                        #     压缩处理耗时 = 17ms(主learner完成)      > 目标达成

                        # ROUND 5 GRAD_BATCH_SIZE=1 同步训练
                        # avg grad send time: 738ms, avg wait time: 0ms, avg handle time: 0ms, mean send size: 41529, mean version diff: 0.00
                        # 优化空间:
                        #     平均等待梯度时间 = 738 - 0 - 0 = 738ms (发送梯度/等待参数推送)
                        log(f"[{idx}] avg grad send time: {int(((time.time() - all_begin_time) / total_count) * 1000)}ms, avg wait time: {int(total_wait_time / total_count * 1000)}ms, avg handle time: {int((total_handle_time - total_wait_time) / total_count * 1000)}ms, mean send size: {int(mean_send_size)}, mean version diff: {(total_version_diff / (total_count * GRAD_BATCH_SIZE)):.2f}")

                    # 清空
                    batch_compressed_results.clear()

                # 统计耗时
                handle_cost_time = time.time() - begin_time 
                total_handle_time += handle_cost_time
                log(f"[{idx}][{send_count}] grad handler done, handle cost time: {int(handle_cost_time * 1000)}ms")

        except Exception as e:
            log(f"[{idx}] grad_coroutine connect to server failed: \n{get_exception_msg()}")
            # 关闭连接
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

        log(f"[{idx}] grad_coroutine done")

    @staticmethod
    async def param_coroutine(async_share_data, ip, train_title, info_data, params_dict, grad_params_value_shape):
        """
        获取服务器参数
        """
        log(f"param_coroutine start")

        # 共享队列
        _temp_dump_data = pickle.dumps(([v for _, v in params_dict.items()], {'full': True}, np.int64(0), np.int64(0)))
        log(f"[{info_data.client_id}] init params_dump_q, buffer size: {len(_temp_dump_data)}")
        params_dump_q = safe_share_memory_queue('param_coroutine_dump_q', len(_temp_dump_data), 4)

        # 统计拉取参数耗时
        begin_time = 0
        total_count = 0
        total_handle_time = 0

        log(f"param_coroutine init done")
        try:
            # 创建异步socket连接
            log(f'param_coroutine connect to server')
            # reader, writer = await asyncio.open_connection(HOST, PORT)
            reader, writer = await connect_and_tune(HOST, PORT)
            log(f'param_coroutine connect to server done')
            # 发送连接验证
            await async_send_msg(writer, f'{CODE}_{ip}')
            log(f'param_coroutine send CODE_IP done')
            # 发送指令类型
            await async_send_msg(writer, f'{train_title}:wait_params')
            log(f'param_coroutine send CMD done')
            last_v = 0
            while True:
                # 停止事件
                if async_share_data['stop']:
                    break

                try:
                    dump_data = await _async_wait_server_weights(reader, timeout=5, loads=False)
                except TimeoutError:
                    continue

                total_count += 1
                t = time.time()
                if begin_time == 0:
                    begin_time = t
                log(f"[{total_count}] recv params push")

                # 加入队列
                while True:
                    try:
                        params_dump_q.put(dump_data, block=False)
                        break
                    except:
                        await asyncio.sleep(0.001)

                # 统计耗时
                handle_cost_time = time.time() - t
                total_handle_time += handle_cost_time

                # 统计耗时
                if total_count % 30 == 0:
                    # 本机接收后处理耗时(avg handle time)
                    # avg param push time: 928ms, avg handle time: 0ms
                    # avg param push time: 616ms, avg handle time: 1ms
                    log(f"[{total_count}] avg param push time: {int(((time.time() - begin_time) / total_count) * 1000)}ms, avg handle time: {int(total_handle_time / total_count * 1000)}ms")

        except Exception as e:
            log(f"param_coroutine connect to server failed: \n{get_exception_msg()}")
            # 关闭连接
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

        log(f"param_coroutine done")

    def compute_gradients(self, *args, **kwargs):
        self.update_count += 1

        if self.client_id == 0:
            # report_memory_usage(f'[{self.update_count}]')
            # 清空learner参数同步事件
            self.sync_learner_param_event.clear()
            # 清空learner同步事件
            self.sync_learner_event.clear()

        # 计算梯度
        g = super().compute_gradients(*args, **kwargs)
        # log(f'[{self.client_id}][{self.update_count}] compute_gradients done')
        return g
    
    def apply_gradients(self, *args, **kwargs):
        self.grads_count += 1

        try:
            if self.client_id == 0:
                # 主learner，压缩梯度
                cpu_gradients = [v.grad.cpu() for _, v in self._params.items()]
                log(f'[{self.client_id}][{self.update_count}] cpu_gradients ready')

                # if self.grads_count == 0:
                #     # 保存第一个梯度，验证梯度是否正确
                #     pickle.dump((cpu_gradients, self.info_data.need_warn_up), open(f'compute_gradient.pkl', 'wb'))

                # 梯度压缩
                t = time.time()
                log(f'[{self.client_id}][{self.update_count}] compress gradients begin, need_warn_up: {self.info_data.need_warn_up}')
                compressed_grads, compress_info = self.gradient_compressor.compress(cpu_gradients, self.info_data.need_warn_up)
                cost = int((time.time() - t) * 1000)
                self.tatal_compress_cost += cost

                # if self.grads_count == 0:
                #     # 保存第一个压缩梯度，验证梯度是否正确
                #     pickle.dump((compressed_grads, compress_info), open(f'compress_gradient.pkl', 'wb'))

                # compress gradients done, cost time: 19ms, avg cost: 18ms
                # compress gradients done, cost time: 17ms, avg cost: 17ms
                log(f'[{self.client_id}][{self.update_count}] compress gradients done, cost time: {cost}ms, avg cost: {int(self.tatal_compress_cost / self.grads_count)}ms')

                # 加入队列
                self.task_queue.put(pickle.dumps((compressed_grads, compress_info)), extra_data=np.int64(self.info_data.version))
                # self.task_queue.put((pickle.dumps((compressed_grads, compress_info)), np.int64(self.info_data.version)))
                log(f'[{self.client_id}][{self.update_count}] task_queue: {self.task_queue.qsize()} / {self.task_queue._maxsize}')

            if self.ready_params_job:
                # 需要准备参数的learner，解压参数
                N = 0
                # 累计 N 个梯度后，需要强制等待新的参数就位
                if N > 0 and self.grads_count % N == 0:
                    # 等待新的参数就位
                    t = time.time()
                    log(f'[{self.client_id}][{self.update_count}] force sync step, wait new params ready')
                    dump_data = self.params_dump_q.get()
                else:
                    # 不需要强制同步参数的step
                    try:
                        dump_data = self.params_dump_q.get(block=False)
                    except Empty:
                        dump_data = None
                    log(f'[{self.client_id}][{self.update_count}] not force sync step, check if param ready: {dump_data is not None}')
                    
                if dump_data is not None:
                    # loads
                    weights, info, version, need_warn_up = pickle.loads(dump_data)
                    # 更新共享信息
                    self.info_data.set(version, need_warn_up)
                    # 解压到 self.params_dict
                    IncrementalCompressor.decompress(weights, info, self.params_dict)
                    # 拷贝到 shared_param_between_learner
                    self.shared_param_between_learner.set_param(self.params_dict)
                    # 触发主learner的参数更新事件
                    self.sync_learner_param_event.set(1)
                    self.update_param_count += 1
                    log(f'[{self.client_id}][{self.update_count}] decompress and ready new params done')

                # 等解压参数（检查参数更新）完毕，全部learner可以继续运行
                self.sync_learner_event.set(self.num_learners - 1)
            else:
                # 非解压参数的learner，等解压参数（检查参数更新）完毕
                self.sync_learner_event.wait()

            if self.sync_learner_param_event.is_set():
                # if self.client_id == 0:
                #     log(f'[{self.update_count}] apply new param to local')

                # 获取参数覆盖本地参数
                p = self.shared_param_between_learner.get_weights()
                self.module._rl_modules['default_policy'].load_state_dict(p)
                log(f'[{self.client_id}][{self.update_count}] apply new param to local')

        except Exception as e:
            log(f'[{self.client_id}][{self.update_count}] apply_gradients failed: \n{get_exception_msg()}')
            raise e

    def after_gradient_based_update(self, *args, **kwargs):
        # 重置
        self.update_count = 0
        return super().after_gradient_based_update(*args, **kwargs)
    
    def before_gradient_based_update(self, *args, **kwargs):
        return super().before_gradient_based_update(*args, **kwargs)

    def set_client_id(self, client_id):
        self.client_id = client_id
        return True
    
    def set_params_job(self):
        self.ready_params_job = True
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
