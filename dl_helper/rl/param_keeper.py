import torch, time, math
import multiprocessing
from multiprocessing.queues import Empty
import asyncio
import gymnasium as gym

import numpy as np
import pickle, requests
from typing import Dict, Any
from collections import deque
import copy
from dl_helper.rl.socket_base import CODE, PORT, send_msg, recv_msg
from dl_helper.rl.rl_utils import ParamCompressor
from dl_helper.deep_gradient_compression import DeepGradientCompression
from dl_helper.param_compression import IncrementalCompressor
from dl_helper.tool import AsyncLockWithLog, LockWithLog, report_memory_usage, AsyncProcessEventReader
from dl_helper.rl.socket_base import async_send_msg, async_recv_msg, GRAD_BATCH_SIZE, ack, wait_ack

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.tune.registry import _global_registry, ENV_CREATOR

from py_ext.tool import log, share_tensor_list, share_tensor, get_exception_msg

class AsyncRLParameterServer:
    def __init__(self,
                 config, env
        ):
        """分布式强化学习参数服务器
        
        Args:
            config: RLlib 配置
            env: 环境
        """
        self.learner = config.build_learner(env=env)
        self.learner.build()
        self.ver = 0
        # self.total_client_version_diff = 0
        # self.total_count = 0

    def apply_gradients(self, gradients_list, client_version):
        """
        更新参数
        gradients_list: 梯度列表, 键为参数名, 值为torch.Tensor
        """
        # log(f'gradients_list length: {len(gradients_list)}')
        params = self.learner._params
        for idx, k in enumerate(params.keys()):
            params[k].grad = gradients_list[idx].to(self.learner._device)
        
        self.learner.apply_gradients({})
        self.ver += 1
        # self.total_client_version_diff += self.ver - client_version
        # self.total_count += 1

    def get_gradients_params(self):
        """获取计算梯度的参数"""
        return copy.deepcopy(self.learner._params)
    
    def get_weights(self):
        """获取参数"""
        # return (self.learner.get_state(components=COMPONENT_RL_MODULE)['rl_module']['default_policy'], self.ver)
        weights = self.learner.module._rl_modules['default_policy'].state_dict()
        return weights, self.ver

    # def get_mean_version_diff(self):
    #     """获取平均版本差"""
    #     return self.total_client_version_diff / self.total_count

class ExperimentHandler:
    """处理单个实验的类"""

    def __init__(self, train_title, config, debug=False, grad_warm_up_steps=10
    ):
        """
        train_title: 训练标题
        config: RLlib 配置
        """
        # 训练标题
        self.train_title = train_title
        self.debug = debug

        # 客户端 IP/id
        self.client_ip_ids = {}

        # 梯度缓存数量
        self.grad_cache_size = GRAD_BATCH_SIZE * 10

        # 版本号
        self.version = 0

        # grad allow version diff
        self.grad_allow_version_diff = 30

        # 梯度预热步数
        self.grad_warm_up_steps = grad_warm_up_steps

        # 参数推送频率
        self.push_params_interval = GRAD_BATCH_SIZE # 每 push_params_interval 步推送一次参数

        # 共享梯度队列
        self.client_nums_q = multiprocessing.Queue()
        # 梯度数据经过稀疏化，形状
        self.gradients_info_share_q = multiprocessing.Queue()
        # 共享参数信息
        self.params_info_share = share_tensor(f'{train_title}_params_info_share', (2,), 'int64')
        # 共享参数float32 ver 队列
        self.params_float32_ver_share_q = multiprocessing.Queue()
        # 共享梯度锁
        self.share_gradients_lock = multiprocessing.Lock()
        # 共享参数锁
        self.share_params_lock = multiprocessing.Lock()
        # 共享参数float32锁
        self.share_params_float32_lock = multiprocessing.Lock()

        # 共享数据新增通知
        self.share_data_new_event = multiprocessing.Event()

        # 添加梯度锁
        self.gradients_add_lock = asyncio.Lock()

        # 独立线程转发 进程任务
        self.ready_params_event = multiprocessing.Event()
        self.aper = AsyncProcessEventReader(self.ready_params_event)
        
        # 启动计算进程
        self.p = multiprocessing.Process(target=ExperimentHandler.gpu_most_task, args=(
            train_title, self.client_nums_q, self.gradients_info_share_q, self.params_float32_ver_share_q,self.share_data_new_event, 
            self.share_gradients_lock, self.share_params_lock, self.share_params_float32_lock,
            config,
            self.debug,
            self.grad_cache_size,
            self.grad_warm_up_steps,
            self.ready_params_event,
        ))
        self.p.start()

        # 等待接受 参数的形状列表
        # 用于初始化共享数据
        _simple_params, _simple_grad_params = self.gradients_info_share_q.get()

        # 共享梯度列表
        self.gradients_cache_share_full = []# 全梯度
        self.gradients_cache_share = []# 用于压缩的梯度使用
        # 共享参数, 只需要维护一份最新的数据
        self.params_cache_share = []
        # 初始化共享梯度
        for idx, (_shape_full, _shape) in enumerate(_simple_grad_params):
            self.gradients_cache_share_full.append(share_tensor_list(f'{self.train_title}_gcsfull_{idx}', _shape_full, 'float32', self.grad_cache_size, debug=self.debug))
            self.gradients_cache_share.append(share_tensor_list(f'{self.train_title}_gcs_{idx}', _shape, 'float32', self.grad_cache_size, debug=self.debug))
        # 初始化共享参数
        for idx, _shape in enumerate(_simple_params):
            # for debug
            self.params_cache_share.append(share_tensor(f'{self.train_title}_pcs_{idx}', _shape, 'float32'))
            # self.params_cache_share.append(share_tensor(f'{self.train_title}_pcs_{idx}', (math.prod(_shape),), 'int8'))
    
        # 增量参数压缩器
        self.params_compressor = IncrementalCompressor()

        # 允许验证的客户端ip
        self.need_val_ip = 0
        # 允许验证的时间戳
        self.need_val_timestamp = 0

        # 客户端数量
        self.client_nums = 0

    def __del__(self):
        self.p.terminate()

    @staticmethod
    def gpu_most_task(
        train_title, client_nums_q, gradients_info_share_q, params_float32_ver_share_q, share_data_new_event, 
        share_gradients_lock, share_params_lock, share_params_float32_lock,
        config, 
        debug,
        grad_cache_size, 
        grad_warm_up_steps,
        ready_params_event,
    ):
        """
        负责 梯度解压/梯度应用更新参数
        """
        def ready_params(param_server, lock, params_cache_share, params_info_share, need_warn_up): 
            """获取参数copy到共享参数中"""
            log(f'[CG] ready params begin')
            # 获取模型参数
            # weights： 参数字典[torch.Tensor]
            weights, version = param_server.get_weights()
            with lock:
                for idx, (k, v) in enumerate(weights.items()):
                    params_cache_share[idx].data[:] = v[:]

                params_info_share.data[0] = version
                params_info_share.data[1] = need_warn_up

            # 通知参数发送任务
            ready_params_event.set()
            log(f'[CG] ready params done v: {version}')

        log(f'[CG]{train_title} calculate gpu init')

        # 共享参数信息
        params_info_share = share_tensor(f'{train_title}_params_info_share', (2,), 'int64')
        
        # 计算步数
        step_count = 0

        # 参数服务器
        config = config.learners(    
            num_learners=1,
            num_gpus_per_learner=0,
            num_cpus_per_learner=0.5,
        )
        env_specifier = config.env
        if _global_registry.contains(ENV_CREATOR, env_specifier):
            # 注册的环境
            env = _global_registry.get(ENV_CREATOR, env_specifier)()
        else:
            # gym 环境
            env = gym.make(env_specifier)
        param_server = AsyncRLParameterServer(config, env)
        _params_dict = param_server.get_weights()[0] 
        _grad_params_dict = param_server.get_gradients_params()

        # 梯度压缩器
        gradient_compressor = DeepGradientCompression()

        # 共享梯度
        gradients_cache_share = []
        gradients_cache_share_full = []
        # 计算用临时梯度
        gradients_cache_temp = []
        gradients_cache_temp_full = []
        # 计算用临时梯度信息
        gradients_cache_info_temp = []
        # 临时梯度的数量(待应用)
        temp_length = 0

        # 共享参数
        params_cache_share = []
        # 初始化共享参数
        _simple_params = []
        for idx, (k, v) in enumerate(_params_dict.items()):
            log(f'{train_title} init params share, idx: {idx}, name: {k}, shape: {v.shape}')
            _shape = v.shape
            # for debug
            params_cache_share.append(share_tensor(f'{train_title}_pcs_{idx}', _shape, 'float32'))
            # params_cache_share.append(share_tensor(f'{train_title}_pcs_{idx}', (math.prod(v.shape),), 'int8'))
            _simple_params.append(_shape)

        # 初始化共享梯度
        _simple_grad_params = []
        for idx, (k, v) in enumerate(_grad_params_dict.items()):
            _compress_shape = gradient_compressor.compress_shape(v.shape)
            log(f'{train_title} init gradients share, idx: {idx}, shape: {v.shape}, compress shape: {_compress_shape}')
            gradients_cache_share.append(share_tensor_list(f'{train_title}_gcs_{idx}', _compress_shape, 'float32', grad_cache_size, debug=debug))
            gradients_cache_share_full.append(share_tensor_list(f'{train_title}_gcsfull_{idx}', v.shape, 'float32', grad_cache_size, debug=debug))
            gradients_cache_temp.append(gradients_cache_share[idx].get_blank_same_data_local())
            gradients_cache_temp_full.append(gradients_cache_share_full[idx].get_blank_same_data_local())
            _simple_grad_params.append((v.shape, _compress_shape))

        # 初始化一个最新的参数/info
        need_warn_up = grad_warm_up_steps > step_count
        ready_params(param_server, share_params_lock, params_cache_share, params_info_share, need_warn_up)
        
        # 回传 参数形状列表 
        # 回传后，共享参数以及初始化完成
        gradients_info_share_q.put((_simple_params, _simple_grad_params))

        # 版本差异统计
        total_client_version_diff = 0
        total_count = 0

        # 客户端数量
        client_nums = 0

        log(f'{train_title} calculate most start')
        while True:

            try:
                log(f'[CG]{train_title} wait gradients event')
                share_data_new_event.wait()
                share_data_new_event.clear()
                t = time.time()

                # 获取最新客户端数量
                try:
                    new_client_nums = client_nums_q.get(block=False)
                    client_nums = new_client_nums
                except Empty:
                    pass

                log(f'[CG]{train_title} active')
                # with LockWithLog(share_gradients_lock, log, '[CG]'):
                with share_gradients_lock:
                    g_length = gradients_cache_share[0].size()
                    fullg_length = gradients_cache_share_full[0].size()
                    temp_length = g_length + fullg_length

                    data_client_nums = temp_length // GRAD_BATCH_SIZE
                    if data_client_nums < client_nums:
                        log(f'{train_title} not enough gradients, keep wait({client_nums - data_client_nums} clients)')
                        continue

                    # 拷贝梯度到临时梯度
                    if g_length:
                        for idx, _g in enumerate(gradients_cache_share):
                            _g.all_copy_slice(gradients_cache_temp[idx], 0)
                    if fullg_length:
                        for idx, _g in enumerate(gradients_cache_share_full):
                            _g.all_copy_slice(gradients_cache_temp_full[idx], 0)
                    # 获取全部的梯度信息
                    for idx in range(temp_length):
                        gradients_cache_info_temp.append(gradients_info_share_q.get())  

                log(f'[CG]{train_title} wait gradients: {temp_length}')

                # 是否是full梯度
                is_full_gradient = [i[0][0]['is_full_gradient'] for i in gradients_cache_info_temp]
                use_full_gradient = any(is_full_gradient)

                # 解压梯度
                g_idx = 0
                g_idx_full = 0
                gss = []
                vs = []
                for idx in range(temp_length):
                    # 获取梯度列表
                    # log(f'get gradients')
                    if is_full_gradient[idx]:
                        # 全梯度 优先使用
                        gs = [i[g_idx_full] for i in gradients_cache_temp_full]
                        g_idx_full += 1
                    elif not use_full_gradient:
                        gs = [i[g_idx] for i in gradients_cache_temp]
                        g_idx += 1
                    else:
                        raise Exception(f'{train_title} use_full_gradient: {use_full_gradient}, is_full_gradient: {is_full_gradient[idx]}')

                    # 解压梯度
                    # log(f'decompress gradients')
                    info, version = gradients_cache_info_temp[idx]
                    # pickle.dump((gs,info, version), open(f'wait_handle_gradients.pkl', 'wb'))
                    gs = gradient_compressor.decompress(gs, info)
                    gss.append(gs)
                    vs.append(version)

                # 梯度平均
                if len(gss) == 1:
                    gs = gss[0]
                    version = vs[0]
                else:
                    # 初始化一个与第一个梯度列表相同形状的张量，用于存储总和
                    total_sum = [torch.zeros_like(grad) for grad in gss[0]]
                    for grads in gss:
                        for i, grad in enumerate(grads):
                            total_sum[i] += grad
                    gs = [grad / len(gss) for grad in total_sum]
                    # 取平均版本号
                    version = sum(vs) // len(vs)

                # 更新梯度
                # log(f'update gradients')
                param_server.apply_gradients(gs, version)

                # 是否需要预热
                need_warn_up = grad_warm_up_steps > step_count
                step_count += 1
                
                if (idx + 1) % GRAD_BATCH_SIZE == 0:  
                    ready_params(param_server, share_params_lock, params_cache_share, params_info_share, need_warn_up)
                                        
                # 记录版本差异
                version_diff = param_server.ver -1 - version
                total_client_version_diff += version_diff
                total_count += 1

                # 清空梯度信息
                gradients_cache_info_temp.clear()

                log(f'[CG]{train_title} done, cost: {int(1000*(time.time() - t))}ms, mean version diff: {total_client_version_diff / total_count :.2f}')   
            except Exception as e:
                log(f'ERROR: \n{get_exception_msg()}')
                report_memory_usage()
                raise e


    async def _get_latest_raw_params(self):
        """获取最新参数(无压缩)"""
        # 获取参数
        params = []
        v = 0
        with self.share_params_lock:
            # 交换
            for i in self.params_cache_share:
                params.append(i.data.clone())

            # 获取参数版本号
            v = self.params_info_share.data[0].item()
            # 获取是否需要预热
            need_warn_up = self.params_info_share.data[1].item()
        return params, v, need_warn_up

    def start(self, loop=None):
        self.aper.start(loop)

    async def async_handle_request(self, ip, msg_header, cmd, writer, reader):
        """异步处理客户端请求

        """
        if cmd.startswith('get@'):
            t = time.time()
            # 单次请求参数
            _client_version = int(cmd.split('@')[1])# TODO 客户端版本号
            log(f'{msg_header} recv get request, client version: {_client_version}')

            # 获取最新参数
            params, v, need_warn_up = await self._get_latest_raw_params()

            # 计算更新增量
            params, info = self.params_compressor.compress(params, ip)

            # 发送参数
            await async_send_msg(writer, pickle.dumps((params, info, v, need_warn_up)))
            log(f'{msg_header} send params, version: {v}, cost: {int(1000*(time.time() - t))}ms')

        elif cmd == 'wait_params':
            # 长连接请求参数
            log(f'{msg_header} recv wait_params request, push params interval: {self.push_params_interval}')
            last_send_v = 0

            begin_time = 0
            push_count = 0
            total_handle_time = 0
            total_wait_time = 0
            mean_send_size = 0
            while True:
                # 等待参数更新事件
                log(f'[{msg_header}] wait_params prepare wait, last_send_v: {last_send_v}')
                await self.aper.wait()
                t = time.time()
                if begin_time == 0:
                    begin_time = t

                log(f'[{msg_header}] wait_params wait active, last_send_v: {last_send_v}')

                # 获取最新参数
                params, v, need_warn_up = await self._get_latest_raw_params()
                self.version = v
                log(f'[{msg_header}] wait_params prepare v: {v}, cost: {int(1000*(time.time() - t))}ms')

                # 控制参数推送频率
                log(f'[{msg_header}] v - last_send_v: {v - last_send_v}, need_send: {(v - last_send_v) >= self.push_params_interval}')
                # 必须腰处理全部的增量更新
                # if (v - last_send_v) >= self.push_params_interval:
                last_send_v = v

                # 计算更新增量
                params, info = self.params_compressor.compress(params, ip)

                log(f'[{msg_header}] prepare params, version: {last_send_v}, cost: {int(1000*(time.time() - t))}ms')
                send_begin_time = time.time()
                data = pickle.dumps((params, info, v, need_warn_up))
                await async_send_msg(writer, data)
                send_size = len(data)
                mean_send_size = (mean_send_size * push_count + send_size) / (push_count + 1)

                log(f'[{msg_header}] send params, version: {last_send_v}, cost: {int(1000*(time.time() - t))}ms')
                # # 等待回复
                # await wait_ack(reader)
                # log(f'[{msg_header}] recv check, version: {last_send_v}, cost: {int(1000*(time.time() - t))}ms')
                wait_time = time.time() - send_begin_time
                total_wait_time += wait_time

                push_count += 1
                if push_count % 30 == 0:
                    # 每次参数推送耗时(avg param push time): 本机处理耗时(avg handle time) + 等待耗时(发送，确认返回, avg wait time) + 等待参数耗时
                    # 网络传输耗时: 等待耗时(发送，确认返回, avg wait time) - 客户端接收后处理耗时(客户端统计)
                    # avg param push time: 925ms, avg wait time: 447ms, avg handle time: 3ms
                    # 优化空间:
                    #     平均等待参数时间 = 925 - 447 - 3 = 475ms
                    #     网络传输耗时 = 447 - 0 = 447ms

                    # avg param push time: 616ms, avg wait time: 417ms, avg handle time: 4ms
                    # 优化空间:
                    #     平均等待参数时间 = 616 - 417 - 4 = 195ms
                    #     网络传输耗时 = 417 - 0 = 417ms

                    log(f'[{msg_header}] avg param push time: {int(((time.time() - begin_time) / push_count) * 1000)}ms, avg wait time: {int(total_wait_time / push_count * 1000)}ms, avg handle time: {int((total_handle_time - total_wait_time) / push_count * 1000)}ms, mean send size: {int(mean_send_size)}')


                handle_cost_time = time.time() - t
                total_handle_time += handle_cost_time

        elif cmd == 'need_val':
            # 若当前时间戳 - 允许验证的时间戳 > 12小时, 则允许验证
            t = time.time()
            current_time = time.time()
            res = b'0'
            if current_time - self.need_val_timestamp > 12 * 3600:
                self.need_val_timestamp = current_time
                self.need_val_ip = ip
                res = b'1'

            await async_send_msg(writer, res)
            log(f'{msg_header} send need_val: {res}, cost: {int(1000*(time.time() - t))}ms')

        elif cmd == 'update_gradients':
            # 梯度传递一定是长连接，不断的接收
            log(f'{msg_header} recv update_gradients request')

            # 客户端数量
            self.client_nums += 1
            self.client_nums_q.put(self.client_nums)

            total_handle_time = 0
            begin_time = 0
            push_count = 0

            try:
                while True:

                    # 获取梯度数据
                    data = await async_recv_msg(reader)
                    t = time.time()
                    if begin_time == 0:
                        begin_time = t
                    log(f'{msg_header} recv gradients({len(data)})')

                    # batch_g_info, version = pickle.loads(data)
                    # data: [((compressed_grads, compress_info), version), ...] / ((compressed_grads, compress_info), version)
                    data = pickle.loads(data)
                    if GRAD_BATCH_SIZE > 1:
                        batch_g_info = [(pickle.loads(i[0]), i[1]) for i in data]
                    else:
                        batch_g_info = [(pickle.loads(data[0]), data[1])]

                    # g, compress_info, version = pickle.loads(data)
                    log(f'{msg_header} loads gradients, cost: {int(1000*(time.time() - t))}ms')
                    # 提交到共享梯度信息队列
                    # log(f'put gradients info')

                    # version diff filter
                    version_diffs = [self.version - i[1] for i in batch_g_info]
                    not_allow_idxs = [i for i, v in enumerate(version_diffs) if v > self.grad_allow_version_diff]
                    # 倒序删除不允许的梯度
                    for idx in sorted(not_allow_idxs, reverse=True):
                        log(f'{msg_header} skip gradients idx: {idx}, version diff: {version_diffs[idx]}')
                        batch_g_info.pop(idx)
                    _update_gradients_length = len(batch_g_info)
                    if _update_gradients_length == 0:
                        log(f'{msg_header} no gradients, keep wait')
                        continue

                    # 提交到共享梯度信息队列
                    for i in range(_update_gradients_length):
                        self.gradients_info_share_q.put((batch_g_info[i][0][1], batch_g_info[i][1]))

                    log(f'{msg_header} add info&version done, cost: {int(1000*(time.time() - t))}ms')
                    # 提交到共享梯度
                    # log(f'put gradients')

                    wait_count = 0

                    # 当前等待处理的梯度数量
                    done_idxs = []
                    gradients_cache_share_lengths = [0] * _update_gradients_length
                    gradients_cache_share_length = 0
                    async with self.gradients_add_lock:
                    # async with AsyncLockWithLog(self.gradients_add_lock, log, msg_header):
                        while True:
                            # with LockWithLog(self.share_gradients_lock, log, msg_header):
                            with self.share_gradients_lock:
                                for idx, ((g, compress_info), v) in enumerate(batch_g_info):
                                    # 是否已经处理过
                                    if idx in done_idxs:
                                        continue

                                    # 是否是全梯度
                                    if compress_info[0]['is_full_gradient']:
                                        cache_share = self.gradients_cache_share_full

                                    else:
                                        cache_share = self.gradients_cache_share

                                    # 拷贝到共享梯度
                                    gradients_cache_share_lengths[idx] = cache_share[0].size()
                                    if gradients_cache_share_lengths[idx] < self.grad_cache_size:
                                        for idx, _g in enumerate(g):
                                            cache_share[idx].append(_g)
                                        done_idxs.append(idx)
                                    else:
                                        break

                            if len(done_idxs) == _update_gradients_length:
                                wait_count = 0
                                break

                            # 等待处理的梯度大于 梯度缓存大小
                            # 释放锁并等待
                            self.share_data_new_event.set()
                            log(f'{msg_header} wait gradients, wait length: {max(gradients_cache_share_lengths)}')
                            await asyncio.sleep(0.1)

                            wait_count += 1
                            if wait_count > 30:
                                log(f'{msg_header} wait gradients timeout')
                                import sys
                                sys.exit()

                    gradients_cache_share_length = max(gradients_cache_share_lengths) + 1
                    log(f'{msg_header} add gradients done, wait length: {gradients_cache_share_length}, cost: {int(1000*(time.time() - t))}ms')

                    # # 回复，避免socket堆积
                    # await ack(writer)

                    # 通知新梯度
                    self.share_data_new_event.set()
                    handle_cost_time = time.time() - t
                    total_handle_time += handle_cost_time
                    log(f'{msg_header} handle gradients done, wait length: {gradients_cache_share_length}, cost: {int(1000*handle_cost_time)}ms')

                    push_count += 1
                    if push_count % 30 == 0:
                        # avg gradients recv time: 923ms, avg handle time: 15ms
                        # avg gradients recv time: 43ms, avg handle time: 9ms
                        log(f'{msg_header} avg gradients recv time: {int(((time.time() - begin_time) / push_count) * 1000)}ms, avg handle time: {int(total_handle_time / push_count * 1000)}ms')

            except Exception as e:
                # 异常处理
                # 客户端数量
                self.client_nums -= 1
                self.client_nums_q.put(self.client_nums)
                raise e


if __name__ == '__main__':
    pass