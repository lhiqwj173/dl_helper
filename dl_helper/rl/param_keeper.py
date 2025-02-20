import torch, time, math
import multiprocessing
from multiprocessing.queues import Empty
import asyncio
import gymnasium as gym

import numpy as np
import pickle, requests
from typing import Dict, Any
from collections import deque, OrderedDict
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

from py_ext.tool import log, share_tensor_list, share_tensor, get_exception_msg, safe_share_memory_queue

GRAD_ALLOW_VERSION_DIFF = 30

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

        self.last_lr_times = 1

    def apply_gradients(self, gradients_list, client_version, lr_times=1):
        """
        更新参数
        gradients_list: 梯度列表, 键为参数名, 值为torch.Tensor
        """
        # log(f'gradients_list length: {len(gradients_list)}')
        params = self.learner._params
        for idx, k in enumerate(params.keys()):
            params[k].grad = gradients_list[idx].to(self.learner._device)
        
        # 修改学习率
        if lr_times != self.last_lr_times:
            _times = lr_times / self.last_lr_times
            log(f'modify lr X{_times:.2f}')
            for module_id, optimizer_names in self.learner._module_optimizers.items():
                for optimizer_name in optimizer_names:
                    optim = self.learner.get_optimizer(module_id, optimizer_name)
                    for param_group in optim.param_groups:
                        param_group['lr'] = param_group['lr'] * _times
            self.last_lr_times = lr_times

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

    def __init__(self, train_title, config, debug=False, grad_warm_up_steps=0
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

        # 梯度预热步数
        self.grad_warm_up_steps = grad_warm_up_steps

        # 等待参数 id 队列，用于传递等待 id/最大dumps大小，用于初始化共享参数队列
        self.wait_params_id_q = multiprocessing.Queue()
        # 客户端id 开始循环等待参数
        self.on_wait_params_id_q = multiprocessing.Queue()

        # 初始化/储存共享队列
        self.share_params_dump_max_size = 0
        self.ip_params_dump_q = {}
        self.share_gradients_dump_max_size = 0
        self.ip_gradients_dump_q = {}

        # 添加梯度锁
        self.gradients_add_lock = asyncio.Lock()

        # 独立线程转发 进程任务
        self.ready_params_event = multiprocessing.Event()
        self.aper = AsyncProcessEventReader(self.ready_params_event)
        
        # 启动计算进程
        self.p = multiprocessing.Process(target=ExperimentHandler.gpu_most_task, args=(
            train_title, 
            config,
            self.grad_warm_up_steps,
            self.wait_params_id_q,self.on_wait_params_id_q,
        ))
        self.p.start()

        # 等待回传的大小数据
        self.share_params_dump_max_size, self.share_gradients_dump_max_size = self.wait_params_id_q.get()

        # 允许验证的客户端ip
        self.need_val_ip = 0

        # 允许验证的时间戳
        self.need_val_timestamp = 0

        # 客户端 ip 列表
        self.clients = set()

    def __del__(self):
        self.p.terminate()

    @staticmethod
    def gpu_most_task(
        train_title, 
        config, 
        grad_warm_up_steps,
        wait_params_id_q,on_wait_params_id_q,
    ):
        """
        负责 梯度解压/梯度应用更新参数/参数压缩
        """
        log(f'[CG]{train_title} calculate gpu init')
        
        # 参数压缩器
        params_compressor = IncrementalCompressor(1e-5)

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

        # 计算最大参数dump大小
        _p_dump = pickle.dumps(([v for _, v in _params_dict.items()], {'full': True}, np.int64(0), np.int64(0)))
        _p_size = len(_p_dump)
        # 计算最大梯度dump大小
        # 1.0 全梯度的压缩数据大小
        _grad_params_list = [v for _, v in _grad_params_dict.items()]
        _compress_grad_info = [{'is_full_gradient': True,} for _ in _grad_params_list]
        _single_grad_dump = pickle.dumps((_grad_params_list, _compress_grad_info))
        # 1.1 GRAD_BATCH_SIZE 多个数据的大小
        if GRAD_BATCH_SIZE >1:
            _g_dump = pickle.dumps(
                [(_single_grad_dump, np.int64(0)) for _ in range(GRAD_BATCH_SIZE)]
            )
        else:
            _g_dump = pickle.dumps((_single_grad_dump, np.int64(0)))
        _g_size = len(_g_dump)

        wait_params_id_q.put((_p_size, _g_size))# 回传大小

        # 等待队列数据被取出
        while not wait_params_id_q.empty():
            time.sleep(0.01)

        # 版本差异统计
        total_client_version_diff = 0
        total_count = 0

        # 客户端共享队列, 用于发送参数 dump 数据
        client_params_q = {}

        # 客户端共享队列, 用于接收 grad dump 数据
        client_grad_q = {}

        # 客户端等待参数的状态 
        # -1: 不需要等待
        # 0: 一次等待
        # 1: 循环等待
        client_wait_state = {}

        begin_time = time.time()
        loop_count = 0

        log(f'{train_title} calculate most start')
        while True:
            try:
                t = time.time()
                ################################################
                # 1.0 接收梯度dump/解压应用梯度
                #   1.1 尝试get梯度，若获取成功继续处理
                #   1.2 解压梯度
                # 2.0 准备/压缩参数
                ################################################

                # 检查是否有新的 等待参数 id
                _q_size = wait_params_id_q.qsize()
                for _ in range(_q_size):
                    try:
                        new_wait_params_id = wait_params_id_q.get(block=False)
                        if new_wait_params_id not in client_params_q:
                            # 初始化共享队列
                            client_params_q[new_wait_params_id] = safe_share_memory_queue(f'dump_q_{new_wait_params_id}', _p_size, 4, len(pickle.dumps(np.int64(0))))# 额外的数据保存版本信息
                            client_grad_q[new_wait_params_id] = safe_share_memory_queue(f'g_dump_q_{new_wait_params_id}', _g_size, 4)
                            # 单次状态
                            client_wait_state[new_wait_params_id] = 0
                    except Empty:
                        break

                # 检查是否有新的 循环等待 id
                _q_size = on_wait_params_id_q.qsize()
                for _ in range(_q_size):
                    try:
                        new_wait_params_id = on_wait_params_id_q.get(block=False)
                        if new_wait_params_id not in client_wait_state:
                            # 循环状态
                            client_wait_state[new_wait_params_id] = 1   
                    except Empty:
                        break

                #####################################
                # 1.1 尝试get梯度，若获取成功继续处理
                #####################################
                # 遍历所有的梯度队列，尝试获取 更新梯度
                for _id, _q in client_grad_q.items():
                    _q_size = _q.qsize()
                    for _ in range(_q_size):
                        try:
                            grad_dump_data = _q.get(block=False)
                        except Empty:
                            break

                        #####################################
                        # 1.2 过滤 / 解压 / 应用梯度
                        #####################################
                        t = time.time()
                        # data: [((compressed_grads, compress_info), version), ...] / ((compressed_grads, compress_info), version)
                        log(f'[CG]{train_title} grad_dump_data: {grad_dump_data}')
                        data = pickle.loads(grad_dump_data)
                        if GRAD_BATCH_SIZE > 1:
                            batch_g_info = [(pickle.loads(i[0]), i[1]) for i in data]
                        else:
                            batch_g_info = [(pickle.loads(data[0]), data[1])]
                        log(f'[CG]{train_title} loads gradients, cost: {int(1000*(time.time() - t))}ms')

                        # version diff 过滤
                        cur_version = param_server.ver
                        version_diffs = [cur_version - i[1] for i in batch_g_info]
                        # 记录版本差异
                        total_client_version_diff += sum(version_diffs)
                        total_count += len(version_diffs)
                        not_allow_idxs = [i for i, v in enumerate(version_diffs) if v > GRAD_ALLOW_VERSION_DIFF]
                        if not_allow_idxs:
                            # 倒序删除不允许的梯度
                            for idx in sorted(not_allow_idxs, reverse=True):
                                log(f'[CG]{train_title} skip gradients idx: {idx}, version diff: {version_diffs[idx]}')
                                batch_g_info.pop(idx)
                        # 数量检查
                        _update_gradients_length = len(batch_g_info)
                        if _update_gradients_length == 0:
                            log(f'[CG]{train_title} version diff filt no gradients, keep wait')
                            continue
                        log(f'[CG]{train_title} version diff filt done, left: {_update_gradients_length}, cost: {int(1000*(time.time() - t))}ms')

                        # 遍历剩下的梯度，逐个应用
                        for idx, ((g, compress_info), v) in enumerate(batch_g_info):
                            # 解压梯度
                            g = DeepGradientCompression.decompress(g, compress_info)
                            param_server.apply_gradients(g, v)
                            step_count += 1

                #####################################
                # 2.0 准备/压缩参数
                #####################################
                # 是否需要预热
                need_warn_up = grad_warm_up_steps > step_count
                weights, version = param_server.get_weights()
                # 转为列表
                weights = [v for _, v in weights.items()]

                # 遍历 client_params_q 压缩准备参数
                for _id in client_wait_state:
                    # 获取等待状态
                    if client_wait_state[_id] == 0:
                        # 一次 > -1
                        client_wait_state[_id] = -1
                    elif client_wait_state[_id] == -1:
                        # 不在等待了
                        continue

                    # 检查队列是否满了，满了则跳过
                    # 说明客户端可能已经断开
                    _q = client_params_q[_id]
                    if _q.is_full():
                        continue

                    log(f'[CG]{train_title} compress params for {_id}, version: {version}')
                    
                    # 压缩
                    compress_data, compress_info = params_compressor.compress(weights, _id)
                    # dumps
                    dump_data = pickle.dumps((compress_data, compress_info, version, need_warn_up))
                    _q.put(dump_data, block=False, extra_data=np.int64(version))

            except Exception as e:
                log(f'ERROR: \n{get_exception_msg()}')
                report_memory_usage()
                raise e
            
            loop_count += 1
            if step_count % 60 == 0 and total_count > 0:
                log(f'[CG]{train_title} avg cost: {int(1000*(time.time() - begin_time)) / loop_count}ms, version diff: {total_client_version_diff / total_count :.2f}')

    def start(self, loop=None):
        self.aper.start(loop)

    async def get_params_dump_data(self, _id, latest=True):
        params_dump_q = self.ip_params_dump_q[_id]
        while True:
            # 等待不为空
            if params_dump_q.is_empty():
                await asyncio.sleep(0.001)
                continue

            if latest:
                # 获取队列长度
                q_size = params_dump_q.qsize()
                # 获取最后一个数据，最新的
                for _ in range(q_size):
                    dump_data, dump_v = params_dump_q.get(block=False)
            else:
                # 获取第一个数据，用于处理 get 请求，不一定是最新的
                dump_data, dump_v = params_dump_q.get(block=False)

            return dump_data, dump_v

    async def put_gradients_dump_data(self, dump_data, _id):
        _q = self.ip_gradients_dump_q[_id]
        while True:
            try:
                _q.put(dump_data, block=False)
                break
            except Empty:
                await asyncio.sleep(0.001)
                continue

    async def async_handle_request(self, ip, msg_header, cmd, writer, reader):
        """异步处理客户端请求
        """
        _id = ip.replace('.', '')
        if cmd.startswith('get@'):
            t = time.time()
            # 单次请求参数
            _client_version = int(cmd.split('@')[1])# TODO 客户端版本号
            log(f'{msg_header} recv get request, client version: {_client_version}')

            # 初始化共享参数队列
            assert _id not in self.ip_params_dump_q, f'{_id} already in ip_params_dump_q'
            self.ip_params_dump_q[_id] = safe_share_memory_queue(f'dump_q_{_id}', self.share_params_dump_max_size, 4, len(pickle.dumps(np.int64(0))))# 额外的数据保存版本信息
            self.ip_params_dump_q[_id].clear()
            self.ip_gradients_dump_q[_id] = safe_share_memory_queue(f'g_dump_q_{_id}', self.share_gradients_dump_max_size, 4)
            self.ip_gradients_dump_q[_id].clear()

            # 通知需要等待的ip  
            self.wait_params_id_q.put(_id)

            # 等待获取参数 dump 数据
            dump_data, dump_v = await self.get_params_dump_data(_id, latest=False)

            # 发送参数
            await async_send_msg(writer, dump_data)
            log(f'{msg_header} send params, version: {dump_v}, cost: {int(1000*(time.time() - t))}ms')

        elif cmd == 'wait_params':
            # 长连接请求参数
            log(f'{msg_header} recv wait_params request')

            # 通知 开始循环等待
            self.on_wait_params_id_q.put(_id)

            last_send_v = 0
            begin_time = 0
            push_count = 0
            total_handle_time = 0
            total_wait_time = 0
            mean_send_size = 0
            while True:
                log(f'[{msg_header}] wait_params prepare wait, last_send_v: {last_send_v}')
                # 等待获取参数 dump 数据
                dump_data, dump_v = await self.get_params_dump_data(_id)
                self.version = max(self.version, dump_v)

                t = time.time()
                if begin_time == 0:
                    begin_time = t

                log(f'[{msg_header}] wait_params wait active, last_send_v: {last_send_v}')

                # 每步都推送
                last_send_v = dump_v

                # 发送参数
                send_begin_time = time.time()
                await async_send_msg(writer, dump_data)
                send_size = len(dump_data)
                mean_send_size = (mean_send_size * push_count + send_size) / (push_count + 1)

                # 9909 
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

            # 客户端数据索引
            self.clients.add(ip)

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

                    # 转发到队列中，不在这里处理
                    await self.put_gradients_dump_data(data, _id)

                    handle_cost_time = time.time() - t
                    total_handle_time += handle_cost_time

                    # 14015
                    log(f'{msg_header} forward gradients done, cost: {int(1000*handle_cost_time)}ms')

                    push_count += 1
                    if push_count % 30 == 0:
                        # avg gradients recv time: 923ms, avg handle time: 15ms
                        # avg gradients recv time: 43ms, avg handle time: 9ms
                        log(f'{msg_header} avg gradients recv time: {int(((time.time() - begin_time) / push_count) * 1000)}ms, avg forward time: {int(total_handle_time / push_count * 1000)}ms')

            except Exception as e:
                # 异常处理
                self.clients.remove(ip)
                raise e


if __name__ == '__main__':
    pass