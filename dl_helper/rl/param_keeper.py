import torch, time, math
import multiprocessing
from multiprocessing.queues import Empty, Full
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
from dl_helper.rl.socket_base import async_send_msg, async_recv_msg, ack, wait_ack, PUSH_INTERVAL

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.tune.registry import _global_registry, ENV_CREATOR

from py_ext.tool import log, share_tensor_list, share_tensor, get_exception_msg, safe_share_memory_queue

GRAD_ALLOW_VERSION_DIFF = 60
# GRAD_ALLOW_VERSION_DIFF = 0

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

    def apply_gradients(self, gradients_list, lr_times=1):
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
    
    def load_weights(self, weights):
        """加载参数"""
        self.learner.module._rl_modules['default_policy'].load_state_dict(weights)

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
        self.grad_cache_size = 10

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

        # # FOR DEBUG
        # self.params_list = self.wait_params_id_q.get()

        # 允许验证的客户端ip
        self.need_val_ip = 0

        # 允许验证的时间戳
        self.need_val_timestamp = 0

        # # FOR DEBUG
        # self.revc_grad_id_dict = {}
        # # 伪造参数增量更新
        # compressed_tensors = []
        # compress_info = {
        #     'update_indices': [],
        #     'full': []
        # }
        # for p in self.params_list:
        #     n = max(int(p.numel() * 0.1), 1)
        #     _, top_indices = torch.topk(p.flatten(), n)
        #     mask = torch.zeros_like(p, dtype=torch.bool)
        #     mask.view(-1)[top_indices] = True
        #     update_indices = torch.where(mask)
        #     update_values = p[mask]
        #     compress_info['update_indices'].append(torch.stack(update_indices, dim=1))
        #     compress_info['full'].append(False)
        #     compressed_tensors.append(update_values)
        # self.dump_data = pickle.dumps((compressed_tensors, compress_info, 1, 0))
        # self.dump_v = 1

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
        params_compressor = IncrementalCompressor()

        # 计算步数
        step_count = 0
        # 上次输出的步数
        last_print_step = 0

        # 参数服务器
        if torch.cuda.is_available():
            config = config.learners(    
                num_learners=1,
                num_gpus_per_learner=1,
            )
        else:
            config = config.learners(    
                num_learners=1
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
        _p_dump = pickle.dumps(([v for _, v in _params_dict.items()], {'full': True}, 0, False))
        _p_size = len(_p_dump)
        # 计算最大梯度dump大小
        # 1.0 全梯度的压缩数据大小
        _grad_params_list = [v for _, v in _grad_params_dict.items()]
        _compress_grad_info = [{'is_full_gradient': True,} for _ in _grad_params_list]
        _single_grad_dump = pickle.dumps((_grad_params_list, _compress_grad_info))
        _g_dump = pickle.dumps((_single_grad_dump, 0))
        _g_size = len(_g_dump)

        wait_params_id_q.put((_p_size, _g_size))# 回传大小

        # # FOR DEBUG
        # wait_params_id_q.put([v for _, v in _params_dict.items()])

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

        # 客户端推送梯度计数
        client_push_grad_count = {}

        # 客户端最近一次更新的计数
        client_last_update_count = {}

        begin_time = time.time()
        update_count = 0
        total_update_time = 0

        log(f'{train_title} calculate most start')
        while True:
            try:
                t = time.time()
                ################################################
                # A 方案
                # 1.0 遍历所有的客户id梯度队列 接收所有梯度dump/解压应用梯度
                #   遍历梯度:
                #       1.1 解压梯度
                #       1.2 应用梯度
                # 2.0 遍历所有的客户id参数队列，准备/压缩/推送参数
                # 
                # B 方案 choose
                # 1.0 遍历所有的客户id梯度队列 接收所有梯度dump/解压应用梯度
                #   1.1 遍历梯度 解压
                #   1.2 平均梯度后应用
                # 2.0 遍历所有的客户id参数队列，准备/压缩/推送参数
                ################################################

                # 检查是否有新的 等待参数 id
                while True:
                    try:
                        new_wait_params_id = wait_params_id_q.get(block=False)
                        if new_wait_params_id not in client_params_q:
                            # 初始化共享队列
                            client_params_q[new_wait_params_id] = safe_share_memory_queue(f'dump_q_{new_wait_params_id}', _p_size, 4, len(pickle.dumps(np.int64(0))))# 额外的数据保存版本信息
                            client_grad_q[new_wait_params_id] = safe_share_memory_queue(f'g_dump_q_{new_wait_params_id}', _g_size, 30)
                            # 单次状态
                            client_wait_state[new_wait_params_id] = 0
                            # 推送梯度计数
                            client_push_grad_count[new_wait_params_id] = 0
                            # 最近一次更新的计数
                            # 初始化为 -1 : 
                            #   会提前一个step推送给客户端，节约客户端等待参数的时间, 假设3step推送，step2时，2-(-1) = 3, 会提前一个step推送
                            #   缺点是客户端接收到的参数会延迟一个step
                            # 可尝试: -1, -2, -3 查看效果
                            client_last_update_count[new_wait_params_id] = 0
                    except Empty:
                        break

                # 检查是否有新的 循环等待 id
                while True:
                    try:
                        new_wait_params_id = on_wait_params_id_q.get(block=False)
                        # 循环状态
                        client_wait_state[new_wait_params_id] = 1   
                    except Empty:
                        break


                if not client_grad_q:
                    # log(f'[CG]{train_title} not client_grad_q, keep wait')
                    time.sleep(0.001)
                    continue
                # else:
                #     log(f'[CG]{train_title} check new client, cost: {int(1000*(time.time() - t))}ms')

                #####################################
                # 1.1 尝试get梯度，若获取成功继续处理
                #####################################
                # 一次获取当前的所有队列中的所有梯度
                # # 遍历所有的梯度队列，获取梯度 > 梯度列表
                # grad_dump_data_list = []
                # for _id, _q in client_grad_q.items():
                #     _q_size = _q.qsize()
                #     # 更新推送梯度计数
                #     client_push_grad_count[_id] += _q_size
                #     for _ in range(_q_size):
                #         try:
                #             grad_dump_data = _q.get(block=False)
                #             grad_dump_data_list.append((_id, grad_dump_data))
                #         except Empty:
                #             break

                # 一次获取当前的所有队列中一个梯度
                # 遍历所有的梯度队列，获取梯度 > 梯度列表
                grad_dump_data_list = []
                for _id, _q in client_grad_q.items():
                    _q_size = _q.qsize()
                    if _q_size > 0:
                        # 更新推送梯度计数
                        try:
                            grad_dump_data = _q.get(block=False)
                            grad_dump_data_list.append((_id, grad_dump_data))
                            client_push_grad_count[_id] += 1
                        except Empty:
                            pass

                #####################################
                # 1.2 过滤 / 解压 / 应用梯度
                #####################################
                if grad_dump_data_list:
                    log(f'[CG]{train_title} collect grads, cost: {int(1000*(time.time() - t))}ms')
                    batch_g_info = []
                    # load 数据
                    # data: [((compressed_grads, compress_info), version), ...] / ((compressed_grads, compress_info), version)
                    for (_id, grad_dump_data) in grad_dump_data_list:
                        compress_data, compress_info, version = pickle.loads(grad_dump_data)
                        batch_g_info.append((compress_data, compress_info, version, _id))
                    log(f'[CG]{train_title} loads gradients, cost: {int(1000*(time.time() - t))}ms')

                    # version diff 过滤
                    cur_version = param_server.ver
                    version_diffs = [(i[3],cur_version - i[2]) for i in batch_g_info]
                    log(f'[CG]{train_title} grad versions: {[(i[3],i[2]) for i in batch_g_info]}')
                    log(f'[CG]{train_title} version diffs: {version_diffs}')
                    # 记录版本差异
                    total_client_version_diff += sum([i[1] for i in version_diffs])
                    total_count += len(version_diffs)
                    if GRAD_ALLOW_VERSION_DIFF > 0:
                        not_allow_idxs = [i for i, (_id, v) in enumerate(version_diffs) if v > GRAD_ALLOW_VERSION_DIFF]
                        if not_allow_idxs:
                            # 倒序删除不允许的梯度
                            for idx in sorted(not_allow_idxs, reverse=True):
                                log(f'[CG]{train_title} skip gradients idx: {idx}, version diff: {version_diffs[idx]}')
                                batch_g_info.pop(idx)

                        # 数量检查
                        _update_gradients_length = len(batch_g_info)
                        if _update_gradients_length == 0:
                            log(f'[CG]{train_title} version diff filt no gradients')
                        log(f'[CG]{train_title} version diff filt done, left: {_update_gradients_length}, cost: {int(1000*(time.time() - t))}ms')

                    # 遍历剩下的梯度，取平均后应用
                    # 解压所有梯度
                    if batch_g_info:
                        all_grads = []
                        for (g, compress_info, v, _id) in batch_g_info:
                            # 解压梯度
                            g = DeepGradientCompression.decompress(g, compress_info)
                            all_grads.append(g)
                        # 计算平均梯度
                        avg_grads = []
                        for grad_idx in range(len(all_grads[0])):
                            # 使用 torch.stack 将同位置的梯度堆叠后求平均
                            stacked_grads = torch.stack([grads[grad_idx] for grads in all_grads])
                            avg_grad = torch.mean(stacked_grads, dim=0)
                            avg_grads.append(avg_grad)
                        # 应用平均梯度
                        param_server.apply_gradients(avg_grads, len(all_grads))
                        step_count += 1

                        log(f'[CG]{train_title} apply grad, latest_version: {param_server.ver}, cost: {int(1000*(time.time() - t))}ms')

                #####################################
                # 2.0 准备/压缩参数
                #####################################
                # 是否需要预热
                need_warn_up = grad_warm_up_steps > step_count
                weights, version = param_server.get_weights()
                # 转为列表
                weights = [v for _, v in weights.items()]

                # 遍历 client_params_q 压缩准备参数
                need_push_ids = []
                for _id in client_wait_state:
                    # 获取等待状态
                    if client_wait_state[_id] == 0:
                        # 一次 > -1
                        client_wait_state[_id] = -1
                    elif client_wait_state[_id] == -1:
                        # 不在等待了
                        continue
                    elif client_wait_state[_id] == 1:
                        # 循环等待, 只推送有更新的参数
                        if len(grad_dump_data_list) == 0:
                            continue

                        # 检查该客户端是否需要推送参数
                        log(f'[CG]{train_title} id:{_id} client_push_grad_count: {client_push_grad_count[_id]} client_last_update_count: {client_last_update_count[_id]}')
                        if client_push_grad_count[_id] - client_last_update_count[_id] < PUSH_INTERVAL:
                            continue
                        else:
                            client_last_update_count[_id] += PUSH_INTERVAL
                            log(f'[CG]{train_title} id:{_id} client_last_update_count > {client_last_update_count[_id]}')

                    # 检查队列是否满了，满了则跳过
                    # 说明客户端可能已经断开
                    if client_params_q[_id].is_full():
                        continue

                    need_push_ids.append(_id)

                if need_push_ids:
                    log(f'[CG]{train_title} compress params for {need_push_ids}, cost: {int(1000*(time.time() - t))}ms')

                    # 压缩
                    res_dict = params_compressor.compress(weights, need_push_ids)
                    log(f'[CG]{train_title} compress params done, cost: {int(1000*(time.time() - t))}ms')

                    for _id, (compress_data, compress_info) in res_dict.items():
                        # dumps
                        dump_data = pickle.dumps((compress_data, compress_info, version, need_warn_up))
                        client_params_q[_id].put(dump_data, block=False, extra_data=np.int64(version))
                        log(f'[CG]{train_title} ready params for {_id}, version: {version}, size: {len(dump_data)}, done, cost: {int(1000*(time.time() - t))}ms')

                    total_update_time += time.time() - t
                    update_count += 1

            except Exception as e:
                log(f'ERROR: \n{get_exception_msg()}')
                report_memory_usage()
                raise e
            
            if step_count % 60 == 0 and total_count > 0 and step_count > last_print_step:
                log(f'[CG]{train_title} avg cost: {(total_update_time * 1000 / update_count):.2f}ms, version diff: {total_client_version_diff / total_count :.2f}')
                last_print_step = step_count

    async def get_params_dump_data(self, _id):
        params_dump_q = self.ip_params_dump_q[_id]
        while True:
            # 等待不为空
            if params_dump_q.is_empty():
                await asyncio.sleep(0.001)
                continue

            dump_data, dump_v = params_dump_q.get(block=False)
            return dump_data, dump_v

    async def put_gradients_dump_data(self, dump_data, _id):
        _q = self.ip_gradients_dump_q[_id]
        while True:
            try:
                _q.put(dump_data, block=False)
                break
            except Full:
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
            self.ip_gradients_dump_q[_id] = safe_share_memory_queue(f'g_dump_q_{_id}', self.share_gradients_dump_max_size, 30)
            self.ip_gradients_dump_q[_id].clear()

            # 通知需要等待的ip  
            self.wait_params_id_q.put(_id)

            # 等待获取参数 dump 数据
            dump_data, dump_v = await self.get_params_dump_data(_id)

            # 发送参数
            await async_send_msg(writer, dump_data)
            log(f'{msg_header} send params, version: {dump_v}, cost: {int(1000*(time.time() - t))}ms')

            # #FOR DEBUG
            # self.revc_grad_id_dict[_id] = 0

        elif cmd == 'wait_params':
            # 长连接请求参数
            log(f'{msg_header} recv wait_params request')

            # 通知 开始循环等待
            # FOR DEBUG
            self.on_wait_params_id_q.put(_id)

            last_send_v = 0
            begin_time = 0
            push_count = 0
            total_handle_time = 0
            total_net_time = 0
            total_wait_time = 0
            mean_send_size = 0
            # # FOR DEBUG
            # last_push_grad_count = 0
            # async def wait_need_push(self, last_push_grad_count):
            #     while True:
            #         if self.revc_grad_id_dict[_id] - last_push_grad_count >= PUSH_INTERVAL:
            #             last_push_grad_count += PUSH_INTERVAL
            #             return last_push_grad_count
            #         else:
            #             await asyncio.sleep(0.001)
            #             continue
            while True:
                t = time.time()
                log(f'{msg_header} wait_params prepare wait, last_send_v: {last_send_v}')
                # 等待获取参数 dump 数据
                # FOR DEBUG
                dump_data, dump_v = await self.get_params_dump_data(_id)
                # last_push_grad_count = await wait_need_push(self, last_push_grad_count)
                # dump_data, dump_v = self.dump_data, self.dump_v

                wait_time = time.time() - t
                total_wait_time += wait_time
                log(f'{msg_header} wait_params wait active, wait time: {int(1000*wait_time)}ms')

                t = time.time()
                if begin_time == 0:
                    begin_time = t

                # 每步都推送
                last_send_v = dump_v

                # 发送参数
                send_begin_time = time.time()
                await async_send_msg(writer, dump_data)
                send_size = len(dump_data)
                mean_send_size = (mean_send_size * push_count + send_size) / (push_count + 1)

                # 9909 
                log(f'{msg_header} send params, version: {last_send_v}, cost: {int(1000*(time.time() - t))}ms')
                # # 等待回复
                # await wait_ack(reader)
                # log(f'[{msg_header}] recv check, version: {last_send_v}, cost: {int(1000*(time.time() - t))}ms')
                net_time = time.time() - send_begin_time
                total_net_time += net_time

                push_count += 1

                if push_count % 30 == 0:
                    # TIME
                    # 服务端只处理网络传输，返回的是模拟的数据 5C
                    # avg param push time: 822ms, avg wait time: 829ms, avg net time: 0ms, avg handle time: 0ms, mean send size: 277300
                    # 服务端完整处理数据 5C 计算耗时约500ms
                    # avg param push time: 1323ms, avg wait time: 1321ms, avg net time: 6ms, avg handle time: 0ms, mean send size: 47731
                    log(f'{msg_header} avg param push time: {int(((time.time() - begin_time) / push_count) * 1000)}ms, avg wait time: {int(total_wait_time / push_count * 1000)}ms, avg net time: {int(total_net_time / push_count * 1000)}ms, avg handle time: {int((total_handle_time - total_net_time) / push_count * 1000)}ms, mean send size: {int(mean_send_size)}')

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
                    # FOR DEBUG
                    await self.put_gradients_dump_data(data, _id)
                    # self.revc_grad_id_dict[_id] += 1

                    handle_cost_time = time.time() - t
                    total_handle_time += handle_cost_time

                    # 14015
                    log(f'{msg_header} forward gradients done, cost: {int(1000*handle_cost_time)}ms')

                    push_count += 1
                    if push_count % 30 == 0:
                        # TIME
                        # 服务端只处理网络传输，返回的是模拟的数据 5C   
                        # avg gradients recv time: 137ms, avg forward time: 0ms
                        # 服务端完整处理数据 5C
                        # avg gradients recv time: 221ms, avg forward time: 3ms
                        log(f'{msg_header} avg gradients recv time: {int(((time.time() - begin_time) / push_count) * 1000)}ms, avg forward time: {int(total_handle_time / push_count * 1000)}ms')

            except Exception as e:
                # 异常处理
                raise e


if __name__ == '__main__':
    pass