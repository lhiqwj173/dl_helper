import torch, time, math
import multiprocessing
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

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.tune.registry import _global_registry, ENV_CREATOR

from py_ext.tool import log, share_tensor_list, share_tensor

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
    
    def get_gradients_params(self):
        """获取计算梯度的参数"""
        return copy.deepcopy(self.learner._params)
    
    def get_weights(self):
        """获取参数"""
        # return (self.learner.get_state(components=COMPONENT_RL_MODULE)['rl_module']['default_policy'], self.ver)
        weights = self.learner.module._rl_modules['default_policy'].state_dict()
        return weights, self.ver

class ExperimentHandler:
    """处理单个实验的类"""
    def __init__(self, train_title, config, debug=False, grad_warm_up_steps=100, grad_cache_size=15
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
        self.grad_cache_size = grad_cache_size

        # 版本号
        self.version = 0

        self.grad_warm_up_steps = grad_warm_up_steps

        # 共享梯度队列
        # 梯度数据经过稀疏化，形状
        self.gradients_info_share_q = multiprocessing.Queue()
        # 共享参数信息队列
        self.params_info_share_q = multiprocessing.Queue()
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

        # 启动计算进程
        self.p = multiprocessing.Process(target=ExperimentHandler.gpu_most_task, args=(
            train_title, self.gradients_info_share_q, self.params_info_share_q, self.params_float32_ver_share_q,self.share_data_new_event, 
            self.share_gradients_lock, self.share_params_lock, self.share_params_float32_lock,
            config,
            self.debug,
            self.grad_cache_size,
        ))
        self.p.start()

        # 等待接受 参数的形状列表
        # 用于初始化共享数据
        _simple_params, _simple_grad_params = self.gradients_info_share_q.get()

        # 启动cpu计算进程
        self.p2 = multiprocessing.Process(target=ExperimentHandler.cpu_most_task, args=(
            train_title, self.params_info_share_q, self.params_float32_ver_share_q,
            self.share_params_lock, self.share_params_float32_lock,
            _simple_params,
            self.grad_warm_up_steps,
        ))
        self.p2.start()

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
    
    def __del__(self):
        self.p.terminate()

    @staticmethod
    def cpu_most_task(
        train_title, params_info_share_q, params_float32_ver_share_q,
        share_params_lock, share_params_float32_lock,
        _simple_params,
        grad_warm_up_steps,
    ):
        """
        参数压缩，生成缓存
        """
        def update_params(lock, q, params_list, info, version, need_warn_up, params_cache_share):
            """更新参数信息"""
            log(f'update params')
            with lock:
                # 清空q
                while not q.empty():
                    q.get()
                # 最新数据
                q.put((info, version, need_warn_up))
                # 更新参数
                for idx, p in enumerate(params_list):
                    params_cache_share[idx].data[:] = p[:]
                    
        log(f'[CC]{train_title} calculate cpu init')

        # 参数压缩器
        param_compressor = ParamCompressor()
        # param_compressor = IncrementalCompressor()
        
        # 共享参数
        params_cache_share_float32 = []
        _params_cache_share_float32 = []
        params_cache_share = []
        # 初始化共享参数
        for idx, _shape in enumerate(_simple_params):
            # for debug
            params_cache_share.append(share_tensor(f'{train_title}_pcs_{idx}', _shape, 'float32'))
            # params_cache_share.append(share_tensor(f'{train_title}_pcs_{idx}', (math.prod(_shape),), 'int8'))
            params_cache_share_float32.append(share_tensor(f'{train_title}_pcs32_{idx}', _shape, 'float32'))
            _params_cache_share_float32.append(torch.zeros(_shape, dtype=torch.float32))

        step_count = 0
        while True:
            # 获取一份待压缩数据
            version = params_float32_ver_share_q.get()
            with share_params_float32_lock:
                for idx, _data in enumerate(params_cache_share_float32):
                    _params_cache_share_float32[idx][:] = _data.data[:]
                while not params_float32_ver_share_q.empty():
                    version = params_float32_ver_share_q.get()

            # 压缩参数
            weights, info = param_compressor.compress_params_dict(_params_cache_share_float32)
            # weights, info = param_compressor.compress(_params_cache_share_float32)

            # 是否需要预热
            need_warn_up = grad_warm_up_steps > step_count

            # 更新到共享参数
            update_params(share_params_lock, params_info_share_q, weights, info, version, need_warn_up, params_cache_share)

            step_count += 1

    @staticmethod
    def gpu_most_task(
        train_title, gradients_info_share_q, params_info_share_q, params_float32_ver_share_q, share_data_new_event, 
        share_gradients_lock, share_params_lock, share_params_float32_lock,
        config, 
        debug,
        grad_cache_size,
    ):
        """
        负责 梯度解压/梯度应用更新参数
        """
        def copy_params(param_server, lock, params_cache_share_float32, params_float32_ver_share_q): 
            """获取参数copy到共享参数中"""
            # log(f'copy params')
            # 获取模型参数
            # weights： 参数字典[torch.Tensor]
            weights, version = param_server.get_weights()
            with lock:
                for idx, (k, v) in enumerate(weights.items()):
                    # log(f'copy params, idx: {idx}, cache shape: {params_cache_share_float32[idx].data.shape} < {v.shape}')
                    params_cache_share_float32[idx].data[:] = v[:]
                params_float32_ver_share_q.put(version)

        log(f'[CG]{train_title} calculate gpu init')

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
        params_cache_share_float32 = []
        params_cache_share = []
        # 初始化共享参数
        _simple_params = []
        for idx, (k, v) in enumerate(_params_dict.items()):
            log(f'{train_title} init params share, idx: {idx}, name: {k}, shape: {v.shape}')
            _shape = v.shape
            # for debug
            params_cache_share.append(share_tensor(f'{train_title}_pcs_{idx}', _shape, 'float32'))
            # params_cache_share.append(share_tensor(f'{train_title}_pcs_{idx}', (math.prod(v.shape),), 'int8'))
            params_cache_share_float32.append(share_tensor(f'{train_title}_pcs32_{idx}', _shape, 'float32'))
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
        # 拷贝一份模型数据，交由cpu压缩生成缓存
        copy_params(param_server, share_params_float32_lock, params_cache_share_float32, params_float32_ver_share_q)
        
        # 回传 参数形状列表 
        # 回传后，共享参数以及初始化完成
        gradients_info_share_q.put((_simple_params, _simple_grad_params))

        log(f'{train_title} calculate most start')
        while True:
            share_data_new_event.wait()
            share_data_new_event.clear()

            # log(f'{train_title} calculate active')

            with share_gradients_lock:
                temp_length = gradients_cache_share[0].size() + gradients_cache_share_full[0].size()
                if temp_length == 0:
                    log(f'{train_title} no gradients, keep wait')
                    continue
                # 获取全部的梯度信息
                for idx in range(temp_length):
                    gradients_cache_info_temp.append(gradients_info_share_q.get())  
                # 拷贝梯度到临时梯度
                for idx, _g in enumerate(gradients_cache_share):
                    _g.all_copy_slice(gradients_cache_temp[idx], 0)
                for idx, _g in enumerate(gradients_cache_share_full):
                    _g.all_copy_slice(gradients_cache_temp_full[idx], 0)

            log(f'[CG]{train_title} wait gradients: {temp_length}')

            # 是否是full梯度
            is_full_gradient = [i[0][0]['is_full_gradient'] for i in gradients_cache_info_temp]

            # 计算梯度
            g_idx = 0
            g_idx_full = 0
            for idx in range(temp_length):
                # 获取梯度列表
                # log(f'get gradients')
                if is_full_gradient[idx]:
                    gs = [i[g_idx_full] for i in gradients_cache_temp_full]
                    g_idx_full += 1
                else:
                    gs = [i[g_idx] for i in gradients_cache_temp]
                    g_idx += 1

                # 解压梯度
                # log(f'decompress gradients')
                info, version = gradients_cache_info_temp[idx]
                # pickle.dump((gs,info, version), open(f'wait_handle_gradients.pkl', 'wb'))
                gs = gradient_compressor.decompress(gs, info)
                # 更新梯度
                # log(f'update gradients')
                param_server.apply_gradients(gs, version)
                
                # # 每4次更新生成一次参数缓存
                # if (idx + 1) % 4 == 0 or idx == temp_length - 1:
                #     # 拷贝一份模型数据，交由cpu压缩生成缓存
                #     copy_params(param_server, share_params_float32_lock, params_cache_share_float32, params_float32_ver_share_q)
                # 拷贝一份模型数据，交由cpu压缩生成缓存
                copy_params(param_server, share_params_float32_lock, params_cache_share_float32, params_float32_ver_share_q)
            
            log(f'[CG]{train_title} done')   

    def handle_request(self, client_socket, msg_header, cmd):
        """处理客户端请求"""
        raise Exception('弃用，使用 async_handle_request 代替')
        try:
            if cmd == 'get':
                # 返回模型参数
                weights, version = self.param_server.get_weights()
                # 压缩参数
                weights = self.param_compressor.compress_params_dict(weights)
                send_msg(client_socket, pickle.dumps((weights, version)))
                log(f'{msg_header} Parameters sent, version: {version}')

            elif cmd == 'update_gradients':
                update_data = recv_msg(client_socket)
                if update_data is None:
                    return
                grads, compress_info, version = pickle.loads(update_data)
                log(f'{msg_header} Received gradients, version: {version}')
                # 解压梯度
                grads = self.gradient_compressor.decompress(grads, compress_info)
                # 更新梯度并返回模型参数
                weights = self.param_server.apply_gradients(grads, version)
                # send_msg(client_socket, pickle.dumps(weights))
                # log(f'{msg_header} Send back weights, version: {weights[1]}')

            elif cmd == 'client_id':
                """
                根据ip分配返回客户端id
                若ip已经存在, 返回 id+1
                """
                client_ip = client_socket.getpeername()[0]
                current_time = time.time()

                # 清理过期的ip-id映射并检查当前ip
                if client_ip in self.client_ip_ids:
                    # ip存在,检查是否过期
                    _, timestamp = self.client_ip_ids[client_ip]
                    if current_time - timestamp <= 12 * 3600:
                        # 未过期,id+1 返回
                        self.client_ip_ids[client_ip][0] += 1
                        send_msg(client_socket, str(self.client_ip_ids[client_ip][0]).encode())
                        log(f'{msg_header} ip:{client_ip} exist, Send back client_id:{self.client_ip_ids[client_ip][0]}')
                        return
                    # 已过期,删除
                    del self.client_ip_ids[client_ip]
                    log(f'{msg_header} ip:{client_ip} out of date')

                # 分配新id
                new_id = 0
                self.client_ip_ids[client_ip] = [new_id, current_time]
                send_msg(client_socket, str(new_id).encode())
                log(f'{msg_header} new ip:{client_ip}, Send back client_id: {new_id}')

        except ConnectionResetError:
            pass

    async def async_handle_request(self, msg_header, cmd, data):
        """异步处理客户端请求"""
        if cmd.startswith('get@'):
            log(f'{msg_header} recv get request')
            _client_version = int(cmd.split('@')[1])# TODO 客户端版本号

            # 返回 共享参数
            params = []
            info = None
            v = 0
            with self.share_params_lock:
                # 交换
                for i in self.params_cache_share:
                    params.append(i.data.clone())
                # 取出再放回， 保证队列中仍有数据
                info, v, need_warn_up = self.params_info_share_q.get()
                self.params_info_share_q.put((info, v, need_warn_up))
            log(f'{msg_header} prepare params, version: {v}')
            return pickle.dumps((params, info, v, need_warn_up))

        elif cmd == 'update_gradients':
            log(f'{msg_header} recv update_gradients request')
            gradients_cache_share_length = 0
            g, compress_info, version = pickle.loads(data)
            # 提交到共享梯度信息队列
            # log(f'put gradients info')
            self.gradients_info_share_q.put((compress_info, version))
            # 提交到共享梯度
            # log(f'put gradients')

            wait_count = 0
            while True:
                # 是否是全梯度
                if compress_info[0]['is_full_gradient']:
                    cache_share = self.gradients_cache_share_full
                else:
                    cache_share = self.gradients_cache_share

                with self.share_gradients_lock:

                    gradients_cache_share_length = cache_share[0].size()
                    if gradients_cache_share_length < self.grad_cache_size:
                        for idx, _g in enumerate(g):
                            cache_share[idx].append(_g)
                        gradients_cache_share_length += 1
                        break

                # 释放锁并等待
                log(f'{msg_header} wait gradients, current length: {gradients_cache_share_length}')
                await asyncio.sleep(0.1)

                wait_count += 1
                if wait_count > 10:
                    log(f'{msg_header} wait gradients timeout')
                    import sys
                    sys.exit()

            # if gradients_cache_share_length > 30:
            #     log(f'{msg_header} gradients_cache_share_length > 15')
            #     import sys
            #     sys.exit()

            # 通知新梯度
            # log(f'set share data new event')
            self.share_data_new_event.set()
            # log(f'{msg_header} Received gradients, gradients_cache_share length: {gradients_cache_share_length}')

        elif cmd == 'client_id':
            pass

if __name__ == '__main__':
    pass