import torch, time, math
import multiprocessing

import numpy as np
import pickle, requests
from typing import Dict, Any
from collections import deque
import copy
from dl_helper.rl.socket_base import CODE, PORT, send_msg, recv_msg
from dl_helper.rl.rl_utils import GradientCompressor, ParamCompressor

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.tune.registry import _global_registry, ENV_CREATOR

from py_ext.tool import log, share_ndarray_list, share_ndarray

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
        return self.get_weights()
    
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
    def __init__(self, train_title, config, debug=False):
        """
        train_title: 训练标题
        config: RLlib 配置
        """
        # 训练标题
        self.train_title = train_title
        self.debug = debug

        # 客户端 IP/id
        self.client_ip_ids = {}

        # 版本号
        self.version = 0

        # 共享梯度队列
        # 梯度数据经过稀疏化，形状
        self.gradients_info_share_q = multiprocessing.Queue()
        # 共享参数信息队列
        self.params_info_share_q = multiprocessing.Queue()

        # 共享梯度锁
        self.share_gradients_lock = multiprocessing.Lock()
        # 共享参数锁
        self.share_params_lock = multiprocessing.Lock()

        # 共享数据新增通知
        self.share_data_new_event = multiprocessing.Event()

        # 启动计算进程
        self.p = multiprocessing.Process(target=ExperimentHandler.cpu_most_task, args=(
            train_title, self.gradients_info_share_q, self.params_info_share_q, self.share_data_new_event, self.share_gradients_lock, self.share_params_lock,
            config,
            self.debug,
        ))
        self.p.start()

        # 等待接受 参数的形状列表
        # 用于初始化共享数据
        _simple_params, _simple_grad_params = self.gradients_info_share_q.get()
        # 共享梯度列表
        self.gradients_cache_share = []
        # 共享参数, 只需要维护一份最新的数据
        self.params_cache_share = []
        # 初始化共享梯度
        for idx, _shape in enumerate(_simple_grad_params):
            self.gradients_cache_share.append(share_ndarray_list(f'{self.train_title}_gcs_{idx}', _shape, 'int8', 30, debug=self.debug))
        # 初始化共享参数
        for idx, _shape in enumerate(_simple_params):
            self.params_cache_share.append(share_ndarray(f'{self.train_title}_pcs_{idx}', _shape, 'int8'))
    
    def __del__(self):
        self.p.terminate()

    @staticmethod
    def cpu_most_task(
        train_title, gradients_info_share_q, params_info_share_q, share_data_new_event, share_gradients_lock, share_params_lock, 
        config, 
        debug,
    ):
        """CPU密集型任务"""
        def produce_params_cache(param_server, param_compressor): 
            """生成发送次数缓存"""
            log(f'produce params cache')
            # 获取模型参数
            # weights： 参数字典[torch.Tensor]
            weights, version = param_server.get_weights()
            # 压缩参数
            weights, info = param_compressor.compress_params_dict(weights)
            return weights, info, version
        
        def update_params(lock, q, params_list, info, version, params_cache_share):
            """更新参数信息"""
            log(f'update params')
            with lock:
                # 清空q
                while not q.empty():
                    q.get()
                # 最新数据
                q.put((info, version))
                # 更新参数
                for idx, p in enumerate(params_list):
                    params_cache_share[idx].data[:] = p[:]

        log(f'{train_title} calculate most init')

        # 参数服务器
        config = config.learners(    
            num_learners=1,
            num_gpus_per_learner=0,
            num_cpus_per_learner=0.5,
        )
        env_creater = _global_registry.get(ENV_CREATOR, config.env)
        env = env_creater()
        param_server = AsyncRLParameterServer(config, env)
        _params_dict = param_server.get_weights()[0] 
        _grad_params_dict = param_server.get_gradients_params()

        # 梯度压缩器
        gradient_compressor = GradientCompressor()

        # 参数压缩器
        param_compressor = ParamCompressor()

        # 共享梯度
        gradients_cache_share = []
        # 计算用临时梯度
        gradients_cache_temp = []
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
            _shape = (math.prod(v.shape),)
            params_cache_share.append(share_ndarray(f'{train_title}_pcs_{idx}', _shape, 'int8'))
            _simple_params.append(_shape)

        # 初始化共享梯度
        _simple_grad_params = []
        for idx, (k, v) in enumerate(_grad_params_dict.items()):
            _compress_shape = gradient_compressor.compress_shape(v.shape)
            log(f'{train_title} init gradients share, idx: {idx}, shape: {v.shape}, compress shape: {_compress_shape}')
            gradients_cache_share.append(share_ndarray_list(f'{train_title}_gcs_{idx}', _compress_shape, 'int8', 30, debug=debug))
            gradients_cache_temp.append(gradients_cache_share[idx].get_blank_same_data_local())
            _simple_grad_params.append(_compress_shape)

        # 初始化一个最新的参数/info
        weights, info, version = produce_params_cache(param_server, param_compressor)
        update_params(share_params_lock, params_info_share_q, weights, info, version, params_cache_share)
        
        # 回传 参数形状列表 
        # 回传后，共享参数以及初始化完成
        gradients_info_share_q.put((_simple_params, _simple_grad_params))

        log(f'{train_title} calculate most start')
        while True:
            share_data_new_event.wait()
            share_data_new_event.clear()

            log(f'{train_title} calculate active')

            with share_gradients_lock:
                temp_length = gradients_cache_share[0].size()
                if temp_length == 0:
                    log(f'{train_title} no gradients, keep wait')
                    continue
                # 拷贝梯度到临时梯度
                for idx, _g in enumerate(gradients_cache_share):
                    _g.all_copy_slice(gradients_cache_temp[idx], 0)
                # 获取全部的梯度信息
                for idx in range(temp_length):
                    gradients_cache_info_temp.append(gradients_info_share_q.get())  

            log(f'{train_title} wait gradients: {temp_length}')

            # 计算梯度
            for idx in range(temp_length):
                # 获取梯度列表
                # log(f'get gradients')
                gs = [i[idx] for i in gradients_cache_temp]
                # 解压梯度
                # log(f'decompress gradients')
                info, version = gradients_cache_info_temp[idx]
                gs = gradient_compressor.decompress(gs, info)
                # 更新梯度
                # log(f'update gradients')
                param_server.apply_gradients(gs, version)
                # 生成参数缓存
                # log(f'produce params cache')
                weights, info, version = produce_params_cache(param_server, param_compressor)
                update_params(share_params_lock, params_info_share_q, weights, info, version, params_cache_share)
                log(f'{train_title} update params, version: {version}')

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
                info, v = self.params_info_share_q.get()
                self.params_info_share_q.put((info, v))
            log(f'{msg_header} send params, version: {v}')
            return pickle.dumps((params, info, v))

        elif cmd == 'update_gradients':
            log(f'{msg_header} update gradients')
            gradients_cache_share_length = 0
            g, compress_info, version = pickle.loads(data)
            # 提交到共享梯度信息队列
            # log(f'put gradients info')
            self.gradients_info_share_q.put((compress_info, version))
            # 提交到共享梯度
            # log(f'put gradients')
            with self.share_gradients_lock:
                for idx, _g in enumerate(g):
                    # log(f'append gradients, idx: {idx}, shape: {_g.shape} > {self.gradients_cache_share[idx]._data[0].shape}')
                    self.gradients_cache_share[idx].append(_g)
                # gradients_cache_share_length = self.gradients_cache_share[0].size()

            # 通知新梯度
            # log(f'set share data new event')
            self.share_data_new_event.set()
            # log(f'{msg_header} Received gradients, gradients_cache_share length: {gradients_cache_share_length}')

        elif cmd == 'client_id':
            pass

if __name__ == '__main__':
    pass