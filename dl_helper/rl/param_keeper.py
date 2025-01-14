import torch, time
import multiprocessing

import numpy as np
import pickle, requests
from typing import Dict, Any
from collections import deque

from dl_helper.rl.socket_base import CODE, PORT, send_msg, recv_msg
from dl_helper.rl.rl_utils import GradientCompressor, ParamCompressor

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.tune.registry import _global_registry, ENV_CREATOR

from py_ext.tool import log, share_ndarray_list

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
        """更新参数"""
        log(f'gradients_list length: {len(gradients_list)}')
        params = self.learner._params
        for idx, k in enumerate(params.keys()):
            params[k].grad = gradients_list[idx].to(self.learner._device)
        
        self.learner.apply_gradients({})
        self.ver += 1
        return self.get_weights()
    
    def get_weights(self):
        """获取参数"""
        return (self.learner.get_state(components=COMPONENT_RL_MODULE)['rl_module']['default_policy'], self.ver)

class ExperimentHandler:
    """处理单个实验的类"""
    def __init__(self, train_title, config):
        """
        train_title: 训练标题
        config: RLlib 配置
        """
        # 训练标题
        self.train_title = train_title

        # 梯度压缩器
        self.gradient_compressor = GradientCompressor()

        # 参数压缩器
        self.param_compressor = ParamCompressor()

        # 客户端 IP/id
        self.client_ip_ids = {}

        # 参数服务器
        config = config.learners(    
            num_learners=1,
            num_gpus_per_learner=0,
            num_cpus_per_learner=0.5,
        )
        env_creater = _global_registry.get(ENV_CREATOR, config.env)
        self.env = env_creater()
        self.param_server = AsyncRLParameterServer(config, self.env)

        # 版本号
        self.version = 0

        # 共享梯度
        self.gradients_cache_share = []
        # 共享参数
        self.params_cache_share = self.produce_params_cache()
        self.params_cache_share_version = 0

        # 共享数据锁
        self.share_gradients_lock = multiprocessing.Lock()
        self.share_params_lock = multiprocessing.Lock()

        # 共享数据新增通知
        self.share_data_new_event = multiprocessing.Event()

        # 启动计算进程
        self.p = multiprocessing.Process(target=self.cpu_most_task)
        self.p.start()
    
    def __del__(self):
        self.p.terminate()

    def produce_params_cache(self): 
        """生成发送次数缓存"""
        # 获取模型参数
        weights, version = self.param_server.get_weights()
        # 压缩参数
        weights = self.param_compressor.compress_params_dict(weights)
        return pickle.dumps((weights, version))

    def cpu_most_task(self):
        """CPU密集型任务"""
        log(f'{self.train_title} calculate most start')
        _gradients_cache = []
        while True:
            self.share_data_new_event.wait()
            self.share_data_new_event.clear()

            log(f'{self.train_title} calculate active')

            with self.share_gradients_lock:
                if len(self.gradients_cache_share) == 0:
                    log(f'{self.train_title} no gradients, keep wait')
                    continue
                # 拷贝梯度到临时梯度
                _gradients_cache = self.gradients_cache_share
                self.gradients_cache_share = []

            log(f'{self.train_title} wait gradients: {len(_gradients_cache)}')

            # 计算梯度
            for data in _gradients_cache:
                # 解压梯度
                g, compress_info, version = pickle.loads(data)
                g = self.gradient_compressor.decompress(g, compress_info)
                # 更新梯度
                self.param_server.apply_gradients(g, version)
                # 生成参数缓存
                res = self.produce_params_cache()
                # 更新到共享参数
                with self.share_params_lock:
                    self.params_cache_share = res
                    self.params_cache_share_version = version
                log(f'{self.train_title} update params, version: {version}')

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
        if cmd == 'get':
            # 返回 共享参数
            with self.share_params_lock:
                # 交换
                res = self.params_cache_share
                v = self.params_cache_share_version
            log(f'{msg_header} send params, version: {v}')
            return res

        elif cmd == 'update_gradients':
            gradients_cache_share_length = 0
            with self.share_gradients_lock:
                # 添加到共享梯度信息中
                self.gradients_cache_share.append(data)
                gradients_cache_share_length = len(self.gradients_cache_share)

            # 通知新梯度
            self.share_data_new_event.set()
            log(f'{msg_header} Received gradients, gradients_cache_share length: {gradients_cache_share_length}')

        elif cmd == 'client_id':
            pass

if __name__ == '__main__':
    pass