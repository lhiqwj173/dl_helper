import torch, time

import numpy as np
import pickle, requests
from typing import Dict, Any
from collections import deque

from dl_helper.rl.socket_base import CODE, PORT, send_msg, recv_msg

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.tune.registry import _global_registry, ENV_CREATOR

from py_ext.tool import log

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
        
    def apply_gradients(self, gradients_dict, client_version):
        """更新参数"""
        self.learner.apply_gradients(gradients_dict)
        self.ver += 1
        return self.get_weights()
    
    def get_weights(self):
        """获取参数"""
        return (self.learner.get_state(components=COMPONENT_RL_MODULE), self.ver)

class ExperimentHandler:
    """处理单个实验的类"""
    def __init__(self, train_title, config):
        """
        train_title: 训练标题
        config: RLlib 配置
        """
        # 训练标题
        self.train_title = train_title

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

    def handle_request(self, client_socket, msg_header, cmd):
        """处理客户端请求"""
        try:
            if cmd == 'get':
                # 返回模型参数
                weights = self.param_server.get_weights()
                send_msg(client_socket, pickle.dumps(weights))
                log(f'{msg_header} Parameters sent, version: {weights[1]}')

            elif cmd == 'update_gradients':
                update_data = recv_msg(client_socket)
                if update_data is None:
                    return
                grads, version = pickle.loads(update_data)
                log(f'{msg_header} Received gradients, version: {version}')
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


if __name__ == '__main__':
    pass