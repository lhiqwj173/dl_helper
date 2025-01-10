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

class AsyncRLParameterServer_0:
    def __init__(self,
                 agent,
                 learning_rate: float = 0.001,
                 staleness_threshold: int = 20,
                 momentum: float = 0.9,
                 importance_decay: float = 0.8,
                 max_version_delay: int = 100
        ):
        """分布式强化学习参数服务器
        
        Args:
            agent: 
            learning_rate: 基准学习率
            staleness_threshold: 时延阈值
            momentum: 动量因子
            importance_decay: 重要性衰减系数
            max_version_delay: 最大允许版本延迟
        """
        raise NotImplementedError("弃用")

        self.agent = agent
        self.params = dict(agent.get_model_to_sync().named_parameters()) # 不拷贝，直接引用
        self.base_lr = learning_rate
        self.staleness_threshold = staleness_threshold
        self.momentum = momentum
        self.importance_decay = importance_decay
        self.max_version_delay = max_version_delay
            
        # 动量相关状态
        self.velocity = {k: v.new_zeros(v.shape) for k, v in self.params.items()}
        
    def process_update(self, grads, importance, client_version) -> bool:
        """处理客户端的梯度更新
        
        Args:
            grads: 梯度
            importance: 重要性
            client_version: 客户端版本
            
        Returns:
            bool: 更新是否成功
        """
        # 1. 计算版本延迟
        version_delay = self.agent.version - client_version
        
        # 如果延迟太大，直接丢弃
        if version_delay > self.max_version_delay:
            log(f'Version delay too large: {version_delay}')
            return False
        
        # 2. 基于延迟调整重要性权重
        adjusted_importance = self._adjust_importance(importance, version_delay)
        
        # 3. 调整学习率
        effective_lr = self._compute_effective_lr(version_delay)
        
        # 4. 应用梯度更新
        self._apply_gradient_update(grads, effective_lr, adjusted_importance)
        
        # 5. 更新版本号
        self.agent.version += 1

        return True
        
    def _adjust_importance(self, importance: float, delay: int) -> float:
        """基于延迟调整重要性权重
        
        使用指数衰减调整重要性，延迟越大权重越小
        """
        if delay <= self.staleness_threshold:
            return importance
        
        decay_factor = self.importance_decay ** (delay - self.staleness_threshold)
        return importance * decay_factor
        
    def _compute_effective_lr(self, delay: int) -> float:
        """计算考虑延迟的有效学习率
        
        随着延迟增加降低学习率
        """
        if delay <= self.staleness_threshold:
            return self.base_lr
            
        # 使用自适应衰减
        decay = 1.0 / (1.0 + delay - self.staleness_threshold)
        return self.base_lr * decay
        
    def _apply_gradient_update(self,
                             grads: Dict[str, np.ndarray],
                             lr: float,
                             importance: float):
        """应用梯度更新

        使用动量方法更新参数，同时考虑重要性权重。直接使用 PyTorch 操作以提高性能。
        """
        for name, grad in grads.items():
            if name not in self.params:
                continue
                
            # 将 numpy 数组转换为 tensor，并确保设备匹配
            grad_tensor = torch.from_numpy(grad).to(self.params[name].device)
            
            # 1. 更新动量 (直接使用 tensor 运算)
            self.velocity[name].mul_(self.momentum).add_(
                grad_tensor * importance * (1 - self.momentum)
            )
            
            # 2. 应用梯度更新 (使用 PyTorch 的 in-place 操作)
            self.params[name].data.add_(self.velocity[name], alpha=-lr)

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
                send_msg(client_socket, pickle.dumps(weights))
                log(f'{msg_header} Send back weights, version: {weights[1]}')

            elif cmd == 'client_id':
                """
                根据ip分配返回客户端id
                若ip已经存在, 返回 ''
                """
                client_ip = msg_header['ip']
                current_time = time.time()

                # 清理过期的ip-id映射并检查当前ip
                if client_ip in self.client_ip_ids:
                    # ip存在,检查是否过期
                    if current_time - int(self.client_ip_ids[client_ip])/1000 <= 12 * 3600:
                        # 未过期,返回空
                        send_msg(client_socket, b'')
                        return
                    # 已过期,删除
                    del self.client_ip_ids[client_ip]

                # 分配新id(毫秒时间戳)
                new_id = str(int(current_time * 1000))
                self.client_ip_ids[client_ip] = new_id
                send_msg(client_socket, new_id.encode())

        except ConnectionResetError:
            pass


if __name__ == '__main__':
    pass