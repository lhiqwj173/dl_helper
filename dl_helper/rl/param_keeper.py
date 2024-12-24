import numpy as np
import pickle
from typing import Dict, Any
from collections import deque

from py_ext.tool import log

class AsyncRLParameterServer:
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
        self.agent = agent
        self.params = dict(agent.get_model_to_sync().named_parameters()) # 不拷贝，直接引用
        self.base_lr = learning_rate
        self.staleness_threshold = staleness_threshold
        self.momentum = momentum
        self.importance_decay = importance_decay
        self.max_version_delay = max_version_delay
            
        # 动量相关状态
        self.velocity = {k: np.zeros_like(v) for k, v in self.params.items()}
        
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
        
        使用动量方法更新参数，同时考虑重要性权重
        """
        for name, grad in grads.items():
            if name not in self.params:
                continue
                
            # 1. 更新动量
            self.velocity[name] = self.momentum * self.velocity[name] + \
                                (1 - self.momentum) * grad * importance
            
            # 2. 应用梯度更新
            self.params[name] -= lr * self.velocity[name]
            