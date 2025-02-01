import torch
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

CompressInfo = Dict[str, List[torch.Tensor]]

class IncrementalCompressor:
    def __init__(self, 
                 threshold: float = 1e-3,
                 sparsity_threshold: float = 0.3  # 稀疏度阈值，超过则全量更新
                ):
        """
        参数:
            threshold: 压缩阈值,只压缩变化大于此值的参数
            sparsity_threshold: 稀疏度阈值，当更新元素比例超过此值时切换为全量更新
        """
        self.threshold = threshold
        self.sparsity_threshold = sparsity_threshold
        self.client_params = {}  # 存储不同客户端的参数 {client_id: List[tensor]}
        
    def _init_reference(self, 
                       client_id: str,
                       tensors: List[torch.Tensor],
                       init: bool = False
                      ) -> None:
        """初始化参考张量"""
        if init or client_id not in self.client_params:
            self.client_params[client_id] = [t.clone().detach() for t in tensors]
        
    def compress(self, 
                tensors: List[torch.Tensor],
                client_id: str,
                init: bool = False
               ) -> Tuple[List[torch.Tensor], CompressInfo]:
        """压缩张量列表"""
        self._init_reference(client_id, tensors, init)
        if init:
            return tensors, {'full': True}
        
        compressed_tensors = []
        compress_info = {
            'update_indices': [],
            'full': []
        }
        
        for curr_t, last_t in zip(tensors, self.client_params[client_id]):
            # 计算变化量
            diff = torch.abs(curr_t - last_t)
            mask = diff > self.threshold
            
            # 计算更新比例
            update_ratio = mask.sum().item() / mask.numel()
            
            # 根据更新比例决定使用全量更新还是增量更新
            if update_ratio > self.sparsity_threshold:
                # 全量更新
                compressed_tensors.append(curr_t)
                compress_info['full'].append(True)
                compress_info['update_indices'].append(None)
                last_t[:] = curr_t[:]
            else:
                # 增量更新
                update_indices = torch.where(mask)
                update_values = curr_t[mask]
                
                compressed_tensors.append(update_values)
                compress_info['update_indices'].append(torch.stack(update_indices, dim=1))
                compress_info['full'].append(False)
                
                # 更新参考张量
                last_t[mask] = curr_t[mask]
            
        return compressed_tensors, compress_info
    
    @staticmethod
    def decompress(
                  compressed_tensors: List[torch.Tensor],
                  compress_info: CompressInfo,
                  param_dict: Dict[str, torch.Tensor]
                ) -> None:
        """解压张量列表并直接更新参数字典"""
        param_names = list(param_dict.keys())
        
        if isinstance(compress_info.get('full'), bool):
            # 全部全量更新
            for param_name, compressed_t in zip(param_names, compressed_tensors):
                param_dict[param_name][:] = compressed_t[:]
            return
            
        # 混合更新模式
        for param_name, compressed_t, is_full, indices in zip(
            param_names,
            compressed_tensors,
            compress_info['full'],
            compress_info['update_indices']
        ):
            if is_full:
                # 全量更新
                param_dict[param_name][:] = compressed_t[:]
            else:
                # 增量更新
                if indices.numel() > 0:
                    param_dict[param_name][tuple(indices[:, i] for i in range(indices.shape[1]))] = compressed_t