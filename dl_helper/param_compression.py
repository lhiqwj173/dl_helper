import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

CompressInfo = Dict[str, List[torch.Tensor]]

class IncrementalCompressor:
    def __init__(self, threshold: float = 1e-3):
        """
        参数:
            threshold: 压缩阈值,只压缩变化大于此值的参数
        """
        self.threshold = threshold
        self.client_params = {}  # 存储不同客户端的参数 {client_id: List[tensor]}
        
    def _init_reference(self, 
                       client_id: str,
                       tensors: List[torch.Tensor],
                       init: bool = False
                      ) -> None:
        """
        初始化参考张量
        
        参数:
            client_id: 客户端ID
            tensors: 要初始化的张量列表
            init: 是否强制初始化
        """
        if init or client_id not in self.client_params:
            self.client_params[client_id] = [t.clone().detach() for t in tensors]
        
    def compress(self, 
                tensors: List[torch.Tensor],
                client_id: str,
                init: bool = False
               ) -> Tuple[List[torch.Tensor], CompressInfo]:
        """
        压缩张量列表
        
        参数:
            tensors: 要压缩的张量列表
            client_id: 客户端ID
            init: 是否初始化客户端参数
            
        返回:
            compressed_tensors: 压缩后的张量列表
            compress_info: 压缩信息字典
        """
        self._init_reference(client_id, tensors, init)
        if init:
            # 全量返回
            return tensors, {'full': True,}
        
        compressed_tensors = []
        compress_info = {
            'update_indices': [],
        }
        
        for curr_t, last_t in zip(tensors, self.client_params[client_id]):
            # 计算变化量
            diff = torch.abs(curr_t - last_t)
            mask = diff > self.threshold
            update_indices = torch.nonzero(mask).squeeze()
            
            # 获取需要更新的值
            update_values = diff[update_indices]
            
            # 保存压缩信息
            compress_info['update_indices'].append(update_indices)
            
            compressed_tensors.append(update_values)
            
            # 更新参考张量
            last_t[mask] = curr_t[mask]
            
        return compressed_tensors, compress_info
    
    def decompress(self,
                  compressed_tensors: List[torch.Tensor],
                  compress_info: CompressInfo,
                  param_dict: Dict[str, torch.Tensor]
                 ) -> None:
        """
        解压张量列表并直接更新参数字典
        
        参数:
            compressed_tensors: 压缩后的张量列表
            compress_info: 压缩信息字典
            param_dict: 模型参数字典，会被直接修改
            
        返回:
            None
        """
        param_names = list(param_dict.keys())
        
        if 'full' not in compress_info:
            for param_name, compressed_t, indices in zip(
                param_names,
                compressed_tensors,
                compress_info['update_indices'],
            ):
                # 获取参数张量
                param_tensor = param_dict[param_name]
                
                # 使用压缩的值更新
                if indices.numel() > 0:  # 有需要更新的值
                    param_tensor[indices[:, 0], *[indices[:, i] for i in range(1, indices.shape[1])]] = compressed_t

        else:
            # 全量更新
            for param_name, compressed_t in zip(param_names, compressed_tensors):
                param_dict[param_name][:] = compressed_t[:]