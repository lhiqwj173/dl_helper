import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class CompressInfo:
    """压缩信息数据类"""
    masks: List[torch.Tensor]  # 更新掩码
    ref_tensors: List[torch.Tensor]  # 参考张量
    update_indices: List[torch.Tensor]  # 更新索引
    shapes: List[torch.Size]  # 原始形状

class IncrementalCompressor:
    def __init__(self, threshold: float = 1e-3):
        """
        参数:
            threshold: 压缩阈值,只压缩变化大于此值的参数
        """
        self.threshold = threshold
        self.last_tensors = None  # 存储上一次的参数
        
    def _init_reference(self, tensors: List[torch.Tensor]) -> None:
        """初始化参考张量"""
        if self.last_tensors is None:
            self.last_tensors = [t.clone().detach() for t in tensors]
        
    def compress(self, 
                tensors: List[torch.Tensor]
               ) -> Tuple[List[torch.Tensor], CompressInfo]:
        """
        压缩张量列表
        
        参数:
            tensors: 要压缩的张量列表
            
        返回:
            compressed_tensors: 压缩后的张量列表
            compress_info: 压缩信息
        """
        self._init_reference(tensors)
        
        masks = []
        compressed_tensors = []
        update_indices = []
        shapes = []
        
        for curr_t, last_t in zip(tensors, self.last_tensors):
            # 计算变化量
            diff = torch.abs(curr_t - last_t)
            mask = diff > self.threshold
            
            # 获取需要更新的值和索引
            update_idx = torch.nonzero(mask, as_tuple=False)
            update_values = curr_t[mask]
            
            # 保存压缩信息
            masks.append(mask)
            compressed_tensors.append(update_values)
            update_indices.append(update_idx)
            shapes.append(curr_t.shape)
            
            # 更新参考张量
            last_t[mask] = curr_t[mask]
            
        compress_info = CompressInfo(
            masks=masks,
            ref_tensors=self.last_tensors,
            update_indices=update_indices,
            shapes=shapes
        )
        
        return compressed_tensors, compress_info
    
    def decompress(self,
                  compressed_tensors: List[torch.Tensor],
                  compress_info: CompressInfo
                 ) -> List[torch.Tensor]:
        """
        解压张量列表
        
        参数:
            compressed_tensors: 压缩后的张量列表
            compress_info: 压缩信息
            
        返回:
            decompressed_tensors: 解压后的张量列表
        """
        decompressed_tensors = []
        
        for compressed_t, ref_t, indices, shape in zip(
            compressed_tensors,
            compress_info.ref_tensors,
            compress_info.update_indices,
            compress_info.shapes
        ):
            # 创建新张量并填充参考值
            decompressed = ref_t.clone()
            
            # 使用压缩的值更新
            if indices.numel() > 0:  # 有需要更新的值
                decompressed[indices[:, 0], *[indices[:, i] for i in range(1, indices.shape[1])]] = compressed_t
                
            decompressed_tensors.append(decompressed)
            
        return decompressed_tensors