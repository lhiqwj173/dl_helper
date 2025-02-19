import torch
from typing import List, Tuple, Dict

CompressInfo = Dict[str, List[torch.Tensor]]

class IncrementalCompressor:
    def __init__(self, 
                 threshold: float = 1e-3,
                 sparsity_threshold: float = 0.3,
                 min_sparsity_threshold: float = 0.01
                ):
        """
        参数:
            threshold: 压缩阈值,只压缩变化大于此值的参数
            sparsity_threshold: 稀疏度阈值，当更新元素比例超过此值时切换为全量更新
            min_sparsity_threshold: 最小稀疏度阈值，当更新元素比例小于此值时，取最大的 n 个元素更新（n>=1 由稀疏度阈值决定）
        """
        self.threshold = threshold
        self.sparsity_threshold = sparsity_threshold
        self.min_sparsity_threshold = min_sparsity_threshold

        self.client_params = {}  # 存储不同客户端的参数 {client_id: List[tensor]}
        
    def _init_reference(self, 
                       client_id: str,
                       tensors: List[torch.Tensor],
                      ) -> bool:
        """初始化参考张量，统一在CPU上存储"""
        if client_id not in self.client_params:
            # 直接在CPU上创建参考张量，使用clone确保内存独立
            self.client_params[client_id] = [t.detach().cpu().clone() for t in tensors]
            return True
        return False
        
    def compress(self, 
                tensors: List[torch.Tensor],
                client_id: str,
               ) -> Tuple[List[torch.Tensor], CompressInfo]:
        """压缩张量列表, 确保压缩结果在CPU上"""
        if self._init_reference(client_id, tensors):
            # 初始化时直接返回参考张量的克隆
            return [t.clone() for t in self.client_params[client_id]], {'full': True}
        
        compressed_tensors = []
        compress_info = {
            'update_indices': [],
            'full': []
        }
        
        for curr_t, ref_t in zip(tensors, self.client_params[client_id]):
            # 在CPU上计算差异，避免多余的设备转换
            with torch.no_grad():  # 使用no_grad减少内存使用
                curr_t_cpu = curr_t.cpu()
                diff = torch.abs(curr_t_cpu - ref_t)
                mask = diff > self.threshold
                
                # 计算更新比例
                update_ratio = mask.sum().item() / mask.numel()
                
                if update_ratio > self.sparsity_threshold:
                    # 全量更新 - 使用clone确保数据独立性
                    compressed_t = curr_t_cpu.clone()
                    compressed_tensors.append(compressed_t)
                    compress_info['full'].append(True)
                    compress_info['update_indices'].append(None)
                    # 更新参考张量
                    ref_t.copy_(curr_t_cpu)
                else:
                    if update_ratio < self.min_sparsity_threshold:
                        # 取最大的 n 个元素更新（n>=1 由稀疏度阈值决定）
                        n = max(1, int(update_ratio * mask.numel()))
                        # 修改 mask
                        _, top_indices = torch.topk(diff.flatten(), n)
                        mask = torch.zeros_like(diff, dtype=torch.bool)
                        mask.view(-1)[top_indices] = True

                    # 增量更新 - 只复制需要更新的值
                    update_indices = torch.where(mask)
                    # 直接使用索引获取更新值，无需额外克隆
                    update_values = curr_t_cpu[mask]
                    
                    compressed_tensors.append(update_values)
                    # 将索引信息存储在CPU上
                    compress_info['update_indices'].append(
                        torch.stack(update_indices, dim=1)
                    )
                    compress_info['full'].append(False)
                    
                    # 更新参考张量中变化的部分
                    ref_t[mask] = update_values
            
        return compressed_tensors, compress_info
    
    @staticmethod
    def decompress(
                  compressed_tensors: List[torch.Tensor],
                  compress_info: CompressInfo,
                  param_dict: Dict[str, torch.Tensor]
                ) -> None:
        """解压张量列表并直接更新参数字典，保持目标张量在原设备上"""
        param_names = list(param_dict.keys())
        
        with torch.no_grad():  # 使用no_grad避免保存不必要的梯度信息
            if isinstance(compress_info.get('full'), bool):
                # 全部全量更新
                for param_name, compressed_t in zip(param_names, compressed_tensors):
                    target_tensor = param_dict[param_name]
                    # 直接将压缩的张量复制到目标设备
                    target_tensor.copy_(compressed_t.to(target_tensor.device, non_blocking=True))
                return
                
            # 混合更新模式
            for param_name, compressed_t, is_full, indices in zip(
                param_names,
                compressed_tensors,
                compress_info['full'],
                compress_info['update_indices']
            ):
                target_tensor = param_dict[param_name]
                target_device = target_tensor.device
                
                if is_full:
                    # 全量更新 - 使用non_blocking=True提高性能
                    target_tensor.copy_(compressed_t.to(target_device, non_blocking=True))
                else:
                    # 增量更新 - 只更新变化的部分
                    if indices.numel() > 0:
                        index_tuple = tuple(indices[:, i] for i in range(indices.shape[1]))
                        # 使用to()时启用non_blocking加速设备间传输
                        target_tensor[index_tuple] = compressed_t.to(
                            target_device, 
                            non_blocking=True
                        )