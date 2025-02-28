import torch
from typing import List, Tuple, Dict
import concurrent.futures
from functools import partial
import pickle
import numpy as np

from py_ext.tool import log

CompressInfo = Dict[str, List[torch.Tensor]]
class IncrementalCompressor:
    def __init__(self, 
                 threshold: float = 1e-3,
                 sparsity_threshold: float = 0.3,
                 min_sparsity_threshold: float = 0.01,
                 max_workers: int = None  # 添加并行工作线程数参数
                ):
        """
        参数:
            threshold: 压缩阈值,只压缩变化大于此值的参数
            sparsity_threshold: 稀疏度阈值，当更新元素比例超过此值时切换为全量更新
            min_sparsity_threshold: 最小稀疏度阈值，当更新元素比例小于此值时，取最大的 n 个元素更新（n>=1 由稀疏度阈值决定）
            max_workers: 最大并行工作线程数，None表示使用默认值(CPU核心数)
        """
        self.threshold = threshold
        self.sparsity_threshold = sparsity_threshold
        self.min_sparsity_threshold = min_sparsity_threshold
        self.max_workers = max_workers

        self.client_params = {}  # 存储不同客户端的参数 {client_id: List[tensor]}
        # 创建一个持久化的线程池
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
    def _calibrate_sparsity_threshold(self, tensors: List[torch.Tensor], default=0.3) -> float:
        """
        Dynamically calibrates the sparsity threshold by measuring the serialized size
        of incremental vs full updates across various update ratios.
        
        Args:
            tensors: List of tensors to use for calibration
            
        Returns:
            optimal_threshold: The update ratio at which incremental updates become
                            less efficient than full updates
        """
        # Create sample tensor with same shape as the first tensor for testing
        sample_tensor = tensors[0].clone()
        ref_tensor = sample_tensor.clone()
        
        # Define update ratios to test (from 0.01 to 0.6)
        update_ratios = np.linspace(0.01, 0.6, 20)
        
        # Store sizes for each update ratio
        incremental_sizes = []
        
        # Get full update size once
        full_update = sample_tensor.clone()
        full_size = len(pickle.dumps((full_update, {'full': True})))
        
        print(f"Full update size: {full_size} bytes")
        
        # Test each update ratio
        for ratio in update_ratios:
            # Create a random mask with the specified update ratio
            mask = torch.zeros_like(sample_tensor, dtype=torch.bool)
            num_elements = sample_tensor.numel()
            num_updates = int(ratio * num_elements)
            
            # Randomly select indices to update
            flat_indices = torch.randperm(num_elements)[:num_updates]
            mask.view(-1)[flat_indices] = True
            
            # Create incremental update
            updated_tensor = ref_tensor.clone()
            updated_tensor[mask] += torch.randn_like(updated_tensor[mask]) * 0.1  # Small random updates
            
            # Get update values and indices
            update_indices = torch.where(mask)
            update_values = updated_tensor[mask]
            
            # Measure serialized size
            compress_info = {
                'update_indices': [torch.stack(update_indices, dim=1)],
                'full': [False]
            }
            incremental_size = len(pickle.dumps((update_values, compress_info)))
            incremental_sizes.append(incremental_size)
            
            print(f"Update ratio: {ratio:.2f}, Incremental size: {incremental_size} bytes")
        
        # Find the threshold where incremental updates are still smaller than full updates
        threshold_indices = np.where(np.array(incremental_sizes) <= full_size * 0.8)[0]
        
        if len(threshold_indices) > 0:
            # Get the last update ratio where incremental update is still efficient
            optimal_threshold = update_ratios[threshold_indices[-1]]
        else:
            optimal_threshold = default
        
        print(f"Optimal sparsity threshold: {optimal_threshold:.4f}")
        return optimal_threshold


    def _init_reference(self, 
                       client_id: str,
                       tensors: List[torch.Tensor],
                      ) -> bool:
        if len(self.client_params) == 0:
            # 动态校准稀疏度阈值
            self.sparsity_threshold = self._calibrate_sparsity_threshold(tensors)
            log(f"Calibrated sparsity threshold: {self.sparsity_threshold}")

        if client_id not in self.client_params:
            # 需要拷贝，不需要额外处理设备
            self.client_params[client_id] = [t.clone() for t in tensors]
            return True
        return False
    
    def _compress_single_client(self,
                              client_id: str,
                              tensors: List[torch.Tensor],
                              return_need_clone: bool = False
                             ) -> Tuple[str, List[torch.Tensor], CompressInfo]:
        """处理单个客户端的压缩操作"""
        if self._init_reference(client_id, tensors):
            # 初始化时直接返回参考张量的克隆
            if return_need_clone:
                return client_id, [t.clone() for t in self.client_params[client_id]], {'full': True}
            else:
                return client_id, self.client_params[client_id], {'full': True}
        
        compressed_tensors = []
        compress_info = {
            'update_indices': [],
            'full': []
        }
        
        for curr_t_cpu, ref_t in zip(tensors, self.client_params[client_id]):
            with torch.no_grad():  # 使用no_grad减少内存使用
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
                        n = max(1, int(self.min_sparsity_threshold * mask.numel()))
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
    
        # 返回结果
        return client_id, compressed_tensors, compress_info
        
    def compress(self, 
                raw_tensors: List[torch.Tensor],
                client_ids: str | List[str],
                return_need_clone: bool = False,
               ) -> Dict[str, Tuple[List[torch.Tensor], CompressInfo]]:
        """
        并行压缩张量列表, 确保压缩结果在CPU上
        返回一个字典
        {
            id: (compressed_tensors, compress_info),
            ...
        }
        """
        # 预先将所有张量都移至CPU，减少各个id重复操作
        tensors = [t.detach().cpu().clone() for t in raw_tensors]
        if not isinstance(client_ids, list):
            client_ids = [client_ids]

        # 如果只有一个客户端，直接处理无需并行
        if len(client_ids) == 1:
            client_id, compressed_tensors, compress_info = self._compress_single_client(
                client_ids[0], tensors, return_need_clone
            )
            return {client_id: (compressed_tensors, compress_info)}
        
        # 使用持久化的线程池并行处理多个客户端
        res = {}
        # 创建一个部分应用的函数，固定tensors和return_need_clone参数
        process_func = partial(
            self._compress_single_client, 
            tensors=tensors, 
            return_need_clone=return_need_clone
        )
        
        # 并行提交所有客户端的压缩任务
        future_to_client = {
            self.thread_pool.submit(process_func, client_id): client_id 
            for client_id in client_ids
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_client):
            client_id, compressed_tensors, compress_info = future.result()
            res[client_id] = (compressed_tensors, compress_info)
        
        return res
    
    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()
    
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