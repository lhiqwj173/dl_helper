import torch
import numpy as np
from typing import List, Dict
from collections import OrderedDict

from dl_helper.param_compression import IncrementalCompressor

def calculate_size(tensors: List[torch.Tensor]) -> int:
    """计算张量列表的总大小（字节）"""
    return sum(t.nelement() * t.element_size() for t in tensors)

def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """计算压缩率"""
    return (original_size - compressed_size) / original_size * 100

def print_compression_stats(name: str, original_size: int, compressed_size: int):
    """打印压缩统计信息"""
    ratio = calculate_compression_ratio(original_size, compressed_size)
    print(f"\n{name}:")
    print(f"原始大小: {original_size/1024:.2f} KB")
    print(f"压缩后大小: {compressed_size/1024:.2f} KB")
    print(f"压缩率: {ratio:.2f}%")

def test_compressor():
    # 初始化压缩器
    compressor = IncrementalCompressor(threshold=1e-3)
    
    # 测试用例1：大规模稀疏更新
    print("\n=== 测试1：大规模稀疏更新 ===")
    large_tensor1 = torch.randn(1000, 1000)
    large_tensor2 = large_tensor1.clone()
    # 只修改1%的值
    indices = torch.randint(0, 1000, (100, 2))
    large_tensor2[indices[:, 0], indices[:, 1]] += 0.1
    
    params = OrderedDict([
        ('large_sparse', large_tensor2)
    ])
    tensors = list(params.values())
    original_size = calculate_size(tensors)
    
    # 压缩
    compressed_tensors, info = compressor.compress(tensors, "client1", init=True)
    compressed_size = calculate_size(compressed_tensors)
    print_compression_stats("大规模稀疏更新", original_size, compressed_size)
    
    # 解压验证
    compressor.decompress(compressed_tensors, info, params)
    diff = torch.abs(large_tensor2 - params['large_sparse']).max()
    print(f"最大误差: {diff}")

    # 测试用例2：小规模密集更新
    print("\n=== 测试2：小规模密集更新 ===")
    small_tensor1 = torch.randn(10, 10)
    small_tensor2 = small_tensor1 + 0.1  # 所有值都有变化
    
    params = OrderedDict([
        ('small_dense', small_tensor2)
    ])
    tensors = [small_tensor2]
    original_size = calculate_size(tensors)
    
    compressed_tensors, info = compressor.compress(tensors, "client2", init=True)
    compressed_size = calculate_size(compressed_tensors)
    print_compression_stats("小规模密集更新", original_size, compressed_size)

    # 测试用例3：混合更新（多个张量）
    print("\n=== 测试3：混合更新（多个张量）===")
    # 创建不同维度的张量
    tensor1 = torch.randn(100, 100)  # 2D
    tensor2 = torch.randn(50, 30, 20)  # 3D
    tensor3 = torch.randn(1000)  # 1D
    
    # 模拟更新
    updated_tensor1 = tensor1.clone()
    updated_tensor2 = tensor2.clone()
    updated_tensor3 = tensor3.clone()
    
    # 不同的更新模式
    updated_tensor1[0:10, 0:10] += 0.1  # 局部块更新
    updated_tensor2[:, 0, :] += 0.1  # 切片更新
    updated_tensor3[::100] += 0.1  # 稀疏更新
    
    params = OrderedDict([
        ('tensor1', updated_tensor1),
        ('tensor2', updated_tensor2),
        ('tensor3', updated_tensor3)
    ])
    tensors = list(params.values())
    original_size = calculate_size(tensors)
    
    compressed_tensors, info = compressor.compress(tensors, "client3", init=True)
    compressed_size = calculate_size(compressed_tensors)
    print_compression_stats("混合更新", original_size, compressed_size)

    # 测试用例4：极端值测试
    print("\n=== 测试4：极端值测试 ===")
    # 创建包含极端值的张量
    extreme_tensor1 = torch.tensor([float('inf'), float('-inf'), float('nan'), 1e38, -1e38, 0])
    extreme_tensor2 = extreme_tensor1.clone()
    extreme_tensor2[0] = 1e39  # 更改一个值
    
    params = OrderedDict([
        ('extreme', extreme_tensor2)
    ])
    tensors = [extreme_tensor2]
    original_size = calculate_size(tensors)
    
    compressed_tensors, info = compressor.compress(tensors, "client4", init=True)
    compressed_size = calculate_size(compressed_tensors)
    print_compression_stats("极端值", original_size, compressed_size)

    # 测试用例5：连续更新测试
    print("\n=== 测试5：连续更新测试 ===")
    base_tensor = torch.randn(100, 100)
    params = OrderedDict([
        ('continuous', base_tensor.clone())
    ])
    
    total_original_size = 0
    total_compressed_size = 0
    
    # 模拟5次连续更新
    for i in range(5):
        print(f"\n更新 {i+1}:")
        # 随机更新一些值
        update_mask = torch.rand(100, 100) > 0.8
        params['continuous'][update_mask] += 0.1
        
        tensors = [params['continuous']]
        original_size = calculate_size(tensors)
        total_original_size += original_size
        
        # 使用相同的client_id进行压缩
        compressed_tensors, info = compressor.compress(tensors, "client5", init=(i==0))
        compressed_size = calculate_size(compressed_tensors)
        total_compressed_size += compressed_size
        
        print_compression_stats(f"连续更新 {i+1}", original_size, compressed_size)
    
    # 打印总体压缩率
    print("\n总体压缩统计:")
    print_compression_stats("所有更新总计", total_original_size, total_compressed_size)

if __name__ == "__main__":
    test_compressor()