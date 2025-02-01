import torch, pickle
import numpy as np
from typing import List, Dict
from collections import OrderedDict

from dl_helper.param_compression import IncrementalCompressor

def calculate_size(*obj) -> int:
    """
    计算 pickle 序列化后的大小
    """
    return len(pickle.dumps(obj))

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

def _test_init_and_update(tensor1, tensor2):
    # 初始化压缩器
    compressor = IncrementalCompressor(threshold=1e-3)

    model_params = OrderedDict([
        ('tensor1', tensor1)
    ])
    
    # 初始化参数记录（全量）
    compressed_tensors, info = compressor.compress([tensor1], "client1")
    
    # 压缩参数
    compressed_tensors, info = compressor.compress([tensor2], "client1")

    # 原始大小 
    original_size = calculate_size(tensor2)
    compressed_size = calculate_size((compressed_tensors, info))
    print_compression_stats("初始化和更新", original_size, compressed_size)
    
    # 解压验证
    compressor.decompress(compressed_tensors, info, model_params)
    diff = torch.abs(tensor2 - model_params['tensor1']).max()
    print(f"最大误差: {diff}")

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
    _test_init_and_update(large_tensor1, large_tensor2)

    # 测试用例2：小规模密集更新
    print("\n=== 测试2：小规模密集更新 ===")
    small_tensor1 = torch.randn(10, 10)
    small_tensor2 = small_tensor1 + 0.1  # 所有值都有变化
    _test_init_and_update(small_tensor1, small_tensor2)

    # 测试用例4：极端值测试
    print("\n=== 测试4：极端值测试 ===")
    # 创建包含极端值的张量
    extreme_tensor1 = torch.tensor([float('inf'), float('-inf'), float('nan'), 1e38, -1e38, 0])
    extreme_tensor2 = extreme_tensor1.clone()
    extreme_tensor2[0] = torch.finfo(torch.float32).max  # 更改一个值
    _test_init_and_update(extreme_tensor1, extreme_tensor2)

    # 测试用例5：无参数更新
    print("\n=== 测试5：无参数更新 ===")
    # 创建包含极端值的张量
    small_tensor1 = torch.randn(100, 100)
    small_tensor2 = small_tensor1.clone()
    _test_init_and_update(small_tensor1, small_tensor2)

if __name__ == "__main__":
    test_compressor()