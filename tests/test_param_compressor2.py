import unittest
import torch
from dl_helper.param_compression import IncrementalCompressor

class TestIncrementalCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = IncrementalCompressor(threshold=0.1, sparsity_threshold=0.3)
        
    def test_initial_compression(self):
        # 创建测试张量
        tensors = [
            torch.randn(100, 100),
            torch.randn(50, 50)
        ]
        client_id = "test_client"
        
        # 测试初始压缩
        compressed_tensors, info = self.compressor.compress(tensors, client_id)
        
        # 验证初始压缩是否完整返回
        self.assertTrue(info.get('full'))
        self.assertEqual(len(compressed_tensors), len(tensors))
        for ct, t in zip(compressed_tensors, tensors):
            self.assertTrue(torch.allclose(ct.to(t.device), t))
            
    def test_incremental_compression(self):
        # 创建初始张量
        tensors = [
            torch.randn(100, 100),
            torch.randn(50, 50)
        ]
        client_id = "test_client"
        
        # 首次压缩
        _, _ = self.compressor.compress(tensors, client_id)
        
        # 小幅修改张量
        tensors[0][0, 0] += 0.2  # 超过阈值的修改
        tensors[1][0, 0] += 0.05  # 低于阈值的修改
        
        # 再次压缩
        compressed_tensors, info = self.compressor.compress(tensors, client_id)
        
        # 验证增量更新
        self.assertFalse(info['full'][0])  # 第一个张量应该是增量更新
        self.assertTrue(len(info['update_indices'][0]) > 0)  # 应该有更新索引
        
    def test_decompression(self):
        # 创建原始张量
        original_tensors = [
            torch.randn(100, 100),
            torch.randn(50, 50)
        ]
        client_id = "test_client"
        
        # 压缩
        compressed_tensors, info = self.compressor.compress(original_tensors, client_id)
        
        # 创建目标参数字典
        param_dict = {
            'layer1.weight': torch.zeros_like(original_tensors[0]),
            'layer2.weight': torch.zeros_like(original_tensors[1])
        }
        
        # 解压缩
        IncrementalCompressor.decompress(compressed_tensors, info, param_dict)
        
        # 验证解压缩结果
        for original, (_, decompressed) in zip(original_tensors, param_dict.items()):
            self.assertTrue(torch.allclose(original, decompressed))

if __name__ == '__main__':
    unittest.main()
