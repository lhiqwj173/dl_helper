import torch
from dl_helper.rl.rl_utils import GradientCompressor

def test_gradient_compressor():
    """测试梯度压缩器在各种情况下的表现"""
    # 初始化压缩器
    compressor = GradientCompressor()
    
    def run_test(test_name, gradients, expected_error=None):
        """运行单个测试用例"""
        print(f"\n测试用例: {test_name}")
        try:
            # 压缩
            compressed_grads, compress_info = compressor.compress(gradients)
            print("压缩成功")
            
            # 解压
            decompressed_grads = compressor.decompress(compressed_grads, compress_info)
            print("解压成功")
            
            # 计算压缩统计信息
            stats = compressor.get_compression_stats(gradients, compressed_grads, compress_info)
            print(f"压缩统计: {stats}")
            
            # 验证解压后的形状与原始形状相同
            for orig, decomp in zip(gradients, decompressed_grads):
                assert orig.shape == decomp.shape, f"形状不匹配: 原始 {orig.shape} vs 解压 {decomp.shape}"
            print("形状验证通过")
            
            return True
            
        except Exception as e:
            if expected_error and isinstance(e, expected_error):
                print(f"预期错误: {str(e)}")
                return True
            else:
                print(f"意外错误: {str(e)}")
                return False
    
    # 测试用例列表
    test_cases = [
        # 1. 基本测试 - 正常梯度
        {
            "name": "正常梯度",
            "gradients": [
                torch.randn(100, 100),
                torch.randn(50, 50)
            ]
        },
        
        # 2. 边界值测试 - 全零梯度
        {
            "name": "全零梯度",
            "gradients": [
                torch.zeros(100, 100),
                torch.zeros(50, 50)
            ]
        },
        
        # 3. 边界值测试 - 全同值梯度
        {
            "name": "全同值梯度",
            "gradients": [
                torch.full((100, 100), 1.0),
                torch.full((50, 50), 2.0)
            ]
        },
        
        # 4. 特殊形状测试 - 高维张量
        {
            "name": "高维张量",
            "gradients": [
                torch.randn(10, 10, 10, 10),
                torch.randn(5, 5, 5, 5)
            ]
        },
        
        # 5. 特殊形状测试 - 一维张量
        {
            "name": "一维张量",
            "gradients": [
                torch.randn(100),
                torch.randn(50)
            ]
        },
        
        # 6. 特殊形状测试 - 不规则形状
        {
            "name": "不规则形状",
            "gradients": [
                torch.randn(63, 1344),
                torch.randn(128, 256)
            ]
        },
        
        # 7. 极端值测试 - 包含极大值
        {
            "name": "包含极大值",
            "gradients": [
                torch.randn(100, 100) * 1e6,
                torch.randn(50, 50) * 1e6
            ]
        },
        
        # 8. 极端值测试 - 包含极小值
        {
            "name": "包含极小值",
            "gradients": [
                torch.randn(100, 100) * 1e-6,
                torch.randn(50, 50) * 1e-6
            ]
        },
        
        # 9. 混合值测试 - 包含正负值和零
        {
            "name": "混合值",
            "gradients": [
                torch.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.0]]),
                torch.tensor([[0.5, -0.5, 0.0], [-1.5, 1.5, 0.0]])
            ]
        },
        
        # 10. 特殊值测试 - 包含NaN/Inf
        {
            "name": "包含NaN/Inf",
            "gradients": [
                torch.tensor([[1.0, float('inf')], [float('nan'), 2.0]]),
                torch.tensor([[3.0, float('-inf')], [float('nan'), 4.0]])
            ]
        }
    ]
    
    # 运行所有测试用例
    results = []
    for test_case in test_cases:
        result = run_test(test_case["name"], test_case["gradients"])
        results.append(result)
    
    # 输出测试结果统计
    success_count = sum(results)
    total_count = len(results)
    print(f"\n测试完成: {success_count}/{total_count} 通过")
    
    return success_count == total_count

if __name__ == "__main__":
    test_gradient_compressor()