import torch, pickle
from dl_helper.deep_gradient_compression import DeepGradientCompression

def test_gradient_compressor(compressor_cls, *args, warm_up_steps=False, **kwargs):
    """测试梯度压缩器在各种情况下的表现和压缩率"""
    # 初始化压缩器
    def calculate_compression_ratio(original_size, compressed_size):
        """计算压缩率"""
        return (original_size - compressed_size) / original_size * 100
    
    def get_tensor_size(tensor):
        """获取张量序列化后的字节数"""
        return len(pickle.dumps(tensor))
    
    def run_test(test_name, gradients, expected_error=None):
        """运行单个测试用例"""
        print(f"\n测试用例: {test_name}")
        compressor = compressor_cls(*args, **kwargs)

        # 计算原始大小
        original_size = get_tensor_size(gradients)
        
        # 压缩
        compressed_grads, compress_info = compressor.compress(gradients, warm_up_steps=warm_up_steps)
        print("压缩成功")
        
        # 计算压缩后大小
        compressed_size = get_tensor_size((compressed_grads, compress_info))
        compression_ratio = calculate_compression_ratio(original_size, compressed_size)
        print(f"压缩率: {compression_ratio:.2f}%")
        
        # 解压
        decompressed_grads = compressor.decompress(compressed_grads, compress_info)
        print("解压成功")
        
        # 计算压缩误差
        error = sum(torch.mean((orig - decomp) ** 2).item() 
                    for orig, decomp in zip(gradients, decompressed_grads))
        print(f"压缩误差 (MSE): {error:.2e}")
        
        # 验证解压后的形状与原始形状相同
        for orig, decomp in zip(gradients, decompressed_grads):
            assert orig.shape == decomp.shape, f"形状不匹配: 原始 {orig.shape} vs 解压 {decomp.shape}"
        print("形状验证通过")
        
        return {
            "success": True,
            "compression_ratio": compression_ratio,
            "error": error,
        }
        
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
        
        # 2. 边界值测试 - 全零梯度 (应该有很高的压缩率)
        {
            "name": "全零梯度",
            "gradients": [
                torch.zeros(100, 100),
                torch.zeros(50, 50)
            ]
        },
        
        # 3. 边界值测试 - 全同值梯度 (应该有很高的压缩率)
        {
            "name": "全同值梯度",
            "gradients": [
                torch.full((100, 100), 1.0),
                torch.full((50, 50), 2.0)
            ]
        },
        
        # 4. 稀疏梯度测试 (应该有较高的压缩率)
        {
            "name": "稀疏梯度",
            "gradients": [
                torch.sparse_coo_tensor(
                    indices=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                    values=torch.tensor([1.0, 2.0, 3.0]),
                    size=(100, 100)
                ).to_dense(),
                torch.sparse_coo_tensor(
                    indices=torch.tensor([[0, 1], [0, 1]]),
                    values=torch.tensor([4.0, 5.0]),
                    size=(50, 50)
                ).to_dense()
            ]
        },
        
        # 5. 高频变化梯度测试 (应该有较低的压缩率)
        {
            "name": "高频变化梯度",
            "gradients": [
                torch.sin(torch.linspace(0, 100*torch.pi, 10000)).reshape(100, 100),
                torch.cos(torch.linspace(0, 50*torch.pi, 2500)).reshape(50, 50)
            ]
        },
        
        # 7. 大规模梯度测试
        {
            "name": "大规模梯度",
            "gradients": [
                torch.randn(1000, 1000),
                torch.randn(500, 500)
            ]
        },
        
        # 8. 量化敏感测试 (小数值梯度)
        {
            "name": "量化敏感梯度",
            "gradients": [
                torch.randn(100, 100) * 1e-5,
                torch.randn(50, 50) * 1e-5
            ]
        },
        
        # 9. 混合分布测试
        {
            "name": "混合分布梯度",
            "gradients": [
                torch.cat([
                    torch.randn(50, 100),  # 正态分布
                    torch.full((50, 100), 1.0)  # 常值
                ]),
                torch.cat([
                    torch.zeros(25, 50),  # 零值
                    torch.randn(25, 50)   # 正态分布
                ])
            ]
        },
        
        # # 10. 特殊数值测试
        # {
        #     "name": "特殊数值梯度",
        #     "gradients": [
        #         torch.tensor([[1.0, float('inf')], [float('nan'), 2.0]]),
        #         torch.tensor([[3.0, float('-inf')], [float('nan'), 4.0]])
        #     ]
        # }
    ]
    
    # 运行所有测试用例
    results = []
    compression_stats = []
    for test_case in test_cases:
        result = run_test(test_case["name"], test_case["gradients"])
        results.append(result["success"])
        if "compression_ratio" in result:
            compression_stats.append({
                "name": test_case["name"],
                "ratio": result["compression_ratio"],
                "error": result.get("error", float('nan'))
            })
    
    # 输出详细的测试结果统计
    success_count = sum(results)
    total_count = len(results)
    print("\n=== 测试结果汇总 ===")
    print(f"测试用例总数: {total_count}")
    print(f"成功用例数: {success_count}")
    print(f"失败用例数: {total_count - success_count}")
    
    if compression_stats:
        print("\n=== 压缩率统计 ===")
        print("测试用例               压缩率(%)    误差(MSE)")
        print("-" * 50)
        for stat in compression_stats:
            print(f"{stat['name']:<20} {stat['ratio']:>8.2f}    {stat['error']:>10.2e}")
        
        # 计算平均压缩率
        avg_ratio = sum(stat['ratio'] for stat in compression_stats) / len(compression_stats)
        print(f"\n平均压缩率: {avg_ratio:.2f}%")
    
    return success_count == total_count

if __name__ == "__main__":
    # test_gradient_compressor(DeepGradientCompression)
    test_gradient_compressor(DeepGradientCompression, warm_up_steps=True)