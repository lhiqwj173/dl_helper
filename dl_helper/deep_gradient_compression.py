import torch, math, copy, pickle
from py_ext.tool import get_exception_msg, log

class DeepGradientCompression:
    def __init__(self, momentum_buffer_size=32, compress_ratio=0.02, 
                 communication_threshold=0.001,
                 momentum_factor=0.9):

        """
        Args:
            momentum_buffer_size (int): 动量缓冲区大小
            compress_ratio (float): 压缩比率
            communication_threshold (float): 通信阈值
            momentum_factor (float): 动量因子
        """

        self.momentum_buffer_size = momentum_buffer_size
        self.compress_ratio = compress_ratio
        self.communication_threshold = communication_threshold
        self.momentum_factor = momentum_factor
        
        self.momentum_states = {}

        # 可压缩的最小元素数量，若小于这个数量的原始数据，取最大的一个梯度进行更新
        self.min_raw_elements = int(math.ceil(1 / compress_ratio))

    def clear(self):
        self.momentum_states = {}

    def compress_shape(self, original_shape):
        """计算压缩后的形状, 确保压缩后长度不为0

        compress_ratio = 0.02
        min_raw_elements = 50

        1. raw_elements = 49
            compressed_size = 1

        2. raw_elements = 50
            compressed_size = 1
        """
        raw_elements = math.prod(original_shape)
        if raw_elements < self.min_raw_elements:
            return (1, )
        compressed_size = int(math.prod(original_shape) * self.compress_ratio)
        return (compressed_size,)

    def try_compress(self, gradients, need_warm_up_steps=False):
        g = [copy.deepcopy(i) for i in gradients]
        try:
            return self.compress(gradients, need_warm_up_steps)
        except Exception as e:
            print(get_exception_msg(e))
            pickle.dump(g, open('error_grads.pkl', 'wb'))
            raise e

    def compress(self, gradients, need_warm_up_steps=False):
        """压缩梯度，确保输出大小固定且不小于最小元素数量"""

        compressed_gradients = []
        compression_infos = []

        # 预热期间不压缩
        if need_warm_up_steps:
            for idx, gradient in enumerate(gradients):
                param_name = f'grad_{idx}'

                # 初始化动量状态
                if param_name not in self.momentum_states:
                    self.momentum_states[param_name] = torch.zeros_like(gradient)
                
                # 计算新的动量
                momentum_grad = (self.momentum_factor * self.momentum_states[param_name]) + gradient
                self.momentum_states[param_name] = momentum_grad
                
                compression_info = {
                    'is_full_gradient': True,
                }
                
                compressed_gradients.append(momentum_grad)
                compression_infos.append(compression_info)
            
        else:
            # 处理每个梯度
            for idx, gradient in enumerate(gradients):
                param_name = f'grad_{idx}'

                # 检查梯度 inf/nan
                if torch.isnan(gradient).any() or torch.isinf(gradient).any():
                    raise ValueError(f"梯度 {param_name} 包含 inf/nan 值")

                # 扁平化梯度
                flat_grad = gradient.view(-1)

                # 初始化动量状态
                if param_name not in self.momentum_states or self.momentum_states[param_name].shape != flat_grad.shape:
                    if param_name in self.momentum_states:
                        log(f"param {param_name} shape {self.momentum_states[param_name].shape} change to {flat_grad.shape}, most likely due to no more need warm up")
                    self.momentum_states[param_name] = torch.zeros_like(flat_grad)
                
                # 计算新的动量
                momentum_grad = (self.momentum_factor * self.momentum_states[param_name]) + flat_grad
                self.momentum_states[param_name] = momentum_grad
                
                # 计算阈值
                abs_grad = torch.abs(momentum_grad)
                threshold = max(
                    torch.quantile(abs_grad, 1 - self.compress_ratio), 
                    self.communication_threshold
                )
            
                # 获取重要梯度的掩码和索引
                important_mask = abs_grad >= threshold
                important_indices = torch.nonzero(important_mask).squeeze()

                # 校正索引
                compressed_size = self.compress_shape(gradient.shape)[0]

                if important_indices.numel() < compressed_size:
                    # 没有重要梯度 / 重要梯度数量不足
                    # 选取 topk k = compressed_size
                    topk = torch.topk(abs_grad, compressed_size)
                    important_indices = topk.indices

                elif important_indices.numel() > compressed_size:
                    # # 随机抽取降采样
                    # important_indices = important_indices[torch.randperm(len(important_indices))[:compressed_size]]
                    
                    # 取最大的 `compressed_size` 个元素，不做降采样
                    topk = torch.topk(abs_grad, compressed_size)
                    important_indices = topk.indices

                # 获取重要梯度
                important_grad = momentum_grad[important_indices]
                
                compression_info = {
                    'indices': important_indices,
                    'is_full_gradient': False,
                    'original_shape': gradient.shape
                }
                
                compressed_gradients.append(important_grad)
                compression_infos.append(compression_info)

        return compressed_gradients, compression_infos

    @staticmethod
    def decompress(compressed_grads, compression_infos):
        decompressed_gradients = []

        for compressed_grad, comp_info in zip(compressed_grads, compression_infos):
            # 如果是全梯度,不需要解压
            if comp_info['is_full_gradient']:
                decompressed_gradients.append(compressed_grad)
                continue

            # 创建零张量
            full_gradient = torch.zeros(
                math.prod(comp_info['original_shape']), 
                device=compressed_grad.device
            )
            
            # 填充压缩后的梯度
            full_gradient[comp_info['indices']] = compressed_grad
            
            # 恢复原始形状
            decompressed_gradients.append(full_gradient.view(comp_info['original_shape']))
        
        return decompressed_gradients

# 使用示例
def example_usage():
    # 创建DGC实例
    dgc = DeepGradientCompression()
    
    for i in range(3):
        # 模拟多个梯度和模型参数
        example_gradients = [
            torch.randn(10000),  # 第一个梯度
            torch.randn(5000),   # 第二个梯度
            torch.randn(8000)    # 第三个梯度
        ]
        
        # 压缩梯度
        compressed_grads, comp_infos = dgc.compress(example_gradients)
        if comp_infos[0]['is_full_gradient']:
            print('预热梯度\n')
            continue

        origin_size = len(pickle.dumps(example_gradients))
        compressed_size = len(pickle.dumps((compressed_grads, comp_infos)))
        compression_ratio = (origin_size - compressed_size) / origin_size * 100
        print("原始大小:", origin_size)
        print("压缩后大小:", compressed_size)
        print(f"压缩比率: {compression_ratio:.2f}%\n")

        # # 解压梯度
        # reconstructed_grads = dgc.decompress(
        #     compressed_grads, comp_infos
        # )

def debug():
    
    file = r"C:\Users\lh\Downloads\error_grads.pkl"
    grad = pickle.load(open(file, 'rb'))
    print(grad)

    dgc = DeepGradientCompression()
    compressed_grads, comp_infos = dgc.compress(grad)

    print(compressed_grads[0].shape)
    print(comp_infos[0])

# 如果直接运行此脚本,执行示例
if __name__ == '__main__':

    # example_usage()
    debug()


