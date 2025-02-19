import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dl_helper.param_compression import IncrementalCompressor
import matplotlib.pyplot as plt
import argparse
import time
from typing import Dict, List, Tuple
from collections import OrderedDict

# 创建一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TrainingStats:
    def __init__(self):
        self.epoch_losses: List[float] = []
        self.compression_ratios: List[float] = []
        
    def add_epoch_stats(self, avg_loss: float, compression_ratio: float):
        self.epoch_losses.append(avg_loss)
        self.compression_ratios.append(compression_ratio)
        
    def plot_results(self, use_compression: bool):
        plt.figure(figsize=(10, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.epoch_losses, label='Average Loss per Epoch: {:.4f}'.format(self.epoch_losses[-1]))
        plt.title('Average Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 压缩率（如果启用压缩）
        if use_compression:
            plt.subplot(1, 2, 2)
            plt.plot(self.compression_ratios, label='Compression Ratio per Epoch')
            plt.title('Compression Ratio per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Compression Ratio')
        
        plt.tight_layout()
        plt.show()
        plt.close()

def calculate_compression_ratio(compress_info: Dict, param_dict: Dict) -> float:
    if isinstance(compress_info.get('full'), bool):
        return 1.0
        
    total_elements = 0
    compressed_elements = 0
    
    for is_full, indices, param in zip(compress_info['full'], compress_info['update_indices'], param_dict.values()):
        if is_full:
            compressed_elements += 1
            total_elements += 1
        else:
            if indices is not None:
                param_total_elements = param.numel()
                compressed_elements += indices.shape[0] / param_total_elements
                total_elements += 1
                
    return compressed_elements / total_elements if total_elements > 0 else 1.0

def train(model: nn.Module, 
          dataloader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          compressor: IncrementalCompressor = None,
          client_id: str = "client_1") -> Tuple[float, float]:
    
    total_loss = 0
    compression_ratio = 1.0

    # 模拟参数传输
    param_dict = OrderedDict()
    for name, param in model.state_dict().items():
        param_dict[name] = param.clone()
    
    for batch_X, batch_y in dataloader:
        # 前向传播
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 如果启用压缩，执行参数压缩和解压缩
        if compressor is not None:
            current_params = [p.clone().cpu() for p in model.state_dict().values()]
            compressed_params, compress_info = compressor.compress(current_params, client_id)
            
            # 模拟参数传输
            IncrementalCompressor.decompress(compressed_params, compress_info, param_dict)

            # param_dict 与 compressor.client_params[client_id] 应该是相同的
            for compressor_t, t in zip(compressor.client_params[client_id], param_dict.values()):
                assert torch.allclose(compressor_t, t)
            
            # 计算压缩率
            compression_ratio = calculate_compression_ratio(compress_info, param_dict)

            # param_dict 覆盖
            model.load_state_dict(param_dict)

            # 模型参数 应该与 compressor.client_params[client_id] 相同
            for compressor_t, t in zip(compressor.client_params[client_id], model.state_dict().values()):
                assert torch.allclose(compressor_t, t)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, compression_ratio

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Neural Network Training with Optional Compression')
    parser.add_argument('--use-compression', action='store_true', default=True,
                        help='Enable parameter compression during training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--threshold', type=float, default=1e-3,
                        help='Compression threshold')
    parser.add_argument('--sparsity-threshold', type=float, default=0.3,
                        help='Sparsity threshold for compression')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模拟数据
    X = torch.randn(1000, 10)
    y = torch.sum(X, dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1
    
    # 创建数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 创建模型、优化器和损失函数
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 创建压缩器（如果启用）
    compressor = None
    if args.use_compression:
        compressor = IncrementalCompressor(
            threshold=args.threshold,
            sparsity_threshold=args.sparsity_threshold
        )
    
    # 创建统计对象
    stats = TrainingStats()
    
    # 训练循环
    print(f"Starting training {'with' if args.use_compression else 'without'} compression...")
    for epoch in range(args.epochs):
        avg_loss, compression_ratio = train(
            model, dataloader, criterion, optimizer, compressor
        )
        
        # 记录统计信息
        stats.add_epoch_stats(avg_loss, compression_ratio)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        if args.use_compression:
            print(f"Compression Ratio: {compression_ratio:.2%}")
        print("-" * 50)
    
    # 绘制结果
    stats.plot_results(args.use_compression)

if __name__ == '__main__':
    main()
