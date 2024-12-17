import torch
import torch.nn as nn

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        
        # 可学习的噪声参数
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 权重和偏置的确定性和噪声部分
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-1 / np.sqrt(in_features), 1 / np.sqrt(in_features)))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-std_init / np.sqrt(in_features), std_init / np.sqrt(in_features)))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features).uniform_(-1 / np.sqrt(in_features), 1 / np.sqrt(in_features)))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features).uniform_(-std_init / np.sqrt(in_features), std_init / np.sqrt(in_features)))
        
        # 噪声因子生成器
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))
    
    def forward(self, x):
        # 重新采样噪声
        self.weight_epsilon = torch.randn(self.out_features, self.in_features)
        self.bias_epsilon = torch.randn(self.out_features)
        
        # 应用噪声
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        
        return F.linear(x, weight, bias)

def replace_linear_with_noisy(model, std_init=0.1):
    """
    将模型中的所有 nn.Linear 层替换为 NoisyLinear 层
    
    参数:
    model (nn.Module): 原始模型
    std_init (float): 噪声初始标准差
    
    返回:
    nn.Module: 替换后的模型
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 获取原线性层的输入和输出维度
            in_features = module.in_features
            out_features = module.out_features
            
            # 创建新的 NoisyLinear 层
            noisy_linear = NoisyLinear(in_features, out_features, std_init)
            
            # 复制原层的权重和偏置到 mu 参数
            noisy_linear.weight_mu.data = module.weight.data
            noisy_linear.bias_mu.data = module.bias.data
            
            # 替换原层
            setattr(model, name, noisy_linear)
        elif len(list(module.children())) > 0:
            # 递归处理子模块
            replace_linear_with_noisy(module, std_init)
    
    return model

if __name__ == '__main__':
    # 使用示例
    class OriginalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # 转换示例
    original_model = OriginalModel()
    noisy_model = replace_linear_with_noisy(original_model)


