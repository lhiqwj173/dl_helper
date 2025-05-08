import torch
import torch.nn as nn

# 设置随机种子保证可重复性
torch.manual_seed(42)

# 示例1：正确的操作（无批次混合）
def correct_version():
    # 定义简单的线性层，输入2维，输出3维
    model = nn.Linear(2, 3)
    
    # 输入数据：batch_size=3, input_dim=2
    x = torch.randn(3, 2, requires_grad=True)
    
    # 前向传播
    out = model(x)  # 输出形状 [3,3]
    
    # 将损失设为第一个样本（i=0）所有输出的和
    loss = out[0].sum()  # 正确：仅使用第0个样本
    
    # 反向传播
    loss.backward()
    
    # 检查输入梯度
    print("正确操作的输入梯度:")
    print(x.grad)

# 示例2：错误的操作（意外混合批次维度）
def incorrect_version():
    model = nn.Linear(2, 3)
    
    # 相同的输入数据
    x = torch.randn(3, 2, requires_grad=True)
    
    # 错误操作：使用view改变形状导致跨样本混合
    # 将输入从[3,2]改为[2,3]，相当于交换了特征和批次维度！
    x_wrong = x.view(2, 3).t()  # 错误的重塑操作
    
    # 前向传播
    out = model(x_wrong)  # 现在输入是[3,2]，但数据被错误重组
    
    loss = out[0].sum()    # 仍然尝试计算第一个样本的损失
    
    loss.backward()
    
    print("\n错误操作的输入梯度:")
    print(x.grad)

# 运行示例
correct_version()
incorrect_version()