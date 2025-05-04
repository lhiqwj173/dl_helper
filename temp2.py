import torch
import torch.nn as nn
import torch.optim as optim
import math

# 假设我们已经修改了OneCycleLR类的load_state_dict方法
from dl_helper.rl.custom_pytorch_module.lrscheduler import OneCycleLR

checkpoint_path = r'C:\Users\lh\Desktop\temp\checkpoint.pth'

# 定义一个简单的模型用于演示
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)

# 模拟训练若干步
def train_for_steps(model, optimizer, scheduler, num_steps):
    lrs = []
    for step in range(num_steps):
        # 模拟数据
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        # 更新学习率
        scheduler.step()
        
        # 记录当前学习率
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        print(f"步骤 {scheduler.last_epoch}: 学习率 = {lr:.6f}")
    
    return lrs

# 第一次运行 (total_steps=100)
def first_run():
    print("=== 第一次训练运行 (total_steps=100) ===")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 创建调度器，总步数为100
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=100,
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=1000.0
    )
    
    # 训练15步 (15%)
    lrs_first = train_for_steps(model, optimizer, scheduler, 15)
    print(f"训练了15步后 (15%完成), 学习率 = {scheduler.get_last_lr()[0]:.6f}")
    
    # 保存模型、优化器和调度器状态
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    
    return lrs_first

# 第二次运行 (total_steps=200)
def second_run():
    print("\n=== 第二次训练运行 (total_steps=200) ===")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    
    # 创建新模型和优化器
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 加载模型和优化器状态
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 创建新调度器，总步数为200
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=200,
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=1000.0
    )
    
    # 加载调度器状态
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    # 调度器现在应该在新周期的15%位置 (步骤30)
    print(f"加载后的当前步骤: {scheduler.last_epoch}")
    
    # 继续训练更多步骤
    lrs_second = train_for_steps(model, optimizer, scheduler, 20)
    print(f"额外训练20步后, 学习率 = {scheduler.get_last_lr()[0]:.6f}")
    
    return lrs_second

# 运行示例并比较学习率变化
lrs_first = first_run()
lrs_second = second_run()

# 打印验证信息
print("\n=== 验证 ===")
# 计算第一次运行中第15步的相对位置
checkpoint = torch.load(checkpoint_path)
old_step = checkpoint['scheduler']['last_epoch']
old_total = checkpoint['scheduler']['total_steps']
relative_pos = old_step / old_total
print(f"第一次运行中的相对位置: {relative_pos:.2f} (步骤 {old_step}/{old_total})")

# 在相同相对位置的新总步数下应该有的步骤
new_total = 200
expected_step = int(relative_pos * new_total)
print(f"在新总步数下的预期步骤: {expected_step}/{new_total}")