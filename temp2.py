import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from dl_helper.scheduler import OneCycle, ReduceLR_slow_loss, ReduceLROnPlateau, WarmupReduceLROnPlateau, LRFinder, ConstantLRScheduler

# 初始化 Accelerator
accelerator = Accelerator()

# 定义简单的模型
model = nn.Linear(10, 1)

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 定义调度器（简单线性衰减）
_scheduler = ConstantLRScheduler(optimizer)

# 使用 Accelerator 准备模型、优化器和调度器
model, optimizer, scheduler = accelerator.prepare(model, optimizer, _scheduler)

# 模拟训练一步并保存检查点
model.train()
input_data = torch.randn(32, 10).to(accelerator.device)
target = torch.randn(32, 1).to(accelerator.device)
optimizer.zero_grad()
output = model(input_data)
loss = nn.MSELoss()(output, target)
accelerator.backward(loss)
optimizer.step()
scheduler.step()

# 保存检查点
checkpoint_folder = "./checkpoint"
accelerator.save_state(checkpoint_folder)
# print("Initial LR:", [group['lr'] for group in optimizer.param_groups])

# 调用 step() 确保调度器更新
scheduler.step()
# 检查学习率是否正确修改
print("before modified Optimizer LR:", [group['lr'] for group in optimizer.param_groups])

# 恢复检查点
accelerator.load_state(checkpoint_folder)

# 修改学习率
new_lr = 0.5
# for param_group in optimizer.param_groups:
#     param_group['lr'] = new_lr
scheduler.scheduler.base_lrs = [new_lr] * len(scheduler.scheduler.base_lrs)
# for opt in scheduler.optimizers:
#       for g in opt.param_groups:
#           g['lr'] = new_lr
# for g in optimizer.param_groups:
#     g['lr'] = new_lr
# for g in scheduler.scheduler.optimizer.param_groups:
#     g['lr'] = new_lr

# 再次调用 step() 确保调度器更新
scheduler.step()
# 检查学习率是否正确修改
print("Modified Optimizer LR:", [group['lr'] for group in optimizer.param_groups])
print("Modified Optimizer LR:", _scheduler.optimizer.param_groups[0]["lr"])

# 保存新检查点（可选）
accelerator.save_state("./new_checkpoint")