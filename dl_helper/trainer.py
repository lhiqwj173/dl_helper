from dl_helper.train_param import match_num_processes

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from accelerate import Accelerator
from accelerate import notebook_launcher
from accelerate.utils import set_seed

if match_num_processes() ==8:
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.data_parallel as dp


class train_base():
    def __init__(self, seed, amp):
        self.amp = amp
        set_seed(seed)
        
    def get_data(self, num_samples, num_classes, batch_size):
        data = torch.randn(num_samples, 3, 32, 32)
        targets = torch.randint(0, num_classes, (num_samples,))

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, targets),
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader
        
    def get_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device
    
    def init_criterion(self, criterion):
        return criterion
        
    def init_model(self, model):
        return model.to(self.device)
    
    def step(self, loss, optimizer):
        loss.backward()
        optimizer.step()
    
    def prepare(self, d, t):
        return d.to(self.device), t.to(self.device)
    
    def print(self, msg):
        print(msg)
        
    def cal_output_loss(self, model, data, criterion):
        outputs = model(data)
        loss = criterion(outputs, targets)
        return outputs, loss
  
class train_gpu(train_base):
    def __init__(self, seed, amp):
        super().__init__(seed, amp)
        self.accelerator = Accelerator(mixed_precision=amp if amp!='no' else 'no')
        
    def get_data(self, num_samples, num_classes, batch_size):
        train_loader = super().get_data(num_samples, num_classes, batch_size)
        train_loader = self.accelerator.prepare(train_loader)
        return train_loader
        
    def get_device(self):
        self.device = self.accelerator.device
        return self.device
    
    def step(self, loss, optimizer):
        self.accelerator.backward(loss)
        optimizer.step()
        
    def init_criterion(self, criterion):
        criterion = self.accelerator.prepare(criterion)
        return criterion
    
    def init_model(self, model):
        model = self.accelerator.prepare(model)
        return model
    
    def prepare(self, d, t):
        return d, t
    
    def print(self, msg):
        self.accelerator.print(msg)
        
    def cal_output_loss(self, model, data, criterion):
        if self.amp != 'no':
            output = self.model(data)
            with self.accelerator.autocast():
                loss = self.loss_fn(output, target)
            return outputs, loss
        else:
            return super().cal_output_loss(model, data, criterion)

class train_tpu(train_base):
    def __init__(self, seed, amp):
        super().__init__(seed, amp)
        dist.init_process_group('xla', init_method='xla://')
        
    def get_data(self, num_samples, num_classes, batch_size):
        train_loader = super().get_data(num_samples, num_classes, batch_size)
        # train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
        # 使用DataParallel对DataLoader进行分布式处理
        train_device_loader = dp.DataLoader(train_loader)
        return train_device_loader
        
    def get_device(self):
        self.device = xla.device()
        return self.device
    
    def step(self, loss, optimizer):
        loss.backward()
        xm.mark_step()
    
    def init_model(self, model):
        model = DDP(model.to(self.device), gradient_as_bucket_view=True)
        return model
    
    def prepare(self, d, t):
        return d, t
    
    def print(self, msg):
        xm.master_print(msg)
        
    def cal_output_loss(self, model, data, criterion):
        if self.amp != 'no':
            with autocast(xm.xla_device()):
                output = self.model(data)
                loss = self.loss_fn(output, target)
            return outputs, loss
        
        else:
            return super().cal_output_loss(model, data, criterion)

def train_fn_sample(i, seed:int=42, batch_size:int=16, lr=3e-2/25, amp=False):
    
    # 设置训练参数
    epochs = 30
    
    if num_processes == 8:
        trainer = train_tpu(seed, amp)
        batch_size //= 8
        # lr*=8
        
    elif num_processes == 0:
        trainer = train_base(seed, amp)

    else:
        trainer = train_gpu(seed, amp)
        if num_processes > 1:
            batch_size //= num_processes
            # lr*=num_processes
    
    device = trainer.get_device()

    # 创建模拟数据
    num_classes = 10
    num_samples = 100000
    train_loader = trainer.get_data(num_samples, num_classes, batch_size)

    model = ResNet()
    model = trainer.init_model(model)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    criterion = trainer.init_criterion(criterion)

    # 初始化优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # TPU  each epoch step: 782
    # P100 each epoch step: 6250
    # T4*2 each epoch step: 3125
    trainer.print(f'batch_size: {batch_size}')
    trainer.print(f'each epoch step: {len(train_loader)}')
    
    # 训练循环
    for epoch in range(epochs):
        idx = 0
        for data, targets in train_loader:
            data, targets = trainer.prepare(data, targets)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            trainer.step(loss, optimizer)
            
        if epoch%5 == 0 and epoch!=0:
            trainer.print(f'epoch {epoch}')


