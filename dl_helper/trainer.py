from dl_helper.train_param import match_num_processes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler

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
    
    from torch_xla.amp import autocast, GradScaler
    try:
      from torch_xla.amp import syncfree
    except ImportError:
      assert False, "Missing package syncfree; the package is available in torch-xla>=1.11"

class train_base():
    def __init__(self, seed, amp):
        self.amp = amp
        set_seed(seed)
        self.device = None
        
    def get_fake_data(self, num_samples, num_classes, batch_size):
        data = torch.randn(num_samples, 3, 32, 32)
        target = torch.randint(0, num_classes, (num_samples,))
        dataset = torch.utils.data.TensorDataset(data, target)
        
        # for debug
        # for i in range(8):
        #     self.print(i, data[i][0][0][:5])
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        return train_loader

    def init_data_loader(self, data_loader):
        return data_loader
        
    def get_device(self):
        if None is self.device:
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
    
    def print(self, *msg):
        print(*msg)
        
    def cal_output_loss(self, model, data, target, criterion):
        output = model(data)
        loss = criterion(output, target)
        return output, loss
  
class train_gpu(train_base):
    def __init__(self, seed, amp):
        super().__init__(seed, amp)
        self.accelerator = Accelerator(mixed_precision=amp if amp!='no' else 'no')

    def init_data_loader(self, data_loader):
        data_loader = self.accelerator.prepare(data_loader)
        return data_loader
        
    def get_device(self):
        if None is self.device:
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
    
    def print(self, *msg):
        self.accelerator.print(*msg)
        
    def cal_output_loss(self, model, data, target, criterion):
        if self.amp != 'no':
            output = model(data)
            with self.accelerator.autocast():
                loss = criterion(output, target)
            return output, loss
        else:
            return super().cal_output_loss(model, data, target, criterion)

class train_tpu(train_base):
    def __init__(self, seed, amp):
        super().__init__(seed, amp)
        dist.init_process_group('xla', init_method='xla://')
          
    def init_data_loader(self, data_loader):
        if xm.xrt_world_size() > 1:
            # 获取dataloader参数
            dataset = data_loader.dataset
            shuffle = isinstance(data_loader.sampler, (RandomSampler, DistributedSampler))
            batch_size = data_loader.batch_size
            drop_last=data_loader.drop_last
            num_workers = data_loader.num_workers
            pin_memory = data_loader.pin_memory

            del data_loader

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle)

            # 新建dataloader
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory)
              
        train_device_loader = pl.MpDeviceLoader(data_loader, self.device)
        return train_device_loader

    def get_device(self):
        if None is self.device:
            self.device = xla.device()
        return self.device
    
    def step(self, loss, optimizer):
        loss.backward()
        optimizer.step()

        # ddp模式，不需要 xm.optimizer_step，会自动同步梯度
        # xm.optimizer_step(optimizer)
        
        # for debug
        # 汇总loss
        # _loss = loss.item()
        # print(_loss)
        # xm.mark_step()
        # losss = xm.all_gather(_loss)
        # print(_loss)
    
    def init_model(self, model):
        model = model.to(self.device)
        # Optional for TPU v4 and GPU
        xm.broadcast_master_param(model)
        # model = DDP(model, gradient_as_bucket_view=True)
        return model
    
    def prepare(self, d, t):
        return d, t
    
    def print(self, *msg):
        xm.master_print(*msg)
        
    def cal_output_loss(self, model, data, target, criterion):
        if self.amp != 'no':
            with autocast(xm.xla_device()):
                output = model(data)
                loss = criterion(output, target)
            return output, loss
        
        else:
            return super().cal_output_loss(model, data, target, criterion)

def train_fn(index, params, model, criterion, optimizer, train_data, val_data, trainer):
    # 训练循环
    for epoch in range(params.epochs):

        # 训练
        model.train()
        for idx, (data, target) in enumerate(train_data):
            
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            data, target = trainer.prepare(data, target)
            optimizer.zero_grad()
            output, loss = trainer.cal_output_loss(model, data, target, criterion)
            trainer.step(loss, optimizer)

        # # 验证
        # model.eval()
        # for idx, (data, target) in enumerate(val_data):
        #     data, target = trainer.prepare(data, target)
        #     output, loss = trainer.cal_output_loss(model, data, target, criterion)
            
        if epoch%5 == 0 and epoch!=0:
            trainer.print(f'epoch {epoch}')

def test_fn(index):
    pass

def run_fn(index, num_processes, test):


    ###########################################
    # 1. 训练
    ###########################################
    # 调整训练参数
    params = test.get_param()
    # 调整batch_size
    if num_processes == 2:
        params.batch_size //= num_processes

    # 调整lr
    if num_processes > 0:
        params.learning_rate *= num_processes

    # 创建训练器
    if num_processes == 8:
        trainer = train_tpu(params.seed, params.amp)
    elif num_processes == 0:
        trainer = train_base(params.seed, params.amp)
    else:
        trainer = train_gpu(params.seed, params.amp)

    trainer.print('准备训练元素')
    model = test.get_model()
    train_data = test.get_data('train', params)
    val_data = test.get_data('val', params)
    criterion = test.get_criterion()
    optimizer = test.get_optimizer(model)

    trainer.print('初始化训练元素')
    model = trainer.init_model(model)
    trainer.print('model')
    train_data = trainer.init_data_loader(train_data)
    trainer.print('train_data')
    val_data = trainer.init_data_loader(val_data)
    trainer.print('val_data')
    criterion = trainer.init_criterion(criterion)
    trainer.print('criterion')

    trainer.print('开始训练')
    train_fn(index, params, model, criterion, optimizer, train_data, val_data, trainer)
    ###########################################
    # 1. 测试
    ###########################################

def run(test):
    # 训练核心数
    # P100 = 1
    # T4x2 = 2
    # TPU  = 8
    # CPU  = 0
    num_processes = match_num_processes()

    if num_processes == 8:
        xmp.spawn(run_fn, args=(num_processes, test), start_method='fork')      
    else:
        notebook_launcher(run_fn, args=(0, num_processes, test), num_processes=num_processes)