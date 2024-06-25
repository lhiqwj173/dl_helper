from dl_helper.train_param import match_num_processes
from dl_helper.tracker import Tracker
from dl_helper.scheduler import ReduceLR_slow_loss
from dl_helper.tool import report_memory_usage
from dl_helper.trainers.gpu import train_gpu
from dl_helper.trainers.tpu import train_tpu

import copy

import multiprocessing as mp

from tqdm import tqdm
import time, os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler

if match_num_processes() ==8:
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla import runtime as xr
    from torch_xla.amp import autocast, GradScaler
    try:
      from torch_xla.amp import syncfree
    except ImportError:
      assert False, "Missing package syncfree; the package is available in torch-xla>=1.11"

from accelerate import notebook_launcher

def train_fn(index, epoch, params, model, criterion, optimizer, train_data, trainer, tracker):
    model.train()
    for idx, (data, target) in tqdm(enumerate(train_data), total=len(train_data), disable=not trainer.is_main_process(), desc=f'epoch {epoch} training'):
        # trainer.wait_for_everyone()
        # if trainer.is_main_process():
        #     report_memory_usage(f'begin')
        
        # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
        if not params.classify and len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        data, target = trainer.prepare(data, target)
        optimizer.zero_grad()
        output, loss = trainer.cal_output_loss(model, data, target, criterion)

        # tracker.track(output, target, loss, 'train')
        trainer.step(loss, optimizer)

        if idx % 10 == 0:
            if trainer.is_main_process():
                report_memory_usage(f'step')
        trainer.wait_for_everyone()

def val_fn(index, epoch, params, model, val_data, trainer, criterion, tracker):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_data), total=len(val_data), disable=not trainer.is_main_process(), desc=f'epoch {epoch} validating'):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            data, target = trainer.prepare(data, target)
            output, loss = trainer.cal_output_loss(model, data, target, criterion)
            # tracker.track(output, target, loss, 'val')

            trainer.wait_for_everyone()
            if trainer.is_main_process():
                report_memory_usage(f'val_{idx}')

def test_fn(index, params, model, test_data, trainer):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_data):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            data, target = trainer.prepare(data, target)
            output = model(data)

def run_fn_0(index, lock, num_processes, test):
    ###########################################
    # 1. 训练/验证
    ###########################################
    # 调整参数
    params = test.get_param()
    # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
    if num_processes == 2:
        params.batch_size //= num_processes

    # 调整lr
    if num_processes > 0:
        params.learning_rate *= num_processes

    # 创建训练器
    if num_processes == 8:
        trainer = train_tpu(params, index, lock)
    else:
        trainer = train_gpu(params, index, lock)
    device = trainer.get_device()

    trainer.print('准备训练元素...')
    model = test.get_model()
    train_data = test.get_data('train', params, trainer.get_data_sampler)
    val_data = test.get_data('val', params, trainer.get_data_sampler)
    criterion = test.get_criterion()
    optimizer = test.get_optimizer(model)
    scheduler = ReduceLR_slow_loss(optimizer)
    trainer.print('done')
                
    # 训练追踪器
    tracker = Tracker(params, trainer, scheduler, num_processes)

    trainer.print('初始化训练元素...')
    model = trainer.init_model(model)
    trainer.print('model')
    train_data = trainer.init_data_loader(train_data)
    trainer.print('train_data')
    val_data = trainer.init_data_loader(val_data)
    trainer.print('val_data')
    criterion = trainer.init_criterion(criterion)
    trainer.print('criterion')
    scheduler = trainer.init_scheduler(scheduler)
    trainer.print('scheduler')
    tracker = trainer.init_tracker(tracker)
    trainer.print('done')

    # 同步
    if trainer.is_main_process():
        report_memory_usage('开始训练')

    # for epoch in tqdm(range(params.epochs), disable=not trainer.is_main_process()):
    for epoch in range(params.epochs):
        # 训练
        train_fn(index, epoch, params, model, criterion, optimizer, train_data, trainer, tracker)
        # 同步
        trainer.wait_for_everyone()
        if trainer.is_main_process():
            report_memory_usage()

        # 验证
        val_fn(index, epoch, params, model, val_data, trainer, criterion, tracker)
        # 同步
        trainer.wait_for_everyone()
        if trainer.is_main_process():
            report_memory_usage()

        # 每epoch更新
        tracker.update()

        # 缓存训练数据
        if tracker.need_save:
            trainer.print('cache train data')
            tracker.save()

        # 同步
        trainer.wait_for_everyone()

    return

    ###########################################
    # 1. 测试
    ###########################################
    trainer.print('开始测试')
    trainer.clear_data_loader()
    del train_data, val_data, optimizer

    test_data = test.get_data('test', params, trainer.get_data_sampler)
    test_data = trainer.init_data_loader(test_data)

    val_fn(index, params, model, test_data, trainer, criterion, tracker)
    
    tracker.update()
    tracker.plot()

def get_data_sampler(data_set):
    train_sampler = None
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            data_set,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)

    return train_sampler

def run_fn_1(index, lock, num_processes, test):
    ###########################################
    # 1. 训练/验证
    ###########################################
    # 调整参数
    params = test.get_param()
    # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
    if num_processes == 2:
        params.batch_size //= num_processes

    # 调整lr
    if num_processes > 0:
        params.learning_rate *= num_processes

    dist.init_process_group('xla', init_method='xla://')

    device = xm.xla_device()

    xm.master_print('准备训练元素...')
    model = test.get_model()
    xm.master_print('model')
    model = model.to(device)
    if xr.using_pjrt():
        xm.broadcast_master_param(model)
    model = DDP(model, gradient_as_bucket_view=True)

    train_data = test.get_data('train', params, get_data_sampler)
    val_data = test.get_data('val', params, get_data_sampler)
    criterion = test.get_criterion()
    optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = ReduceLR_slow_loss(optimizer)
    xm.master_print('done')

    xm.master_print('train_data')
    train_data = pl.MpDeviceLoader(train_data, device)
    xm.master_print('val_data')
    val_data = pl.MpDeviceLoader(val_data, device)
    xm.master_print('done')

    if xm.is_master_ordinal():
        report_memory_usage('开始训练')
  
    for epoch in range(params.epochs):
        # 训练
        model.train()
        for idx, (data, target) in tqdm(enumerate(train_data), total=len(train_data), disable=not xm.is_master_ordinal(), desc=f'epoch {epoch} training'):
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if idx % 3 == 0:
                if xm.is_master_ordinal():
                    report_memory_usage(f'step {idx}')

            # xm.mark_step()

        # 同步
        xm.rendezvous("wait_for_everyone")

        # 验证
        model.eval()
        with torch.no_grad():
            for idx, (data, target) in tqdm(enumerate(val_data), total=len(val_data), disable=not xm.is_master_ordinal(), desc=f'epoch {epoch} validating'):
                # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                if not params.classify and len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)

                output = model(data)
                loss = criterion(output, target)

                if idx % 3 == 0:
                    if xm.is_master_ordinal():
                        report_memory_usage(f'step {idx}')

                # xm.mark_step()

        # 同步
        xm.rendezvous("wait_for_everyone")

        # # 每epoch更新
        # tracker.update()

        # # 缓存训练数据
        # if tracker.need_save:
        #     xm.master_print('cache train data')
        #     # tracker.save()


    return

    ###########################################
    # 1. 测试
    ###########################################
    xm.master_print('开始测试')
    trainer.clear_data_loader()
    del train_data, val_data, optimizer

    test_data = test.get_data('test', params, trainer.get_data_sampler)
    test_data = trainer.init_data_loader(test_data)

    val_fn(index, params, model, test_data, trainer, criterion, tracker)
    
    tracker.update()
    tracker.plot()

from dl_helper.models.binctabl import m_bin_ctabl

# 定义简单的 ResNet 模型
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = out.mean([2, 3])  # 平均池化
        out = self.fc(out)
        return out
        
def run_fn(index, num_processes, test, fake_data=False):
    # 调整参数
    params = test.get_param()
    # 调整batch_size, 多gpu时的batch_size指的是每个gpu的batch_size
    if num_processes == 2:
        params.batch_size //= num_processes

    # 调整lr
    if num_processes > 0:
        params.learning_rate *= num_processes

    # 设置训练参数
    epochs = 30

    # dist.init_process_group('xla', init_method='xla://')

    device = xm.xla_device()

    batch_size = 1024

    ddp = False

    if fake_data:
        # 创建模拟数据
        num_classes = 3

        # for debug
        num_samples = 272955
        # num_samples = 100000

        data = torch.randn(num_samples, 40, 100)
        # data = torch.randn(num_samples, 3, 64, 64)
        target = torch.randint(0, num_classes, (num_samples,))
        train_dataset = torch.utils.data.TensorDataset(data, target)

        train_sampler = None
        if xm.xrt_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True
            # shuffle=True,
        )

        test_samples = int(num_samples / 6)
        test_data = torch.randn(test_samples, 40, 100)
        test_target = torch.randint(0, num_classes, (test_samples,))
        test_dataset = torch.utils.data.TensorDataset(test_data, test_target)

        test_sampler = None
        if xm.xrt_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
        
        # 创建数据加载器
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            drop_last=True
            # shuffle=True,
        )

    else:
        # 真实数据
        train_loader = test.get_data('train', params, get_data_sampler)
        test_loader = test.get_data('val', params, get_data_sampler)

    xm.master_print(f'data_set len: {len(train_loader.dataset)}')
    print(f'{index} {device}')
    xm.rendezvous("init train_loader")
    if xm.is_master_ordinal():
        report_memory_usage(f'train_loader')

    train_loader = pl.MpDeviceLoader(train_loader, device)
    test_loader = pl.MpDeviceLoader(test_loader, device)

    # model = ResNet()
    model = m_bin_ctabl(60, 40, 100, 40, 120, 10, 3, 1)
    model = model.to(device)
    if ddp:
        if xr.using_pjrt():
            xm.master_print('broadcast_master_param')
            xm.broadcast_master_param(model)
        model = DDP(model, gradient_as_bucket_view=True)
    
    criterion = nn.CrossEntropyLoss()

    # 初始化优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    xm.master_print(f'batch_size: {batch_size}')
    xm.master_print(f'each epoch step: {len(train_loader)}')
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), disable=not xm.is_master_ordinal()):
            # if not ddp:
            #     data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if ddp:
                optimizer.step()
            else:
                xm.optimizer_step(optimizer)

            if xm.is_master_ordinal() and idx % 10 == 0:
                report_memory_usage(f'train {epoch} {idx}')

        xm.rendezvous(f"train {epoch} done")
        if xm.is_master_ordinal():
            report_memory_usage(f'train {epoch} done')

        model.eval()
        with torch.no_grad():
            for idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), disable=not xm.is_master_ordinal()):
                output = model(data)
                loss = criterion(output, target)

                if xm.is_master_ordinal() and idx % 10 == 0:
                    report_memory_usage(f'val {epoch} {idx}')

        xm.rendezvous(f"val {epoch} done")
        if xm.is_master_ordinal():
            report_memory_usage(f'val {epoch} done')

    if xm.is_master_ordinal():
        report_memory_usage('STOP')


def run(test, fake_data=False):
    num_processes = match_num_processes()
    # lock = mp.Manager().Lock()

    if num_processes == 8:
        xmp.spawn(run_fn, args=(num_processes, copy.deepcopy(test), fake_data), start_method='fork')      
    else:
        notebook_launcher(run_fn, args=(0, num_processes, copy.deepcopy(test), fake_data), num_processes=num_processes)