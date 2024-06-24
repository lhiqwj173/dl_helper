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

        trainer.print('prepare data')
        data, target = trainer.prepare(data, target)
        trainer.print('zero_grad')
        optimizer.zero_grad()
        trainer.print('output, loss')
        output, loss = trainer.cal_output_loss(model, data, target, criterion)

        trainer.wait_for_everyone()
        if trainer.is_main_process():
            report_memory_usage(f'output, loss')

        # tracker.track(output, target, loss, 'train')
        trainer.step(loss, optimizer)

        trainer.wait_for_everyone()
        if trainer.is_main_process():
            report_memory_usage(f'step')

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

def run_fn(index, lock, num_processes, test):
    print(f'run_fn {index}')

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

def run(test):
    # 训练核心数
    # P100 = 1
    # T4x2 = 2
    # TPU  = 8
    # CPU  = 0
    num_processes = match_num_processes()

    lock = mp.Manager().Lock()

    if num_processes == 8:
        xmp.spawn(run_fn, args=(lock, num_processes, test), start_method='fork')      
    else:
        notebook_launcher(run_fn, args=(0, lock, num_processes, copy.deepcopy(test)), num_processes=num_processes)