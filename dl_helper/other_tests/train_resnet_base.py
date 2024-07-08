from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import time, sys
import psutil
import itertools

import torch
import torch_xla
import torchvision
import torch.optim as optim
import torch.nn as nn

def get_data_sampler(data_set, _type):
    train_sampler = None
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            data_set,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True if _type == 'train' else False
            )

    return train_sampler

def report_memory_usage(msg=''):
  if xm.is_master_ordinal():
    memory_usage = psutil.virtual_memory()
    print(f"{msg} CPU 内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")

class TrainResNetBase():

  def __init__(self, batch_size, normal_data=False):
    self.batch_size = batch_size
    self.normal_data = normal_data

    self.img_dim = 224
    self.num_epochs = 50
    self.train_dataset_len = 12000  # Roughly the size of Imagenet dataset.
    
    if not self.normal_data:
      train_loader = xu.SampleGenerator(
          data=(torch.zeros(self.batch_size, 3, self.img_dim, self.img_dim),
                torch.zeros(self.batch_size, dtype=torch.int64)),
          sample_count=self.train_dataset_len // self.batch_size //
          xr.world_size())
    else:
      data = torch.randn(self.train_dataset_len, 3, self.img_dim, self.img_dim)
      target = torch.zeros(self.train_dataset_len, dtype=torch.int64)
      dataset = torch.utils.data.TensorDataset(data, target)

      train_sampler = None
      if xm.xrt_world_size() > 1:
          train_sampler = torch.utils.data.distributed.DistributedSampler(
              dataset,
              num_replicas=xm.xrt_world_size(),
              rank=xm.get_ordinal(),
              )

      # 创建数据加载器
      loader = torch.utils.data.DataLoader(
          dataset,
          batch_size=self.batch_size,
          sampler=train_sampler,
          drop_last=True,
      )

    self.device = torch_xla.device()
    self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
    self.model = torchvision.models.resnet50().to(self.device)
    self.optimizer = optim.SGD(self.model.parameters(), weight_decay=1e-4)
    self.loss_fn = nn.CrossEntropyLoss()

  def run_optimizer(self):
    self.optimizer.step()

  def train_loop_fn(self, loader, epoch):
    self.model.train()
    for step, (data, target) in enumerate(loader):
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.run_optimizer()

  def start_training(self):
    xm.master_print('train begin {}'.format(
        time.strftime('%l:%M%p %Z on %b %d, %Y')))

    for epoch in range(1, self.num_epochs + 1):
      self.train_loop_fn(self.train_device_loader, epoch)
      if epoch % 5 == 0:
        report_memory_usage(f'epoch {epoch}')

    xm.master_print('train end {}'.format(
        time.strftime('%l:%M%p %Z on %b %d, %Y')))
    xm.wait_device_ops()


if __name__ == '__main__':
  batch_size = 128
  normal_data=False

  args = sys.argv
  for i in range(1, len(args)):
    if args[i] == '--batch_size':
      batch_size = int(args[i + 1])
    
    elif args[i] == '--normal_data':
      normal_data = True

  base = TrainResNetBase(batch_size, normal_data)
  base.start_training()