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
    self.train_dataset_len = 12000 // 2  # Roughly the size of Imagenet dataset.
    
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
  """
  === Batch Size: 2 ===
  train begin  1:46AM UTC on Jul 09, 2024
  epoch 5 CPU 内存占用：2.9% (7.589GB/334.562GB)
  epoch 10 CPU 内存占用：3.0% (7.807GB/334.562GB)
  epoch 15 CPU 内存占用：3.1% (8.001GB/334.562GB)
  epoch 20 CPU 内存占用：3.1% (8.115GB/334.562GB)
  epoch 25 CPU 内存占用：3.1% (8.187GB/334.562GB)
  epoch 30 CPU 内存占用：3.2% (8.312GB/334.562GB)
  epoch 35 CPU 内存占用：3.2% (8.449GB/334.562GB)
  epoch 40 CPU 内存占用：3.2% (8.551GB/334.562GB)
  epoch 45 CPU 内存占用：3.3% (8.667GB/334.562GB)
  epoch 50 CPU 内存占用：3.3% (8.797GB/334.562GB)
  train end  3:14AM UTC on Jul 09, 2024

  === Batch Size: 2 ===
  === Batch Size: 4 ===
  train begin  3:14AM UTC on Jul 09, 2024
  epoch 5 CPU 内存占用：3.3% (8.709GB/334.562GB)
  epoch 10 CPU 内存占用：3.3% (8.854GB/334.562GB)
  epoch 15 CPU 内存占用：3.3% (8.918GB/334.562GB)
  epoch 20 CPU 内存占用：3.4% (9.038GB/334.562GB)
  epoch 25 CPU 内存占用：3.4% (9.071GB/334.562GB)
  epoch 30 CPU 内存占用：3.4% (9.137GB/334.562GB)
  epoch 35 CPU 内存占用：3.4% (9.201GB/334.562GB)
  epoch 40 CPU 内存占用：3.5% (9.365GB/334.562GB)
  epoch 45 CPU 内存占用：3.5% (9.318GB/334.562GB)
  epoch 50 CPU 内存占用：3.5% (9.364GB/334.562GB)
  train end  3:58AM UTC on Jul 09, 2024

  === Batch Size: 4 ===
  === Batch Size: 8 ===
  train begin  3:58AM UTC on Jul 09, 2024
  epoch 5 CPU 内存占用：3.2% (8.328GB/334.562GB)
  epoch 10 CPU 内存占用：3.2% (8.378GB/334.562GB)
  epoch 15 CPU 内存占用：3.2% (8.399GB/334.562GB)
  epoch 20 CPU 内存占用：3.2% (8.420GB/334.562GB)
  epoch 25 CPU 内存占用：3.2% (8.431GB/334.562GB)
  epoch 30 CPU 内存占用：3.2% (8.414GB/334.562GB)
  epoch 35 CPU 内存占用：3.2% (8.428GB/334.562GB)
  epoch 40 CPU 内存占用：3.2% (8.454GB/334.562GB)
  epoch 45 CPU 内存占用：3.2% (8.433GB/334.562GB)
  epoch 50 CPU 内存占用：3.2% (8.481GB/334.562GB)
  train end  4:21AM UTC on Jul 09, 2024

  === Batch Size: 8 ===
  === Batch Size: 16 ===
  train begin  4:21AM UTC on Jul 09, 2024
  epoch 5 CPU 内存占用：3.3% (8.653GB/334.562GB)
  epoch 10 CPU 内存占用：3.3% (8.802GB/334.562GB)
  epoch 15 CPU 内存占用：3.3% (8.897GB/334.562GB)
  epoch 20 CPU 内存占用：3.4% (9.116GB/334.562GB)
  epoch 25 CPU 内存占用：3.4% (9.198GB/334.562GB)
  epoch 30 CPU 内存占用：3.4% (9.275GB/334.562GB)
  epoch 35 CPU 内存占用：3.5% (9.446GB/334.562GB)
  epoch 40 CPU 内存占用：3.5% (9.529GB/334.562GB)
  epoch 45 CPU 内存占用：3.5% (9.601GB/334.562GB)
  epoch 50 CPU 内存占用：3.6% (9.749GB/334.562GB)
  train end  4:33AM UTC on Jul 09, 2024

  === Batch Size: 16 ===
  === Batch Size: 32 ===
  train begin  4:33AM UTC on Jul 09, 2024
  epoch 5 CPU 内存占用：3.6% (9.889GB/334.562GB)
  epoch 10 CPU 内存占用：4.0% (11.087GB/334.562GB)
  epoch 15 CPU 内存占用：4.2% (11.853GB/334.562GB)
  epoch 20 CPU 内存占用：4.6% (13.112GB/334.562GB)
  epoch 25 CPU 内存占用：5.0% (14.337GB/334.562GB)
  epoch 30 CPU 内存占用：5.2% (15.002GB/334.562GB)
  epoch 35 CPU 内存占用：5.2% (15.099GB/334.562GB)
  epoch 40 CPU 内存占用：5.5% (16.232GB/334.562GB)
  epoch 45 CPU 内存占用：6.0% (17.682GB/334.562GB)
  epoch 50 CPU 内存占用：6.3% (18.744GB/334.562GB)
  train end  4:43AM UTC on Jul 09, 2024

  === Batch Size: 32 ===
  === Batch Size: 64 ===
  train begin  4:43AM UTC on Jul 09, 2024
  epoch 5 CPU 内存占用：3.3% (8.660GB/334.562GB)
  epoch 10 CPU 内存占用：3.3% (8.661GB/334.562GB)
  epoch 15 CPU 内存占用：3.3% (8.670GB/334.562GB)
  epoch 20 CPU 内存占用：3.3% (8.651GB/334.562GB)
  epoch 25 CPU 内存占用：3.3% (8.698GB/334.562GB)
  epoch 30 CPU 内存占用：3.3% (8.640GB/334.562GB)
  epoch 35 CPU 内存占用：3.3% (8.658GB/334.562GB)
  epoch 40 CPU 内存占用：3.3% (8.668GB/334.562GB)
  epoch 45 CPU 内存占用：3.3% (8.671GB/334.562GB)
  epoch 50 CPU 内存占用：3.3% (8.670GB/334.562GB)
  train end  4:52AM UTC on Jul 09, 2024

  === Batch Size: 64 ===
  === Batch Size: 128 ===
  train begin  4:52AM UTC on Jul 09, 2024
  epoch 5 CPU 内存占用：3.3% (8.822GB/334.562GB)
  epoch 10 CPU 内存占用：3.3% (8.790GB/334.562GB)
  epoch 15 CPU 内存占用：3.3% (8.775GB/334.562GB)
  epoch 20 CPU 内存占用：3.3% (8.770GB/334.562GB)
  epoch 25 CPU 内存占用：3.3% (8.813GB/334.562GB)
  epoch 30 CPU 内存占用：3.3% (8.835GB/334.562GB)
  epoch 35 CPU 内存占用：3.3% (8.830GB/334.562GB)
  epoch 40 CPU 内存占用：3.3% (8.822GB/334.562GB)
  epoch 45 CPU 内存占用：3.3% (8.799GB/334.562GB)
  epoch 50 CPU 内存占用：3.3% (8.792GB/334.562GB)
  train end  5:00AM UTC on Jul 09, 2024

  === Batch Size: 128 ===
  === Batch Size: 256 ===
  train begin  5:00AM UTC on Jul 09, 2024
  epoch 5 CPU 内存占用：3.3% (8.808GB/334.562GB)
  epoch 10 CPU 内存占用：3.3% (8.739GB/334.562GB)
  epoch 15 CPU 内存占用：3.3% (8.792GB/334.562GB)
  epoch 20 CPU 内存占用：3.3% (8.747GB/334.562GB)
  epoch 25 CPU 内存占用：3.3% (8.776GB/334.562GB)
  epoch 30 CPU 内存占用：3.3% (8.808GB/334.562GB)
  epoch 35 CPU 内存占用：3.3% (8.776GB/334.562GB)
  epoch 40 CPU 内存占用：3.3% (8.809GB/334.562GB)
  epoch 45 CPU 内存占用：3.3% (8.756GB/334.562GB)
  epoch 50 CPU 内存占用：3.3% (8.783GB/334.562GB)
  train end  5:08AM UTC on Jul 09, 2024

  === Batch Size: 256 ===
  === Batch Size: 512 ===
  train begin  5:08AM UTC on Jul 09, 2024

  === Batch Size: 512 ===
  === Batch Size: 1024 ===
  train begin  5:09AM UTC on Jul 09, 2024

  === Batch Size: 1024 ===
  """
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