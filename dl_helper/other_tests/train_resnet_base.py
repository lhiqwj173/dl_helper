from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import time
import itertools

import torch
import torch_xla
import torchvision
import torch.optim as optim
import torch.nn as nn

def report_memory_usage(msg=''):
  if xm.is_master_ordinal():
    memory_usage = psutil.virtual_memory()
    print(f"{msg} CPU 内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")

class TrainResNetBase():

  def __init__(self):
    self.img_dim = 224
    self.batch_size = 128
    self.num_epochs = 1
    self.train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    # For the purpose of this example, we are going to use fake data.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, 3, self.img_dim, self.img_dim),
              torch.zeros(self.batch_size, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size //
        xr.world_size())

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
      report_memory_usage(f'epoch {epoch}')

    xm.master_print('train end {}'.format(
        time.strftime('%l:%M%p %Z on %b %d, %Y')))
    xm.wait_device_ops()


if __name__ == '__main__':
  base = TrainResNetBase()
  base.start_training()