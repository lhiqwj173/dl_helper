"""
训练的基类
"""
from dl_helper.data import read_data, Dataset_cahce, DistributedSampler, DataLoaderDevice
from dl_helper.transforms.base import transform
from dl_helper.scheduler import ReduceLR_slow_loss, ReduceLROnPlateau

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class test_base():
    def __init__(self, idx, data_folder='', amp='no', debug=False):
        self.idx = idx
        self.data_folder = data_folder if data_folder else './data'
        self.amp = amp
        self.debug = debug
        self.para = None

    # 获取训练参数
    def get_param(self):
        assert self.para, 'should init param in __init__()'
        if self.debug:
            self.para.epochs = 2
            self.para.debug = True
        return self.para

    # 初始化数据
    # 返回一个 torch dataloader
    def get_data(self, _type, params, data_sample_getter_func=None):
        return read_data(_type = _type, params=params, data_sample_getter_func = data_sample_getter_func)

    # 返回 局部缓存的数据
    # 按需读取数据，节省内存
    # 效率略低
    def get_cache_data(self, _type, params, accelerator):
        dataset = Dataset_cahce(params, _type, accelerator.device)
        sampler = DistributedSampler(dataset, accelerator, shuffle=True if _type == 'train' else False)
        dataloader = DataLoaderDevice(
            dataset,
            batch_size=params.batch_size, sampler=sampler,
            device=accelerator.device
        )
        return dataloader

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        raise NotImplementedError("must override get_model")

    # 初始化损失函数
    # 返回一个 torch loss
    def get_criterion(self):
        assert self.para, 'should init param in __init__()'
        return nn.CrossEntropyLoss(label_smoothing=self.para.label_smoothing) if self.para.classify else nn.MSELoss()

    # 初始化优化器
    # 返回一个 torch optimizer
    def get_optimizer(self, model):
        assert self.para, 'should init param in __init__()'
        return torch.optim.AdamW(
            model.parameters(), lr=self.para.learning_rate, weight_decay=self.para.weight_decay)

    def get_transform(self, device):
        return transform()

    def get_lr_scheduler(self, optimizer, params, *args, **kwargs):
        if 'ReduceLR_slow_loss' == self.lr_scheduler_class:
            lr_scheduler_class = ReduceLR_slow_loss
        elif 'ReduceLROnPlateau' == self.lr_scheduler_class:
            lr_scheduler_class = ReduceLROnPlateau

        return lr_scheduler_class(optimizer, *args, patience=params.learning_rate_scheduler_patience, **kwargs)