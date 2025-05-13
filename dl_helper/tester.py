"""
训练的基类
"""
import functools

from dl_helper.data import read_data, Dataset_cahce, DistributedSampler, DataLoaderDevice
from dl_helper.transforms.base import transform
from dl_helper.scheduler import OneCycle, ReduceLR_slow_loss, ReduceLROnPlateau, WarmupReduceLROnPlateau, LRFinder, blank_scheduler
from py_ext.tool import log

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from py_ext.tool import debug

class test_base():
    def __init__(self, idx, *args, data_folder='', amp='no', debug=False, test=False, findbest_lr=False, **kwargs):
        self.idx = idx
        log(f'train begin :{self.idx}')

        self.data_folder = data_folder if data_folder else './data'

        # GPU: fp16
        self.amp = amp
        self.debug = debug
        self.test = test
        self.para = None
        self.findbest_lr = findbest_lr

    # 获取训练参数
    def get_param(self):
        assert self.para, 'should init param in __init__()'
        if self.debug:
            self.para.epochs = 1
            self.para.debug = True

        if self.test:
            self.para.epochs = 6
            self.para.test = True

        if self.findbest_lr:
            self.para.epochs = 45
            self.para.root += f'_findlr'
            self.para.train_title += f'_findlr'

        return self.para

    # 初始化数据
    # 返回一个 torch dataloader
    def get_data(self, _type, data_sample_getter_func=None):
        raise NotImplementedError("必须在子类中实现 get_data 方法")

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
            model.parameters(), lr=self.para.learning_rate if not self.para.abs_learning_rate else self.para.abs_learning_rate, weight_decay=self.para.weight_decay)

    def get_transform(self, device):
        # 此处返回一个空白的 transform
        return transform()
    
    def get_lr_scheduler_class(self):
        return ''

    def get_lr_scheduler(self, optimizer, *args, **kwargs):
        """
        可以在子类中实现 get_lr_scheduler_class 方法， 
        可以是字符串 ['ReduceLR_slow_loss', 'ReduceLROnPlateau', 'WarmupReduceLROnPlateau'] 
        或一个自定义类，
        用于实例化 lr_scheduler
        若无特别需求，则返回 None, 不使用 lr_scheduler
        """
        lr_scheduler_class = self.get_lr_scheduler_class()

        if self.findbest_lr:
            lr_scheduler_class = LRFinder
        elif isinstance(lr_scheduler_class, str):
            if 'ReduceLR_slow_loss' == lr_scheduler_class:
                lr_scheduler_class = ReduceLR_slow_loss
            elif 'ReduceLROnPlateau' == lr_scheduler_class:
                lr_scheduler_class = ReduceLROnPlateau
            elif 'WarmupReduceLROnPlateau' == lr_scheduler_class:
                lr_scheduler_class = WarmupReduceLROnPlateau
            else:
                lr_scheduler_class = blank_scheduler
        elif isinstance(lr_scheduler_class, type) or isinstance(lr_scheduler_class, functools.partial):
            lr_scheduler_class = lr_scheduler_class
        else:
            lr_scheduler_class = blank_scheduler

        return lr_scheduler_class(optimizer, *args, **kwargs)