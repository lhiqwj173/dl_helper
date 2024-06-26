"""
训练的基类
"""
from dl_helper.data import read_data

import torch
import torch.nn as nn

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
            self.para.epochs = 30
        return self.para

    # 初始化数据
    # 返回一个 torch dataloader
    def get_data(self, _type, params, data_sample_getter_func=None):
        if self.debug:
            return read_data(_type = _type, params=params, max_num=1, data_sample_getter_func = data_sample_getter_func)
        return read_data(_type = _type, params=params, data_sample_getter_func = data_sample_getter_func)

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