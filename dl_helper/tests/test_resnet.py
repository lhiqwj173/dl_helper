from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str

import torch
import torch.nn as nn

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

class test(test_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        title = f'test_resnet_v{self.idx}'

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'aa.7z',
            learning_rate=0.00013, batch_size=1024, epochs=3,

            # 3分类
            classify=True,
        )

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        return ResNet()

    # 初始化数据
    # 返回一个 torch dataloader
    def get_data(self, _type, params, data_sample_getter_func=None):

        # 创建模拟数据
        num_classes = 3

        # for debug
        num_samples = 30720

        data = torch.randn(num_samples, 3, 64, 64)
        target = torch.randint(0, num_classes, (num_samples,))
        dataset = torch.utils.data.TensorDataset(data, target)

        train_sampler = None
        if not None is data_sample_getter_func:
            train_sampler = data_sample_getter_func(dataset, _type)

        # 创建数据加载器
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            sampler=train_sampler,
            drop_last=True,
            shuffle=False if not None is train_sampler else True if _type == 'train' else False,
        )

        return loader
