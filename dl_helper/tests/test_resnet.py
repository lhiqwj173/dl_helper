from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str

import torch
import torch.nn as nn

# 定义简单的 ResNet 模型
class ResNet(nn.Module):
    def __init__(self, img_dim):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, img_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(img_dim, 3)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = out.mean([2, 3])
        out = self.fc(out)
        return out

class test(test_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        title = f'test_resnet_v{self.idx}'
        self.img_dim = 224
        # self.img_dim = 64

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
        return ResNet(self.img_dim)

    # 初始化数据
    # 返回一个 torch dataloader
    def get_data(self, _type, params, data_sample_getter_func=None):

        # 创建模拟数据
        num_classes = 3
        
        # for debug
        num_samples = 3069

        data = torch.randn(num_samples, 3, self.img_dim, self.img_dim)
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

    def get_cache_data(self, _type, params, accelerator):
        world_size = accelerator.num_processes
        rank = accelerator.process_index

        def get_data_sampler(data_set, _type):
            train_sampler = None
            if world_size > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    data_set,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True if _type == 'train' else False
                    )

            return train_sampler

        return self.get_data(_type, params, get_data_sampler)