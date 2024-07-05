from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str

import torch
import torch.nn as nn
import torchvision

class test(test_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        title = f'test_resnet50_v{self.idx}'

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'aa.7z',
            learning_rate=0.00013, batch_size=64, epochs=3,

            # 3分类
            classify=True,
        )

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        return torchvision.models.resnet50()

    # 初始化数据
    # 返回一个 torch dataloader
    def get_data(self, _type, params, data_sample_getter_func=None):

        self.img_dim = 224
        self.batch_size = 128
        self.train_dataset_len = 3069 # Roughly the size of Imagenet dataset.

        data = torch.zeros(self.train_dataset_len, 3, self.img_dim, self.img_dim)
        target = torch.zeros(self.train_dataset_len, dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(data, target)
        sampler = None
        if not None is data_sample_getter_func:
            sampler = data_sample_getter_func(dataset, _type)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
            sampler=sampler,
        )

        return train_loader
