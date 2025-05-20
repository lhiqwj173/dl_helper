import sys, torch, os
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import functools

from accelerate.utils import set_seed

from dl_helper.tester import test_base
from dl_helper.train_param import Params, match_num_processes
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.base import transform
from dl_helper.trainer import run
from dl_helper.tool import model_params_num

"""
手写 mnist 数据集 5样本 过拟合测试
"""
# 4. 定义简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

class mini_dataset(torchvision.datasets.MNIST):
    def __init__(self, limit_num=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get one sample per class (0-9)
        self.indices = []
        class_seen = set()
        for idx in range(super().__len__()):
            _, label = super().__getitem__(idx)
            if label not in class_seen:
                self.indices.append(idx)
                class_seen.add(label)
            if len(class_seen) == limit_num:
                break

        assert len(self.indices) == limit_num
        self.limit_num = limit_num

    def __getitem__(self, index):
        # Fetch item using the filtered indices
        return super().__getitem__(self.indices[index])[0]

    def __len__(self):
        return self.limit_num

class test(test_base):
    title_base = '20250520_mnist_overfit_5_sample'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params_kwargs = {k: v for k, v in kwargs.items()}
        params_kwargs['y_n'] = 5
        params_kwargs['classify'] = True
        params_kwargs['batch_size'] = 5
        params_kwargs['no_better_stop'] = 0

        # 实例化 参数对象
        self.para = Params(
            **params_kwargs
        )

        # 2. 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差
        ])

        # 3. 下载并加载MNIST数据集
        self.train_dataset = mini_dataset(5, root='data', train=True, 
                                                transform=transform, download=True)
        self.val_dataset = mini_dataset(5, root='data', train=False, 
                                        transform=transform)

    def get_model(self):
        return SimpleCNN()
    
    def get_data(self, _type, data_sample_getter_func=None):
        if _type == 'train':
            return DataLoader(dataset=self.train_dataset, batch_size=self.para.batch_size, shuffle=True, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'val':
            return DataLoader(dataset=self.val_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'test':
            return None
        
if '__main__' == __name__:
    model = SimpleCNN()
    print(f"模型参数量: {model_params_num(model)}")

    # input_folder = r'/kaggle/input'
    # # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
    # data_folder_name = os.listdir(input_folder)[0]
    # data_folder = os.path.join(input_folder, data_folder_name)

    run(
        test, 
    )