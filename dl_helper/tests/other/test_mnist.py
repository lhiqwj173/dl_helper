import sys, torch, os
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
from torch.utils.data import DataLoader
import functools

from accelerate.utils import set_seed

from dl_helper.tester import test_base
from dl_helper.train_param import Params
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.base import transform
from dl_helper.trainer import run
from dl_helper.tool import model_params_num

from py_ext.tool import log, init_logger

train_folder = train_title = f'20250513_test_mnist'
init_logger(train_title, home=train_folder, timestamp=False)

"""
手写 mnist 数据集训练测试
"""
# 4. 定义简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class test(test_base):
    title_base = train_title
    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        # 实例化 参数对象
        self.para = Params(
            train_title=train_title, 

            # 10分类
            classify=True,
            y_n=10, 
        )

        # 2. 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差
        ])
        # 3. 下载并加载MNIST数据集
        self.train_dataset = torchvision.datasets.MNIST(root=os.path.join(self.para.root, 'data'), train=True, 
                                                transform=transform, download=True)
        self.test_dataset = torchvision.datasets.MNIST(root=os.path.join(self.para.root, 'data'), train=False, 
                                        transform=transform)

    def get_model(self):
        return SimpleCNN()
    
    def get_data(self, _type, data_sample_getter_func=None):
        if _type == 'train':
            return DataLoader(dataset=self.train_dataset, batch_size=self.para.batch_size, shuffle=True)
        elif _type == 'test':
            return DataLoader(dataset=self.test_dataset, batch_size=self.para.batch_size, shuffle=False)

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