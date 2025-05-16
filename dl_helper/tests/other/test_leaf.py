import sys, torch, os, copy
import pandas as pd
import numpy as np
from PIL import Image
import functools

import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.models as models

from accelerate.utils import set_seed

from dl_helper.tester import test_base
from dl_helper.train_param import Params
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.base import transform
from dl_helper.trainer import run
from dl_helper.tool import model_params_num

"""
leaf 数据集训练测试
"""
# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  #header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))
        
        if mode == 'train':
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])  #self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image 
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])  
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image
            
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        self.real_len = len(self.image_arr)

        labels_dataframe = pd.read_csv('../input/classify-leaves/train.csv')
        # 把label文件排个序
        leaves_labels = sorted(list(set(labels_dataframe['label'])))
        n_classes = len(leaves_labels)
        print(n_classes)
        leaves_labels[:10]
        self.class_to_num = dict(zip(leaves_labels, range(n_classes)))
        # 再转换回来，方便最后预测的时候使用
        self.num_to_class = {v : k for k, v in self.class_to_num.items()}

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)
        img_as_img = self.transform(img_as_img)
        
        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = self.class_to_num[label]

            return img_as_img, number_label  #返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len

class test(test_base):
    title_base = '20250516_test_leaf'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params_kwargs = copy.deepcopy(kwargs)
        params_kwargs['y_n'] = 176
        params_kwargs['classify'] = True

        # 实例化 参数对象
        self.para = Params(
            **params_kwargs
        )

        # # 加载数据集
        # train_path = '../input/classify-leaves/train.csv'
        # test_path = '../input/classify-leaves/test.csv'
        # img_path = '../input/classify-leaves/'
        # self.train_dataset = LeavesData(train_path, img_path, mode='train')
        # self.val_dataset = LeavesData(train_path, img_path, mode='valid')
        # self.test_dataset = LeavesData(test_path, img_path, mode='test')

    def get_model(self):
        # # EfficientNet-B0         模型参数量: 11689512
        # model = models.efficientnet_b0(weights=None)
        # model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.para.y_n)

        # # resnet18                模型参数量: 11266800
        # model = models.resnet18(weights=None)
        # model.fc = nn.Linear(model.fc.in_features, self.para.y_n)

        # # mobilenet_v3_small      模型参数量:  1698256
        # model = models.mobilenet_v3_small(weights=None)
        # model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.para.y_n)

        # mobilenet_v3_large        模型参数量:  4427488
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.para.y_n)

        return model
    
    def get_data(self, _type, data_sample_getter_func=None):
        if _type == 'train':
            return DataLoader(dataset=self.train_dataset, batch_size=self.para.batch_size, shuffle=True)
        elif _type == 'val':
            return DataLoader(dataset=self.val_dataset, batch_size=self.para.batch_size, shuffle=False)
        elif _type == 'test':
            return DataLoader(dataset=self.test_dataset, batch_size=self.para.batch_size, shuffle=False)
        
if '__main__' == __name__:
    t = test(idx=0)
    model = t.get_model()
    print(f"模型参数量: {model_params_num(model)}")

    run(
        test, 
    )