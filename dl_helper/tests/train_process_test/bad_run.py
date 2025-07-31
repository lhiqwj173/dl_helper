import sys, torch, os, pickle
from torch.nn.init import xavier_uniform_, zeros_
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import functools
from math import prod
from itertools import product

import numpy as np
import pandas as pd

from accelerate.utils import set_seed

from dl_helper.rl.custom_imitation_module.dataset import LobTrajectoryDataset
from dl_helper.rl.rl_env.lob_trade.lob_const import DATA_FOLDER
from dl_helper.tester import test_base
from dl_helper.train_param import Params, match_num_processes
from dl_helper.trainer import run
from dl_helper.tool import model_params_num
from torch.utils.data import DataLoader, Dataset

# CNN Model Architecture
class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(  1, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28 
        self.conv2 = nn.Conv2d( 64,128, kernel_size=3, padding=1)  # 14x14 -> 14x14 
        self.conv3 = nn.Conv2d(128,256, kernel_size=3, padding=1)  #  7x7  ->  7x7 
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)  # After pooling: 28->14->7
        self.fc2 = nn.Linear(512, num_classes)
        # self.to(device)
        
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, 28, 28) -> (batch, 1, 28, 28)
        
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Third conv block
        x = F.relu(self.conv3(x))
        x = self.dropout1(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: (batch, 128*7*7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class MNISTDataset(Dataset):
    def __init__(self, filename='/kaggle/input/digit-recognizer/train.csv', _type='train'):
        df = pd.read_csv(filename)

        # 分割数据集，使用固定的随机种子，保证每次分割结果相同
        rng = np.random.default_rng(seed=42)
        if _type in ['train', 'val']:
            df_train = df.sample(frac=0.8, random_state=rng)
            df_val = df.drop(df_train.index)
            if _type == 'train':
                df = df_train.reset_index(drop=True)
            else:
                df = df_val.reset_index(drop=True)
        
        pixel_cols = [col for col in df.columns if col.startswith('pixel')]
        pixel_data = df[pixel_cols].values
        pixel_tensor = torch.tensor(pixel_data, dtype=torch.float32)
        pixel_tensor = pixel_tensor.reshape(-1, 28, 28)  # Shape: (num_images, 28, 28)
        pixel_tensor = pixel_tensor / 255.               # Normalize [0,1]
        
        self.pixels = pixel_tensor
        self.labels = torch.tensor(df['label'].values, dtype=torch.long) if 'label' in df.columns else None

        self.transform_all = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))  # MINST Dataset Normalization Values
        ])
        # # Apply 5% Offsets (Translation Augmentation) = lower score on MINST
        # self.transform_train = transforms.Compose([
        #     # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0),
        # ])
        # self.transform_test = transforms.Compose([
        # ])        

    def __len__(self):
        return len(self.pixels)
      
    def __getitem__(self, idx):
        image = self.pixels[idx].unsqueeze(0) # Add channel dimension for transforms (1, 28, 28)

        image = self.transform_all(image)
        if self.labels is not None:
            # image = self.transform_train(image)
            label = self.labels[idx]
            return image, label
        else:
            # image = self.transform_test(image)
            return image, -1  # Return dummy label for test set

class test(test_base):
    title_base = 'bad_run'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params_kwargs['y_n'] = 10
        self.params_kwargs['classify'] = True
        self.params_kwargs['no_better_stop'] = 0
        self.params_kwargs['batch_n'] = 32
        self.params_kwargs['epochs'] = 100
        self.params_kwargs['learning_rate'] = 3e-4
        self.params_kwargs['no_better_stop'] = 0
        self.params_kwargs['label_smoothing'] = 0

        seeds = range(5)
        self.model_cls = MNISTNet
        self.seed = seeds[self.idx]
        self.params_kwargs['seed'] = self.seed

        # 实例化 参数对象
        self.para = Params(
            **self.params_kwargs
        )

        # 准备数据集
        self.train_dataset = MNISTDataset('/kaggle/input/digit-recognizer/train.csv', _type='train')
        self.val_dataset   = MNISTDataset('/kaggle/input/digit-recognizer/train.csv', _type='val')
        self.test_dataset  = MNISTDataset('/kaggle/input/digit-recognizer/test.csv', _type='test')

    def get_title_suffix(self):
        """获取后缀"""
        return f'{self.model_cls.__name__}_seed{self.seed}'

    def get_model(self):
        return self.model_cls()
    
    def get_data(self, _type, data_sample_getter_func=None):
        if _type == 'train':
            return DataLoader(dataset=self.train_dataset, batch_size=self.para.batch_size, shuffle=True, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'val':
            return DataLoader(dataset=self.val_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'test':
            return DataLoader(dataset=self.test_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        
if '__main__' == __name__:

    run(
        test, 
    )