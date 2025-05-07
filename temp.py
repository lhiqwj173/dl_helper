%%writefile ddp.py

# 首先导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import OneCycleLR

# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """
        
        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

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

        #如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
#         if img_as_img.mode != 'L':
#             img_as_img = img_as_img.convert('L')

        #设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),   #随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
        img_as_img = transform(img_as_img)
        
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

# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

# resnet34模型
def res_model(num_classes, feature_extract = False):
    model_ft = models.resnet34(weights=None)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft

def ddp_setup(rank, world_size):
    """初始化DDP进程组"""
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 选择一个空闲端口
    
    # 初始化进程组
    init_process_group(
        backend='nccl',  # 在GPU上使用nccl
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前设备
    torch.cuda.set_device(rank)

def train(rank, world_size, train_dataset, val_dataset, model_path, num_classes, num_epoch, batch_size, batch_n, max_lr, weight_decay, use_amp):
    # 初始化DDP环境
    if world_size > 1:
        ddp_setup(rank, world_size)
    
    # 创建模型
    model = res_model(num_classes)
    
    # 将模型移动到指定设备并包装为DDP模型
    model = model.to(rank)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(rank)
    
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001 * batch_n, weight_decay=weight_decay)
    
    if world_size > 1:
        # 使用DistributedSampler创建数据加载器
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size * batch_n, 
        sampler=train_sampler, 
        num_workers=4 if world_size == 1 else 1,
        pin_memory=True
    )
    
    # 计算总步数用于学习率调度器
    total_steps = ((len(train_dataset) // (batch_size * batch_n)) * num_epoch) // world_size
    
    # 初始化学习率调度器
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr= max_lr * batch_n * world_size,
        total_steps=total_steps
    )
    
    # 初始化混合精度训练的GradScaler
    scaler = GradScaler(enabled=use_amp)
    
    # 用于记录训练指标
    all_train_loss = []
    all_train_accs = []
    all_lr = []
    
    # 训练循环
    for epoch in range(num_epoch):
        if train_sampler:
            # 确保每个epoch的数据分布不同
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = []
        train_accs = []
        
        # 使用tqdm显示进度（仅在主进程中）
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}")
            loader = pbar
        else:
            loader = train_loader
            
        for batch in loader:
            imgs, labels = batch
            imgs = imgs.to(rank)
            labels = labels.to(rank)
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=use_amp, dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            # 梯度缩放与优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 计算当前batch的准确率
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            
            # 记录损失和准确率
            train_loss.append(loss.item())
            train_accs.append(acc.item())
            
            # 更新学习率
            lr_scheduler.step()
        
        # 计算平均损失和准确率
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_accs) / len(train_accs)
        
        # 仅在rank 0上记录和打印
        if rank == 0:
            all_train_loss.append(avg_train_loss)
            all_train_accs.append(avg_train_acc)
            all_lr.append(optimizer.param_groups[0]['lr'])
            print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {avg_train_loss:.5f}, acc = {avg_train_acc:.5f}, lr = {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存模型（每5个epoch保存一次）
            if (epoch + 1) % 5 == 0:
                if world_size > 1:
                    torch.save(model.module.state_dict(), f"{model_path}.{epoch+1}")
                else:
                    torch.save(model.state_dict(), f"{model_path}.{epoch+1}")
    
    # 保存最终模型（仅在rank 0）
    if rank == 0:
        torch.save(model.module.state_dict(), model_path)
        
        # 保存训练结果
        results = (all_train_loss, all_train_accs, all_lr)
        with open("training_results.pkl", "wb") as f:
            pickle.dump(results, f)
    
    # 清理进程组
    destroy_process_group()

def main():
    # 检测可用GPU数量
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("没有检测到可用的GPU，请确保CUDA正常工作")
    print(f"检测到 {world_size} 个GPU，开始分布式训练")
        
    # 超参数
    max_lr = 1e-3
    weight_decay = 1e-3
    num_epoch = 100
    use_amp = False if world_size == 1 else True
    model_path = './pre_res_model.ckpt'
    batch_size = 8
    # batch_n = 2**5 + 28 if world_size == 1 else 2**5 + 20
    batch_n = 2**5
    print(f'batch_size: {batch_n * batch_size}')
    
    # 加载数据集
    train_path = '../input/classify-leaves/train.csv'
    img_path = '../input/classify-leaves/'

    train_dataset = LeavesData(train_path, img_path, mode='train')
    val_dataset = LeavesData(train_path, img_path, mode='valid')

    # 获取类别数
    num_classes = 176  # 从您的数据集中获得的类别数
    
    args = (world_size, train_dataset, val_dataset, model_path, num_classes, 
                num_epoch, batch_size, batch_n, max_lr, weight_decay, use_amp)
    if world_size > 1:
        # 设置多进程启动方法（如果尚未设置）
        if not torch.multiprocessing.get_start_method() in ['spawn', 'forkserver']:
            torch.multiprocessing.set_start_method('spawn', force=True)

        # 启动多进程训练
        mp.spawn(
            train,
            args=args,
            nprocs=world_size,
            join=True  # 等待所有进程完成
        )
    else:
        train(0, *args)

if __name__ == "__main__":
    main()