from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
from torch.utils import data
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torch
from tqdm import tqdm
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import roc_auc_score as auc
import multiprocessing
from IPython import display
import math
import dataframe_image as dfi
import dill
import itertools
import random

# 设置启动方法为'spawn'
multiprocessing.set_start_method('spawn', force=True)

root = ''
data_path = ''
epochs = 0
batch_n = 0
batch_size = 0
learning_rate = 0
warm_up_epochs = 0
device = None
workers = None
logger = None
wx = None
no_better_stop = 0
amp = False
random_wipe_pct = 0
random_wipe_pct_row = 0
weight_decay = 0
label_smoothing = 0
title = ''
init_learning_ratio=0
increase_ratio=0
cache_epoch = 30

amp_ratio = 2

def set_golbal_var(_root, _data_path, _epochs, _batch_n, _batch_size, _learning_rate, _warm_up_epochs, _device, _workers, _logger, _wx, _no_better_stop, _amp, _random_wipe_pct, _random_wipe_pct_row,_weight_decay, _label_smoothing, _title, _init_learning_ratio, _increase_ratio, _cache_epoch):
    global root, data_path, epochs, batch_n, batch_size, learning_rate, warm_up_epochs, device, workers, logger, wx, no_better_stop, amp, random_wipe_pct, random_wipe_pct_row, weight_decay, label_smoothing, title, init_learning_ratio, increase_ratio, cache_epoch

    root = _root
    data_path = _data_path
    epochs = _epochs
    batch_n = _batch_n
    batch_size = _batch_size
    learning_rate = _learning_rate
    warm_up_epochs = _warm_up_epochs
    device = _device
    workers = _workers
    logger = _logger
    wx = _wx
    no_better_stop = _no_better_stop
    amp = _amp
    random_wipe_pct = _random_wipe_pct
    random_wipe_pct_row = _random_wipe_pct_row
    weight_decay = _weight_decay
    label_smoothing = _label_smoothing
    title = _title
    init_learning_ratio, increase_ratio = _init_learning_ratio, _increase_ratio
    cache_epoch = _cache_epoch

    # amp 模式下学习率调整
    if amp:
        learning_rate *= amp_ratio


class Increase_ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    按照指定速率，初始学习率在每个迭代中增加学习率，单触发 ReduceLROnPlateau 衰减后则不在增加学习率
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        super().__init__(optimizer, mode, factor, patience,
                 threshold, threshold_mode, cooldown, min_lr, eps, verbose)

        self.init_learning_ratio = init_learning_ratio
        self.increase = increase_ratio * init_learning_ratio
        self.need_increase = True
        self.init_lr = False

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            # 触发调整学习率
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

            # 停止增加学习率
            self.need_increase = False

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def warn_up(self, init = False):
        if not self.need_increase:
            # 停止增加
            return

        if init:
            if not self.init_lr:
                logger.debug(f'init warn up')
                self.init_lr = True
            else:
                # 已经初始化过了
                return

        for i, param_group in enumerate(self.optimizer.param_groups):
            if init:
                param_group['lr'] = self.init_learning_ratio
            else:
                param_group['lr'] += self.increase

class warm_up_ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    warm_up + ReduceLROnPlateau 调整学习率
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, warm_up_epoch=0, iter_num_each_epoch=0, init_warm_up_ratio=25):
        super(warm_up_ReduceLROnPlateau, self).__init__(optimizer, mode, factor, patience,
                 threshold, threshold_mode, cooldown, min_lr, eps, verbose)

        self.warm_up_iters = warm_up_epoch * iter_num_each_epoch
        self.init_warm_up_ratio = init_warm_up_ratio
        self.cur_warm_up_iter = 0
        self.lr_diff = {}

    def step(self, metrics, epoch=None):
        if self.cur_warm_up_iter < self.warm_up_iters:
            # 存在warn up
            return

        super().step(metrics, epoch)

    def warn_up(self, init = False):
        if self.cur_warm_up_iter >= self.warm_up_iters:
            # 不在warn up
            return

        if init:
            if self.lr_diff == {}:
                logger.debug(f'init warn up')
            else:
                # 已经初始化过了
                return
        else:
            self.cur_warm_up_iter+=1

        for i, param_group in enumerate(self.optimizer.param_groups):
            if i not in self.lr_diff:
                original_lr = float(param_group['lr'])
                param_group['lr'] = original_lr / self.init_warm_up_ratio#初始化学习率
                self.lr_diff[i] = (original_lr - param_group['lr']) / self.warm_up_iters# 记录每次迭代的学习率差
            else:
                # 每次增加一个差值
                param_group['lr'] += self.lr_diff[i]

# 定义随机遮挡函数
def random_mask(tensor, mask_prob):
    mask = torch.rand(tensor.size()) < torch.rand(1)*mask_prob
    tensor.masked_fill_(mask, 0)
    return tensor

# 随机遮挡行
def random_mask_row(tensor, mask_prob):
    mask = torch.rand(tensor.size(0)) < torch.rand(1)*mask_prob
    tensor[mask] = 0
    return tensor

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, raw_data, x, y, mean_std, ids=[]):
        """Initialization"""

        # 原始数据
        # self.data = torch.tensor(np.array(raw_data), dtype=torch.float)
        self.data = torch.from_numpy(raw_data.values)
        self.data = torch.unsqueeze(self.data, 0)  # 增加一个通道维度

        # id
        self.ids = ids

        # 数据长度
        self.length = len(x)

        # x 切片索引
        self.x_idx = x
        # x = torch.tensor(np.array(x), dtype=torch.float)

        # y
        # 标签onehot 编码
        # self.y = torch.tensor(pd.get_dummies(np.array(y)).values, dtype=torch.int64)
        self.y = torch.tensor(np.array(y), dtype=torch.int64)

        # 标准化数据
        self.mean_std = mean_std

        # 区分价量列
        self.price_cols = [i*2 for i in range(20)] + [42, 45]
        self.vol_cols = [i*2+1 for i in range(20)]

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        # 切片范围
        a, b = self.x_idx[index]

        # 获取切片
        # 50 -> 49
        x = self.data[:, a:b, :].clone().to(device)

        # 获取均值方差
        mean_std = torch.tensor(
            self.mean_std[index], dtype=torch.float).to(device)

        # mid_price
        mid = (float(x[0, -1, 0]) + float(x[0, -1, 2])) / 2

        # 价格标准化
        x[0, :, self.price_cols] /= mid
        # x[0, :, self.price_cols] -= mean_std[:, 0]
        # x[0, :, self.price_cols] /= mean_std[:, 1]
        x[0, :, :] -= mean_std[:, 0]
        x[0, :, :] /= mean_std[:, 1]

        # 随机mask
        if random_wipe_pct>0:
            x = random_mask(x, random_wipe_pct)

        if random_wipe_pct_row>0:
            x = random_mask_row(x, random_wipe_pct_row)

        # return x, (self.y[index], self.ids[index])
        return x, self.y[index]

def re_blance_sample(ids, price_mean_std, test_x, test_y, test_raw):

    # 索引数组
    idx = np.arange(len(test_x))
    need_reindex = False

    # 标签平衡
    logger.debug('标签平衡')
    need_reindex = True

    labels = set(test_y)
    sy = pd.Series(test_y)
    min_num = sy.value_counts().min()

    idx = []
    for label in labels:
        origin_idx = sy[sy == label].index

        if len(origin_idx) > min_num:
            idx += np.random.choice(origin_idx, min_num,
                                    replace=False).tolist()
        else:
            idx += origin_idx.tolist()

    # # 打乱
    # logger.debug('样本随机')
    # need_reindex = True
    # np.random.shuffle(idx)

    # 重新索引
    if need_reindex:
        ids = [ids[i] for i in idx]
        test_x = [test_x[i] for i in idx]
        test_y = [test_y[i] for i in idx]
        price_mean_std = [price_mean_std[i] for i in idx]

    return ids, price_mean_std, test_x, test_y, test_raw

class ResumeSample():
    """
    支持恢复的采样器
    随机/顺序采样
    """

    def __init__(self, length=0, shuffle=True):
        # 随机产生种子
        self.shuffle = shuffle
        self.step = random.randint(0, 100)
        self.seed = random.randint(0, 100)
        self.size = length
        self.idx = 0
        self.data = []
        self._loop = False

    def state_dict(self):
        return {'step': self.step, 'seed': self.seed, 'shuffle': self.shuffle, 'size': self.size, 'idx': self.idx, 'loop': self._loop}

    def load_state_dict(self, state_dict):
        self.step = state_dict['step']
        self.seed = state_dict['seed']
        self.shuffle = state_dict['shuffle']
        self.size = state_dict['size']
        self.idx = state_dict['idx']
        self._loop = state_dict['loop']

        # 使用原种子
        random.seed(self.seed)
        self._init_data()

        return self

    def _init_data(self):
        if self.shuffle:

            self.data = random.sample(range(self.size), self.size)
        else:
            self.data = list(range(self.size))

    def __iter__(self):
        if not self._loop:
            # 更新种子
            self.seed += self.step
            random.seed(self.seed)
            self._init_data()

            self.idx = 0
            self._loop = True

        return self

    def __next__(self):
        if self.idx >= self.size:
            self._loop = False
            self.idx = 0
            raise StopIteration

        v = self.data[self.idx]
        self.idx += 1
        return v

    def __len__(self):
        return self.size


def read_data(_type, reblance=False, shuffle=False, max_num=100, head_n=0, pct=100, need_id=False):
    # # 读取测试数据
    # price_mean_std, x, y, raw = pickle.load(open(os.path.join(data_path, f'{_type}.pkl'), 'rb'))

    # 获取数据分段
    files = []
    for file in os.listdir(data_path):
        if _type in file:
            files.append(file)
    files.sort()
    logger.debug(f'{files}')

    # 读取分段合并
    ids, mean_std, x, y, raw = [], [], [], [], pd.DataFrame()
    diff_length = 0
    count = 0
    for file in files:
        count += 1
        if count > max_num:
            break
        _id, _mean_std, _x, _y, _raw = pickle.load(
            open(os.path.join(data_path, file), 'rb'))
        ids += _id
        mean_std += _mean_std
        y += _y
        x += [(i[0] + diff_length, i[1] + diff_length) for i in _x]
        raw = pd.concat([raw, _raw], axis=0, ignore_index=True)
        diff_length += len(_raw)

    if head_n == 0 and pct < 100 and pct > 0:
        head_n = int(len(x) * (pct / 100))

    if head_n > 0:
        logger.debug(f"控制样本数量 -> {head_n} / {len(x)}")
        raw = raw.iloc[:head_n, :]
        to_del_idx = [i for i in range(len(x)) if x[i][-1] > head_n]

        x = [x[i] for i in range(len(x)) if i not in to_del_idx]
        y = [y[i] for i in range(len(y)) if i not in to_del_idx]
        mean_std = [mean_std[i]
                    for i in range(len(mean_std)) if i not in to_del_idx]
        ids = [ids[i] for i in range(len(ids)) if i not in to_del_idx]

    if reblance:
        logger.debug(f"样本均衡")
        ids, mean_std, x, y, raw = re_blance_sample(ids, mean_std, x, y, raw)

    logger.debug(f"nan值样本数量 {raw.isna().sum().sum()}")
    logger.debug(f'\n标签分布\n{pd.Series(y).value_counts()}')

    if not need_id:
        ids = []
    dataset_test = Dataset(raw, x, y, mean_std, ids)
    del ids, x, y, raw

    data_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size if not (amp and _type == 'train') else int(
        batch_size*amp_ratio), sampler=ResumeSample(len(dataset_test), shuffle=shuffle), num_workers=workers, pin_memory=True if workers>0 else False)
    del dataset_test

    return data_loader

def last_value(data):
    """返回最后一个非nan值"""
    for i in range(len(data)-1, -1, -1):
        if not math.isnan(data[i]):
            return "{:.3e}".format(data[i])
    raise ValueError("没有找到非nan值")

def debug_plot():
    epochs, train_losses, test_losses, train_acc, test_acc, lrs, f1_scores = pickle.load(
        open(os.path.join(root, 'var', f'plot_datas.pkl'), 'rb'))
    
    plot_loss(epochs, train_losses, test_losses, train_acc, test_acc, lrs, f1_scores)

def plot_loss(epochs, train_losses, test_losses, train_acc, test_acc, lrs, f1_scores):
    # pickle.dump((train_losses, test_losses, train_acc, test_acc, lrs, f1_scores), open(os.path.join(root, 'var', f'plot_datas.pkl'), 'wb'))

    # 计算误差最低点
    min_train_loss = min(train_losses)
    min_test_loss = min(test_losses)
    min_train_x = train_losses.tolist().index(min_train_loss)
    min_test_x = test_losses.tolist().index(min_test_loss)

    # 计算acc最高点
    max_train_acc = max(train_acc)
    max_test_acc = max(test_acc)
    max_train_acc_x = train_acc.tolist().index(max_train_acc)
    max_test_acc_x = test_acc.tolist().index(max_test_acc)

    # 创建图形和坐标轴
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [7, 3]})

    # 1 图左侧坐标轴
    ax1 = axs[0]

    line_train_loss, = ax1.plot(list(range(epochs)), train_losses, label=f'train loss {last_value(train_losses)}', c='b')
    line_test_loss, = ax1.plot(list(range(epochs)), test_losses, label=f'validation loss {last_value(test_losses)}', c='#00BFFF')

    line_train_acc, = ax1.plot(list(range(epochs)), train_acc, label=f'train acc {last_value(train_acc)}', c='r')
    line_test_acc, = ax1.plot(list(range(epochs)), test_acc, label=f'validation acc {last_value(test_acc)}', c='#FFA07A')
        
    # 标记损失最低点
    train_loss_min = ax1.scatter(min_train_x, min_train_loss, c='b',
                label=f'train loss min: {min_train_loss:.4f}')
    test_loss_min = ax1.scatter(min_test_x, min_test_loss, c='#00BFFF',
                label=f'validation loss min: {min_test_loss:.4f}')

    # 标记准确率最高点
    train_acc_max = ax1.scatter(max_train_acc_x, max_train_acc, c='r',
                label=f'train acc max: {max_train_acc:.4f}')
    train_val_max = ax1.scatter(max_test_acc_x, max_test_acc, c='#FFA07A',
                label=f'validation acc max: {max_test_acc:.4f}')

    # 创建右侧坐标轴
    ax2 = ax1.twinx()

    # 绘制学习率
    line_lr, = ax2.plot(list(range(epochs)), lrs, label='lr', c='#87CEFF',linewidth=2,alpha =0.5)

    # 添加图例
    ax1.legend(handles=[line_train_loss, line_test_loss, line_train_acc, line_test_acc, train_loss_min, test_loss_min, train_acc_max, train_val_max])
    # ax2.legend(handles=[line_lr], loc='upper left')

    # 显示横向和纵向的格线
    ax1.grid(True)
    ax1.set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值

    # 图2
    t2_handles = []
    # 绘制f1曲线
    for i in f1_scores:
        _line, = axs[1].plot(list(range(epochs)), f1_scores[i], label=f'f1 {i} {last_value(f1_scores[i])}')
        t2_handles.append(_line)
    axs[1].grid(True)
    axs[1].set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值
    axs[1].legend(handles=t2_handles)

    plt.title(f'{title} {datetime.now().strftime("%Y%m%d_%H_%M_%S")}')
    plt.savefig(f"pic_{title}.png")
    wx.send_file(f"pic_{title}.png")
    display.clear_output(wait=True)
    plt.pause(0.00000001)


def log_grad(model):
    '''Print the grad of each layer'''
    max_grad = 0
    min_grad = 0
    has_nan = False
    with open(os.path.join(root, 'grad'), 'a') as f:
        f.write('--------------------------------------------\n')
        for name, parms in model.named_parameters():
            f.write(f'-->name: {name}\n')
            f.write(f'-->grad_requirs: {parms.requires_grad}\n')
            f.write(f'-->grad_value: {parms.grad}\n')

            # 记录最大最小梯度
            if parms.grad is not None:
                max_grad = max(
                    max(max_grad, parms.grad.max().item()), max_grad)
                min_grad = min(
                    min(min_grad, parms.grad.min().item()), min_grad)

                # 检查是否有nan
                if torch.isnan(parms.grad).any():
                    has_nan = True

        f.write('--------------------------------------------\n')
        f.write(f"记录梯度 max: {max_grad}, min: {min_grad}, has_nan: {has_nan}")
    logger.debug(f"记录梯度 max: {max_grad}, min: {min_grad}, has_nan: {has_nan}")


def count_correct_predictions(predictions, labels):
    softmax_predictions = F.softmax(predictions, dim=1)
    predicted_labels = torch.argmax(softmax_predictions, dim=1)
    correct_count = torch.sum(predicted_labels == labels).item()
    # logger.debug(f'correct: {correct_count} / {len(labels)}')
    return correct_count

# A function to encapsulate the training loop


def batch_gd(model, criterion, optimizer_class, lr_lambda, train_loader, test_loader, epochs, result_dict):

    train_losses = np.full(epochs, np.nan)
    test_losses = np.full(epochs, np.nan)
    train_acc = np.full(epochs, np.nan)
    test_acc = np.full(epochs, np.nan)
    lrs = np.full(epochs, np.nan)

    all_targets, all_predictions = [], []
    f1_scores = {}

    best_test_loss = np.inf
    best_test_epoch = 0
    begin = 0
    train_loss = []
    test_loss = []
    train_correct = 0
    train_all = 0
    test_correct = 0
    test_all = 0
    step_in_epoch = 0

    # optimizer
    optimizer = None
    scheduler = None

    # Automatic Mixed Precision
    scaler = GradScaler()

    # 检查是否有缓存文件
    if os.path.exists(os.path.join(root, 'var', f'datas.pkl')):
        logger.debug(f"使用缓存文件继续训练")
        train_losses, test_losses, train_acc, test_acc,lrs, f1_scores,all_targets, all_predictions, best_test_loss, best_test_epoch, begin, train_loss, test_loss, train_correct, test_correct, train_all, test_all, step_in_epoch, scaler = pickle.load(
            open(os.path.join(root, 'var', f'datas.pkl'), 'rb'))

        # logger.debug(f'train_losses: \n{train_losses}')
        # logger.debug(f'test_losses: \n{test_losses}')

        model = torch.load(os.path.join(root, 'var', f'model.pkl'))
    else:
        logger.debug(f"新的训练")

    # 构造优化器
    optimizer = optimizer_class(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if None is lr_lambda:
        if init_learning_ratio > 0:
            scheduler = Increase_ReduceLROnPlateau(optimizer)
        else:
            scheduler = warm_up_ReduceLROnPlateau(optimizer, warm_up_epoch=warm_up_epochs, iter_num_each_epoch=len(train_loader))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

    # 恢复 scheduler/optmizer
    if os.path.exists(os.path.join(root, 'var', f'helper.pkl')):
        sd_scheduler, sd_optimizer, sd_train_loader, sd_test_loader = pickle.load(
            open(os.path.join(root, 'var', f'helper.pkl'), 'rb'))
        scheduler.load_state_dict(sd_scheduler)
        optimizer.load_state_dict(sd_optimizer)
        train_loader.sampler.load_state_dict(sd_train_loader)
        test_loader.sampler.load_state_dict(sd_test_loader)

    # 初始化warnup
    if isinstance(scheduler, warm_up_ReduceLROnPlateau) or isinstance(scheduler, Increase_ReduceLROnPlateau):
        scheduler.warn_up(init=True)

    t = time.time()
    for it in range(begin, epochs):
        # 早停检查
        if best_test_epoch > 0 and best_test_epoch + no_better_stop < it:
            break

        msg = f'Epoch {it+1}/{epochs} '

        t0 = datetime.now()
        # 训练
        if step_in_epoch == 0:
            logger.debug(f'{msg}开始训练')

            model.train()

            count = 0
            if train_loader.sampler.idx==0:
                logger.debug("重置训练记录")
                train_correct = 0
                train_all = 0

            for inputs, targets in tqdm(train_loader, initial=int(train_loader.sampler.idx / batch_size), total=len(train_loader)):
                # move data to GPU
                inputs, targets = inputs.to(device, dtype=torch.float), targets.to(
                    device, dtype=torch.int64)
                optimizer.zero_grad()

                outputs = None
                loss = None
                if amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    scaler.scale(loss).backward()
                    # print_grad(model)
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    # log_grad(model)
                    optimizer.step()

                train_loss.append(loss.item())

                with torch.no_grad():
                    train_correct += count_correct_predictions(
                        outputs, targets)
                    train_all += len(targets)
                    # logger.debug(f'[{count}] train_correct: {train_correct}')

                count += 1
                if count % cache_epoch == 0:
                    logger.debug(f"[{count}] 缓存数据")
                    pickle.dump((train_losses, test_losses, train_acc, test_acc, lrs,f1_scores,all_targets, all_predictions,  best_test_loss, best_test_epoch, it, train_loss, test_loss,
                                train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(root, 'var', f'datas.pkl'), 'wb'))
                    torch.save(model, os.path.join(root, 'var', f'model.pkl'))
                    pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
                    ), test_loader.sampler.state_dict()), open(os.path.join(root, 'var', f'helper.pkl'), 'wb'))

                # warnup
                if isinstance(scheduler, warm_up_ReduceLROnPlateau) or isinstance(scheduler, Increase_ReduceLROnPlateau):
                    scheduler.warn_up()

            step_in_epoch += 1

            # Get train loss and test loss
            train_loss = np.mean(train_loss)  # a little misleading

            # 缓存数据
            pickle.dump((train_losses, test_losses, train_acc, test_acc, lrs,f1_scores, all_targets, all_predictions, best_test_loss, best_test_epoch, it, train_loss, test_loss,
                        train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(root, 'var', f'datas.pkl'), 'wb'))
            torch.save(model, os.path.join(root, 'var', f'model.pkl'))
            pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
            ), test_loader.sampler.state_dict()), open(os.path.join(root, 'var', f'helper.pkl'), 'wb'))

        torch.cuda.empty_cache()
        logger.debug(f'{msg}训练完成')
        # logger.debug(f'{msg}训练完成\ntrain_loss: {train_loss}')

        # 验证
        if step_in_epoch == 1:
            logger.debug(f'{msg}开始验证')
            model.eval()

            if test_loader.sampler.idx==0:
                logger.debug("重置验证记录")
                all_targets = []
                all_predictions = []

                test_correct = 0
                test_all = 0
            with torch.no_grad():
                count = 0
                for inputs, targets in tqdm(test_loader, initial=int(test_loader.sampler.idx / batch_size), total=len(test_loader)):
                    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(
                        device, dtype=torch.int64)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss.append(loss.item())
                    # logger.debug(f'test_loss: {loss.item()}')
                    
                    test_correct += count_correct_predictions(
                        outputs, targets)
                    test_all += len(targets)

                    # 转成概率
                    p = torch.softmax(outputs, dim=1)

                    # Get prediction
                    # torch.max returns both max and argmax
                    _, predictions = torch.max(p, 1)

                    all_targets.append(targets.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())

                    count += 1
                    if count % cache_epoch == 0:
                        logger.debug(f"[{count}] 缓存数据")
                        pickle.dump((train_losses, test_losses, train_acc, test_acc, lrs, f1_scores, all_targets, all_predictions, best_test_loss, best_test_epoch, it, train_loss, test_loss,
                                    train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(root, 'var', f'datas.pkl'), 'wb'))
                        torch.save(model, os.path.join(
                            root, 'var', f'model.pkl'))
                        pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
                        ), test_loader.sampler.state_dict()), open(os.path.join(root, 'var', f'helper.pkl'), 'wb'))

            all_targets = np.concatenate(all_targets)
            all_predictions = np.concatenate(all_predictions)

            report = classification_report(
                all_targets, all_predictions, digits=4, output_dict=True)
            # 将分类报告转换为DataFrame
            df_report = pd.DataFrame(report).transpose()
            _f1_scores_dict = df_report.iloc[:-3, 2].to_dict()
            # 存入 f1_scores
            for i in _f1_scores_dict:
                if i not in f1_scores:
                    f1_scores[i] = np.full(epochs, np.nan)
                f1_scores[i][it] = _f1_scores_dict[i]

            step_in_epoch += 1

            test_loss = np.mean(test_loss)

            pickle.dump((train_losses, test_losses, train_acc, test_acc, lrs,f1_scores,all_targets, all_predictions,  best_test_loss, best_test_epoch, it, train_loss, test_loss,
                        train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(root, 'var', f'datas.pkl'), 'wb'))
            torch.save(model, os.path.join(root, 'var', f'model.pkl'))
            pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
            ), test_loader.sampler.state_dict()), open(os.path.join(root, 'var', f'helper.pkl'), 'wb'))

        torch.cuda.empty_cache()
        # logger.debug(f'{msg}验证完成\ntest_loss: {test_loss}')
        logger.debug(f'{msg}验证完成')

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        logger.debug(
            f'train_correct: {train_correct}, train_loader: {train_all}')
        train_acc[it] = train_correct / train_all

        logger.debug(f'test_correct: {test_correct}, test_loader: {test_all}')
        test_acc[it] = test_correct / test_all

        if test_loss < best_test_loss:
            torch.save(model, os.path.join(root, f'best_val_model'))
            best_test_loss = test_loss
            best_test_epoch = it

        # 更新学习率
        lrs[it] = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            logger.debug(f'ReduceLROnPlateau 更新学习率')
            scheduler.step(train_loss)
        else:
            scheduler.step()

        msg += f'lr: {lrs[it]:.8f} -> {optimizer.param_groups[0]["lr"]:.8f}\n'

        dt = datetime.now() - t0
        msg += f'Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}, Train acc: {train_acc[it]:.4f}, Val acc: {test_acc[it]:.4f} \nDuration: {dt}, Best Val Epoch: {best_test_epoch}'
        logger.debug(msg)

        if it % 10 == 0:
            wx.send_message(f'[{title}] {msg}')

        # 缓存数据
        train_loss = []
        test_loss = []
        step_in_epoch = 0
        pickle.dump((train_losses, test_losses, train_acc, test_acc, lrs,f1_scores, all_targets, all_predictions, best_test_loss, best_test_epoch, it+1, train_loss, test_loss,
                    train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(root, 'var', f'datas.pkl'), 'wb'))
        torch.save(model, os.path.join(root, 'var', f'model.pkl'))
        pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
        ), test_loader.sampler.state_dict()), open(os.path.join(root, 'var', f'helper.pkl'), 'wb'))

        # 更新最佳数据
        result_dict['train_loss'] = min(train_losses)
        result_dict['val_loss'] = min(test_losses)
        result_dict['train_acc'] = max(train_acc)
        result_dict['val_acc'] = max(test_acc)

    wx.send_message(f'[{title}] 训练完成')
    plot_loss(epochs, train_losses, test_losses, train_acc, test_acc, lrs, f1_scores)
    return min(train_losses), min(test_losses), max(train_acc), max(test_acc)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Residual(nn.Module):  # @save
    def __init__(self, num_channels, dp_rate):
        super().__init__()
        mid_channels = num_channels * 4
        # depthwise conv
        self.conv0 = nn.Conv2d(num_channels, num_channels,kernel_size=(7, 1), padding=(3, 0), stride=(1, 1), groups=num_channels)
        self.norm = LayerNorm(num_channels, eps=1e-6, data_format="channels_last")
        self.conv1 = nn.Linear(num_channels, mid_channels)
        self.conv2 = nn.Linear(mid_channels, num_channels)

        self.gamma = nn.Parameter(1e-6 * torch.ones((num_channels,)),requires_grad=True)
        self.grn = GRN(mid_channels)
        self.cbam = CBAMLayer(num_channels)

        self.drop_path = DropPath(dp_rate)

    def forward(self, X):
        Y = self.conv0(X)
        Y = Y.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        Y = self.norm(Y)
        Y = F.relu(self.conv1(Y))
        Y = self.conv2(Y)
        Y = Y.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        # 注意力
        Y = self.cbam(Y)

        Y = X + self.drop_path(Y)
        return Y

        # Y = self.conv0(X)
        # Y = Y.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        # Y = self.norm(Y)
        # Y = F.relu(self.conv1(Y))
        # Y = self.grn(Y)
        # Y = self.conv2(Y)
        # Y = Y.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        # Y = X + self.drop_path(Y)
        # return Y

        # Y = self.conv0(X)
        # Y = Y.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        # Y = self.norm(Y)
        # Y = F.relu(self.conv1(Y))
        # Y = self.conv2(Y)
        # Y = self.gamma * Y
        # Y = Y.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        # Y = X + self.drop_path(Y)
        # return Y

def make_block(layer_n, in_channels, out_channels, dp_rate):
    layers = []
    if out_channels > in_channels:
        # 降维
        layers.append(LayerNorm(in_channels, eps=1e-6, data_format="channels_first"))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(2, 1), stride=(2, 1)))

    for i in range(layer_n):
        layers.append(Residual(out_channels, dp_rate))
    return nn.Sequential(*layers)

class ConvNeXt_block(nn.Module):
    def __init__(self, y_len, in_channel, layer_list, channel_list, drop_path_rate=0.0):
        super().__init__()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, len(channel_list))]

        self.stages = nn.ModuleList()
        for layer_n, out_channel, dp_rate in zip(layer_list,channel_list, dp_rates):
            self.stages.append(make_block(layer_n, in_channel, out_channel, dp_rate))
            in_channel = out_channel

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.f = nn.Flatten()
        self.ln = nn.LayerNorm(channel_list[-1], eps=1e-6)
        self.l = nn.Linear(channel_list[-1], y_len)

    def forward(self, X):
        for stage in self.stages:
            X = stage(X)

        X = self.global_avg_pool(X)
        return self.l(self.ln(X.mean([-2, -1])))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # logger.debug(i, (kernel_size-1) * dilation_size, dilation_size)
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(
            normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(
            normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class inception_resnet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        inception_in = channels
        inception_out = inception_in // 4
        # inception moduels  
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=inception_in, out_channels=inception_out, kernel_size=(1,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_out),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=inception_out, out_channels=inception_in, kernel_size=(3,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_in),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=inception_in, out_channels=inception_out, kernel_size=(1,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_out),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=inception_out, out_channels=inception_in, kernel_size=(5,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_in),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=inception_in, out_channels=inception_in, kernel_size=(1,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_in),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.conv1_1 = nn.Conv2d(inception_in, inception_in*3,kernel_size=1,padding='same')

    def forward(self, x):
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x_inp = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = self.conv1_1(x) 
        return F.relu(x_inp+x)

def test_model(test_loader, select='best'):
    """
    模型可选 最优/最终 best/final
    """
    model = None

    if 'best' == select:
        model = torch.load(os.path.join(root, f'best_val_model'))
    elif 'final' == select:
        model = torch.load(os.path.join(root, 'var', f'model.pkl'))
    else:
        return

    n_correct = 0.
    n_total = 0.

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(
                device, dtype=torch.int64)

            # Forward pass
            outputs = model(inputs)

            # 转成概率
            p = torch.softmax(outputs, dim=1)

            # Get prediction
            # torch.max returns both max and argmax
            _, predictions = torch.max(p, 1)

            # update counts
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

    test_acc = n_correct / n_total
    logger.debug(f"Test acc: {test_acc:.4f}")

    # model = torch.load('best_val_model_pytorch')
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(
                device, dtype=torch.int64)

            # Forward pass
            outputs = model(inputs)

            # 转成概率
            p = torch.softmax(outputs, dim=1)

            # Get prediction
            # torch.max returns both max and argmax
            _, predictions = torch.max(p, 1)

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    logger.debug(
        f'accuracy_score: {accuracy_score(all_targets, all_predictions)}')
    report = classification_report(
        all_targets, all_predictions, digits=4, output_dict=True)
    # 将分类报告转换为DataFrame
    df = pd.DataFrame(report).transpose()
    logger.debug(f'\n{df}')
    dfi.export(df, 'test_result.png', table_conversion="matplotlib")
    wx.send_file("test_result.png")

def test_ps(data_loader, ps):
    """
    data_loader 数据
    ps     概率阈值
    """
    # 载入模型
    model = torch.load(os.path.join(root, f'best_val_model'))

    # 真实标签 预测概率
    true_label = []
    p = []

    for inputs, targets in data_loader:
        # Move to GPU
        inputs = inputs.to(device, dtype=torch.float)

        # Forward pass
        outputs = model(inputs)

        # 转成概率
        # 0， 1， 2
        _p = torch.softmax(outputs, dim=1)

        true_label.append(targets.numpy())
        p.append(_p.detach().cpu().numpy())

    true_label = np.concatenate(true_label)
    p = np.concatenate(p)
    _, predictions = torch.max(torch.tensor(p), 1)
    logger.debug(pd.DataFrame({'label': predictions})['label'].value_counts())

    # 标签类别
    n = p.shape[1]

    predict_labels = np.zeros(p.shape)
    for i in range(n):
        # 生成预测
        func = np.vectorize(lambda x: 1 if x > ps[i] else 0)
        predict_labels[:, i] = func(p[:, i])

    predicts = [0 if predict_labels[i][0] else 2 if predict_labels[i]
                [2] else 1 for i in range(predict_labels.shape[0])]
    # predicts = [1 if predict_labels[i][1] else 0 if predict_labels[i][0] else 2 for i in range(predict_labels.shape[0])]

    logger.debug(pd.DataFrame({'label': predicts})['label'].value_counts())

    report = classification_report(
        true_label, predicts, digits=4, output_dict=True)
    # 将分类报告转换为DataFrame
    df = pd.DataFrame(report).transpose()
    logger.debug(f'\n{df}')
    dfi.export(df, 'pic.png', table_conversion="matplotlib")
    wx.send_file("pic.png")


def plot_roc(data_loader):
    """
    data_loader 数据
    """
    # 载入模型
    model = torch.load(os.path.join(root, f'best_val_model'))

    # 针对 0 标签（下跌）
    # label = 0

    # 真实标签 预测概率
    true_label = []
    p = []

    for inputs, targets in data_loader:
        # Move to GPU
        inputs = inputs.to(device, dtype=torch.float)

        # Forward pass
        outputs = model(inputs)

        # 转成概率
        # 0， 1， 2
        _p = torch.softmax(outputs, dim=1)

        true_label.append(targets.numpy())
        p.append(_p.detach().cpu().numpy())

    true_label = np.concatenate(true_label)
    p = np.concatenate(p)

    # 标签类别
    n = p.shape[1]

    # 子图
    fig, axes = plt.subplots(1, n, figsize=(15, 5))

    bests = []
    for i in range(n):
        # 计算混淆矩阵
        c_m = confusion_matrix(true_label, np.argmax(p, axis=1), labels=[i])
        logger.debug(f'类别:{i} 混淆矩阵:\n{c_m}')

        # 计算roc曲线数值
        fpr, tpr, thresholds = roc(true_label, p[:, i], pos_label=i)

        # 找到最佳的阈值
        best_idx = (tpr - fpr).tolist().index(max(tpr - fpr))
        pbest = thresholds[best_idx]
        bests.append(pbest)
        logger.debug(f'类别:{i} 最佳阈值:{pbest:.3f}')

        # 绘制roc曲线
        axes[i].plot(fpr, tpr)
        axes[i].set_title(f'ROC label:{i} best:{pbest:.3f}')
        axes[i].scatter(fpr[best_idx], tpr[best_idx], c='r')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

    return bests


def predict_test_y():
    # 载入模型
    model = torch.load(os.path.join(root, f'best_val_model'))

    # 读取测试数据
    mean_std, test_x, test_y, test_raw = pickle.load(
        open(os.path.join(data_path, f'test.pkl'), 'rb'))

    # 修正标签 -1,0,1 -> 0,1,2
    test_y = [i+1 for i in test_y]

    dataset_test = Dataset(test_raw, test_x, test_y, mean_std)
    # , num_workers=workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size)

    results = {'id': [], 'predict': [], 'target': []}

    idx = 0
    for inputs, targets in test_loader:
        a = idx*batch_size
        idx += 1

        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(
            device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)

        # 转成概率
        p = torch.softmax(outputs, dim=1)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(p, 1)

        results['predict'] += predictions.tolist()
        results['target'] += targets.tolist()

    results = pd.DataFrame(results)
    # 储存预测
    results.to_csv(os.path.join(root, 'predict.csv'), index=False)


"""
1, 1, 70, 46
width_ratio=0.3 
Total params: 247,965
FLOPs: 4.71M
"""
class m_convnext(nn.Module):
    def __init__(self, y_len, width_ratio=0.3, layer_ratio=1, use_trade_data=True):
        super().__init__()
        self.y_len = y_len
        self.use_trade_data = use_trade_data

        self.pk_stem = None
        self.trade_stem = None
        if use_trade_data:
            self.pk_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=4, out_channels=8,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=8, out_channels=16,
                        kernel_size=(1, 10)),
                LayerNorm(16, eps=1e-6, data_format="channels_first")
            )

            self.trade_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4,
                        kernel_size=(1, 3), stride=(1, 3)),
                nn.Conv2d(in_channels=4, out_channels=8,
                        kernel_size=(1, 2), stride=(1, 2)),
                LayerNorm(8, eps=1e-6, data_format="channels_first")
            )

        else:
            self.pk_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=6, out_channels=12,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=12, out_channels=24,
                        kernel_size=(1, 10)),
                LayerNorm(24, eps=1e-6, data_format="channels_first")
            ) 

        channel_list = [24]
        for i in range(3):
            channel_list.append(int((1*width_ratio + 1) * channel_list[-1]))
        print(f'channel_list: {channel_list}')

        self.block = ConvNeXt_block(
            self.y_len, 24, 
            [int(i*layer_ratio) for i in [3,3,9,3]],
            channel_list,
            0.3
        )

    def forward(self, combine_x):
        # 盘口数据
        x = combine_x[:, :, :, :40]  # torch.Size([1, 1, 70, 40])
        x = self.pk_stem(x)  # torch.Size([1, 16, 70, 1])

        # 成交数据
        if self.use_trade_data:
            x_2 = combine_x[:, :, :, 40:]  # torch.Size([1, 1, 70, 6])
            x_2 = self.trade_stem(x_2)  # torch.Size([1, 8, 70, 1])

            # 合并
            x = torch.cat((x, x_2), dim=1)# torch.Size([1, 24, 70, 1])

        return self.block(x)

"""
1, 1, 70, 46
Total params: 143,842
FLOPs: 24.36M
"""
class m_deeplob(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        x = x[:, :, :, :40]

        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(device)
        c0 = torch.zeros(1, x.size(0), 64).to(device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        
        # x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    # device = 'cuda'
    # print(device)

    # # model = m_deeplob(y_len=2)
    # model = m_convnext(y_len=2, width_ratio=0.3)
    # print(model)

    # summary(model, (1, 1, 70, 46), device=device)
    
    # from thop import profile
    # from thop import clever_format

    # model = model.to(device)
    # input = torch.randn((1, 1, 70, 46)).to(device)
    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params])
    # print(f"FLOPs: {flops} Params: {params}")
    pass