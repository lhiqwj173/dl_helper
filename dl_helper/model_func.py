from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
from torch.utils import data
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torch
from torchmetrics import R2Score
from tqdm import tqdm
import pandas as pd
import pickle
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import mean_squared_error
import multiprocessing
from IPython import display
import math
import dataframe_image as dfi
import dill
import itertools
import random, psutil

from accelerate import Accelerator
from accelerate.utils import set_seed

from py_ext.wechat import wx
from py_ext.lzma import compress_folder

from .tg import tg_download_async, tg_download, tg_upload, tg_del_file
from .train_param import init_param, logger, params, data_parm2str, data_str2parm
from .data import read_data
from .data_map import DATA_MAP
from .tool import report_memory_usage, check_nan

ses = '1BVtsOKABu6pKio99jf7uqjfe5FMXfzPbEDzB1N5DFaXkEu5Og5dJre4xg4rbXdjRQB7HpWw7g-fADK6AVDnw7nZ1ykiC5hfq-IjDVPsMhD7Sffuv0lTGa4-1Dz2MktHs3e_mXpL1hNMFgNm5512K1BWQvij3xkoiHGKDqXLYzbzeVMr5e230JY7yozEZRylDB_AuFeBGDjLcwattWnuX2mnTZWgs-lS1A_kZWomGl3HqV84UsoJlk9b-GAbzH-jBunsckkjUijri6OBscvzpIWO7Kgq0YzxJvZe_a1N8SFG3Gbuq0mIOkN3JNKGTmYLjTClQd2PIJuFSxzYFPQJwXIWZlFg0O2U='

# 当训练的损失序列区域平缓时，减低学习率
class ReduceLR_slow_loss():
    def __init__(self, optimizer, min_pct=-0.00005, patience=10, factor=0.1, min_lr=0, eps=1e-8, debug=False):
        self.optimizer = optimizer

        self.min_pct = min_pct
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.eps = eps
        self.wait = 0

        self.debug = debug

    def step(self, array_loss):
        if self.wait > 0:
            self.wait -= 1
            return

        # 计算损失均线，ma=self.patience
        # 均线变动率 大于 min_pct 则减少学习率
        loss = pd.DataFrame({'loss': array_loss}).dropna()
        if len(loss) < self.patience+1:
            return

        loss['ma'] = loss['loss'].rolling(self.patience).mean()
        loss['pct'] = loss['ma'].pct_change()
        loss['match'] = loss['pct']>=self.min_pct

        if loss.iloc[-1]['match']:
            self._reduce_lr()
        elif self.debug:
            print('pass')

    def _reduce_lr(self):
        self.wait = self.patience
        if self.debug:
            print('reduce_lr')
        if None is self.optimizer:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class Increase_ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    按照指定速率，初始学习率在每个迭代中增加学习率，单触发 ReduceLROnPlateau 衰减后则不在增加学习率
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        super().__init__(optimizer, mode, factor, patience,
                 threshold, threshold_mode, cooldown, min_lr, eps, verbose)

        self.init_learning_ratio = params.init_learning_ratio
        self.increase = params.increase_ratio * params.init_learning_ratio
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

def last_value(data):
    """返回最后一个非nan值"""
    for i in range(len(data)-1, -1, -1):
        if not math.isnan(data[i]):
            return "{:.3e}".format(data[i])
    raise ValueError("没有找到非nan值")

def plot_loss(help_vars, cost_hour, send_wx=True, folder=''):
    epochs = params.epochs

    # 创建图形和坐标轴
    fig, axs = None, None
    ax1 = None
    if params.classify:
        # 分类模型
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [7, 3]})
        ax1 = axs[0]
    else:
        fig, axs = plt.subplots(figsize=(15, 10))
        ax1 = axs

    # 用于添加图例
    ax1_handles = []

    # 计算误差最低点
    min_train_loss = min(help_vars.train_losses)
    min_test_loss = min(help_vars.test_losses)
    min_train_x = help_vars.train_losses.tolist().index(min_train_loss)
    min_test_x = help_vars.test_losses.tolist().index(min_test_loss)
    # 绘制loss曲线
    ax1_handles.append(ax1.plot(list(range(epochs)), help_vars.train_losses, label=f'train loss {last_value(help_vars.train_losses)}', c='b')[0])
    ax1_handles.append(ax1.plot(list(range(epochs)), help_vars.test_losses, label=f'validation loss {last_value(help_vars.test_losses)}', c='#00BFFF')[0])
    # 标记损失最低点
    ax1_handles.append(ax1.scatter(min_train_x, min_train_loss, c='b',label=f'train loss min: {min_train_loss:.4f}'))
    ax1_handles.append(ax1.scatter(min_test_x, min_test_loss, c='#00BFFF',label=f'validation loss min: {min_test_loss:.4f}'))

    if params.classify:
        # 分类模型
        # 计算acc最高点
        max_train_acc = max(help_vars.train_acc)
        max_test_acc = max(help_vars.test_acc)
        max_train_acc_x = help_vars.train_acc.tolist().index(max_train_acc)
        max_test_acc_x = help_vars.test_acc.tolist().index(max_test_acc)
        # 绘制acc曲线
        ax1_handles.append(ax1.plot(list(range(epochs)), help_vars.train_acc, label=f'train acc {last_value(help_vars.train_acc)}', c='r')[0])
        ax1_handles.append(ax1.plot(list(range(epochs)), help_vars.test_acc, label=f'validation acc {last_value(help_vars.test_acc)}', c='#FFA07A')[0])
        # 标记准确率最高点
        ax1_handles.append(ax1.scatter(max_train_acc_x, max_train_acc, c='r',label=f'train acc max: {max_train_acc:.4f}'))
        ax1_handles.append(ax1.scatter(max_test_acc_x, max_test_acc, c='#FFA07A',label=f'validation acc max: {max_test_acc:.4f}'))
    else:
        # 回归模型
        colors = ['#C39BD3', '#F1948A', '#F9E79F', '#F5CBA7', '#AEB6BF', '#48C9B0']
        for i in range(params.y_n):
            # 计算r2最高点
            max_train_r2 = max(help_vars.train_r2s[i])
            max_test_r2 = max(help_vars.test_r2s[i])
            max_train_r2_x = help_vars.train_r2s[i].tolist().index(max_train_r2)
            max_test_r2_x = help_vars.test_r2s[i].tolist().index(max_test_r2)
            # 绘制r2曲线
            c1, c2 = colors[0], colors[1]
            colors = colors[2:]
            ax1_handles.append(ax1.plot(list(range(epochs)), help_vars.train_r2s[i], label=f'train r2 {i} {last_value(help_vars.train_r2s[i])}', c=c1)[0])
            ax1_handles.append(ax1.plot(list(range(epochs)), help_vars.test_r2s[i], label=f'validation r2 {i} {last_value(help_vars.test_r2s[i])}', c=c2)[0])
            # 标记r2最高点
            ax1_handles.append(ax1.scatter(max_train_r2_x, max_train_r2, c=c1,label=f'train r2 {i} max: {max_train_r2:.4f}'))
            ax1_handles.append(ax1.scatter(max_test_r2_x, max_test_r2, c=c2,label=f'validation r2 {i} max: {max_test_r2:.4f}'))

    # 创建右侧坐标轴
    ax2 = ax1.twinx()

    # 绘制学习率
    line_lr, = ax2.plot(list(range(epochs)), lrs, label='lr', c='#87CEFF',linewidth=2,alpha =0.5)

    # 添加图例
    ax1.legend(handles=ax1_handles)
    # ax2.legend(handles=[line_lr], loc='upper left')

    # 显示横向和纵向的格线
    ax1.grid(True)
    ax1.set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值

    # 图2
    if params.classify:
        # 分类模型
        t2_handles = []
        # 绘制f1曲线
        for i in f1_scores:
            _line, = axs[1].plot(list(range(epochs)), f1_scores[i], label=f'f1 {i} {last_value(f1_scores[i])}')
            t2_handles.append(_line)
        axs[1].grid(True)
        axs[1].set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值
        axs[1].legend(handles=t2_handles)

    plt.title(f'{params.train_title} | {params.describe} | {datetime.now().strftime("%Y%m%d")}          cost:{cost_hour:.2} hours')

    pic_file = os.path.join(folder if folder else params.root, f"{params.train_title}.png")
    plt.savefig(pic_file)

    if send_wx:
        wx.send_file(pic_file)

def log_grad(model):
    '''Print the grad of each layer'''
    max_grad = 0
    min_grad = 0
    has_nan = False
    with open(os.path.join(params.root, 'grad'), 'a') as f:
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

debug = False
# A function to encapsulate the training loop

def pack_folder():
    # 打包训练文件夹 zip 
    file = params.root+".7z"

    if os.path.exists(file):
        # 删除
        os.remove(file)

    compress_folder(params.root, file, 9, inplace=False)

    if not debug:
        # 删除当前的训练文件，如果存在
        tg_del_file(ses, f'{params.train_title}.7z')
        # 上传到tg
        tg_upload(ses, file)


class helper:
    def __init__(self):
        # 统计变量
        self.begin_time = time.time()
        self.train_losses = np.full(params.epochs, np.nan)
        self.test_losses = np.full(params.epochs, np.nan)
        self.train_r2s = [np.full(params.epochs, np.nan) for i in range(params.y_n)] if not params.classify else None
        self.test_r2s = [np.full(params.epochs, np.nan) for i in range(params.y_n)] if not params.classify else None
        self.train_acc = np.full(params.epochs, np.nan)
        self.test_acc = np.full(params.epochs, np.nan)
        self.lrs = np.full(params.epochs, np.nan)
        self.all_targets, all_predictions = [], []
        self.f1_scores = {}
        self.best_test_loss = np.inf
        self.best_test_epoch = 0
        self.begin = 0
        self.it = 0
        self.resume_train_step = 0
        self.resume_val_step = 0
        self.train_loss = []
        self.test_loss = []
        self.train_r_squared = [R2Score() for i in range(params.y_n)] if not params.classify else None
        self.test_r_squared = [R2Score() for i in range(params.y_n)] if not params.classify else None
        self.train_correct = 0
        self.train_all = 0
        self.test_correct = 0
        self.test_all = 0
        self.step_in_epoch = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

def batch_gd(accelerator, result_dict, cnn, seed):
    # 检查下载tg上的 checkpoints 数据
    resume_from_checkpoint = False
    if not params.debug and accelerator.is_local_main_process:
        tg_download(
            ses,
            f'{params.train_title}.7z',
            '/kaggle/working/tg'
        )

        # 如果存在 checkpoints ，拷贝到正确的路径以继续训练
        if os.path.exists(os.path.join('/kaggle/working/tg', params.train_title, 'checkpoints')):
            wx.send_message(f'[{params.train_title}] 使用缓存文件继续训练')
            logger.debug(f"使用缓存文件继续训练")
            resume_from_checkpoint = True
            shutil.copytree(os.path.join('/kaggle/working/tg', params.train_title), params.root, dirs_exist_ok=True)

    # 训练过程变量
    help_vars = helper()

    # 模型
    model = params.model

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing) if params.classify else nn.MSELoss()

    # 构造优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # 获取训练数据
    train_loader = read_data('train', cnn=cnn, seed=seed, log=accelerator.is_local_main_process)
    assert len(train_loader) > 0, "没有训练数据"
    # 获取验证数据
    val_loader = read_data('val', cnn=cnn, seed=seed, log=accelerator.is_local_main_process)
    assert len(val_loader) > 0, "没有验证数据"
    
    # 获取输入数据形状
    input_shape = train_loader.dataset.input_shape
    
    # 构造调度器
    scheduler = Increase_ReduceLROnPlateau(optimizer) if params.init_learning_ratio > 0 else warm_up_ReduceLROnPlateau(optimizer, warm_up_epoch=params.warm_up_epochs, iter_num_each_epoch=len(train_loader))
    scheduler2 = ReduceLR_slow_loss(optimizer)# 新增一个调度器

    # 交由Accelerator处理
    model, optimizer, train_loader, val_loader, scheduler, scheduler2, help_vars = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler, scheduler2, help_vars
    )

    # 检查是否有缓存文件
    if resume_from_checkpoint:
        if accelerator.is_local_main_process:
            logger.debug(f"加载缓存文件")
        accelerator.load_state(os.path.join(params.root, 'checkpoints'))
        help_vars.begin = help_vars.it

    # 初始化warnup
    if isinstance(scheduler, warm_up_ReduceLROnPlateau) or isinstance(scheduler, Increase_ReduceLROnPlateau):
        scheduler.warn_up(init=True)

    for it in range(help_vars.begin, params.epochs):
        # 记录当前轮数
        help_vars.it = it

        # 早停检查
        if help_vars.best_test_epoch > 0 and params.no_better_stop>0 and help_vars.best_test_epoch + params.no_better_stop < it:
            break

        msg = f'Epoch {it+1}/{params.epochs} '

        t0 = datetime.now()
        # 训练
        if help_vars.step_in_epoch == 0:
            if accelerator.is_local_main_process:
                logger.debug(f"开始训练")

            # 跳过训练过的步骤
            if resume_from_checkpoint and it == help_vars.begin and help_vars.resume_train_step:
                active_dataloader = accelerator.skip_first_batches(train_loader, help_vars.resume_train_step)
            else:
                active_dataloader = train_loader
            step_length = len(active_dataloader)

            model.train()
            idx = 0
            train_last = time.time()
            for inputs, targets in tqdm(active_dataloader, disable=not accelerator.is_local_main_process):

                # 记录恢复步数
                help_vars.resume_train_step += idx + 1

                # TODO 在此处标准化数据 > 使用gpu

                # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                if not params.classify and len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # 检查loss 是否nan
                check_nan(loss, inputs=inputs, targets=targets, outputs=outputs)

                # 记录loss
                help_vars.train_loss.append(loss.detach().float())

                # 记录正确率/r方
                with torch.no_grad():
                    if accelerator.is_local_main_process:
                        all_outputs, all_targets = accelerator.gather_for_metrics((outputs, targets))
                        if params.classify:
                            # 分类模型 统计acc
                            help_vars.train_correct += count_correct_predictions(
                                all_outputs, all_targets)
                            help_vars.train_all += len(all_targets)
                        else:
                            if accelerator.is_local_main_process:
                                logger.debug(f'loss: {help_vars.train_loss[-1]:.3f}')
                            # 回归模型 统计 r方
                            for i in range(params.y_n):
                                help_vars.train_r_squared[i].update(all_outputs[:, i], all_targets[:, i])

                # warnup
                if isinstance(scheduler, warm_up_ReduceLROnPlateau) or isinstance(scheduler, Increase_ReduceLROnPlateau):
                    scheduler.warn_up()

                if idx%100 == 0 or idx == step_length - 1:
                    t1 = time.time()
                    if t1 - train_last >= 30*60 or idx == step_length - 1:
                        train_last = t1
                        
                        # 30min，缓存数据
                        accelerator.save_state(os.path.join(params.root, 'checkpoints'))

                        # 打包文件
                        if accelerator.is_local_main_process:
                            pack_folder()

                idx += 1

            help_vars.step_in_epoch += 1

            # Get train loss and test loss
            help_vars.train_loss = np.mean(help_vars.train_loss)

        if accelerator.is_local_main_process:
            logger.debug(f'{msg}训练完成')

        # 验证
        if help_vars.step_in_epoch == 1:
            if accelerator.is_local_main_process:
                logger.debug(f'{msg}开始验证')

            # 跳过验证过的步骤
            if resume_from_checkpoint and it == help_vars.begin and help_vars.resume_val_step:
                active_dataloader = accelerator.skip_first_batches(val_loader, help_vars.resume_val_step)
            else:
                active_dataloader = val_loader
            step_length = len(active_dataloader)

            model.eval()
            with torch.no_grad():

                idx = 0
                val_last = time.time()
                for inputs, targets in tqdm(active_dataloader, disable=not accelerator.is_local_main_process):
                    # 记录恢复步数
                    help_vars.resume_val_step += idx + 1

                    # TODO 在此处标准化数据 > 使用gpu

                    # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
                    if not params.classify and len(targets.shape) == 1:
                        targets = targets.unsqueeze(1)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # 检查loss 是否nan
                    check_nan(loss, inputs=inputs, targets=targets, outputs=outputs)

                    # 记录loss
                    help_vars.test_loss.append(loss.detach().float())

                    # 记录正确率/r方
                    if accelerator.is_local_main_process:
                        all_outputs, all_targets = accelerator.gather_for_metrics((outputs, targets))
                        if params.classify:
                            # 分类模型 统计acc / f1 score
                            help_vars.test_correct += count_correct_predictions(
                                all_outputs, all_targets)
                            help_vars.test_all += len(all_targets)

                            # 转成概率
                            p = torch.softmax(all_outputs, dim=1)

                            # Get prediction
                            # torch.max returns both max and argmax
                            _, predictions = torch.max(p, 1)

                            help_vars.all_targets.append(all_targets.numpy())
                            help_vars.all_predictions.append(predictions.numpy())
                        else:
                            logger.debug(f'loss: {loss.item()}')
                            # 回归模型 统计 r方
                            for i in range(params.y_n):
                                help_vars.test_r_squared[i].update(all_outputs[:, i], all_targets[:, i])

                    if idx%100 == 0 or idx == step_length - 1:
                        t1 = time.time()
                        if t1 - val_last >= 30*60 or idx == step_length - 1:
                            val_last = t1
                            
                            # 30min，缓存数据
                            accelerator.save_state(os.path.join(params.root, 'checkpoints'))

                            # 打包文件
                            if accelerator.is_local_main_process:
                                pack_folder()

                    idx += 1

            if params.classify and accelerator.is_local_main_process:
                # 分类模型 统计 f1 score
                help_vars.all_targets = np.concatenate(help_vars.all_targets)
                help_vars.all_predictions = np.concatenate(help_vars.all_predictions)

                report = classification_report(
                    help_vars.all_targets, help_vars.all_predictions, digits=4, output_dict=True)
                # 将分类报告转换为DataFrame
                df_report = pd.DataFrame(report).transpose()
                _f1_scores_dict = df_report.iloc[:-3, 2].to_dict()
                # 存入 f1_scores
                for i in _f1_scores_dict:
                    if i not in help_vars.f1_scores:
                        help_vars.f1_scores[i] = np.full(params.epochs, np.nan)
                    help_vars.f1_scores[i][it] = _f1_scores_dict[i]

            help_vars.step_in_epoch += 1
            help_vars.test_loss = np.mean(help_vars.test_loss)

        if accelerator.is_local_main_process:
            logger.debug(f'{msg}验证完成')

        # Save losses
        help_vars.train_losses[it] = help_vars.train_loss
        help_vars.test_losses[it] = help_vars.test_loss

        if params.classify:
            help_vars.train_acc[it] = help_vars.train_correct / help_vars.train_all
            help_vars.test_acc[it] = help_vars.test_correct / help_vars.test_all
        else:
            for i in range(params.y_n):
                help_vars.train_r2s[i][it] = help_vars.train_r_squared[i].compute()
                help_vars.test_r2s[i][it] = help_vars.test_r_squared[i].compute()

        model.eval()
        if help_vars.test_loss < help_vars.best_test_loss:

            help_vars.best_test_loss = help_vars.test_loss
            help_vars.best_test_epoch = it

            model_save_path = os.path.join(params.root, f'best_val_model')
            onnex_model_save_path = os.path.join(params.root, f'best_val_model.onnx')

        else:
            model_save_path = os.path.join(params.root, f'final_model')
            onnex_model_save_path = os.path.join(params.root, f'final_model.onnx')


        if accelerator.is_local_main_process:
            # 保存模型
            accelerator.save_model(model, model_save_path)

            # 导出onnx
            try:
                torch.onnx.export(model, torch.randn(input_shape).to(params.device), onnex_model_save_path, do_constant_folding=False,
                input_names=['input'], output_names=['output'])
            except:
                logger.debug('导出onnx失败')
                logger.debug(f"模型的设备：{next(model.parameters()).device}")
                logger.debug(f"数据的设备：{torch.randn(input_shape).to(params.device).device}")

        # 更新学习率
        help_vars.lrs[it] = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if accelerator.is_local_main_process:
                logger.debug(f'ReduceLROnPlateau 更新学习率')
            scheduler.step(help_vars.train_loss)
        else:
            scheduler.step()
        scheduler2.step(help_vars.train_losses)

        msg += f'lr: {help_vars.lrs[it]:.8f} -> {optimizer.param_groups[0]["lr"]:.8f}\n'

        dt = datetime.now() - t0
        msg += f'Train Loss: {help_vars.train_loss:.4f}, Val Loss: {help_vars.test_loss:.4f}'
        if params.classify:
            msg += f', Train acc: {help_vars.train_acc[it]:.4f}, Val acc: {help_vars.test_acc[it]:.4f}'
        else:
            for i in range(params.y_n):
                msg += f', Train R2 {i}: {help_vars.train_r2s[i][it]:.4f}, Val R2 {i}: {help_vars.test_r2s[i][it]:.4f}'

        msg += f'\nDuration: {dt}, Best Val Epoch: {help_vars.best_test_epoch}'

        if accelerator.is_local_main_process:
            logger.debug(msg)

        # 重置记录变量
        help_vars.train_loss = []
        help_vars.test_loss = []
        if accelerator.is_local_main_process:
            logger.debug("重置训练记录")
        help_vars.train_correct = 0
        help_vars.train_all = 0
        if accelerator.is_local_main_process:
            logger.debug("重置验证记录")
        help_vars.all_targets = []
        help_vars.all_predictions = []
        help_vars.test_correct = 0
        help_vars.test_all = 0
        if not params.classify:
            # 重置
            for i in range(params.y_n):
                help_vars.train_r_squared[i].reset()
                help_vars.test_r_squared[i].reset()

        help_vars.step_in_epoch = 0

        # 更新最佳数据
        best_idx = help_vars.test_losses.tolist().index(min(help_vars.test_losses))
        result_dict['train_loss'] = help_vars.train_losses[best_idx]
        result_dict['val_loss'] = help_vars.test_losses[best_idx]

        if params.classify:
            # 分类模型 记录最佳模型的acc / f1 score
            result_dict['train_acc'] = help_vars.train_acc[best_idx]
            result_dict['val_acc'] = help_vars.test_acc[best_idx]

            for idx, i in enumerate(help_vars.f1_scores):
                result_dict[f'F1_{idx}'] = help_vars.f1_scores[i][best_idx]
        else:
            # 回归模型 记录最佳模型的 R方
            for i in range(params.y_n):
                result_dict['train_r2_{i}'] = help_vars.train_r2s[i][best_idx]
                result_dict['val_r2_{i}'] = help_vars.test_r2s[i][best_idx]

        # 缓存数据
        accelerator.save_state(os.path.join(params.root, 'checkpoints'))

        if accelerator.is_local_main_process:
            # 保存损失图
            cost_hour = (time.time() - help_vars.begin_time) / 3600
            plot_loss(help_vars, cost_hour, send_wx=False)

            # 打包文件
            pack_folder()

    if accelerator.is_local_main_process:
        cost_hour = (time.time() - help_vars.begin_time) / 3600
        plot_loss(help_vars, cost_hour)

    return cost_hour

def test_model(model, accelerator, result_dict, cnn, select='best'):
    """
    模型可选 最优/最终 best/final
    """
    model = None

    if 'best' == select:
        model = torch.load(os.path.join(params.root, f'best_val_model'))
    elif 'final' == select:
        model = torch.load(os.path.join(params.root, 'var', f'model.pkl'))
    else:
        return

    # 读取数据
    test_loader = read_data('test', need_id=True, cnn=cnn)

    # model = torch.load('best_val_model_pytorch')
    all_targets = []
    all_predictions = []
    r2score = [R2Score().to(params.device) for i in range(params.y_n)]

    total_times = 0
    total_counts = 0

    y_data_type = torch.int64 if params.classify else torch.float

    logger.debug(f'测试模型')
    with torch.no_grad():
        for inputs, targets in test_loader:
            # 如果是  torch.Size([512]) 则调整为 torch.Size([512, 1])
            if not params.classify and len(targets.shape) == 1:
                targets = targets.unsqueeze(1)

            # Move to GPU
            inputs, targets = inputs.to(params.device, dtype=torch.float), targets.to(
                params.device, dtype=torch.int64)

            t0 = time.time()
            # Forward pass
            outputs = model(inputs)

            # 记录耗时
            total_times += time.time() - t0
            total_counts += len(targets)

            if params.classify:
                # 分类模型
                # 转成概率
                p = torch.softmax(outputs, dim=1)

                # Get prediction
                # torch.max returns both max and argmax
                _, predictions = torch.max(p, 1)
                all_predictions.append(predictions.cpu().numpy())
            else:
                # 回归模型 统计 r方
                for i in range(params.y_n):
                    r2score[i].update(outputs[:, i], targets[:, i])
                all_predictions.append(outputs.cpu().numpy())

            all_targets.append(targets.cpu().numpy())
            
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    ids = test_loader.dataset.ids# code_timestamp: btcusdt_1710289478588

    del test_loader

    # 分类预测
    datas = {}
    for i in range(len(ids)):
        symbol, timestamp = ids[i].split('_')
        if symbol not in datas:
            datas[symbol] = []
        
        datas[symbol].append((timestamp, all_targets[i], all_predictions[i]))

    # 储存预测结果
    # symbol_begin_end.csv
    for symbol in datas:
        data_list = datas[symbol]
        begin = data_list[0][0]
        end = data_list[-1][0]
        with open(os.path.join(params.root, f'{symbol}_{begin}_{end}.csv'), 'w') as f:
            f.write('timestamp,target,predict\n')
            for timestamp, target, pre,  in data_list:
                f.write(f'{timestamp},{target},{pre}\n')

    if params.classify:
        # 分类模型
        logger.debug(
            f'accuracy_score: {accuracy_score(all_targets, all_predictions)}')
        report = classification_report(
            all_targets, all_predictions, digits=4, output_dict=True)
        # 将分类报告转换为DataFrame
        df = pd.DataFrame(report).transpose()
        logger.debug(f'测试结果:\n{df}')
        
        # 储存测试acc
        result_dict['test_acc'] = df.iloc[-3, -1]

        # 储存平均f1
        result_dict['wa_f1'] = df.iloc[-1, -2]

        _f1_scores_dict = df.iloc[:-3, 2].to_dict()
        # 存入 result_dict
        for idx, i in enumerate(_f1_scores_dict):
            result_dict[f'test_f1_{idx}'] = _f1_scores_dict[i]

        dfi.export(df, os.path.join(params.root, 'test_result.png'), table_conversion="matplotlib")
    else:
        # 回归模型 统计 R2, MSE, RMSE
        mse = mean_squared_error(all_targets, all_predictions, multioutput='raw_values')
        rmse = np.sqrt(mse)
        result_dict['test_mse'] = mse
        result_dict['test_rmse'] = rmse

        for i in range(params.y_n):
            r2 = r2score[i].compute()
            result_dict[f'test_r2_{i}'] = r2

    # 记录预测平均耗时 ms 
    result_dict['predict_ms'] = (total_times / total_counts) * 1000

class trainer:
    def __init__(self, idx, debug=False, cnn=True, workers=3, custom_param={}):
        self.idx = idx
        self.debug = debug
        self.cnn = cnn
        self.workers = workers
        self.custom_param = custom_param

        # # 开启CuDNN自动优化
        # torch.backends.cudnn.benchmark = True

        # 训练结果
        self.result_dict = {}

    def test_data(self):
        data_parm = {
            'predict_n': 5,
            'pass_n': 70,
            'y_n': 3,
            'begin_date': '2024-04-08',
            'data_rate': (1, 1, 1),
            'total_hours': int(3),
            'symbols': 1,
            'target': 'same paper',
            'std_mode': '4h'  # 4h/1d/5d
        }

        return data_parm

    def init_param(self):
        raise NotImplementedError("must override init_param")

    async def download_dataset_async(self, session):
        if self.debug:
            # 测试 使用最小数据
            params.data_set = f'{data_parm2str(self.test_data())}.7z'

        # params.data_set: pred_5_pass_40_y_1_bd_2024-04-08_dr_8@2@2_th_72_s_2_t_samepaper.7z
        # 替换 pass_40 -> pass_100
        # 都使用 pass_100/pass_155 的数据, 在Dataset中按需截取
        params_data_set = params.data_set.split('_y_')
        for i in ['100', '105','155']:
            real_data_set = '_'.join(params_data_set[0].split('_')[:-1]) + f"_{i}_y_" + params_data_set[1]

            # data 映射, 有些文件名太长，tg无法显示
            if real_data_set in DATA_MAP:
                real_data_set = DATA_MAP[real_data_set]

            ret = await download_dataset_async(session, real_data_set)
            if ret:
                return
                
        raise '下载数据集失败'

    def train(self, num_processes, mixed_precision='no', only_test=False, seed=42):
        """
        num_processes: int 8(tpu)/1(p100)/2(t4*2)/0(cpu)
        mixed_precision: 'fp16' or 'bf16' or 'no'        
        """
        accelerator = Accelerator()

        # Set the seed before splitting the data.
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        set_seed(42)

        if self.debug:
            # 使用最小数据进行测试
            params.data_parm['data_rate'] = (44,2,2)
            params.data_parm['total_hours'] = 48

            params.data_parm['data_rate'] = (2,1,1)
            params.data_parm['total_hours'] = 2*4

            params.data_set = f'{data_parm2str(params.data_parm)}.7z'
            params.epochs = 2
            params.workers = 0
            params.debug = True

        # 客制化参数
        if self.custom_param:
            for i in self.custom_param:
                # 检查是否存在
                if hasattr(params, i):
                    setattr(params, i, self.custom_param[i])

        # 混合精度
        if mixed_precision:
            params.amp = mixed_precision

        try:
            t0 = datetime.now()

            ## 训练模型
            if not only_test:
                if accelerator.is_local_main_process:
                    wx.send_message(f'[{params.train_title}] 开始训练')
                cost_hour = batch_gd(accelerator, result_dict=self.result_dict, cnn=self.cnn, seed=seed)

            ## 测试模型
            if accelerator.is_local_main_process:
                logger.debug(f'[{params.train_title}] 开始验证')
            test_model(accelerator, self.result_dict, self.cnn)

            ## 储存结果
            if accelerator.is_local_main_process:
                logger.debug('训练相关运行结束')
                report_memory_usage()

                ## 记录结果
                result_file = os.path.join(params.root, 'result.csv')

                # 数据参数
                data_dict =  data_str2parm(params.data_set)
                data_dict['y_n'] = params.y_n
                data_dict['classify'] = params.classify
                data_dict['regress_y_idx'] = params.regress_y_idx
                data_dict['classify_y_idx'] = params.classify_y_idx
                if not os.path.exists(result_file):
                    # 初始化列名
                    with open(result_file, 'w') as f:
                        # 训练参数
                        f.write('time,epochs,batch_size,learning_rate,warm_up_epochs,no_better_stop,random_mask,random_scale,random_mask_row,amp,label_smoothing,weight_decay,,')

                        # 数据参数
                        for i in data_dict:
                            f.write(f'{i},')

                        # 模型
                        f.write('model,describe,')

                        # 训练结果
                        for i in self.result_dict:
                            f.write(f'{i},')
                        f.write('cost,folder\n')
                
                # 写入结果
                with open(result_file, 'a') as f:
                    # 训练参数
                    f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},{params.epochs},{params.batch_size},{params.learning_rate},{params.warm_up_epochs},{params.no_better_stop},{params.random_mask},{params.random_scale},{params.random_mask_row},{params.amp},{params.label_smoothing},{params.weight_decay},,')

                    # 数据参数
                    for i in data_dict:
                        if isinstance(data_dict[i], list) or isinstance(data_dict[i], tuple):
                            f.write(f'{"@".join([str(i) for i in data_dict[i]])},')
                        else:
                            f.write(f'{data_dict[i]},')

                    # 模型
                    f.write(f'{params.model.model_name()},{params.describe},')

                    # 训练结果
                    for i in self.result_dict:
                        f.write(f'{self.result_dict[i]},')

                    # 文件夹 
                    f.write(f"{cost_hour:.2f}h,{params.root}\n")

                # 删除数据文件
                msg = f'[{params.train_title}] 训练完成, 耗时 {(datetime.now()-t0).seconds/3600:.2f} h'
                logger.debug(msg)
                wx.send_message(msg)

                # 打包文件
                pack_folder()

                # 发送到ex
                wx.send_file(params.root+".7z")

        except Exception as e:
            wx.send_message(f'[{params.train_title}] 训练失败 {e}')
            raise e
