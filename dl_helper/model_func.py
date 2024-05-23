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

from py_ext.wechat import wx
from py_ext.lzma import compress_folder

from .tg import download_dataset_async
from .train_param import init_param, logger, params, data_parm2str, data_str2parm
from .data import read_data
from .data_map import DATA_MAP

# 设置启动方法为'spawn'
multiprocessing.set_start_method('spawn', force=True)

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

def debug_plot():
    epochs, train_losses, test_losses, train_r2s, test_r2s, train_acc, test_acc, lrs, f1_scores = pickle.load(
        open(os.path.join(params.root, 'var', f'plot_datas.pkl'), 'rb'))
    
    plot_loss(epochs, train_losses, test_losses, train_r2s, test_r2s, train_acc, test_acc, lrs, f1_scores, cost_hour)

def plot_loss(epochs, train_losses, test_losses, train_r2s, test_r2s, train_acc, test_acc, lrs, f1_scores, cost_hour):

    # 创建图形和坐标轴
    fig, axs = None, None
    if params.y_n != 1:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [7, 3]})
    else:
        fig, axs = plt.subplots(figsize=(15, 10))

    # 1 图左侧坐标轴
    ax1 = axs[0]

    # 用于添加图例
    ax1_handles = []

    # 计算误差最低点
    min_train_loss = min(train_losses)
    min_test_loss = min(test_losses)
    min_train_x = train_losses.tolist().index(min_train_loss)
    min_test_x = test_losses.tolist().index(min_test_loss)
    # 绘制loss曲线
    ax1_handles.append(ax1.plot(list(range(epochs)), train_losses, label=f'train loss {last_value(train_losses)}', c='b')[0])
    # line_train_loss, = ax1.plot(list(range(epochs)), train_losses, label=f'train loss {last_value(train_losses)}', c='b')
    ax1_handles.append(ax1.plot(list(range(epochs)), test_losses, label=f'validation loss {last_value(test_losses)}', c='#00BFFF')[0])
    # line_test_loss, = ax1.plot(list(range(epochs)), test_losses, label=f'validation loss {last_value(test_losses)}', c='#00BFFF')
    # 标记损失最低点
    ax1_handles.append(ax1.scatter(min_train_x, min_train_loss, c='b',label=f'train loss min: {min_train_loss:.4f}'))
    # train_loss_min = ax1.scatter(min_train_x, min_train_loss, c='b',
    #             label=f'train loss min: {min_train_loss:.4f}')
    ax1_handles.append(ax1.scatter(min_test_x, min_test_loss, c='#00BFFF',label=f'validation loss min: {min_test_loss:.4f}'))
    # test_loss_min = ax1.scatter(min_test_x, min_test_loss, c='#00BFFF',
    #             label=f'validation loss min: {min_test_loss:.4f}')

    if params.y_n != 1:
        # 分类模型
        # 计算acc最高点
        max_train_acc = max(train_acc)
        max_test_acc = max(test_acc)
        max_train_acc_x = train_acc.tolist().index(max_train_acc)
        max_test_acc_x = test_acc.tolist().index(max_test_acc)
        # 绘制acc曲线
        ax1_handles.append(ax1.plot(list(range(epochs)), train_acc, label=f'train acc {last_value(train_acc)}', c='r')[0])
        # line_train_acc, = ax1.plot(list(range(epochs)), train_acc, label=f'train acc {last_value(train_acc)}', c='r')
        ax1_handles.append(ax1.plot(list(range(epochs)), test_acc, label=f'validation acc {last_value(test_acc)}', c='#FFA07A')[0])
        # line_test_acc, = ax1.plot(list(range(epochs)), test_acc, label=f'validation acc {last_value(test_acc)}', c='#FFA07A')
        # 标记准确率最高点
        ax1_handles.append(ax1.scatter(max_train_acc_x, max_train_acc, c='r',label=f'train acc max: {max_train_acc:.4f}'))
        # train_acc_max = ax1.scatter(max_train_acc_x, max_train_acc, c='r',label=f'train acc max: {max_train_acc:.4f}')
        ax1_handles.append(ax1.scatter(max_test_acc_x, max_test_acc, c='#FFA07A',label=f'validation acc max: {max_test_acc:.4f}'))
        # train_val_max = ax1.scatter(max_test_acc_x, max_test_acc, c='#FFA07A',label=f'validation acc max: {max_test_acc:.4f}')
    else:
        # 回归模型
        # 计算r2最高点
        max_train_r2 = max(train_r2s)
        max_test_r2 = max(test_r2s)
        max_train_r2_x = train_r2s.tolist().index(max_train_r2)
        max_test_r2_x = test_r2s.tolist().index(max_test_r2)
        # 绘制r2曲线
        ax1_handles.append(ax1.plot(list(range(epochs)), train_r2s, label=f'train r2 {last_value(train_r2s)}', c='r')[0])
        # line_train_r2, = ax1.plot(list(range(epochs)), train_r2s, label=f'train r2 {last_value(train_r2s)}', c='r')
        ax1_handles.append(ax1.plot(list(range(epochs)), test_r2s, label=f'validation r2 {last_value(test_r2s)}', c='#FFA07A')[0])
        # line_test_r2, = ax1.plot(list(range(epochs)), test_r2s, label=f'validation r2 {last_value(test_r2s)}', c='#FFA07A')
        # 标记r2最高点
        ax1_handles.append(ax1.scatter(max_train_r2_x, max_train_r2, c='r',label=f'train r2 max: {max_train_r2:.4f}'))
        # train_r2_max = ax1.scatter(max_train_r2_x, max_train_r2, c='r',label=f'train r2 max: {max_train_r2:.4f}')
        ax1_handles.append(ax1.scatter(max_test_r2_x, max_test_r2, c='#FFA07A',label=f'validation r2 max: {max_test_r2:.4f}'))
        # test_r2_max = ax1.scatter(max_test_r2_x, max_test_r2, c='#FFA07A',label=f'validation r2 max: {max_test_r2:.4f}')

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
    if params.y_n != 1:
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
    plt.savefig(os.path.join(params.root, f"{params.train_title}.png"))
    wx.send_file(os.path.join(params.root,f"{params.train_title}.png"))
    # display.clear_output(wait=True)
    # plt.pause(0.00000001)

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

# A function to encapsulate the training loop


def batch_gd(model, criterion, optimizer_class, lr_lambda, train_loader, test_loader, epochs, result_dict):

    train_losses = np.full(epochs, np.nan)
    test_losses = np.full(epochs, np.nan)
    train_r2s = np.full(epochs, np.nan)
    test_r2s = np.full(epochs, np.nan)
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
    train_r_squared = R2Score()
    test_r_squared = R2Score()
    train_correct = 0
    train_all = 0
    test_correct = 0
    test_all = 0
    step_in_epoch = 0

    # optimizer
    optimizer = None
    scheduler = None

    # Automatic Mixed Precision
    scaler = None if not params.amp else GradScaler()

    # 检查是否有缓存文件
    if os.path.exists(os.path.join(params.root, 'var', f'datas.pkl')):
        wx.send_message(f'[{params.train_title}] 使用缓存文件继续训练')
        logger.debug(f"使用缓存文件继续训练")
        train_losses, test_losses, train_r2s, test_r2s, train_r_squared, test_r_squared, train_acc, test_acc,lrs, f1_scores,all_targets, all_predictions, best_test_loss, best_test_epoch, begin, train_loss, test_loss, train_correct, test_correct, train_all, test_all, step_in_epoch, scaler = pickle.load(
            open(os.path.join(params.root, 'var', f'datas.pkl'), 'rb'))

        # logger.debug(f'train_losses: \n{train_losses}')
        # logger.debug(f'test_losses: \n{test_losses}')

        model = torch.load(os.path.join(params.root, 'var', f'model.pkl'))
    else:
        logger.debug(f"新的训练")

    # 构造优化器
    optimizer = optimizer_class(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = None
    if None is lr_lambda:
        if params.init_learning_ratio > 0:
            scheduler = Increase_ReduceLROnPlateau(optimizer)
        else:
            scheduler = warm_up_ReduceLROnPlateau(optimizer, warm_up_epoch=params.warm_up_epochs, iter_num_each_epoch=len(train_loader))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

    # 恢复 scheduler/optmizer
    if os.path.exists(os.path.join(params.root, 'var', f'helper.pkl')):
        sd_scheduler, sd_optimizer, sd_train_loader, sd_test_loader = pickle.load(
            open(os.path.join(params.root, 'var', f'helper.pkl'), 'rb'))
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
        if best_test_epoch > 0 and best_test_epoch + params.no_better_stop < it:
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

            idx = 0
            t0 = time.time()
            for inputs, targets in tqdm(train_loader, initial=int(train_loader.sampler.idx / params.batch_size), total=len(train_loader)):
                # move data to GPU
                inputs, targets = inputs.to(params.device, dtype=torch.float), targets.to(
                    params.device, dtype=torch.int64)
                optimizer.zero_grad()

                outputs = None
                loss = None
                if params.amp:
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
                    if params.y_n != 1:
                        # 分类模型 统计acc
                        train_correct += count_correct_predictions(
                            outputs, targets)
                        train_all += len(targets)
                    else:
                        # 回归模型 统计 r方
                        train_r_squared.update(outputs, targets)

                # warnup
                if isinstance(scheduler, warm_up_ReduceLROnPlateau) or isinstance(scheduler, Increase_ReduceLROnPlateau):
                    scheduler.warn_up()

                if idx%100 == 0:
                    t1 = time.time()
                    if t1 - t0 >= 15*60:
                        t0 = t1
                        
                        # 15min，缓存数据
                        pickle.dump((train_losses, test_losses, train_r2s, test_r2s, train_r_squared, test_r_squared, train_acc, test_acc, lrs,f1_scores, all_targets, all_predictions, best_test_loss, best_test_epoch, it, train_loss, test_loss,
                                    train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(params.root, 'var', f'datas.pkl'), 'wb'))
                        torch.save(model, os.path.join(params.root, 'var', f'model.pkl'))
                        pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
                        ), test_loader.sampler.state_dict()), open(os.path.join(params.root, 'var', f'helper.pkl'), 'wb'))

                        # 打包文件
                        pack_folder()

                idx += 1

            step_in_epoch += 1

            # Get train loss and test loss
            train_loss = np.mean(train_loss)  # a little misleading

            # 缓存数据
            pickle.dump((train_losses, test_losses, train_r2s, test_r2s, train_r_squared, test_r_squared, train_acc, test_acc, lrs,f1_scores, all_targets, all_predictions, best_test_loss, best_test_epoch, it, train_loss, test_loss,
                        train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(params.root, 'var', f'datas.pkl'), 'wb'))
            torch.save(model, os.path.join(params.root, 'var', f'model.pkl'))
            pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
            ), test_loader.sampler.state_dict()), open(os.path.join(params.root, 'var', f'helper.pkl'), 'wb'))

            # 打包文件
            pack_folder()

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
                for inputs, targets in tqdm(test_loader, initial=int(test_loader.sampler.idx / params.batch_size), total=len(test_loader)):
                    inputs, targets = inputs.to(params.device, dtype=torch.float), targets.to(
                        params.device, dtype=torch.int64)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss.append(loss.item())
                    # logger.debug(f'test_loss: {loss.item()}')
                    
                    if params.y_n != 1:
                        # 分类模型 统计acc / f1 score
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
                    else:
                        # 回归模型 统计 r方
                        test_r_squared.update(outputs, targets)

            if params.y_n != 1:
                # 分类模型 统计 f1 score
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

            pickle.dump((train_losses, test_losses, train_r2s, test_r2s, train_r_squared, test_r_squared, train_acc, test_acc, lrs,f1_scores,all_targets, all_predictions,  best_test_loss, best_test_epoch, it, train_loss, test_loss,
                        train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(params.root, 'var', f'datas.pkl'), 'wb'))
            torch.save(model, os.path.join(params.root, 'var', f'model.pkl'))
            pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
            ), test_loader.sampler.state_dict()), open(os.path.join(params.root, 'var', f'helper.pkl'), 'wb'))
            
            # 打包文件
            pack_folder()

        torch.cuda.empty_cache()
        logger.debug(f'{msg}验证完成')

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if params.y_n != 1:
            train_acc[it] = train_correct / train_all
            test_acc[it] = test_correct / test_all
        else:
            train_r2s[it] = train_r_squared.compute()
            test_r2s[it] = test_r_squared.compute()

        if test_loss < best_test_loss:
            torch.save(model, os.path.join(params.root, f'best_val_model'))
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

        # 缓存数据
        train_loss = []
        test_loss = []
        train_r_squared.reset()
        test_r_squared.reset()
        step_in_epoch = 0
        pickle.dump((train_losses, test_losses, train_r2s, test_r2s, train_r_squared, test_r_squared, train_acc, test_acc, lrs,f1_scores, all_targets, all_predictions, best_test_loss, best_test_epoch, it+1, train_loss, test_loss,
                    train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(params.root, 'var', f'datas.pkl'), 'wb'))
        torch.save(model, os.path.join(params.root, 'var', f'model.pkl'))
        pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
        ), test_loader.sampler.state_dict()), open(os.path.join(params.root, 'var', f'helper.pkl'), 'wb'))
            
        # 打包文件
        pack_folder()

        # 更新最佳数据
        best_idx = test_losses.tolist().index(min(test_losses))
        result_dict['train_loss'] = train_losses[best_idx]
        result_dict['val_loss'] = test_losses[best_idx]

        if params.y_n != 1:
            # 分类模型 记录最佳模型的acc / f1 score
            result_dict['train_acc'] = train_acc[best_idx]
            result_dict['val_acc'] = test_acc[best_idx]

            for idx, i in enumerate(f1_scores):
                result_dict[f'F1_{idx}'] = f1_scores[i][best_idx]
        else:
            # 回归模型 记录最佳模型的 R方
            result_dict['train_r2'] = train_r2s[best_idx]
            result_dict['val_r2'] = test_r2s[best_idx]

    cost_hour = (time.time() - t) / 3600
    plot_loss(epochs, train_losses, test_losses, train_r2s, test_r2s, train_acc, test_acc, lrs, f1_scores, cost_hour)

    return cost_hour


def test_model(test_loader, result_dict, select='best'):
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

    # model = torch.load('best_val_model_pytorch')
    all_targets = []
    all_predictions = []
    r2score = R2Score()

    total_times = 0
    total_counts = 0

    logger.debug(f'测试模型')
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move to GPU
            inputs, targets = inputs.to(params.device, dtype=torch.float), targets.to(
                params.device, dtype=torch.int64)

            t0 = time.time()
            # Forward pass
            outputs = model(inputs)

            # 记录耗时
            total_times += time.time() - t0
            total_counts += len(targets)

            if params.y_n != 1:
                # 分类模型
                # 转成概率
                p = torch.softmax(outputs, dim=1)

                # Get prediction
                # torch.max returns both max and argmax
                _, predictions = torch.max(p, 1)
                all_predictions.append(predictions.cpu().numpy())
            else:
                # 回归模型 统计 r方
                r2score.update(outputs, targets)
                all_predictions.append(outputs.cpu().numpy())

            all_targets.append(targets.cpu().numpy())
            
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    ids = test_loader.dataset.ids# code_timestamp: btcusdt_1710289478588

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

    if params.y_n != 1:
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
        r2 = r2score.compute()
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = torch.sqrt(mse)

        result_dict['test_r2'] = r2
        result_dict['test_mse'] = mse
        result_dict['test_rmse'] = rmse

    # 记录预测平均耗时 ms 
    result_dict['predict_ms'] = (total_times / total_counts) * 1000

def test_ps(data_loader, ps):
    """
    data_loader 数据
    ps     概率阈值
    """
    # 载入模型
    model = torch.load(os.path.join(params.root, f'best_val_model'))

    # 真实标签 预测概率
    true_label = []
    p = []

    for inputs, targets in data_loader:
        # Move to GPU
        inputs = inputs.to(params.device, dtype=torch.float)

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
    # dfi.export(df, 'pic.png', table_conversion="matplotlib")
    # wx.send_file("pic.png")

def plot_roc(data_loader):
    """
    data_loader 数据
    """
    # 载入模型
    model = torch.load(os.path.join(params.root, f'best_val_model'))

    # 针对 0 标签（下跌）
    # label = 0

    # 真实标签 预测概率
    true_label = []
    p = []

    for inputs, targets in data_loader:
        # Move to GPU
        inputs = inputs.to(params.device, dtype=torch.float)

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

def pack_folder():
    # 打包训练文件夹 zip 
    file = params.root+".7z"

    if os.path.exists(file):
        # 删除
        os.remove(file)

    compress_folder(params.root, file, 9, inplace=False)

def report_memory_usage():
    # 获取设备的总内存（以GB为单位）
    total_memory = psutil.virtual_memory().total / (1024 ** 3)

    pct = psutil.virtual_memory().percent
    logger.debug(f'memory usage: {pct}% of {total_memory:.2f}GB')

class trainer:
    def __init__(self, idx, debug=False):
        self.idx = idx
        self.debug = debug

        # 开启CuDNN自动优化
        torch.backends.cudnn.benchmark = True

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

    def train(self, only_test=False):
        if self.debug:
            # 使用最小数据进行测试
            params.data_parm['data_rate'] = (1,1,1)
            params.data_parm['total_hours'] = 6
            params.data_set = f'{data_parm2str(params.data_parm)}.7z'
            params.epochs = 5

        try:
            t0 = datetime.now()
            if not only_test:
                wx.send_message(f'[{params.train_title}] 开始训练')

                ### 训练
                ## 获取数据
                train_loader = read_data('train', max_num=1 if self.debug else 10000)
                val_loader = read_data('val', max_num=1 if self.debug else 10000)
                assert len(train_loader) > 0, "没有训练数据"
                assert len(val_loader) > 0, "没有验证数据"

                report_memory_usage()

                ## 模型
                _model = params.model.to(params.device)
                if torch.cuda.device_count() > 1:
                    logger.debug("使用多gpu")
                    _model = nn.DataParallel(_model)

                # 损失函数
                criterion = None
                if params.y_n != 1:
                    # 分类模型
                    criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
                else:
                    # 回归模型
                    criterion = nn.MSELoss()

                # 优化器
                optimizer_class = torch.optim.AdamW

                # 训练
                cost_hour = batch_gd(_model, criterion, optimizer_class, None, train_loader, val_loader, epochs=params.epochs, result_dict=self.result_dict)

            ## 测试模型
            test_loader = read_data('test', need_id=True)
            test_model(test_loader, self.result_dict)

            ## 记录结果
            result_file = os.path.join(params.root, 'result.csv')

            # 数据参数
            data_dict =  data_str2parm(params.data_set)
            data_dict['y_n'] = params.y_n
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
            shutil.rmtree(os.path.join(params.root, 'data'))
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