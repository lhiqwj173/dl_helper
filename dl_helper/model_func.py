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

from py_ext.wechat import wx

from .tg import download_dataset_async
from .train_param import init_param, logger, params
from .data import read_data

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
    epochs, train_losses, test_losses, train_acc, test_acc, lrs, f1_scores = pickle.load(
        open(os.path.join(params.root, 'var', f'plot_datas.pkl'), 'rb'))
    
    plot_loss(epochs, train_losses, test_losses, train_acc, test_acc, lrs, f1_scores)

def plot_loss(epochs, train_losses, test_losses, train_acc, test_acc, lrs, f1_scores):

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

    plt.title(f'{params.train_title} {datetime.now().strftime("%Y%m%d_%H_%M_%S")}')
    plt.savefig(f"{params.train_title}.png")
    wx.send_file(f"{params.train_title}.png")
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
    if os.path.exists(os.path.join(params.root, 'var', f'datas.pkl')):
        logger.debug(f"使用缓存文件继续训练")
        train_losses, test_losses, train_acc, test_acc,lrs, f1_scores,all_targets, all_predictions, best_test_loss, best_test_epoch, begin, train_loss, test_loss, train_correct, test_correct, train_all, test_all, step_in_epoch, scaler = pickle.load(
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
                    train_correct += count_correct_predictions(
                        outputs, targets)
                    train_all += len(targets)

                # warnup
                if isinstance(scheduler, warm_up_ReduceLROnPlateau) or isinstance(scheduler, Increase_ReduceLROnPlateau):
                    scheduler.warn_up()

            step_in_epoch += 1

            # Get train loss and test loss
            train_loss = np.mean(train_loss)  # a little misleading

            # 缓存数据
            pickle.dump((train_losses, test_losses, train_acc, test_acc, lrs,f1_scores, all_targets, all_predictions, best_test_loss, best_test_epoch, it, train_loss, test_loss,
                        train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(params.root, 'var', f'datas.pkl'), 'wb'))
            torch.save(model, os.path.join(params.root, 'var', f'model.pkl'))
            pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
            ), test_loader.sampler.state_dict()), open(os.path.join(params.root, 'var', f'helper.pkl'), 'wb'))

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
                        train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(params.root, 'var', f'datas.pkl'), 'wb'))
            torch.save(model, os.path.join(params.root, 'var', f'model.pkl'))
            pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
            ), test_loader.sampler.state_dict()), open(os.path.join(params.root, 'var', f'helper.pkl'), 'wb'))

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
        step_in_epoch = 0
        pickle.dump((train_losses, test_losses, train_acc, test_acc, lrs,f1_scores, all_targets, all_predictions, best_test_loss, best_test_epoch, it+1, train_loss, test_loss,
                    train_correct, test_correct, train_all, test_all, step_in_epoch, scaler), open(os.path.join(params.root, 'var', f'datas.pkl'), 'wb'))
        torch.save(model, os.path.join(params.root, 'var', f'model.pkl'))
        pickle.dump((scheduler.state_dict(), optimizer.state_dict(), train_loader.sampler.state_dict(
        ), test_loader.sampler.state_dict()), open(os.path.join(params.root, 'var', f'helper.pkl'), 'wb'))

        # 更新最佳数据
        best_idx = test_losses.index(min(test_losses))
        result_dict['train_loss'] = train_losses[best_idx]
        result_dict['val_loss'] = test_losses[best_idx]
        result_dict['train_acc'] = train_acc[best_idx]
        result_dict['val_acc'] = test_acc[best_idx]
        for idx, i in emumerate(f1_scores):
            result_dict[f'F1_{idx}'] = f1_scores[i][best_idx]

    plot_loss(epochs, train_losses, test_losses, train_acc, test_acc, lrs, f1_scores)


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

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move to GPU
            inputs, targets = inputs.to(params.device, dtype=torch.float), targets.to(
                params.device, dtype=torch.int64)

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
    logger.debug(f'测试结果:\n{df}')
    
    _f1_scores_dict = df.iloc[:-3, 2].to_dict()

    # 存入 result_dict
    for idx, i in emumerate(_f1_scores_dict):
        result_dict[f'TEST_F1_{idx}'] = _f1_scores_dict[i]


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


def predict_test_y():
    # 载入模型
    model = torch.load(os.path.join(params.root, f'best_val_model'))

    # 读取测试数据
    mean_std, test_x, test_y, test_raw = pickle.load(
        open(os.path.join(data_path, f'test.pkl'), 'rb'))

    # 修正标签 -1,0,1 -> 0,1,2
    test_y = [i+1 for i in test_y]

    dataset_test = Dataset(test_raw, test_x, test_y, mean_std)
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=params.batch_size)

    results = {'id': [], 'predict': [], 'target': []}

    idx = 0
    for inputs, targets in test_loader:
        a = idx*params.batch_size
        idx += 1

        # Move to GPU
        inputs, targets = inputs.to(params.device, dtype=torch.float), targets.to(
            params.device, dtype=torch.int64)

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
    results.to_csv(os.path.join(params.root, 'predict.csv'), index=False)


class trainer:
    def __init__(self, idx):
        self.idx = idx

        # 开启CuDNN自动优化
        torch.backends.cudnn.benchmark = True

        # 训练结果
        self.result_dict = {}

    @classmethod
    def data_parm2str(cls, parm):
        return f"pred_{parm['predict_n']}_pass_{parm['pass_n']}_y_{parm['y_n']}_bd_{parm['begin_date'].replace('-', '_')}_dr_{'@'.join([str(i) for i in parm['data_rate']])}_th_{parm['total_hours']}_s_{parm['symbols']}_t_{parm['taget'].replace(' ', '')}"

    @classmethod
    def data_str2parm(cls, s):
        s = s.split('.')[0]
        p = s.split('_')
        return {
            'predict_n': int(p[1]),
            'pass_n': int(p[3]),
            'y_n': int(p[5]),
            'begin_date': p[7],
            'data_rate': tuple([int(i) for i in p[9].split('@')]),
            'total_hours': int(p[11]),
            'symbols': int(p[13]),
            'taget': p[15]
        }

    def init_param(self):
        raise NotImplementedError("must override init_param")

    async def download_dataset_async(self, session):
        await download_dataset_async(session, params.data_set)

    def train(self):
        t0 = datetime.now()
        wx.send_message(f'[{params.train_title}] 开始训练')

        ### 训练
        ## 获取数据
        train_loader = read_data(os.path.join(params.root, 'data'), 'train', shuffle=True)
        val_loader = read_data(os.path.join(params.root, 'data'), 'val')

        ## 模型
        _model = params.model.to(params.device)
        if torch.cuda.device_count() > 1:
            logger.debug("使用多gpu")
            _model = nn.DataParallel(_model)
        # 损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
        optimizer_class = torch.optim.AdamW
        # 训练
        batch_gd(_model, criterion, optimizer_class, None, train_loader, val_loader, epochs=params.epochs, result_dict=self.result_dict)

        ## 测试模型
        test_loader = read_data('test')
        test_model(test_loader, self.result_dict)

        ## 记录结果
        result_file = os.path.join(params.root, 'result.csv')
        if not os.path.exists(result_file):
            # 初始化列名
            with open(result_file, 'w') as f:
                # 训练参数
                f.write('time,epochs,batch_n,batch_size,learning_rate,warm_up_epochs,no_better_stop,random_mask,random_mask_row,amp,label_smoothing,weight_decay,,')

                # 数据参数
                data_dict =  self.data_str2parm(params.data_set)
                for i in data_dict:
                    f.write(f'{i},')

                # 模型
                f.write('model,')

                # 训练结果
                for i in self.result_dict:
                    f.write(f'{i},')
                f.write('folder\n')
        
        # 写入结果
        with open(result_file, 'w') as f:
            # 训练参数
            f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},{params.epochs},{params.batch_n},{params.batch_size},{params.learning_rate},{params.warm_up_epochs},{params.no_better_stop},{params.random_mask},{params.random_mask_row},{params.amp},{params.label_smoothing},{params.weight_decay},,')

            # 数据参数
            data_dict =  self.data_str2parm(params.data_set)
            for i in data_dict:
                f.write(f'{data_dict[i]},')

            # 模型
            f.write(f'{params.model.model_name()},')

            # 训练结果
            for i in self.result_dict:
                f.write(f'{self.result_dict[i]},')

            # 文件夹
            f.write(f"{root}\n")

        # 删除数据文件
        shutil.rmtree(os.path.join(params.root, 'data'))

        wx.send_message(f'[{params.train_title}] 训练完成, 耗时 {(datetime.now()-t0).seconds/3600:.2f} h')
