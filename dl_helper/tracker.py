"""
用于追踪训练过程评价指标
"""
import time, math, os, copy
from datetime import datetime
import torch
import torch.nn.functional as F
from torchmetrics import F1Score, R2Score

import numpy as np
import matplotlib.pyplot as plt

from accelerate.utils import broadcast, gather_object

from dl_helper.scheduler import ReduceLR_slow_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dl_helper.train_param import tpu_available, data_str2parm
if tpu_available():
    import torch_xla.core.xla_model as xm

def last_value(data):
    """返回最后一个非nan值"""
    for i in range(len(data)-1, -1, -1):
        if not math.isnan(data[i]):
            return data[i]
    raise ValueError("没有找到非nan值")

def cal_balance_acc(y_pred, y_true, y_n):
    unique_labels = [torch.tensor(i, device=y_pred.device) for i in range(y_n)]
    recall_values = []

    for label in unique_labels:
        true_positives = torch.sum((y_true == label) & (y_pred == label))
        false_negatives = torch.sum((y_true == label) & (y_pred != label))
        recall = true_positives / (true_positives + false_negatives)
        recall_values.append(recall)

    # 计算均衡 ACC
    balanced_acc = torch.mean(torch.stack(recall_values))
    return balanced_acc
    
class Tracker_None():
    def __init__(self, *args, **kwargs):
        self.epoch_count = 0
        self.step_in_epoch = 0
        self.step_count = 0
        self.need_test = False

    def plot(self):
        pass

    def update(self):
        pass

    def track(self, *args, **kwargs):
        pass

class Tracker():
    def __init__(self, model_name, params, accelerator, scheduler, num_processes, printer):
        self.model_name = model_name
        # 时间统计
        self.begin_time = time.time()
        self.epoch_count = 0
        self.mini_epoch_count = 0
        self.step_count = 0
        # 每个epoch训练中的阶段
        # 0: 训练 1: 验证
        self.step_in_epoch = 0
        self.run_limit_hour = 12 if  num_processes != 8 else 9
        self.need_test = False
        # self.need_test = True

        self.params = params
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.printer = printer

        # 最终数据
        self.data = {}
        for i in ['train', 'val', 'test']:
            # 训练损失
            self.data[f'{i}_loss'] = None

            if params.classify: 
                # 分类模型 acc
                self.data[f'{i}_acc'] = None

                # 暂时只使用加权f1
                self.data[f'{i}_f1'] = None
            else:
                # 回归模型 r2
                # 暂时只使用加权r2
                self.data[f'{i}_r2'] = None

        # 训练学习率
        self.data['lr'] = []

        # 跟新类别
        self.track_update = ''

        # 计算变量
        self.temp = {}

        self.reset_temp()

    def update_mini_batch(self):
        self.mini_epoch_count += 1

    def update(self, test_dataloader=None):
        # 计算变量 -> data
        # 主进程计算data
        if self.accelerator.is_main_process:
            # 计算数据
            _loss = torch.mean(self.temp['_loss']).unsqueeze(0)
            if self.params.classify:
                # 改用 Balanced Accuracy
                balance_acc = cal_balance_acc(
                    self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n
                ).unsqueeze(0)
                # self.printer.print('balance_acc')
                
                # 计算加权 F1 分数
                f1_score = F1Score(num_classes=self.params.y_n, average='weighted', task='multiclass').to(self.temp['_y_pred'].device)
                weighted_f1 = f1_score(self.temp['_y_pred'], self.temp['_y_true']).unsqueeze(0)
                # self.printer.print('weighted_f1')
            else:
                # 计算方差加权 R2
                r2_score = R2Score(multioutput='variance_weighted').to(self.temp['_y_pred'].device)
                variance_weighted_r2 = r2_score(self.temp['_y_pred'], self.temp['_y_true'])
                # self.printer.print('variance_weighted_r2')
        
            # self.printer.print(f'_loss: {_loss.shape}')
            # self.printer.print(f'balance_acc: {balance_acc.shape}')
            # self.printer.print(f'weighted_f1: {weighted_f1.shape}')

            # 记录数据
            if self.data[f'{self.track_update}_loss'] is None:
                self.data[f'{self.track_update}_loss'] = _loss
                if self.params.classify:
                    self.data[f'{self.track_update}_acc'] = balance_acc
                    self.data[f'{self.track_update}_f1'] = weighted_f1
                else:
                    self.data[f'{self.track_update}_r2'] = variance_weighted_r2
            else:
                self.data[f'{self.track_update}_loss'] = torch.cat([self.data[f'{self.track_update}_loss'], _loss])
                if self.params.classify:
                    self.data[f'{self.track_update}_acc'] = torch.cat([self.data[f'{self.track_update}_acc'], balance_acc])
                    self.data[f'{self.track_update}_f1'] = torch.cat([self.data[f'{self.track_update}_f1'], weighted_f1])
                else:
                    self.data[f'{self.track_update}_r2'] = torch.cat([self.data[f'{self.track_update}_r2'], variance_weighted_r2])
            # self.printer.print('record data done')

        # self.printer.print('update tracker...')
        if 'train' == self.track_update:
            # self.printer.print('update train round')
            # train 结束，指向验证阶段
            self.step_in_epoch = 1

            lr_change = torch.tensor(0, device=self.accelerator.device)
            if self.accelerator.is_main_process:
                # self.printer.print('scheduler.step')

                # 记录学习率
                self.data['lr'].append(self.scheduler.optimizer.param_groups[0]["lr"])
                # self.printer.print('append lr')

                # 更新 学习率
                self.scheduler.step(self.data['train_loss'])

                # self.printer.print('step done')
                if self.data['lr'][-1] != self.scheduler.optimizer.param_groups[0]["lr"]:
                    lr_change += 1
            # self.printer.print('step done')

            # 同步学习率
            self.accelerator.wait_for_everyone()
            lr_change = broadcast(lr_change)
            # self.printer.print('lr_change')

            if tpu_available():
                xm.mark_step()

            if lr_change.item() == 1:
                # self.printer.print('broadcast lr')
                cur_lr = torch.tensor(self.scheduler.optimizer.param_groups[0]["lr"], device=self.accelerator.device)

                self.accelerator.wait_for_everyone()
                cur_lr = broadcast(cur_lr)

                # 在其他设备上应用学习率
                self.printer.print(f'apply not main lr -> {cur_lr}')
                if not self.accelerator.is_main_process:
                    self.scheduler.use_lr(cur_lr)

        if 'val' == self.track_update:
            # val 结束，重置为训练阶段
            # self.printer.print('update val round')
            self.step_in_epoch = 0
            self.epoch_count += 1

        if 'test' == self.track_update:
            if self.accelerator.is_main_process:
                # self.printer.print('update test round')
                # 保存测试数据预测结果
                all_ids = self.temp['_ids']# code_timestamp: btcusdt_1710289478588
                all_predictions = self.temp['_y_pred'].to('cpu')
                all_targets = self.temp['_y_true'].to('cpu')

                # 按标的分类预测
                # self.printer.print('sort prediction')
                # self.printer.print(all_predictions.shape)
                # self.printer.print(all_targets.shape)
                # self.printer.print(len(all_ids))
                datas = {}
                for i in range(all_predictions.shape[0]):
                    symbol, timestamp = all_ids[i].split('_')
                    if symbol not in datas:
                        datas[symbol] = []
                    datas[symbol].append((timestamp, all_targets[i], all_predictions[i]))

                # 储存预测结果
                # symbol_begin_end.csv
                self.printer.print('save prediction')
                for symbol in datas:
                    data_list = datas[symbol]
                    begin = data_list[0][0]
                    end = data_list[-1][0]
                    with open(os.path.join(self.params.root, f'{symbol}_{begin}_{end}.csv'), 'w') as f:
                        f.write('timestamp,target,predict\n')
                        for timestamp, target, pre,  in data_list:
                            f.write(f'{timestamp},{target},{pre}\n')
                # self.printer.print('update test round done')

        if 'test' != self.track_update and self.accelerator.is_main_process:
            # 判断是否需要储存 训练数据
            # self.printer.print('check if need save')
            last_time_hour = ((time.time() - self.begin_time) / 3600)
            each_epoch_time_cost = last_time_hour / (self.epoch_count if self.epoch_count > 0 else 1)
            free_time = self.run_limit_hour - last_time_hour
            if free_time < each_epoch_time_cost * 1.2:
                self.printer.print('run time out, need test/predict')
                self.need_test = True

        self.reset_temp()
        # self.print_state()
        self.step_count = 0
        # self.printer.print('update done')

    def reset_temp(self):
        # 重置计算变量
        self.track_update = ''
        self.temp = {}

        self.temp['_ids'] = []

        self.temp['_loss'] = None
        self.temp['_num'] = 0
        self.temp['_y_true'] = None
        self.temp['_y_pred'] = None

    def print_state(self):
        self.printer.print(f"------------tracker data------------")
        """
        # 时间统计
        self.begin_time = time.time()
        self.epoch_count = 0
        self.step_count = 0
        # 每个epoch训练中的阶段
        # 0: 训练 1: 验证
        self.step_in_epoch = 0
        self.run_limit_hour = 12 if  num_processes != 8 else 9
        self.need_test = False
        """
        self.printer.print(f'[train state]')
        self.printer.print(f'begin time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.begin_time))}')
        self.printer.print(f'epoch done: {self.epoch_count}')
        self.printer.print(f'step in epoch: {"train" if self.step_in_epoch == 0 else "val"}')
        self.printer.print(f'step done: {self.step_count}')
        self.printer.print(f'run limit hour: {self.run_limit_hour}')
        self.printer.print(f'need save: {self.need_test}')

        self.printer.print(f'')
        self.printer.print(f'[train temp]')
        for i in self.temp:
            if 'y' in i:
                self.printer.print(f"{i}: {len(self.temp[i]) if not None is self.temp[i] else None}")
            else:
                self.printer.print(f"{i}: {self.temp[i]}")

        self.printer.print(f'')
        self.printer.print(f'[train data]')
        for i in self.data:
            self.printer.print(f"{i}: {self.data[i]}")

        self.printer.print(f"------------tracker data------------")

    def track(self, output, target, loss, _type, test_dataloader=None):
        # assert _type in ['train', 'val', 'test'], f'error: _type({_type}) should in [train, val, test]'
        # self.printer.print(self.temp[f'{_type}_y_true'], main=False)
        # self.printer.print(self.temp[f'{_type}_y_pred'], main=False)

        self.track_update = _type

        # epoch内的迭代次数
        self.step_count += 1

        # 统计损失 tensor
        predict = output
        if self.params.classify:
            softmax_predictions = F.softmax(output, dim=1)
            predict = torch.argmax(softmax_predictions, dim=1)

        # 汇总所有设备上的数据
        # self.printer.print('sync track...')
        self.accelerator.wait_for_everyone()
        
        # self.printer.print(f"{loss}, {type(loss)}, {loss.device}")
        # self.printer.print(f"{target}")
        # self.printer.print(f"{predict}")
        # self.printer.print(f"{correct_count}")
        _loss, _y_true, _y_pred = self.accelerator.gather_for_metrics((loss, target, predict))
        if _type == 'test':
            _ids = gather_object(test_dataloader.dataset.use_data_id)
            test_dataloader.dataset.use_data_id = []
        else:
            _ids = []

        if len(_loss.shape) == 0:
            _loss = _loss.unsqueeze(0)

        # self.printer.print('main cal track...')
        if self.accelerator.is_main_process:
            if None is self.temp['_y_true']:
                self.temp['_y_true'] = _y_true
                self.temp['_y_pred'] = _y_pred
                self.temp['_loss'] = _loss
            else:
                self.temp['_y_true'] = torch.cat([self.temp['_y_true'], _y_true])
                self.temp['_y_pred'] = torch.cat([self.temp['_y_pred'], _y_pred])
                self.temp['_loss'] = torch.cat([self.temp['_loss'], _loss])

            if _type == 'test':
                # self.printer.print(f"更新self.temp['_ids']: {len(self.temp['_ids'])} type: {type(self.temp['_ids'])}")
                self.temp['_ids'] += _ids
                # self.printer.print(f"更新self.temp['_ids']: {len(self.temp['_ids'])} type: {type(self.temp['_ids'])}")

            self.temp['_num'] += _y_true.shape[0]


    def save_result(self):
        self._plot()
        self._save_result()

    def _plot(self):
        if self.accelerator.is_main_process:
            params = self.params
            self.cost_hour = (time.time() - self.begin_time) / 3600

            # x 数量
            epochs = self.params.epochs

            # 标准化数据，nan补气数据
            data = {}
            for i in self.data:
                data[i] = [] if None is self.data[i] else copy.deepcopy(self.data[i]) if isinstance(self.data[i], list) else self.data[i].cpu().tolist()

                if 'test' in i:
                    data[i] = [data[i][-1]] * epochs if len(data[i]) else []
                else:
                    data[i] = data[i] + (epochs - len(data[i])) * [np.nan]

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
            min_train_loss = min(data['train_loss'])
            min_test_loss = min(data['val_loss'])
            min_train_x = data['train_loss'].index(min_train_loss)
            min_test_x = data['val_loss'].index(min_test_loss)
            # 测试集损失
            if data['test_loss']:
                ax1_handles.append(ax1.plot(list(range(epochs)), data['test_loss'], label=f"test loss {last_value(data['test_loss']):.4f}", c='b', linestyle='--')[0])
            # 绘制loss曲线
            ax1_handles.append(ax1.plot(list(range(epochs)), data['train_loss'], label=f"train loss {last_value(data['train_loss']):.4f}", c='#7070FF')[0])
            ax1_handles.append(ax1.plot(list(range(epochs)), data['val_loss'], label=f"validation loss {last_value(data['val_loss']):.4f}", c='b')[0])
            # 标记损失最低点
            ax1_handles.append(ax1.scatter(min_train_x, min_train_loss, c='#7070FF',label=f'train loss min: {min_train_loss:.4f}'))
            ax1_handles.append(ax1.scatter(min_test_x, min_test_loss, c='b',label=f'validation loss min: {min_test_loss:.4f}'))
            # self.printer.print(f'plot loss')

            if params.classify:
                # 分类模型
                # 计算acc最高点
                max_train_acc = max(data['train_acc'])
                max_test_acc = max(data['val_acc'])
                max_train_acc_x = data['train_acc'].index(max_train_acc)
                max_test_acc_x = data['val_acc'].index(max_test_acc)
                # 测试集准确率
                if data['test_acc']:
                    ax1_handles.append(ax1.plot(list(range(epochs)), data['test_acc'], label=f"test acc {last_value(data['test_acc']):.4f}", c='r', linestyle='--')[0]) 
                # 绘制acc曲线
                ax1_handles.append(ax1.plot(list(range(epochs)), data['train_acc'], label=f"train acc {last_value(data['train_acc']):.4f}", c='#FF7E7E')[0])
                ax1_handles.append(ax1.plot(list(range(epochs)), data['val_acc'], label=f"validation acc {last_value(data['val_acc']):.4f}", c='r')[0])
                # 标记准确率最高点
                ax1_handles.append(ax1.scatter(max_train_acc_x, max_train_acc, c='#FF7E7E',label=f'train acc max: {max_train_acc:.4f}'))
                ax1_handles.append(ax1.scatter(max_test_acc_x, max_test_acc, c='r',label=f'validation acc max: {max_test_acc:.4f}'))
                # self.printer.print(f'plot acc')

            else:
                # 回归模型
                # 计算r2最高点
                max_train_r2 = max(data['train_r2'])
                max_test_r2 = max(data['val_r2'])
                max_train_r2_x = data['train_r2'].index(max_train_r2)
                max_test_r2_x = data['val_r2'].index(max_test_r2)
                # 测试集r2
                if data['test_r2']:
                    ax1_handles.append(ax1.plot(list(range(epochs)), data['test_r2'], label=f"test r2 {last_value(data['test_r2']):.4f}", c='r', linestyle='--')[0])
                # 绘制r2曲线
                ax1_handles.append(ax1.plot(list(range(epochs)), data['train_r2'], label=f"train r2 {last_value(data['train_r2']):.4f}", c='#FF7E7E')[0])
                ax1_handles.append(ax1.plot(list(range(epochs)), data['val_r2'], label=f"validation r2 {last_value(data['val_r2']):.4f}", c='r')[0])
                # 标记r2最高点
                ax1_handles.append(ax1.scatter(max_train_r2_x, max_train_r2, c='#FF7E7E',label=f'train r2 max: {max_train_r2:.4f}'))
                ax1_handles.append(ax1.scatter(max_test_r2_x, max_test_r2, c='r',label=f'validation r2 max: {max_test_r2:.4f}'))
                # self.printer.print(f'plot r2')

            # 创建右侧坐标轴
            ax2 = ax1.twinx()

            # 绘制学习率
            line_lr, = ax2.plot(list(range(epochs)), data['lr'], label='lr', c='#87CEFF',linewidth=2,alpha =0.5)
            # self.printer.print(f'plot lr')

            # 添加图例
            ax1.legend(handles=ax1_handles)

            # 启用次刻度
            ax1.minorticks_on()

            # 启用网格线，并设置淡显的风格
            ax1.grid(True, which='both', alpha=0.3)
            ax1.set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值

            # 图2
            if params.classify:
                # 分类模型
                t2_handles = []

                # 计算f1最高点
                max_train_f1 = max(data["train_f1"])
                max_test_f1 = max(data["val_f1"])
                max_train_f1_x = data["train_f1"].index(max_train_f1)
                max_test_f1_x = data["val_f1"].index(max_test_f1)
                # 测试集f1
                if data["test_f1"]:
                    t2_handles.append(axs[1].plot(list(range(epochs)), data["test_f1"], label=f'test f1 {last_value(data["test_f1"]):.4f}', c='#57C838', linestyle='--')[0])
                # 绘制f1曲线
                t2_handles.append(axs[1].plot(list(range(epochs)), data["train_f1"], label=f'train f1 {last_value(data["train_f1"]):.4f}', c='#8DE874')[0])
                t2_handles.append(axs[1].plot(list(range(epochs)), data["val_f1"], label=f'val f1 {last_value(data["val_f1"]):.4f}', c='#57C838')[0])
                # 标记f1最高点
                t2_handles.append(axs[1].scatter(max_train_f1_x, max_train_f1, c='#8DE874',label=f'train f1 max: {max_train_f1:.4f}'))
                t2_handles.append(axs[1].scatter(max_test_f1_x, max_test_f1, c='#57C838',label=f'val f1 max: {max_test_f1:.4f}'))

                # 启用次刻度
                axs[1].minorticks_on()

                # 启用网格线，并设置淡显的风格
                axs[1].grid(True, which='both', alpha=0.3)
                # 取消y轴的次刻度
                axs[1].yaxis.set_minor_locator(plt.NullLocator())
                axs[1].set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值
                axs[1].legend(handles=t2_handles)
                # self.printer.print(f'plot f1 score')

            title = f'{params.train_title}'
            if params.describe:
                title += f' | {params.describe}'
            title+= f' | {datetime.now().strftime("%Y%m%d")}              cost:{self.cost_hour:.2f} hours'
            plt.title(title)

            pic_file = os.path.join(params.root, f"{params.train_title}.png")
            plt.savefig(pic_file)
            # self.printer.print(f'plot done: {pic_file}')

        self.accelerator.wait_for_everyone()

    def _save_result(self):
        ## 记录结果
        result_file = os.path.join(self.params.root, 'result.csv')
        self.printer.print('1')

        # 数据参数
        data_dict =  data_str2parm(self.params.data_set)
        data_dict['y_n'] = self.params.y_n
        data_dict['classify'] = self.params.classify
        data_dict['regress_y_idx'] = self.params.regress_y_idx
        data_dict['classify_y_idx'] = self.params.classify_y_idx
        self.printer.print('2')

        # 初始化列名
        with open(result_file, 'w') as f:
            # 训练参数
            for key in self.params.__dict__:
                f.write(f'{key},')
            self.printer.print('3')

            # 数据参数
            for i in data_dict:
                f.write(f'{i},')
            self.printer.print('4')
            # 模型
            f.write('model,describe,')
            # 训练结果
            for i in self.data:
                f.write(f'{i},')
            self.printer.print('5')
            f.write('cost,folder\n')

        # 写入结果
        with open(result_file, 'a') as f:
            # 训练参数
            for key in self.params.__dict__:
                f.write(f'{self.params.__dict__[key]},')
            self.printer.print('6')
            # 数据参数
            for i in data_dict:
                if isinstance(data_dict[i], list) or isinstance(data_dict[i], tuple):
                    f.write(f'{"@".join([str(i) for i in data_dict[i]])},')
                else:
                    f.write(f'{data_dict[i]},')
            self.printer.print('7')
            # 模型
            f.write(f'{self.model_name},{self.params.describe},')
            # 训练结果
            # 选择val_loss 最小的点
            best_idx = data['val_loss'].index(min(data['val_loss']))
            for i in self.data:
                if not None is self.data[i] and len(self.data[i]) > best_idx+1:
                    f.write(f'{self.data[i][best_idx]},')
                else:
                    f.write(f',')
            self.printer.print('8')
            # 文件夹 
            f.write(f"{self.cost_hour:.2f}h,{self.params.root}\n")
            self.printer.print('9')

    def state_dict(self):
        # self.params = params
        # self.accelerator = accelerator
        # self.scheduler = scheduler
        # self.printer = printer
        return {key: value for key, value in self.__dict__.items() if key not in [
            'params',
            'accelerator',
            'scheduler',
            'printer',
        ]}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

        if '_ids' not in self.temp:
            self.temp['_ids'] = []
        
        for i in self.temp:
            if isinstance(self.temp[i], torch.Tensor):
                self.temp[i] = self.temp[i].to(self.accelerator.device)

        for i in self.data:
            if isinstance(self.data[i], torch.Tensor):
                self.data[i] = self.data[i].to(self.accelerator.device)

if __name__ == '__main__':
    import torch
    from accelerate import Accelerator
    import multiprocessing as mp

    from dl_helper.trainer import printer
    from dl_helper.scheduler import ReduceLR_slow_loss

    class p:
        classify = True
        # classify = False
        epochs = 20
        train_title = 'test_title'
        describe = 'test_describe'
        root = './'

    def random_classify_data():
        output = torch.randn(10, 3)
        target = torch.randint(0, 3, (10,))
        loss = torch.nn.CrossEntropyLoss()(output, target)
        return output, target, loss
    
    def random_regress_data():
        output = torch.randn(10, 1)
        target = torch.randn(10, 1)
        loss = torch.nn.MSELoss()(output, target)
        return output, target, loss
    
    state_dict = None

    params = p()
    accelerator = Accelerator()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = ReduceLR_slow_loss(optimizer)
    lock = mp.Manager().Lock()
    p = printer(lock, accelerator)
    t = Tracker(params, accelerator, scheduler, 1, p)
    for i in range(15):
        for _type in ['train', 'val']:
            for j in range(10):
                if params.classify:
                    output, target, loss = random_classify_data()
                else:
                    output, target, loss = random_regress_data()
                t.track(output, target, loss, _type)
        
        if i%10 == 0:
            state_dict = t.state_dict()

        t.update()
        
    for j in range(10):
        if params.classify:
            output, target, loss = random_classify_data()
        else:
            output, target, loss = random_regress_data()
        t.track(output, target, loss, 'test')

    t.update()
    t.plot()

    t.load_state_dict(state_dict)
    t.update()