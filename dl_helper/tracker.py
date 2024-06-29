"""
用于追踪训练过程评价指标
"""
import time, math, os
from datetime import datetime
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

from accelerate.utils import broadcast

def last_value(data):
    """返回最后一个非nan值"""
    for i in range(len(data)-1, -1, -1):
        if not math.isnan(data[i]):
            return data[i]
    raise ValueError("没有找到非nan值")

class Tracker():
    def __init__(self, params, accelerator, scheduler, num_processes, printer):
        # 时间统计
        self.begin_time = time.time()
        self.epoch_count = 0
        self.step_count = 0
        # 每个epoch训练中的阶段
        # 0: 训练 1: 验证
        self.step_in_epoch = 0
        self.run_limit_hour = 12 if  num_processes != 8 else 9
        self.need_save = False
        self.need_save = True

        self.params = params
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.printer = printer

        # 最终数据
        self.data = {}
        for i in ['train', 'val', 'test']:
            # 训练损失
            self.data[f'{i}_loss'] = []

            if params.classify: 
                # 分类模型 acc
                self.data[f'{i}_acc'] = []

                # 暂时只使用加权f1
                self.data[f'{i}_f1'] = []
            else:
                # 回归模型 r2
                # 暂时只使用加权r2
                self.data[f'{i}_r2'] = []

        # 训练学习率
        self.data['lr'] = []

        # 跟新类别
        self.track_update = ''

        # 计算变量
        self.temp = {}

        self.reset_temp()

    def update(self):
        # 计算变量 -> data
        # 主进程计算data
        if self.accelerator.is_main_process:
            self.data[f'{self.track_update}_loss'].append((self.temp['_loss'] / self.temp['_num']).cpu().item())
            self.printer.print('data loss')

            if self.params.classify:
                self.data[f'{self.track_update}_acc'].append((self.temp['_correct'] / self.temp['_num']).cpu().item())
                self.printer.print('data acc')
                self.data[f'{self.track_update}_f1'].append(
                    f1_score(self.temp['_y_true'].to('cpu').numpy(), self.temp['_y_pred'].to('cpu').numpy(), average='weighted')
                )
                self.printer.print('data f1')
            else:
                self.data[f'{self.track_update}_r2'].append(r2_score(self.temp['_y_true'].to('cpu').numpy(), self.temp['_y_pred'].to('cpu').numpy(), multioutput='variance_weighted'))
                self.printer.print('data r2')

        # TODO fail
        self.printer.print('cal done')

        self.printer.print('update tracker...')
        if 'train' == self.track_update:
            self.printer.print('update train')
            # train 结束，指向验证阶段
            self.step_in_epoch = 1

            if self.accelerator.is_main_process:
                self.printer.print('scheduler.step')

                # 记录学习率
                self.data['lr'].append(self.scheduler.optimizer.param_groups[0]["lr"])

                # 更新 学习率
                self.scheduler.step(self.data['train_loss'])

            # 同步学习率
            self.printer.print('broadcast lr')
            cur_lr = torch.tensor(self.scheduler.optimizer.param_groups[0]["lr"], device=self.accelerator.device)
            self.accelerator.wait_for_everyone()
            cur_lr = broadcast(cur_lr)

            # 在其他设备上应用学习率
            self.printer.print('apply not main lr')
            if not self.accelerator.is_main_process:
                self.scheduler.use_lr(cur_lr)

        if 'val' == self.track_update:
            # val 结束，重置为训练阶段
            self.step_in_epoch = 0
            self.epoch_count += 1

        if 'test' != self.track_update and self.accelerator.is_main_process:
            # 判断是否需要储存 训练数据
            self.printer.print('check need save')
            last_time_hour = ((time.time() - self.begin_time) / 3600)
            each_epoch_time_cost = last_time_hour / self.epoch_count
            free_time = self.run_limit_hour - last_time_hour
            if free_time < each_epoch_time_cost * 1.2:
                self.need_save = True

        self.printer.print('reset_temp')
        self.reset_temp()
        self.printer.print(self.data)
        self.step_count = 0

    def reset_temp(self):
        # 重置计算变量
        self.track_update = ''
        self.temp = {}

        self.temp['_loss'] = 0.0
        self.temp['_num'] = 0
        self.temp['_correct'] = 0
        self.temp['_y_true'] = None
        self.temp['_y_pred'] = None

    def track(self, output, target, loss, _type):
        assert _type in ['train', 'val', 'test'], f'error: _type({_type}) should in [train, val, test]'
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
            correct_count = torch.sum(predict == target)

        # 汇总所有设备上的数据
        self.accelerator.wait_for_everyone()
        self.printer.print('sync track...')
        self.printer.print(f'loss: {loss}')
        self.printer.print(f'target: {target}')
        self.printer.print(f'predict: {predict}')
        self.accelerator.wait_for_everyone()

        # _loss, _y_true, _y_pred = self.accelerator.gather_for_metrics((loss, target, predict))
        # if self.params.classify:
        #     _correct = self.accelerator.gather_for_metrics(correct_count)  

        self.printer.print('main cal track...')
        if self.accelerator.is_main_process:
            if None is self.temp['_y_true']:
                self.temp['_y_true'] = _y_true
                self.temp['_y_pred'] = _y_pred
            else:
                self.temp['_y_true'] = torch.cat([self.temp['_y_true'], _y_true])
                self.temp['_y_pred'] = torch.cat([self.temp['_y_pred'], _y_pred])
            self.temp['_loss'] += torch.sum(_loss)
            self.temp['_num'] += self.temp['_y_true'].shape[0]
            if self.params.classify:
                self.temp['_correct'] += torch.sum(_correct)   

    def plot(self):
        if not self.accelerator.is_main_process:
            return

        params = self.params
        cost_hour = (time.time() - self.begin_time) / 3600

        # x 数量
        epochs = self.params.epochs

        # 标准化数据，nan补气数据
        data = self.data.copy()
        for i in data:
            if 'test' in i:
                data[i] = [data[i][-1]] * epochs
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
        ax1_handles.append(ax1.plot(list(range(epochs)), data['test_loss'], label=f"test loss {last_value(data['test_loss']):.4f}", c='#B0C4DE')[0])
        # 绘制loss曲线
        ax1_handles.append(ax1.plot(list(range(epochs)), data['train_loss'], label=f"train loss {last_value(data['train_loss']):.4f}", c='b')[0])
        ax1_handles.append(ax1.plot(list(range(epochs)), data['val_loss'], label=f"validation loss {last_value(data['val_loss']):.4f}", c='#00BFFF')[0])
        # 标记损失最低点
        ax1_handles.append(ax1.scatter(min_train_x, min_train_loss, c='b',label=f'train loss min: {min_train_loss:.4f}'))
        ax1_handles.append(ax1.scatter(min_test_x, min_test_loss, c='#00BFFF',label=f'validation loss min: {min_test_loss:.4f}'))

        if params.classify:
            # 分类模型
            # 计算acc最高点
            max_train_acc = max(data['train_acc'])
            max_test_acc = max(data['val_acc'])
            max_train_acc_x = data['train_acc'].index(max_train_acc)
            max_test_acc_x = data['val_acc'].index(max_test_acc)
            # 测试集准确率
            ax1_handles.append(ax1.plot(list(range(epochs)), data['test_acc'], label=f"test acc {last_value(data['test_acc']):.4f}", c='#F5DEB3')[0]) 
            # 绘制acc曲线
            ax1_handles.append(ax1.plot(list(range(epochs)), data['train_acc'], label=f"train acc {last_value(data['train_acc']):.4f}", c='r')[0])
            ax1_handles.append(ax1.plot(list(range(epochs)), data['val_acc'], label=f"validation acc {last_value(data['val_acc']):.4f}", c='#FFA07A')[0])
            # 标记准确率最高点
            ax1_handles.append(ax1.scatter(max_train_acc_x, max_train_acc, c='r',label=f'train acc max: {max_train_acc:.4f}'))
            ax1_handles.append(ax1.scatter(max_test_acc_x, max_test_acc, c='#FFA07A',label=f'validation acc max: {max_test_acc:.4f}'))

        else:
            # 回归模型
            # 计算r2最高点
            max_train_r2 = max(data['train_r2'])
            max_test_r2 = max(data['val_r2'])
            max_train_r2_x = data['train_r2'].index(max_train_r2)
            max_test_r2_x = data['val_r2'].index(max_test_r2)
            # 测试集r2
            ax1_handles.append(ax1.plot(list(range(epochs)), data['test_r2'], label=f"test r2 {last_value(data['test_r2']):.4f}", c='#F5DEB3')[0])
            # 绘制r2曲线
            ax1_handles.append(ax1.plot(list(range(epochs)), data['train_r2'], label=f"train r2 {last_value(data['train_r2']):.4f}", c='r')[0])
            ax1_handles.append(ax1.plot(list(range(epochs)), data['val_r2'], label=f"validation r2 {last_value(data['val_r2']):.4f}", c='#FFA07A')[0])
            # 标记r2最高点
            ax1_handles.append(ax1.scatter(max_train_r2_x, max_train_r2, c='r',label=f'train r2 max: {max_train_r2:.4f}'))
            ax1_handles.append(ax1.scatter(max_test_r2_x, max_test_r2, c='#FFA07A',label=f'validation r2 max: {max_test_r2:.4f}'))

        # 创建右侧坐标轴
        ax2 = ax1.twinx()

        # 绘制学习率
        line_lr, = ax2.plot(list(range(epochs)), data['lr'], label='lr', c='#87CEFF',linewidth=2,alpha =0.5)

        # 添加图例
        ax1.legend(handles=ax1_handles)

        # 显示横向和纵向的格线
        ax1.grid(True)
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
            t2_handles.append(axs[1].plot(list(range(epochs)), data["test_f1"], label=f'test f1 {last_value(data["test_f1"]):.4f}', c='#89BC7A')[0])
            # 绘制f1曲线
            t2_handles.append(axs[1].plot(list(range(epochs)), data["train_f1"], label=f'train f1 {last_value(data["train_f1"]):.4f}', c='#8DE874')[0])
            t2_handles.append(axs[1].plot(list(range(epochs)), data["val_f1"], label=f'val f1 {last_value(data["val_f1"]):.4f}', c='#57C838')[0])
            # 标记f1最高点
            t2_handles.append(axs[1].scatter(max_train_f1_x, max_train_f1, c='#8DE874',label=f'train f1 max: {max_train_f1:.4f}'))
            t2_handles.append(axs[1].scatter(max_test_f1_x, max_test_f1, c='#57C838',label=f'val f1 max: {max_test_f1:.4f}'))

            axs[1].grid(True)
            axs[1].set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值
            axs[1].legend(handles=t2_handles)

        title = f'{params.train_title}'
        if params.describe:
            title += f' | {params.describe}'
        title+= f' | {datetime.now().strftime("%Y%m%d")}              cost:{cost_hour:.2f} hours'
        plt.title(title)

        pic_file = os.path.join(params.root, f"{params.train_title}.png")
        plt.savefig(pic_file)

    def state_dict(self):
        # self.params = params
        # self.accelerator = accelerator
        # self.scheduler = scheduler
        # self.printer = printer
        return {key: value for key, value in self.__dict__.items() if key not in [
            'params',
            'accelerator',
            'scheduler',
            'printer'
        ]}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        
        for i in self.temp:
            if isinstance(self.temp[i], torch.Tensor):
                self.temp[i].to(self.accelerator.device)


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