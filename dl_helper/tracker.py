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

def last_value(data):
    """返回最后一个非nan值"""
    for i in range(len(data)-1, -1, -1):
        if not math.isnan(data[i]):
            return data[i]
    raise ValueError("没有找到非nan值")


class Tracker():
    def __init__(self, params, trader):
        self.begin_time = time.time()
        self.params = params
        self.trader = trader

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
        self.track_update = set()

        # 计算变量
        self.temp = {}
        self.reset_temp()

    def update(self):
        # 计算变量 -> data
        for i in self.track_update:
            self.temp[f'{i}_y_true'] = torch.stack(self.temp[f'{i}_y_true'])
            self.temp[f'{i}_y_pred'] = torch.stack(self.temp[f'{i}_y_pred'])

            # 汇总所有设备上的数据
            _loss, _num, _y_true, _y_pred = self.trader.gather_for_metrics(self.temp[f'{i}_loss'], self.temp[f'{i}_num'], self.temp[f'{i}_y_true'], self.temp[f'{i}_y_pred'])
            if self.params.classify:
                _correct = self.trader.gather_for_metrics(self.temp[f'{i}_correct'])

            if self.trader.is_main_process():
                self.data[f'{i}_loss'].append((_loss / _num).cpu().item())

                if self.params.classify:
                    self.data[f'{i}_acc'].append((_correct / _num).cpu().item())
                    self.data[f'{i}_f1'].append(
                        f1_score(_y_true, _y_pred, average='weighted')
                    )
                else:
                    self.data[f'{i}_r2'].append(r2_score(_y_true, _y_pred, multioutput='variance_weighted'))

        self.reset_temp()
        self.trader.print(self.data)

    def reset_temp(self):
        # 重置计算变量
        self.track_update.clear()
        self.temp = {}
        for i in ['train', 'val', 'test']:
            self.temp[f'{i}_loss'] = 0.0
            self.temp[f'{i}_num'] = 0

            if self.params.classify:
                self.temp[f'{i}_correct'] = 0

            self.temp[f'{i}_y_true'] = []
            self.temp[f'{i}_y_pred'] = []

    def track(self, output, target, loss, _type):
        assert _type in ['train', 'val', 'test'], f'error: _type({_type}) should in [train, val, test]'
        self.track_update.add(_type)

        # 统计损失 tensor
        self.temp[f'{_type}_loss'] += loss
        self.temp[f'{_type}_num'] += torch.tensor(target.shape[0], device=target.device)

        if self.params.classify:
            softmax_predictions = F.softmax(output, dim=1)
            predicted_labels = torch.argmax(softmax_predictions, dim=1)
            correct_count = torch.sum(predicted_labels == target)

            # 统计acc
            self.temp[f'{_type}_correct'] += correct_count

            # 统计f1
            self.temp[f'{_type}_y_true'].extend(target)
            self.temp[f'{_type}_y_pred'].extend(predicted_labels)

        else:
            # 统计r2
            self.temp[f'{_type}_y_true'].extend(target)
            self.temp[f'{_type}_y_pred'].extend(output)

    def plot(self):
        if not self.trader.is_main_process():
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




if __name__ == '__main__':
    import torch
    from dl_helper.trainer import train_base

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
    
    _p = p()
    t = tracker(_p, train_base(1, 'no'))
    for i in range(15):
        for _type in ['train', 'val']:
            for j in range(10):
                if _p.classify:
                    output, target, loss = random_classify_data()
                else:
                    output, target, loss = random_regress_data()
                t.track(output, target, loss, _type)
        
        t.update()
        
    for j in range(10):
        if _p.classify:
            output, target, loss = random_classify_data()
        else:
            output, target, loss = random_regress_data()
        t.track(output, target, loss, 'test')

    t.update()
    t.plot()