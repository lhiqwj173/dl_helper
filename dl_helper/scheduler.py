import pandas as pd
import matplotlib.pyplot as plt
from py_ext.tool import debug, log
from torch.optim.lr_scheduler import ReduceLROnPlateau as _ReduceLROnPlateau

from dl_helper.train_param import tpu_available
if tpu_available():
    import torch_xla.core.xla_model as xm
def lr_lambda(x, min_lr, max_lr, total_iters):
    return min_lr * (max_lr / min_lr) ** (x / total_iters)

class LRFinder:
    def __init__(self, optimizer, *args, total_iters: int=60, min_lr: float=1e-7, max_lr: float=1, **kwargs):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iters = total_iters
        # self.lr_lambda = lambda x: self.min_lr * (self.max_lr / self.min_lr) ** (x / self.total_iters)
        self.iteration = 0
        self.history = {'lr': [], 'loss': []}
        
        # 初始化学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = min_lr

    def step(self, loss_array):
        loss = loss_array[-1]

        self.iteration += 1

        # lr = self.lr_lambda(self.iteration)
        lr = lr_lambda(self.iteration, self.min_lr, self.max_lr, self.total_iters)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.history['lr'].append(lr)
        self.history['loss'].append(loss)
        
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def use_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr   

class ReduceLROnPlateau(_ReduceLROnPlateau):
    def step(self, loss_array):
        loss = loss_array[-1]
        super().step(loss)

    def use_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr   

class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_epochs=10, **kwargs):
        super(WarmupReduceLROnPlateau, self).__init__(optimizer, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.warmup_lrs = [lr / warmup_epochs for lr in self.base_lrs]

        # 初始化学习率
        self.step(None)

    def step(self, metrics):
        debug('step')
        if self.current_epoch < self.warmup_epochs:
            debug(f"Warmup epoch, {self.current_epoch}, {self.warmup_epochs}")
            lr = [(self.current_epoch + 1) * warmup_lr for warmup_lr in self.warmup_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, lr):
                param_group['lr'] = lr
            self.current_epoch += 1
        else:
            # Pass the call to the parent class (ReduceLROnPlateau)
            debug(f'ReduceLROnPlateau')
            super(WarmupReduceLROnPlateau, self).step(metrics)

    def state_dict(self):
        d = super().state_dict()
        debug(f'state_dict: {d}')
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

class OneCycle():
    def __init__(self, optimizer, total_iters: int, min_lr: float, max_lr: float, *args, **kwargs):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iters = total_iters

        one_cycle_epochs = int(total_iters * 0.85)
        self.max_lr_epoch_idx = one_cycle_epochs // 2
        self.final_epoch_idx = self.max_lr_epoch_idx * 2

        # 每次调整的学习率
        self.each_diff_lr = (max_lr - min_lr) / self.max_lr_epoch_idx
        self.each_diff_lr_final = (min_lr - min_lr / 100) / (total_iters - self.final_epoch_idx - 1)

        self.iteration = 0
        
        # 初始化学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = min_lr

    def step(self, loss_array):
        loss = loss_array[-1]

        self.iteration += 1

        # lr = self.lr_lambda(self.iteration)
        cur_lr = self.optimize.param_groups[0]["lr"]
        if self.iteration <= self.max_lr_epoch_idx:
            lr = cur_lr + self.each_diff_lr
        elif self.iteration <= self.final_epoch_idx:
            lr = cur_lr - self.each_diff_lr
        else:
            # 最终阶段
            lr = cur_lr - self.each_diff_lr_final
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def use_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr   


# 当训练的损失序列区域平缓时，减低学习率
class ReduceLR_slow_loss():
    def __init__(self, optimizer, min_pct=-0.00005, patience=20, factor=0.1, min_lr=0, eps=1e-8, debug=False):
        self.optimizer = optimizer

        self.min_pct = min_pct
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.eps = eps
        self.wait = 0

        self.debug = debug

    def step(self, array_loss):
        # print('step')
        if self.wait > 0:
            self.wait -= 1
            return

        # # 计算损失均线，ma=self.patience
        # # 均线变动率 大于 min_pct 则减少学习率
        # loss = pd.DataFrame({'loss': array_loss}).dropna()
        # if len(loss) < self.patience+1:
        #     return
        # loss['ma'] = loss['loss'].rolling(self.patience).mean()
        # loss['pct'] = loss['ma'].pct_change()
        # loss['match'] = loss['pct']>=self.min_pct
        # if loss.iloc[-1]['match']:
        #     self._reduce_lr()
        # elif self.debug:
        #     print('pass')

        # 改用torch
        # print(array_loss.shape)
        if array_loss.shape[0] < self.patience+1:
            return

        # 计算损失均线
        # print(array_loss)
        loss_ma = array_loss.unfold(dimension=0, size=self.patience, step=1).mean(dim=1)
        # print(loss_ma)
        # 计算均线变动率
        loss_pct_change = loss_ma[1:] / loss_ma[:-1] - 1
        # print(loss_pct_change)
        # 判断是否满足减少学习率条件
        match = (loss_pct_change >= self.min_pct)
        # print(match)

        if tpu_available():
            xm.mark_step()
        if match[-1].item():
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
    
    def use_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr   

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


