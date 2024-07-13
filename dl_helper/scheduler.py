import pandas as pd
from dl_helper.train_param import tpu_available
if tpu_available():
    import torch_xla.core.xla_model as xm


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
        if array_loss.size()[0] < self.patience+1:
            return
        # 计算损失均线
        loss_ma = array_loss.unfold(dimension=0, size=self.patience, step=1).mean(dim=1)
        # 计算均线变动率
        loss_pct_change = loss_ma[1:] / loss_ma[:-1] - 1
        # 判断是否满足减少学习率条件
        match = (loss_pct_change >= self.min_pct)

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
