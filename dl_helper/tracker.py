"""
用于追踪训练过程评价指标
"""
import time

class tracker():
    def __init__(self, params, trader):
        self.begin_time = time.time()
        self.params = params
        self.trader = trader

        # 最终数据
        self.data = {}
        # 计算变量
        self.temp = {}
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
                # 暂时只使用平均r2
                sself.data[f'{i}_r2'] = []

        # 训练学习率
        self.data['lr'] = []




    def track(self, output, target, loss, _type):
        pass

    def plot(self):
        pass