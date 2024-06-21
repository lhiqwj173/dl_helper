"""
用于追踪训练过程评价指标
"""

class tracker():
    def __init__(self, params, trader):
        self.params = params
        self.trader = trader

        # 训练损失
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        if params.classify: 
            # 分类模型 acc
            self.train_acc = []
            self.val_acc = []
            self.test_acc = []

        else:
            # 回归模型 r2
            self.train_r2s = [[]]
            self.val_r2s = [[]]
            self.test_r2s = [[]]

        # 训练学习率
        self.lr = []

        

    def track(self, output, target, loss):
        pass

    def plot(self):
        pass