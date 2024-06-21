"""
用于追踪训练过程评价指标
"""

class tracker():
    def __init__(self, params, trader):
        self.params = params
        self.trader = trader

    def track(self, output, target, loss):
        pass

    def plot(self):
        pass