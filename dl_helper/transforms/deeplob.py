import torch

class transform():
    def __init__(self):
        self.price_cols = [i*2 for i in range(20)]
        self.other_cols = [i*2 + 1 for i in range(20)]

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch
            
            # x: [batch_size, time_length, num_rows]
            x = x[:, -100:, :40]

            # 中间价格 / 中间量
            mid_price = ((x[:, -1, 0] + x[:, -1, 2]) / 2).unsqueeze(1).unsqueeze(1).clone()
            mid_vol = ((x[:, -1, 1] + x[:, -1, 3]) / 2).unsqueeze(1).unsqueeze(1).clone()

            # 价归一化
            x[:, self.price_cols, :] /= mid_price

            # 量归一化
            x[:, self.other_cols, :] /= mid_vol

            # 增加一个维度 [batch_size, time_length, num_rows] -> [batch_size, 1，time_length, num_rows]
            x = x.unsqueeze(1)
            
            return x, y