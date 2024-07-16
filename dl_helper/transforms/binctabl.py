import torch
from dl_helper.train_param import tpu_available
from py_ext.tool import debug

class transform():
    def __init__(self, device, param, raw_time_length, scale_prob=0.005, min_scale=0.97, max_scale=1.03):
        assert not param.cnn, '暂不支持cnn'

        self.device = device
        self.param = param
        self.time_length = int(param.data_set.split('_')[3])
        self.raw_time_length = raw_time_length

        self.scale_prob = scale_prob
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.num_rows = 46 if self.param.use_pk and self.param.use_trade else 40 if self.param.use_pk else 6
        self.batch_size = param.batch_size

        # 用于random_mask_row
        self.rand_cols = torch.randint(0, 1, (self.batch_size, self.time_length), device=self.device)
        for i in range(self.batch_size):
            r = torch.randperm(self.raw_time_length,  device='cpu')[:self.time_length].sort().values
            self.rand_cols[i] = r.to(self.device)
        self.rand_cols = self.rand_cols.unsqueeze(1).expand(-1, self.num_rows, -1)

        # 用于random_scale
        self.rand_scales = torch.rand(self.batch_size, self.num_rows, self.time_length, device=self.device) < scale_prob
        # 只用vol_cols
        self.vol_cond = torch.tensor([False, True] * 20, device=self.device).unsqueeze(0).unsqueeze(2).expand(self.batch_size, -1, self.time_length)
        self.rand_scales = torch.where(self.vol_cond, self.rand_scales, torch.tensor(False, device=self.device))

    def random_mask_row(self, tensor):
        """
        随机删除行
        """
        return torch.gather(tensor, dim=2, index=self.rand_cols)

    def random_scale(self, tensor):
        """随机按照scale_prob选择值 随机缩放min_scale - max_scale"""
        new_tensor = (torch.rand(tensor.size(),device=self.device)*(self.max_scale-self.min_scale)+self.min_scale) * tensor
        return torch.where(self.rand_scales, new_tensor, tensor)

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch
            debug(x.shape, y.shape, mean_std.shape)

            # not cnn -> (batchsize, 40, 100)
            x = torch.transpose(x, 1, 2)

            # random_mask_row
            if train and self.param.random_mask_row:
                x = self.random_mask_row(x)
            else:
                if self.raw_time_length > self.time_length:
                    x = x[:, :, -self.time_length:]
            debug('random_mask_row')

            # 调整价格
            mp = (x[:, 0, -1] * x[:, 2, -1] / 2).unsqueeze(1).unsqueeze(1)
            x = torch.where(~self.vol_cond, x / mp, x)
            # 标准化
            x -= mean_std[:, :, :1]
            x /= mean_std[:, :, 1:]
            debug('std')

            # random_scale
            if train and self.param.random_scale>0:
                x = self.random_scale(x)
            debug('random_scale')

            return x, y

if __name__ == '__main__':
    from dl_helper.tests.test_m_bin_ctabl_real_data import test

    t = test(
        idx = 0,
        debug=True,
        data_folder=r'D:\code\featrue_data\notebook\20240413_滚动标准化\20240612', 
    )
    d = t.get_data('train',t.para)
    batch = next(iter(d))
    print(batch[0].shape, batch[1].shape, batch[2].shape)

    trans = transform('cpu', t.para, 105)
    batch = trans(batch, train=True)
    print(batch[0].shape, batch[1].shape)

