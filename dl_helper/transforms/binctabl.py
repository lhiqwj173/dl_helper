import torch, os
from dl_helper.train_param import tpu_available
from py_ext.tool import debug

class fi2010_transform():
    def __init__(self, device, param, raw_time_length,num_rows=40, scale_prob=0.005, min_scale=0.97, max_scale=1.03):
        pass

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch
            x = torch.transpose(x, 1, 2)

            # nan 替换为 -1
            x = torch.where(torch.isnan(x), torch.tensor(-1.0, device=x.device), x)

            return x, y


class transform():
    def __init__(self, device, param, raw_time_length,num_rows=40, scale_prob=0.005, min_scale=0.97, max_scale=1.03):
        """
        如果 param 中random_mask_row为0, 则raw_time_length无作用, 根据输入的形状进行切片
        """
        assert not param.cnn, '暂不支持cnn'

        self.device = device
        self.param = param
        self.time_length = int(param.data_set.split('_')[3])
        self.raw_time_length = raw_time_length

        self.scale_prob = scale_prob
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.num_rows = num_rows
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

        if self.num_rows in [40, 44]:
            self.vol_cond = torch.tensor([False, True] * (20 if self.num_rows==40 else 22), device=self.device).unsqueeze(0).unsqueeze(2).expand(self.batch_size, -1, self.time_length)
        elif self.num_rows in [41, 21, 71, 91]:
            self.vol_cond = torch.tensor([True] * self.num_rows, device=self.device).unsqueeze(0).unsqueeze(2).expand(self.batch_size, -1, self.time_length)
        elif self.num_rows == 146:
            self.vol_cond = torch.tensor(
                [False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True], 
                device=self.device).unsqueeze(0).unsqueeze(2).expand(self.batch_size, -1, self.time_length)
        else:
            raise ValueError('num_rows not supported')

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
            # debug('x', x.shape, x.device)
            # debug('y', y.shape, y.device)
            # debug('mean_std', mean_std.shape, mean_std.device)

            # not cnn -> (batchsize, 40, 100)
            x = torch.transpose(x, 1, 2)

            # random_mask_row
            if train and self.param.random_mask_row:
                if x.shape[2] > self.raw_time_length:
                    x = x[:, :, -self.raw_time_length:]
                x = self.random_mask_row(x)
            else:
                if x.shape[2] > self.time_length:
                    x = x[:, :, -self.time_length:]
            # debug('random_mask_row')

            # debug('trans 0', torch.isnan(x).any().item())

            # 调整价格
            # import pickle
            # pickle.dump(x, open(os.path.join(self.param.root, 'raw_x.pkl'), 'wb'))
            mp = ((x[:, 0, -1] + x[:, 2, -1]) / 2).unsqueeze(1).unsqueeze(1)
            # debug('mp',mp.shape)
            # debug('trans mp nan', torch.isnan(mp).any().item())
            # debug('trans mp 0', (mp == 0.0).any().item())
            # if (mp == 0.0).any().item():
            #     raise ValueError('mp == 0.0')
            x = torch.where(~self.vol_cond, x / mp, x)
            # debug('trans 1', torch.isnan(x).any().item())
            # debug('x 0',x.shape, x.device)
            # 标准化
            # debug('mean',mean_std[:, :, :1].shape, mean_std.device)
            x -= mean_std[:, :, :1]
            # debug('x 1',x.shape)
            # debug('std',mean_std[:, :, 1:].shape)
            # debug('trans 2', torch.isnan(x).any().item())
            x /= mean_std[:, :, 1:]
            # debug('x 2',x.shape)
            # debug('std done')
            # debug('trans 3', torch.isnan(x).any().item())

            # random_scale
            if train and self.param.random_scale>0:
                x = self.random_scale(x)
            # debug('random_scale')
            # debug('trans 4', torch.isnan(x).any().item())

            # nan 替换为 -1
            x = torch.where(torch.isnan(x), torch.tensor(-1.0, device=x.device), x)

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

