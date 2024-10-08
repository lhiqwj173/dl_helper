import functools
import sys, torch

from dl_helper.tester import test_base
from dl_helper.train_param import Params
from dl_helper.scheduler import OneCycle
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.binctabl import transform
from dl_helper.trainer import run

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')

"""
测试数据集:
 100w
 250w
 1000w
 2000w

最佳数据集:
 2000w
"""

class transform_simple_std(transform):

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

            x -= mean_std[:, :, :1]
            x /= mean_std[:, :, 1:]
            if train and self.param.random_scale>0:
                x = self.random_scale(x)

            # nan 替换为 -1
            x = torch.where(torch.isnan(x), torch.tensor(-1.0, device=x.device), x)

            return x, y


def yfunc(y):
    if y > 0.5:
        return 0
    elif y < -0.5:
        return 1
    else:
        return 2

class test(test_base):

    @classmethod
    def title_base(cls):
        return f'binctabl_base_t0_simple_std_{data_type}'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.y_n = 3

        batch_n = 16 * 2

        epochs = 80

        predict_n = [3, 5, 10, 20, 30, 40, 50, 100]

        min_lr = 9.6e-6
        max_lr = 4.6e-3
        self.lr_scheduler_class = functools.partial(OneCycle, total_iters=epochs, min_lr=min_lr, max_lr=max_lr)

        self.predict_n = 3
        classify_idx, targrt_name = 0 + self.idx , f'{self.predict_n}_label_{self.idx}'

        title = self.title_base() + f'_v{self.idx}'
        data_parm = {
            'predict_n': [3, 5, 10, 20, 30, 40, 50, 100],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': '成交量 >= 250w',
            'target': targrt_name,
            'std_mode': '5d'  # 4h/1d/5d
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*batch_n, epochs=epochs,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=self.data_folder,

            describe=f'label_{self.idx} n=3',
            amp=self.amp
        )

    def get_in_out_shape(self):
        return (1, 44, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # T: 100, 200, 100, 1
        # D: 44, 100, 200, 3
        # return m_bin_ctabl(100, 44, 100, 200, 200, 100, self.y_n, 1)

        # T: 100, 140, 60, 1
        # D: 44, 80, 160, 3
        # return m_bin_ctabl(80, 44, 100, 140, 160, 60, self.y_n, 1)

        # T: 100, 100, 50, 1
        # D: 44, 60, 120, 3
        return m_bin_ctabl(60, 44, 100,  100,  120,  50, self.y_n, 1)

    def get_transform(self, device):
        return transform_simple_std(device, self.para, 103, num_rows=44)

if '__main__' == __name__:

    # ##########################
    # # A股
    # # 20231212.pkl
    # # 20240116.pkl
    # # 20240206.pkl
    # # 20240221.pkl
    # # 20240228.pkl
    # # 20240304.pkl
    # # 20240322.pkl
    # # 20240401.pkl
    # # 20240408.pkl
    # # 20240618.pkl
    # # 20240709.pkl
    # # 20240723.pkl
    # ##########################
    # import os, random
    # files = os.listdir(r'Z:\L2_DATA\train_data\20240818\train')
    # print(f'files num: {len(files)}')

    # # 随机抽取10 天的数据作为验证集
    # random.shuffle(files)
    # valid_files = files[:10]
    # valid_files.sort()
    # for file in valid_files:
    #     print(file)

    import sys
    data_type = '100w'
    for i in sys.argv[1:]:
        if i.startswith('100w'):
            data_type = i
            break
        elif i.startswith('250w'):
            data_type = i
            break
        elif i.startswith('1000w'):
            data_type = i
            break
        elif i.startswith('2000w'):
            data_type = i
            break

    data_type = data_type.replace('_', '-')

    run(
        test, 
        # findbest_lr=True,
        amp='fp16',
        mode='cache_data',
        # data_folder=r'/kaggle/input/lh-q-t0-data-extra-250w-2'
        data_folder=rf'/kaggle/input/lh-q-t0-data-extra-{data_type}'
    )