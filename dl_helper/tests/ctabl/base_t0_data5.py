import functools
import sys, torch, os

from accelerate.utils import set_seed

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
测试 simple std 数据集


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
        return f'test_t0_datas'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        datas = [
            '10y_each_sample',
            '10y_simple',
        ]
        for data in datas:
            for seed in [42, 55, 109, 123]:
                vars.append((data, seed))
                    
        self.data_set, seed = vars[self.idx]
        set_seed(seed)

        classify_idx, predict_n, label_idx = 0, 20, 2

        self.y_n = 3

        batch_n = 16 * 2

        epochs = 100

        min_lr = 9.6e-6
        max_lr = 4.6e-3
        self.lr_scheduler_class = functools.partial(OneCycle, total_iters=epochs, min_lr=min_lr, max_lr=max_lr)

        input_folder = r'/kaggle/input'
        data_folder_name = os.listdir(input_folder)[0]
        self.data_folder = os.path.join(input_folder, data_folder_name, self.data_set)

        title = self.title_base() + f"_{self.data_set}_{seed}"
        data_parm = {
            'predict_n': [20],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': '成交额 >= 10亿',
            'target': self.data_set,
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

            describe=f"dataset:{self.data_set}",
            amp=self.amp
        )

    def get_in_out_shape(self):
        return (1, 44, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # # T: 100, 100, 50, 1
        # # D: 44, 60, 120, 3
        return m_bin_ctabl(60, 44, 100,  100,  120,  50, self.y_n, 1)

    def get_transform(self, device):
        if self.data_set == '10y_simple':
            return transform_simple_std(device, self.para, 103, num_rows=44)
        else:
            return transform(device, self.para, 103, num_rows=44)

if '__main__' == __name__:

    run(
        test, 
        # findbest_lr=True,
        amp='fp16',
        mode='cache_data',
    )