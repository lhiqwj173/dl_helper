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
predict_n 100
标签 0-4
标签阈值 0.7/0.9

取消 label_smooth
"""

class transform_simple_std(transform):

    def __init__(self, *args, nan_replace=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_replace = nan_replace

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

            # nan 替换为 self.nan_replace
            x = torch.where(torch.isnan(x), torch.tensor(float(self.nan_replace), device=x.device), x)

            return x, y

def yfunc(threshold, y):
    if y > threshold:
        return 0
    elif y < -threshold:
        return 1
    else:
        return 2

class test(test_base):

    @classmethod
    def title_base(cls):
        return f'no_label_smooth'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        classify_idx = 0
        for predict_n in [10, 30, 60, 100]:
            for label in range(5):
                if predict_n == 100:
                    for threshold in [0.1, 0.3]:
                        vars.append((predict_n, label, classify_idx, threshold))
                classify_idx+=1

        predict_n, label_idx, classify_idx, threshold = vars[self.idx]

        self.y_n = 3

        batch_n = 16 * 2

        epochs = 100

        min_lr = 9.6e-6
        max_lr = 4.6e-3
        self.lr_scheduler_class = functools.partial(OneCycle, total_iters=epochs, min_lr=min_lr, max_lr=max_lr)

        title = self.title_base() + f"_threshold{threshold}_label_idx{label_idx}"
        data_parm = {
            'predict_n': [10, 30, 60, 100],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': '成交额 >= 10亿',
            'target': f'label {label_idx}',
            'std_mode': '5d'  # 4h/1d/5d
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*batch_n, epochs=epochs, label_smoothing=0.0,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=functools.partial(yfunc, threshold),

            data_folder=self.data_folder,

            describe=f"threshold{threshold} label_idx{label_idx}",
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
        return transform_simple_std(device, self.para, 103, num_rows=44)

if '__main__' == __name__:

    input_folder = r'/kaggle/input'
    data_folder_name = os.listdir(input_folder)[0]
    data_folder = os.path.join(input_folder, data_folder_name)

    run(
        test, 
        # findbest_lr=True,
        amp='fp16',
        mode='cache_data',
        data_folder=data_folder
    )