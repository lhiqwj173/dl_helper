import functools
import sys, torch, os

from accelerate.utils import set_seed

from dl_helper.tester import test_base
from dl_helper.train_param import Params
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.models.deeplob_stable_data import m_deeplob

from dl_helper.transforms.binctabl import transform
from dl_helper.trainer import run

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')

"""
稳健移动窗口表示法
100 * 21

predict_n 100
标签 4
依据市值top 20/10/5选股

标准化
量: 简单标准化
pct: 简单标准化

batch_size=128

测试模型:
deeplob
"""

class transform_deeplob(transform):

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch
            # debug('x', x.shape, x.device)
            # debug('y', y.shape, y.device)
            # debug('mean_std', mean_std.shape, mean_std.device)

            # random_mask_row
            if train and self.param.random_mask_row:
                if x.shape[1] > self.raw_time_length:
                    x = x[:, -self.raw_time_length:, :]
                x = self.random_mask_row(x)
            else:
                if x.shape[1] > self.time_length:
                    x = x[:, -self.time_length:, :]
            
            if x.shape[2] > self.num_rows:
                x = x[:, :,:self.num_rows]


            # 标准化
            # -> (batchsize, 2, 21)
            mean_std = torch.transpose(mean_std, 1, 2)
            x -= mean_std[:, :1, :]
            x /= mean_std[:, 1:, :]

            if train and self.param.random_scale>0:
                x = self.random_scale(x)

            # 增加一个 通道维度
            x = x.unsqueeze(1)

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
        return f'stable_rolling_window_{dataset_type}'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        classify_idx = 0
        for predict_n in [3, 30, 60, 100]:
            vars.append((predict_n, classify_idx))
            classify_idx+=1

        predict_n, classify_idx = vars[self.idx]

        self.y_n = 3

        epochs = 30

        self.lr_scheduler_class = functools.partial(OneCycle_fast, total_iters=epochs)

        title = self.title_base() + f"_predict_n{predict_n}"

        data_parm = {
            'predict_n': [3, 30, 60, 100],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': f'{dataset_type}',
            'target': f'label 4',
            'std_mode': '简单标准化'
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*2, epochs=epochs,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=functools.partial(yfunc, 0.5),

            data_folder=self.data_folder,

            describe=f"predict_n{predict_n}",
            amp=self.amp
        )

    def get_in_out_shape(self):
        return (1, 21, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        return m_deeplob(3)

    def get_transform(self, device):
        return transform_deeplob(device, self.para, 103, num_rows=21)

if '__main__' == __name__:

    input_folder = r'/kaggle/input'

    data_folder_name = os.listdir(input_folder)[0]
    data_folder = os.path.join(input_folder, data_folder_name)

    # 按照数据集分类
    dataset_type = ''
    if 'top5' in data_folder_name:
        dataset_type = 'top5'
    elif 'top10' in data_folder_name:
        dataset_type = 'top10'
    elif 'top20' in data_folder_name:
        dataset_type = 'top20'
    else:
        raise Exception('dataset type not found')

    run(
        test, 
        # findbest_lr=True,
        amp='fp16',
        mode='cache_data',
        data_folder=data_folder,

        # debug=True,
        # idx=0
    )