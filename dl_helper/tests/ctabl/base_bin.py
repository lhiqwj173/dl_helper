import functools
import sys

from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.binctabl import transform
from dl_helper.trainer import run

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')

"""
- 使用新的标签 
    价差/最小变单位 >=  1 -上涨 0
    价差/最小变单位 <= -1 -下跌 1
    else                -不变 2
- 验证数据集为训练数据集同期内随机7天的数据，在训练集中排除

降采样      -> 5

不使用数据增强

数据增加 37d
训练:验证:测试
28:7:2
"""

def yfunc(y):
    if y >= 1:
        return 0
    elif y <= -1:
        return 1
    else:
        return 2

class test(test_base):

    @classmethod
    def title_base(cls):
        return 'binctabl_base_bin'

    def __init__(self, *args, target_type=1, lr_scheduler_class='WarmupReduceLROnPlateau', **kwargs):
        super().__init__(*args, **kwargs)

        self.lr_scheduler_class = lr_scheduler_class
        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        classify_idx, targrt_name = 4, '10_target_mid_diff'
        self.y_n = 3

        batch_n = 16 * 2

        # T: 100, 40, 10, 1
        model_vars = [
            (100, 40, 10, 1),
            (100, 60, 20, 1),
            (100, 80, 30, 1),
            (100, 100, 40, 1),
            (100, 120, 50, 1),
            (100, 140, 60, 1),
        ]
        self.model_var = model_vars[self.idx]

        title = self.title_base() + f'_v{self.idx}'
        data_parm = {
            'predict_n': [10, 20, 30],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': '@'.join(symbols),
            'target': targrt_name,
            'std_mode': '5d'  # 4h/1d/5d
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.0001*batch_n, batch_size=64*batch_n, epochs=150,

            # 学习率衰退延迟
            learning_rate_scheduler_patience=10,

            # 每 5 个样本取一个数据
            down_freq=5,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=self.data_folder,

            describe=f't: {"@".join([str(i) for i in self.model_var])}',
            amp=self.amp
        )

    def get_in_out_shape(self):
        return (1, 40, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # T: 100, 40, 10, 1
        # D: 40, 60, 120, 3
        return m_bin_ctabl(60, 40, self.model_var[0], self.model_var[1], 120, self.model_var[2], self.y_n, 1)

    def get_transform(self, device):
        return transform(device, self.para, 103)

if '__main__' == __name__:

    # ##########################
    # # 币安
    # ##########################
    # # 2024-05-08
    # # 2024-05-10
    # # 2024-05-14
    # # 2024-05-19
    # # 2024-07-19
    # # 2024-07-22
    # # 2024-07-30
    # import datetime
    # dates = []

    # # 0501 + 20
    # begin_date = datetime.datetime.strptime('2024-05-01', '%Y-%m-%d')
    # for i in range(20):
    #     dates.append(begin_date + datetime.timedelta(days=i))

    # # 0716 + 17
    # begin_date = datetime.datetime.strptime('2024-07-16', '%Y-%m-%d')
    # for i in range(17):
    #     dates.append(begin_date + datetime.timedelta(days=i))

    # print(f'训练数据天数: {len(dates)}')

    # # 随机抽 7 天作为验证数据集
    # import random
    # random.shuffle(dates)
    # valid_dates = dates[:7]
    # valid_dates.sort()
    # for date in valid_dates:
    #     print(date.date())

    run(
        test,
        findbest_lr=True, 
        mode='cache_data',
        data_folder=r'/kaggle/input/lh-q-bin-data-20240805',
    )