import functools
import sys

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
- 使用新的标签 
    价差/最小变单位 >  0.5 -上涨 0
    价差/最小变单位 < -0.5 -下跌 1
    else                    -不变 2
- 验证数据集为训练数据集同期内随机 10 天的数据，在训练集中排除

数据密度较小，不使用降采样

不使用数据增强

数据增加 187d
训练:验证:测试
170:12:5
"""

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
        return 'binctabl_base_t0_filter_extra_100w'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.y_n = 3

        batch_n = 16 * 2

        epochs = 80

        predict_n = [10, 20, 30, 40, 50, 100]
        predict_n_vars = [10, 30 , 50, 100]
        min_lr_vars = [6.6e-7, 5.0e-7, 5.0e-7, 5.0e-7]
        max_lr_vars = [2.1e-3, 2.1e-3, 5.0e-7, 5.0e-7]

        min_lr = min_lr_vars[self.idx]
        max_lr = max_lr_vars[self.idx]
        self.lr_scheduler_class = functools.partial(OneCycle, total_iters=epochs, min_lr=min_lr, max_lr=max_lr)

        self.predict_n = predict_n_vars[self.idx]
        classify_idx, targrt_name = predict_n.index(self.predict_n) , f'{self.predict_n}_target_mid_diff'

        title = self.title_base() + f'_v{self.idx}'
        data_parm = {
            'predict_n': [10, 20, 30, 40, 50, 100],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': '成交量 >= 100w',
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

            describe=f'filter extra 100w predict_n:{self.predict_n}',
            amp=self.amp
        )

    def get_in_out_shape(self):
        return (1, 44, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # T: 100, 140, 60, 1
        # D: 44, 80, 160, 3
        return m_bin_ctabl(80, 44, 100, 140, 160, 60, self.y_n, 1)

    def get_transform(self, device):
        return transform(device, self.para, 103, num_rows=44)

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

    run(
        test, 
        findbest_lr=True,
        amp='fp16',
        mode='cache_data',
        data_folder=r'/kaggle/input/lh-q-t0-data-filter-extra-100w'
    )