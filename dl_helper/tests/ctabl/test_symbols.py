import functools

from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.binctabl import transform

"""
测试 特定标的 训练
增大batchsize
"""

def yfunc_target_long_short(x):
    # long/ short
    x1, x2 = x
    if x1 > 0:# 多头盈利
        return 0
    elif x2 > 0:# 空头盈利
        return 1
    else:
        return 2

def yfunc_target_simple(ratio, x):
    if x > ratio:
        return 0
    elif x < -ratio:
        return 1
    else:
        return 2

class test(test_base):
    def __init__(self, *args, target_type=1, lr_scheduler_class='ReduceLROnPlateau', **kwargs):
        super().__init__(*args, **kwargs)

        self.lr_scheduler_class = lr_scheduler_class
        symbols_vars = [
            ['BTCFDUSD', 'BTCUSDT'],
            ['BTCUSDT'],
            ['ETHUSDT', 'BTCUSDT']
        ]
        symbols = symbols_vars[self.idx]

        vars = []
        for classify_idx, name, yfunc in zip(
            [[0, 1], 2, 3], 
            ['10_target_long_short', '10_target_mid', '10_target_5_period_point'],
            [yfunc_target_long_short, yfunc_target_simple, yfunc_target_simple]
        ):
            vars.append((classify_idx, name, yfunc))

        classify_idx, targrt_name, yfunc = vars[target_type]
        self.y_n = 3

        batch_n = 16 * 4# 7.9GB
        batch_n = 16 * 8# 7.8 - 8GB
        batch_n = 16 * (2**6)

        yfunc = functools.partial(yfunc, 4e-05)

        title = f'binctabl_symbols_{"@".join(symbols)}'
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
            learning_rate=0.0001*batch_n, batch_size=64*batch_n, epochs=200,

            # 学习率衰退延迟
            learning_rate_scheduler_patience=10,

            # 数据增强
            random_scale=0.05, random_mask_row=1,

            # 每 2 个样本取一个数据
            down_freq=2,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=self.data_folder,

            describe=f'',
            amp=self.amp
        )

    def get_in_out_shape(self):
        return (1, 40, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # 100, 40, 10, 1
        return m_bin_ctabl(60, 40, 100, 40, 120, 10, self.y_n, 1)

    def get_transform(self, device):
        return transform(device, self.para, 103)

