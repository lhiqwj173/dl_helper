from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.binctabl import transform

"""
epochs
    100
    200 X

target_type
    1
    2 X

lr_scheduler_class
    ReduceLR_slow_loss X
    ReduceLROnPlateau 

down_freq 
    2
    1 X

pass_n
    100
    80 
    60 X
    40 X
    20 X

##########################################

pass_n
    100
    90
    80
    70
    60
    50

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

def yfunc_target_simple(x):
    if x > 0:
        return 0
    elif x < 0:
        return 1
    else:
        return 2

class test(test_base):
    def __init__(self, *args, target_type=1, lr_scheduler_class='ReduceLROnPlateau', **kwargs):
        super().__init__(*args, **kwargs)

        self.lr_scheduler_class = lr_scheduler_class
        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        vars = []
        for classify_idx, name, yfunc in zip(
            [[0, 1], 2, 3], 
            ['10_target_long_short', '10_target_mid', '10_target_5_period_point'],
            [yfunc_target_long_short, yfunc_target_simple, yfunc_target_simple]
        ):
            vars.append((classify_idx, name, yfunc))

        classify_idx, targrt_name, yfunc = vars[target_type]
        self.y_n = 3

        batch_n = 16

        model_t_vars = [
            (90, 40, 10, 1),
            (70, 40, 10, 1),
            (60, 40, 10, 1),
            (50, 40, 10, 1),
        ]
        self.model_t = model_t_vars[self.idx]

        title = f'binctabl_pass_n_v2_{self.model_t[0]}'
        data_parm = {
            'predict_n': [10, 20, 30],
            'pass_n': self.model_t[0],
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

            # 数据增强
            random_scale=0.05, random_mask_row=1,

            # 每2个样本取一个数据
            down_freq=2,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=self.data_folder,

            describe=f'model_ts={self.model_t}',
            amp=self.amp
        )

    def get_in_out_shape(self):
        return (1, 40, self.model_t[0]), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # 100, 40, 10, 1
        return m_bin_ctabl(60, 40, self.model_t[0], self.model_t[1], 120, self.model_t[2], self.y_n, self.model_t[3])

    def get_transform(self, device):
        pass_n = self.model_t[0]
        return transform(device, self.para, pass_n + int(pass_n/20))

