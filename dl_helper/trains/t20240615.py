from functools import partial

from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.binctabl import m_bin_ctabl
from ..data import data_parm2str

class trainer(trainer_base):
    """
    binctabl 模型

    使用0612新数据
    标签:
    [0] 10_target_long
    [1] 10_target_short
    [2] 10_target_mid
    [3] 10_target_5_period_point
    [4] 20_target_long
    [5] 20_target_short
    [6] 20_target_mid
    [7] 20_target_5_period_point
    [8] 30_target_long
    [9] 30_target_short
    [10] 30_target_mid
    [11] 30_target_5_period_point


    启用数据增强:
    random_scale=0.05
    random_mask_row=0.7

    batch n = 8
    batch size = 64
    lr=0.00013
    workers=3

    测试标签:
        3分类:
        [0] 10_target_long
            10_target_short
            --------------------------
            0: 10_target_long > 0
            1: 10_target_short > 0
            2: other
        
        3分类:
        [2] 10_target_mid
            --------------------------
            0: 10_target_mid >= minchange
            1: 10_target_mid <= -minchange
            2: other

        3分类:
        [3] 10_target_5_period_point
            --------------------------
            0: 10_target_mid > 0
            1: 10_target_mid < 0
            2: other
    """
    def __init__(self, idx, workers=3, debug=False):
        super().__init__(idx, debug, False, workers)

        self.minchange = {
            'BTC': 0.00001,
            'ETH': 0.0001
        }

    def init_param(self, data_folder=''):
        print('init_param')

        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        def yfunc_target_long_short(x):
            # long/ short
            x1, x2 = x
            if x1 > 0:# 多头盈利
                return 0
            elif x2 > 0:# 空头盈利
                return 1
            else:
                return 2

        def yfunc_target_mid(x):
            if x > 0:
                return 0
            elif x < 0:
                return 1
            else:
                return 2

        def yfunc_target_5_period_point(x):
            if x > 0:
                return 0
            elif x < 0:
                return 1
            else:
                return 2

        # 0 - 2
        vars = []
        for classify_idx, name, yfunc in zip(
            [[0, 1], 2, 3], 
            ['10_target_long_short', '10_target_mid', '10_target_5_period_point'],
            [yfunc_target_long_short, yfunc_target_mid, yfunc_target_5_period_point]
        ):
            vars.append((classify_idx, name, yfunc))

        assert self.idx < len(vars)

        classify_idx, targrt_name, yfunc = vars[self.idx]
        y_n = 3

        title = f'binctabl_{targrt_name}'
        data_parm = {
            'predict_n': [10, 20, 30],
            'pass_n': 105,
            'y_n': y_n,
            'begin_date': '2024-05-01',
            'data_rate': (7, 2, 3),
            'total_hours': int(24*7),
            'symbols': '@'.join(symbols),
            'target': targrt_name,
            'std_mode': '5d'  # 4h/1d/5d
        }

        batch_n = 8
        model = m_bin_ctabl(60, 40, 100, 40, 120, 10, y_n, 1)
        init_param(
            train_title=title, root=f'./{title}', model=model, data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.00013*batch_n, batch_size=64*batch_n, workers=self.workers,

            # 数据增强
            random_scale=0.05, random_mask_row=0.7,

            # 3分类
            classify=True,
            y_n=y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=data_folder,

            describe=f'target={targrt_name}'
        )
