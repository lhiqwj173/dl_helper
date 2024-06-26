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
    
    workers=4
    23112/23112 [1:05:32<00:00,  5.88it/s]
    
    workers=3
    23112/23112 [1:09:27<00:00,  5.55it/s]
    3759 /28391 [20:05<1:11:33,  5.74it/s] 7.5
   *23031/30505 [1:11:29<21:58,  5.67it/s] 8

    workers=2
    23112/23112 [1:15:59<00:00,  5.07it/s]

    测试标签:
        双回归:
        [0] 10_target_long
            10_target_short
        
        单回归:
        [2] 10_target_mid

        单回归:
        [3] 10_target_5_period_point
    """
    def __init__(self, idx, workers=3, debug=False):
        super().__init__(idx, debug, False, workers)

    def init_param(self, data_folder=''):
        print('init_param')

        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        # 0 - 5
        vars = []
        for regress_idx, name, y_n in zip(
            [[0, 1], 2, 3], 
            ['10_target_long_short', '10_target_mid', '10_target_5_period_point'],
            [2, 1, 1]
        ):
            for idx in range(2):
                vars.append((regress_idx, name, y_n, idx))

        assert self.idx < len(vars)

        regress_idx, targrt_name, y_n, idx = None, None, None, None
        if self.idx == -1:
            # 测试半精度训练
            regress_idx, targrt_name, y_n, idx = vars[0]
        else:
            regress_idx, targrt_name, y_n, idx = vars[self.idx]

        title = f'binctabl_{targrt_name}_v{idx}' if self.workers==3 else f'binctabl_{targrt_name}_v{idx}_w{self.workers}'
        data_parm = {
            'predict_n': [10, 20, 30],
            'pass_n': 100,
            'y_n': y_n,
            'begin_date': '2024-05-01',
            'data_rate': (7, 2, 3),
            'total_hours': int(24),
            # 'total_hours': int(24*7),
            'symbols': '@'.join(symbols),
            'target': targrt_name,
            'std_mode': '5d'  # 4h/1d/5d
        }

        batch_n = 8
        model = m_bin_ctabl(60, 40, 100, 40, 120, 10, y_n, 1)
        init_param(
            train_title=title, root=f'./{title}', model=model, data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.00013*batch_n, batch_size=64*batch_n, workers=self.workers,

            # 使用半精度
            amp = self.idx==-1,

            # 数据增强
            random_scale=0.05, random_mask_row=0.7,

            # 3分类
            y_n=y_n, regress_y_idx=regress_idx,

            data_folder=data_folder,

            describe=f'target={targrt_name}'
        )
