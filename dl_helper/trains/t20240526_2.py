from functools import partial

from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.binctabl import m_bin_ctabl
from ..data import data_parm2str

class trainer(trainer_base):
    """
    使用0501新数据
    启用数据增强:
    random_scale=0.01
    random_mask_row=0.5

    batch size = 64
    lr=0.0001
    alpha = 0.5
    
    试验结果作为基线性能
    predict_n = 10
    """
    def __init__(self, idx, debug=False):
        super().__init__(idx, debug, False)

    def init_param(self, data_folder=''):
        print('init_param')

        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        y_n = 3
        title = f'binctabl_10d_2w_v{self.idx}'
        data_parm = {
            'predict_n': [10, 20, 30, 40, 50, 60],
            'pass_n': 100,
            'y_n': 1,
            'begin_date': '2024-04-27',
            'data_rate': (9, 1, 2),
            'total_hours': int(24*11),
            'symbols': '@'.join(symbols),
            'target': 'same paper',
            'std_mode': '1d'  # 4h/1d/5d
        }

        model = m_bin_ctabl(60, 40, 100, 40, 120, 10, 3, 1)
        init_param(
            train_title=title, root=f'./{title}', model=model, data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.0001, batch_size=64, 

            # 数据增强
            random_scale=0.01, random_mask_row=0.5,

            # 3分类
            y_n=y_n, classify_y_idx=0,classify_func=lambda x:0 if x>0 else 1 if x<0 else 2,

            data_folder=data_folder,

            describe='binctabl'
        )
