from functools import partial

from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.binctabl import m_bin_ctabl
from ..data import data_parm2str

class trainer(trainer_base):
    """
    binctabl 模型

    使用0501新数据
    启用数据增强:
    random_scale=0.05
    random_mask_row=0.7

    batch n = 8
    batch size = 64
    lr=0.00013
    
    测试 predict_n [10, 30, 50] 基准
    """
    def __init__(self, idx, workers=4, debug=False):
        super().__init__(idx, debug, False, workers)

    def init_param(self, data_folder=''):
        print('init_param')

        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        vars = []
        for i in [10, 30, 50]:
            for idx in range(2):
                vars.append((i, idx))

        assert self.idx < len(vars)

        predict_n, idx = vars[self.idx]

        predict_ns = [10, 20, 30, 40, 50, 60]
        predict_idx = predict_ns.index(predict_n)

        y_n = 3
        title = f'binctabl_p{predict_n}_v{idx}'
        data_parm = {
            'predict_n': predict_ns,
            'pass_n': 100,
            'y_n': 1,
            'begin_date': '2024-04-27',
            'data_rate': (7, 2, 3),
            'total_hours': int(24*7),
            'symbols': '@'.join(symbols),
            'target': 'same paper',
            'std_mode': '1d'  # 4h/1d/5d
        }

        batch_n = 8
        model = m_bin_ctabl(60, 40, 100, 40, 120, 10, 3, 1)
        init_param(
            train_title=title, root=f'./{title}', model=model, data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.00013*batch_n, batch_size=64*batch_n, workers=self.workers,

            # 数据增强
            random_scale=0.05, random_mask_row=0.7,

            # 3分类
            y_n=y_n, classify_y_idx=predict_idx,classify_func=lambda x:0 if x>0 else 1 if x<0 else 2,

            data_folder=data_folder,

            describe=f'binctabl predict_n={predict_n}'
        )
