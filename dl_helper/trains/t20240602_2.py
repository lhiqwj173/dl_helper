from functools import partial

from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.deeplob import m_deeplob
from ..data import data_parm2str

class trainer(trainer_base):
    """
    使用a股 t0_etf/kzz 数据
    启用数据增强:
    random_scale=0.01
    random_mask_row=0.5

    batch size = 64
    lr=0.0001
    alpha = 0.5
    """
    def __init__(self, idx, workers=2, debug=False):
        super().__init__(idx, debug, False, workers)

    def init_param(self, data_folder=''):
        print('init_param')

        vars = []
        for predict_n in [10, 30, 60]:
            for idx in range(3):
                vars.append((predict_n, idx))

        assert self.idx < len(vars)
        predict_n, idx = vars[self.idx]
        predict_ns = [10, 20, 30, 40, 50, 60]
        predict_idx = predict_ns.index(predict_n)

        y_n = 3
        title = f'a_binctabl_p{predict_n}_v{idx}'
        target_parm = {
            'predict_n': [10, 20, 30, 40, 50, 60],
            'pass_n': 105,
            'y_n': 1,
            'begin_date': '2023-12-28',
            'data_rate': (10, 2, 3),
            'total_hours': int(24*45),
            'symbols': 'T0ETF@KZZ',
            'taget': 'same paper',
            'std_mode': '5d'  # 4h/1d/3d/5d
        }

        model = m_deeplob(y_n)
        init_param(
            train_title=title, root=f'./{title}', model=model, data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.0001, batch_size=64, workers=self.workers,

            # 数据增强
            random_scale=0.01, random_mask_row=0.5,

            # 3分类
            y_n=y_n, classify_y_idx=predict_idx,classify_func=lambda x:0 if x>0 else 1 if x<0 else 2,

            data_folder=data_folder,

            describe=f'a deeplob predict_n={predict_n} idx={idx}',
        )
