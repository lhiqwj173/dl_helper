from functools import partial

from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.mobilenet import m_mobilenet_v2

class trainer(trainer_base):
    """
    使用全部数据 24*3

    batch size = 1024
    lr=0.0016
    alpha = 0.5
    
    实验变量: 
        stem_alpha/alpha
        [(1.3, 0.7), (1.4, 0.8), (1.5, 0.9)]

    """
    def init_param(self):
        print('init_param')

        var_list = [(1.3, 0.7), (1.4, 0.8), (1.5, 0.9)]
        assert self.idx < len(var_list)

        title = f'mobilenet_v2_2alpha2_v{self.idx}'

        data_parm = {
            'predict_n': 5,
            'pass_n': 70,
            'y_n': 3,
            'begin_date': '2024-04-08',
            'data_rate': (8, 2, 2),
            'total_hours': int(24*3),
            'symbols': 2,
            'taget': 'same paper'
        }

        stem_alpha, alpha = var_list[self.idx]
        model = m_mobilenet_v2(data_parm['y_n'], alpha=alpha, stem_alpha=stem_alpha)
        init_param(
            title,
            f'./{title}',
            100,
            1024,
            0.0016,
            3,
            15,
            0,
            0,
            False,
            0.1,
            0.01,
            0,
            0,
            f'{self.data_parm2str(data_parm)}.7z',
            model,
            f'2alpha={"/".join([str(i) for i in var_list[self.idx]])}'
        )