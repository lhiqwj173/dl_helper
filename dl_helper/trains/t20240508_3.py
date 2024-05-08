from functools import partial

from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.mobilenet import m_mobilenet_v2, m_mobilenet
from ..models.convnext import m_convnext
from ..models.moderntcn import m_moderntcn


class trainer(trainer_base):
    """
    使用全部数据
    增加训练数据 24*7
    
    实验变量: 
        m_mobilenet_v2(0.4)/m_mobilenet/m_convnext(0.3)/m_moderntcn(D=16, num_layers=4) 模型 
    """
    def init_param(self):
        print('init_param')

        var_list = [
            partial(m_convnext, width_ratio=0.3), 
            m_mobilenet, 
            partial(m_mobilenet_v2, alpha=0.4), 
            partial(m_moderntcn, M=46, L=100, D=16, num_layers=4, dropout=0)
        ]
        assert self.idx < len(var_list)

        title = f'more_data_v{self.idx}'

        data_parm = {
            'predict_n': 5,
            'pass_n': 70,
            'y_n': 3,
            'begin_date': '2024-04-08',
            'data_rate': (8, 2, 2),
            'total_hours': int(24*7),
            'symbols': 2,
            'taget': 'same paper'
        }

        model = var_list[self.idx](data_parm['y_n'])
        init_param(
            title,
            f'./{title}',
            100,
            640,
            0.001,
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
            f'more data model={model.model_name()}'
        )