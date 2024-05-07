from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.moderntcn import m_moderntcn

class trainer(trainer_base):
    """
    m_moderntcn 模型 

    不使用交易数据
    use_trade_data=False
    
    实验变量: 
        模型 D
        16 24 32
    """
    def init_param(self):
        print('init_param')

        D_list = [16 24 32]
        assert self.idx < len(D_list)

        title = f'moderntcn_alpha_v{self.idx}'

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
            m_moderntcn(data_parm['y_n'], 40, 70, D=D_list[self.idx], use_trade_data=False),
            f'D(var channel)={D_list[self.idx]}'
        )