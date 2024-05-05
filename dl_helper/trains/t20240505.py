from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.deeplob import m_deeplob


class trainer(trainer_base):
    """
    deeplob 模型
    实验变量: predict_n
    5, 10, 15
    """
    def init_param(self):
        print('_init_param')

        title = f'deeplob_predict_n_v{self.idx}'
        n_list = [5, 10, 15]

        data_parm = {
            'predict_n': n_list[self.idx],
            'pass_n': 70,
            'y_n': 3,
            'begin_date': '2024-04-08',
            'data_rate': (8, 2, 2),
            'total_hours': int(24*3),
            'symbols': 2,
            'taget': 'same paper'
        }

        data_parm = self.test_data()    

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
            m_deeplob(data_parm['y_n'])
        )