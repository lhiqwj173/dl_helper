from functools import partial

from ..model_func import trainer as trainer_base
from ..train_param import init_param
from ..models.mobilenet import m_mobilenet_v2

class trainer(trainer_base):
    """
    使用0501新数据

    batch size = 256
    lr=0.0004
    alpha = 0.5
    
    实验变量: 
        不同参数组合的数据
    """
    def init_param(self):
        print('init_param')

        var_list = []
        for day in [5, 4, 3, 2, 1]:
            for symbols in [
                ['BTCUSDT'],                # 单标的
                ['BTCFDUSD'],               # 单标的
                ['BTCUSDT', 'ETHUSDT'],     # 相同货币对 USDT
                ['BTCFDUSD', 'ETHFDUSD'],   # 相同货币对 FDUSD
                ['BTCFDUSD', 'BTCUSDT'],    # 相同交易对 BTC
                ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT'],  # 所有标的
            ]:
                var_list.append((day, symbols))

        assert self.idx < len(var_list)

        title = f'mobilenet_v2_data_v{self.idx}'

        day, symbols = var_list[self.idx]
        des_str = str(day) + '/' + '@'.join(symbols)

        data_parm = {
            'predict_n': 5,
            'pass_n': 100,
            'y_n': 3,
            'begin_date': '2024-05-01',
            'data_rate': (7, 2, 3),
            'total_hours': int(24*day),
            'symbols': '@'.join(symbols),
            'taget': 'same paper'
        }

        model = m_mobilenet_v2(data_parm['y_n'], alpha=0.5)
        init_param(
            title,
            f'./{title}',
            100,
            256,
            0.0004,
            5,
            20,
            0,
            0,
            0,
            False,
            0.1,
            0.01,
            0,
            0,
            f'{self.data_parm2str(data_parm)}.7z',
            model,
            f'data={des_str}'
        )