import functools

from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.binctabl import transform

"""
- 使用新的标签 
    价差/最小变单位 >=  1 -上涨 0
    价差/最小变单位 <= -1 -下跌 1
    else                -不变 2
- 验证数据集为训练数据集同期内随机6天的数据，在训练集中排除

随机遮蔽    -> 3
降采样      -> 1

数据增加 33d
训练:验证:测试
25:6:2
"""

def yfunc(y):
    if y >= 1:
        return 0
    elif y <= -1:
        return 1
    else:
        return 2

class test(test_base):
    def __init__(self, *args, target_type=1, lr_scheduler_class='ReduceLROnPlateau', **kwargs):
        super().__init__(*args, **kwargs)

        self.lr_scheduler_class = lr_scheduler_class
        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        classify_idx, targrt_name = 4, '10_target_mid_diff'
        self.y_n = 3

        batch_n = 16 * 2

        title = f'binctabl_base3_v{self.idx}'
        data_parm = {
            'predict_n': [10, 20, 30],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': '@'.join(symbols),
            'target': targrt_name,
            'std_mode': '5d'  # 4h/1d/5d
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.0001*batch_n, batch_size=64*batch_n, epochs=200,

            # 学习率衰退延迟
            learning_rate_scheduler_patience=10,

            # 数据增强
            random_scale=0.05, random_mask_row=1,

            # 每 1 个样本取一个数据
            down_freq=1,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=self.data_folder,

            describe=f'',
            amp=self.amp
        )

    def get_in_out_shape(self):
        return (1, 40, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # 100, 40, 10, 1
        return m_bin_ctabl(60, 40, 100, 40, 120, 10, self.y_n, 1)

    def get_transform(self, device):
        return transform(device, self.para, 103)

if '__main__' == __name__:
    # 2024-05-06
    # 2024-05-09
    # 2024-05-11
    # 2024-05-20
    # 2024-07-16
    # 2024-07-26
    import datetime
    dates = []

    # 0501 + 20
    begin_date = datetime.datetime.strptime('2024-05-01', '%Y-%m-%d')
    for i in range(20):
        dates.append(begin_date + datetime.timedelta(days=i))

    # 0716 + 11
    begin_date = datetime.datetime.strptime('2024-07-16', '%Y-%m-%d')
    for i in range(11):
        dates.append(begin_date + datetime.timedelta(days=i))

    print(f'训练数据天数: {len(dates)}')

    # 随机抽 6 天作为验证数据集
    import random
    random.shuffle(dates)
    valid_dates = dates[:6]
    valid_dates.sort()
    for date in valid_dates:
        print(date.date())