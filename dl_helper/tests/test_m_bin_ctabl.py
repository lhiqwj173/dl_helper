import torch

from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl

def yfunc_target_long_short(x):
    # long/ short
    x1, x2 = x
    if x1 > 0:# 多头盈利
        return 0
    elif x2 > 0:# 空头盈利
        return 1
    else:
        return 2

def yfunc_target_simple(x):
    if x > 0:
        return 0
    elif x < 0:
        return 1
    else:
        return 2

class test(test_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        vars = []
        for classify_idx, name, yfunc in zip(
            [[0, 1], 2, 3], 
            ['10_target_long_short', '10_target_mid', '10_target_5_period_point'],
            [yfunc_target_long_short, yfunc_target_simple, yfunc_target_simple]
        ):
            vars.append((classify_idx, name, yfunc))

        classify_idx, targrt_name, yfunc = vars[2]
        self.y_n = 3

        batch_n = 16 * 8
        title = f'binctabl_{targrt_name}_v{self.idx}'
        data_parm = {
            'predict_n': [10, 20, 30],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (7, 2, 3),
            'total_hours': int(12),
            'symbols': '@'.join(symbols),
            'target': targrt_name,
            'std_mode': '5d'  # 4h/1d/5d
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.00013*batch_n, batch_size=64*batch_n, epochs=10,

            # # 数据增强
            # random_scale=0.05, random_mask_row=0.7,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=self.data_folder,

            describe=f'target={targrt_name}',
            amp=self.amp
        )

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        return m_bin_ctabl(60, 40, 100, 40, 120, 10, self.y_n, 1)


    # 初始化数据
    # 返回一个 torch dataloader
    def get_data(self, _type, params, data_sample_getter_func=None):

        # 创建模拟数据
        num_classes = 3

        # for debug
        num_samples = 16384

        data = torch.randn(num_samples, 40, 100)
        target = torch.randint(0, num_classes, (num_samples,))
        dataset = torch.utils.data.TensorDataset(data, target)

        train_sampler = None
        if not None is data_sample_getter_func:
            train_sampler = data_sample_getter_func(dataset, _type)

        # 创建数据加载器
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            drop_last=True,
            sampler=train_sampler,
            shuffle=False if not None is train_sampler else True if _type == 'train' else False,
        )

        return loader
