import functools
import sys, torch, os

from accelerate.utils import set_seed

from dl_helper.tester import test_base
from dl_helper.train_param import Params
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.binctabl import transform
from dl_helper.trainer import run

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')

"""
predict_n 100
标签 4
依据市值top 5选股

标准化
价格: (d / mid_price) / 0.001
量: d / mid_vol 

小模型
T: 100, 30, 10, 1
D: 40, 20, 10, 3

只使用订单簿价量数据 
100*40

batch_size=128

测试 过滤时间 09:30 - 14:57
"""
price_cols = [i*2 for i in range(20)]
other_cols = [i*2 + 1 for i in range(20)]

class transform_mid_pv(transform):

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch
            # debug('x', x.shape, x.device)
            # debug('y', y.shape, y.device)
            # debug('mean_std', mean_std.shape, mean_std.device)

            # not cnn -> (batchsize, 40, 100)
            x = torch.transpose(x, 1, 2)

            # random_mask_row
            if train and self.param.random_mask_row:
                if x.shape[2] > self.raw_time_length:
                    x = x[:, :, -self.raw_time_length:]
                x = self.random_mask_row(x)
            else:
                if x.shape[2] > self.time_length:
                    x = x[:, :, -self.time_length:]
            
            if x.shape[1] > self.num_rows:
                x = x[:, :self.num_rows, :]

            # 中间价格 / 中间量
            mid_price = ((x[:, 0, -1] + x[:, 2, -1]) / 2).unsqueeze(1).unsqueeze(1).clone()
            mid_vol = ((x[:, 1, -1] + x[:, 3, -1]) / 2).unsqueeze(1).unsqueeze(1).clone()

            # 价归一化
            x[:, price_cols, :] /= mid_price
            x[:, price_cols, :] /= 0.001

            # 量归一化
            x[:, other_cols, :] /= mid_vol

            if train and self.param.random_scale>0:
                x = self.random_scale(x)

            return x, y


def yfunc(threshold, y):
    if y > threshold:
        return 0
    elif y < -threshold:
        return 1
    else:
        return 2

class test(test_base):

    @classmethod
    def title_base(cls):
        return f'base_filter_time_{dataset_type}'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        classify_idx = 0
        for predict_n in [3, 30, 60, 100]:
            # 同一个训练使用4个随机种子，最终取均值
            for seed in range(4):
                vars.append((predict_n, classify_idx, seed))
            classify_idx+=1

        predict_n, classify_idx, seed = vars[self.idx]

        self.y_n = 3

        epochs = 30

        self.lr_scheduler_class = functools.partial(OneCycle_fast, total_iters=epochs)

        title = self.title_base() + f"_predict_n{predict_n}_seed{seed}"

        data_parm = {
            'predict_n': [3, 30, 60, 100],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': f'{dataset_type}',
            'target': f'label 4',
            'std_mode': '中间价量标准化'
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*2, epochs=epochs,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=functools.partial(yfunc, 0.5),

            data_folder=self.data_folder,

            describe=f"predict_n{predict_n}",
            amp=self.amp,
            seed=seed,
            no_better_stop=0,
        )

    def get_in_out_shape(self):
        return (1, 40, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # SMALL
        t1, t2, t3, t4 = [100, 30, 10, 1]
        d1, d2, d3, d4 = [40, 20, 10, self.y_n]
        return m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)

    def get_transform(self, device):
        return transform_mid_pv(device, self.para, 103, num_rows=40)

if '__main__' == __name__:

    # from dl_helper.trainer import test_train_func
    # test_train_func(r"D:\L2_DATA_T0_ETF\train_data\market_top_20\train\20240726.pkl", '159920_1721973243', test)

    input_folder = r'/kaggle/input'
    # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'

    data_folder_name = os.listdir(input_folder)[0]
    data_folder = os.path.join(input_folder, data_folder_name)

    # 按照数据集分类
    dataset_type = ''
    if 'top5' in data_folder_name:
        dataset_type = 'top5'
    elif 'top10' in data_folder_name:
        dataset_type = 'top10'
    elif 'top20' in data_folder_name:
        dataset_type = 'top20'
    else:
        raise Exception('dataset type not found')

    run(
        test, 
        # findbest_lr=True,
        amp='fp16',
        mode='cache_data',
        data_folder=data_folder,

        # debug=True,
        # idx=0
    )