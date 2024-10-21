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
稳健市场深度表示法
100 * 21

predict_n 100
依据市值top5 选股

标准化
量: d / mid_vol 
中间价: (d / mid_price) / 0.001

batch_size=128

测试 全知标签 可盈利数量 win_v_long 二分类
"""

class transform_stable(transform):

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch

            # not cnn -> (batchsize, 21, 100)
            x = torch.transpose(x, 1, 2)

            # random_mask_row
            if train and self.param.random_mask_row:
                if x.shape[2] > self.raw_time_length:
                    x = x[:, :, -self.raw_time_length:]
                x = self.random_mask_row(x)
            else:
                if x.shape[2] > self.time_length:
                    x = x[:, :, -self.time_length:]

            # 中间价格 / 中间量
            mid_price = x[:, 20, -1].unsqueeze(1).unsqueeze(1).clone()
            mid_vol = x[:, 21, -1].unsqueeze(1).unsqueeze(1).clone()

            # 价归一化
            x[:, 20:21, :] /= mid_price
            x[:, 20:21, :] /= 0.001

            # 量标准化
            x[:, :20, :] /= mid_vol
                        
            x = x[:, :21, :]
            return x, y


def yfunc(threshold, y):
    if y > threshold:
        return 0
    else:
        return 1

class test(test_base):

    @classmethod
    def title_base(cls):
        return f'depth_god_label_win_long'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        classify_idx = 0
        for predict_n in [3, 30, 60, 100]:
            # 同一个训练使用4个随机种子，最终取均值
            for seed in range(4):
                vars.append((predict_n, classify_idx, seed))
            classify_idx+=2

        predict_n, classify_idx, seed = vars[self.idx]

        self.y_n = 2

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
            'symbols': 'top5',
            'target': f'god label win long',
            'std_mode': '中间标准化'
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*2, epochs=epochs,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=functools.partial(yfunc, 0),

            data_folder=self.data_folder,

            describe=f"predict_n{predict_n}_seed{seed}",
            amp=self.amp,
            seed=seed,
            no_better_stop=0,
        )

    def get_in_out_shape(self):
        return (1, 21, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # SMALL
        t1, t2, t3, t4 = [100, 30, 10, 1]
        d1, d2, d3, d4 = [21, 20, 10, self.y_n]
        return m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)

    def get_transform(self, device):
        return transform_stable(device, self.para, 103, num_rows=21)

if '__main__' == __name__:

    input_folder = r'/kaggle/input'
    # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'

    data_folder_name = os.listdir(input_folder)[0]
    data_folder = os.path.join(input_folder, data_folder_name)

    run(
        test, 
        # findbest_lr=True,
        amp='fp16',
        mode='cache_data',
        data_folder=data_folder,

        # debug=True,
        # idx=0
    )