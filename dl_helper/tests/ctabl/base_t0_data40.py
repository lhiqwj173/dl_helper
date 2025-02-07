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
from dl_helper.tool import model_params_num

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')

"""
稳健市场深度表示法
增加成交数据
100 * 41

predict_n 100
标签 4
依据市值top 5选股

标准化
量: d / mid_vol 
中间价: (d / mid_price) / 0.001

batch_size=128

过滤时间 09:30 - 14:57

测试 模型复杂度
"""

class transform_stable(transform):

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch

            # not cnn -> (batchsize, 41, 100)
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
            mid_price = x[:, 40, -1].unsqueeze(1).unsqueeze(1).clone()
            mid_vol = x[:, 41, -1].unsqueeze(1).unsqueeze(1).clone()

            # 价归一化
            x[:, 40:41, :] /= mid_price
            x[:, 40:41, :] /= 0.001

            # 量标准化
            x[:, :40, :] /= mid_vol
                        
            x = x[:, :41, :]
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
        return f'train_depth_deal_model'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        classify_idx = 0
        for predict_n in [3, 30, 60, 100]:
            if predict_n == 100:
                for model_arg_idx in range(4):
                    # 同一个训练使用4个随机种子，最终取均值
                    for seed in range(4):
                        vars.append((predict_n, classify_idx, seed, model_arg_idx))
            classify_idx+=1

        predict_n, classify_idx, seed, self.model_arg_idx = vars[self.idx]

        self.y_n = 3

        epochs = 30

        self.lr_scheduler_class = functools.partial(OneCycle_fast, total_iters=epochs)

        title = self.title_base() + f"_predict_n{predict_n}_model{self.model_arg_idx}_seed{seed}"

        data_parm = {
            'predict_n': [3, 30, 60, 100],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': f'top5',
            'target': f'label 4',
            'std_mode': '中间标准化'
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*2, epochs=epochs,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=functools.partial(yfunc, 0.5),

            data_folder=self.data_folder,

            describe=f"predict_n{predict_n}_seed{seed}",
            amp=self.amp,
            seed=seed,
            no_better_stop=0,
        )

    def get_in_out_shape(self):
        return (1, 41, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # # DEFAULT
        # t1, t2, t3, t4 = [100, 30, 10, 1]
        # d1, d2, d3, d4 = [41, 20, 10, 3]

        if self.model_arg_idx == 0:
            # 模型参数量: 7598
            t1, t2, t3, t4 = [100, 30, 10, 1]
            d1, d2, d3, d4 = [41, 40, 20, 3]
        elif self.model_arg_idx == 1:
            # 模型参数量: 14518
            t1, t2, t3, t4 = [100, 60, 30, 1]
            d1, d2, d3, d4 = [41, 40, 20, 3]
        elif self.model_arg_idx == 2:
            # 模型参数量: 21618
            t1, t2, t3, t4 = [100, 60, 30, 1]
            d1, d2, d3, d4 = [41, 80, 40, 3]
        elif self.model_arg_idx == 3:
            # 模型参数量: 39758
            t1, t2, t3, t4 = [100, 100, 50, 1]
            d1, d2, d3, d4 = [41, 80, 80, 3]
        else:
            raise ValueError(f'model_arg_idx={self.model_arg_idx}')

        return m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)

    def get_transform(self, device):
        return transform_stable(device, self.para, 103, num_rows=41)

if '__main__' == __name__:

    for idx in range(4):
        if idx == 0:
            t1, t2, t3, t4 = [100, 30, 10, 1]
            d1, d2, d3, d4 = [41, 40, 20, 3]
        elif idx == 1:
            t1, t2, t3, t4 = [100, 60, 30, 1]
            d1, d2, d3, d4 = [41, 40, 20, 3]
        elif idx == 2:
            t1, t2, t3, t4 = [100, 60, 30, 1]
            d1, d2, d3, d4 = [41, 80, 40, 3]
        elif idx == 3:
            t1, t2, t3, t4 = [100, 100, 50, 1]
            d1, d2, d3, d4 = [41, 80, 80, 3]

        model = m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)
        print(f"模型参数量: {model_params_num(model)}")

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