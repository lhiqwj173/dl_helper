import functools
import sys, torch, os

from accelerate.utils import set_seed

from dl_helper.tester import test_base
from dl_helper.train_param import Params
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.deeplob import m_deeplob_dropout
from dl_helper.transforms.deeplob import transform
from dl_helper.trainer import run
from dl_helper.tool import model_params_num

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')

"""
predict_n 100
标签 4
依据市值top 5选股

标准化
价格: d / bid_10_price - 1
量: d / mid_vol 

只使用订单簿价量数据 lh_q_t0_base_top5_filter_time
100*40

batch_size=128

测试 m_deeplob_dropout
"""
price_cols = [i*2 for i in range(20)]
other_cols = [i*2 + 1 for i in range(20)]
class transform_bid_10_price(transform):

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch
            
            x = x[:, -100:, :40]

            # 买10价格 / 中间量
            bid_10_price = (x[:, -1, 38]).unsqueeze(1).unsqueeze(1).clone()
            mid_vol = ((x[:, -1, 1] + x[:, -1, 3]) / 2).unsqueeze(1).unsqueeze(1).clone()

            # 价归一化
            x[:, : , price_cols] /= bid_10_price
            x[:, : , price_cols] -= 1

            # 量归一化
            x[:, : , other_cols] /= mid_vol

            # 增加一个维度 [batch_size, time_length, num_rows] -> [batch_size, 1，time_length, num_rows]
            x = x.unsqueeze(1)

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
        return f'base_deeplob_dropout_top5'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        classify_idx = 0
        for predict_n in [3, 10, 30, 60, 100]:
            if predict_n == 100:
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
            'symbols': f'top5',
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
        return m_deeplob_dropout(self.y_n)

    def get_transform(self, device):
        return transform_bid_10_price()

if '__main__' == __name__:

    model = m_deeplob_dropout(3)
    print(f"模型参数量: {model_params_num(model)}")

    input_folder = r'/kaggle/input'
    # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'

    data_folder_name = os.listdir(input_folder)[0]
    data_folder = os.path.join(input_folder, data_folder_name)

    run(
        test, 
        # findbest_lr=True,
        # amp='fp16',
        mode='cache_data',
        data_folder=data_folder,

        # debug=True,
        # idx=0
    )