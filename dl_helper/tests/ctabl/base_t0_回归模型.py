import functools
import sys, torch, os

from accelerate.utils import set_seed

from dl_helper.tester import test_base
from dl_helper.train_param import Params
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.base import transform
from dl_helper.trainer import run
from dl_helper.tool import model_params_num

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')

"""
predict_n 100
标签 4
依据市值top 5选股

历史均值方差标准化

of 数据 lh_q_t0_base_top5_of
100*20

batch_size=128

模型 binctabl

测试 回归模型
"""
class transform_of(transform):

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch

            x = torch.transpose(x, 1, 2)
            
            x = x[:, :20, -100:]

            # 标准化
            x -= mean_std[:, :, :1]
            x /= mean_std[:, :, 1:]

            return x, y


class test(test_base):

    @classmethod
    def title_base(cls):
        return f'base_of_top5_bincatbl_reg'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        regress_idx = 0
        for predict_n in [3, 30, 60, 100]:
            if predict_n == 100:
                # 同一个训练使用4个随机种子，最终取均值
                for seed in range(4):
                    vars.append((predict_n, regress_idx, seed))
            regress_idx+=1

        predict_n, regress_idx, seed = vars[self.idx]

        self.y_n = 1

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
            'std_mode': '简单标准化'
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*2, epochs=epochs,

            # 回归模型
            classify=False,
            regress_y_idx=regress_idx,

            data_folder=self.data_folder,

            describe=f"predict_n{predict_n}",
            amp=self.amp,
            seed=seed,
            no_better_stop=0,
        )

    def get_in_out_shape(self):
        return (1, 20, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        t1, t2, t3, t4 = [100, 30, 10, 1]
        d1, d2, d3, d4 = [20, 20, 10, self.y_n]
        return m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)

    def get_transform(self, device):
        return transform_of()

if '__main__' == __name__:
    t1, t2, t3, t4 = [100, 30, 10, 1]
    d1, d2, d3, d4 = [20, 20, 10, 1]
    model = m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)
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