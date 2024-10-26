import functools
import sys, torch, os, pickle

from accelerate.utils import set_seed

from dl_helper.tester import test_base
from dl_helper.train_param import Params
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.meta import m_meta
from dl_helper.transforms.base import transform
from dl_helper.trainer import run
from dl_helper.tool import model_params_num

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')
"""
模型融合
"""
class blank(transform):
    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch
            return x, y

class test(test_base):

    @classmethod
    def title_base(cls):
        return f'train_model_stack'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.children_num = 12

        # 同一个训练使用4个随机种子，最终取均值
        seed = range(4)[self.idx]

        self.y_n = 3

        epochs = 30

        self.lr_scheduler_class = functools.partial(OneCycle_fast, total_iters=epochs)

        title = self.title_base() + f"_predict_n100_seed{seed}"

        data_parm = {
            'predict_n': [3, 30, 60, 100],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': f'top5',
            'target': f'label 4',
            'std_mode': '-'
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*2, epochs=epochs,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=0,

            data_folder=self.data_folder,

            describe=f"predict_n100",
            amp=self.amp,
            seed=seed,
            no_better_stop=0,
        )

    def get_in_out_shape(self):
        return (1, self.children_num*3), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        return m_meta(self.children_num*3, 3)

    def get_transform(self, device):
        return blank()

if '__main__' == __name__:

    model = m_meta(9, 3)
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