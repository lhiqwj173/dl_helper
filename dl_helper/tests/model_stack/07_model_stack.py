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
lh_q_t0_meta_data_all

测试 基模型对 模型融合 的影响
所有模型性能显著高于随机
"""

class blank(transform):
    def __init__(self, children_num):
        self.children_num = children_num

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch

            # torch.Size([128, 1, 36]) -> torch.Size([128, 36])
            x = x.squeeze(1)

            # 取用前 self.children_num * 4 * 3 个数据
            # 4 个seed
            # 3 个类别输出
            x =  x[:, :self.children_num * 4 * 3]

            return x, y


class test(test_base):

    @classmethod
    def title_base(cls):
        return f'train_model_stack'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        vars = []
        for i in range(9):
            for seed in range(4):
                # 同一个训练使用4个随机种子，最终取均值
                vars.append((i+1, seed))

        self.children_num, seed = vars[self.idx]

        self.y_n = 3

        epochs = 30

        self.lr_scheduler_class = functools.partial(OneCycle_fast, total_iters=epochs)

        title = self.title_base() + f"_base_top{self.children_num}_seed{seed}"

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
            need_meta_output = False,
        )

    def get_in_out_shape(self):
        return (1, self.children_num*4*3), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        return m_meta(self.children_num*4*3, 3)

    def get_transform(self, device):
        return blank(self.children_num)

if '__main__' == __name__:

    # model = m_meta(3*children_num, 3)
    # print(f"模型参数量: {model_params_num(model)}")

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