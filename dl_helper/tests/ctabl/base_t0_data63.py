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
from dl_helper.tool import model_params_num, cal_symbol_y_idx_thresholds, cal_symbol_class_pct

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

测试 按照标的 训练模型,给定阈值 +-0.5
"""
class transform_of(transform):

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mm = batch

            x = x[:, -100:, :20]
            x = torch.transpose(x, 1, 2)

            # 归一化
            x -= mm[:, :20, :1]
            x /= (mm[:, :20, 1:] -  mm[:, :20, :1])

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
        return f'once_of_symbol_0.5'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_n = 3
        filter_class_pct_res = cal_symbol_class_pct(os.path.join(self.data_folder, 'train'), self.y_n)

        vars = []
        classify_idx = 0
        for predict_n in [3, 5, 10, 15, 30, 60, 100]:
            for label in ['paper', 'paper_pct', 'label_1', 'label_1_pct']:
                if predict_n in [60, 100] and label == 'paper':
                    # 检查 classify_idx 是否存在阈值
                    if classify_idx in filter_class_pct_res:
                        for code in [
                            '513050',
                            '513330',
                            '518880',
                            '159941',
                            '513180'
                        ]:

                            # 检查 code 是否存在阈值
                            if code not in filter_class_pct_res[classify_idx]:
                                continue
                            
                            # 同一个训练使用 5 个随机种子，最终取均值
                            for seed in range( 5 ):
                                vars.append((predict_n, classify_idx, seed, code))
                classify_idx+=1
        # 将 vars 反转, 从predict_n=100开始
        vars.reverse()
        vars = vars[:8*5]

        predict_n, classify_idx, seed, code = vars[self.idx]

        epochs = 30

        self.lr_scheduler_class = functools.partial(OneCycle_fast, total_iters=epochs)

        title = self.title_base() + f"_predict_n{predict_n}_{code}_seed{seed}"

        data_parm = {
            'predict_n': [3, 30, 60, 100],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (8, 3, 1),
            'total_hours': int(24*20),
            'symbols': f'{code}',# 过滤标的
            'target': f'label 4',
            'std_mode': '简单标准化'
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
    d1, d2, d3, d4 = [20, 20, 10, 3]
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