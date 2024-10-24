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
增加订单数据
100 * 50

predict_n 100
标签 4
依据市值top 5选股

标准化
量: d / mid_vol 

batch_size=128

过滤时间 09:30 - 14:57

测试 只使用订单数据
"""
class transform_stable(transform):

    def __call__(self, batch, train=False):
        with torch.no_grad():
            x, y, mean_std = batch

            # not cnn -> (batchsize, 50, 100)
            x = torch.transpose(x, 1, 2)

            # 中间价格 / 中间量
            mid_vol = x[:, 91, -1].unsqueeze(1).unsqueeze(1).clone()

            # 删除其他数据数据
            x = x[:, :50, :]

            ####################################
            # fix 在某个时点上所有数据都为0的情况，导致模型出现nan的bug
            # 检查每一行是否全为零
            all_zero_rows = (x == 0).all(dim=1)  # 这将返回一个布尔张量，每个元素表示对应行是否全为零
            # 找到全为零的行
            zero_rows = all_zero_rows.nonzero(as_tuple=False).squeeze()
            # 遍历这些行，并在维度 1 的 39 和 40 位置上填充为 1
            # OBC买10量, OSC卖10量 -> 1
            # 对盘口的影响最小, 同时避免模型产生nan
            if zero_rows.numel() > 0:
                for row_index in zero_rows:
                    x[row_index, 39] += 1
                    x[row_index, 40] += 1
            ####################################

            # random_mask_row
            if train and self.param.random_mask_row:
                if x.shape[2] > self.raw_time_length:
                    x = x[:, :, -self.raw_time_length:]
                x = self.random_mask_row(x)
            else:
                if x.shape[2] > self.time_length:
                    x = x[:, :, -self.time_length:]

            # 量标准化
            x[:, :, :] /= mid_vol
                        
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
        return f'train_depth_only_order'

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
        return (1, 50, 100), (1, self.y_n)

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        # SMALL
        t1, t2, t3, t4 = [100, 30, 10, 1]
        d1, d2, d3, d4 = [50, 20, 10, 3]
        return m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)

    def get_transform(self, device):
        return transform_stable(device, self.para, 103, num_rows=50)

if '__main__' == __name__:

    t1, t2, t3, t4 = [100, 30, 10, 1]
    d1, d2, d3, d4 = [50, 20, 10, 3]
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