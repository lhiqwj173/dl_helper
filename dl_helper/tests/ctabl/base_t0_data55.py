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
from dl_helper.tool import model_params_num, cal_symbol_y_idx_thresholds

from py_ext.tool import log, init_logger
init_logger('base', level='INFO')

"""
稳健市场深度表示法
lh_q_t0_depth_clear_keep_dup
100 * 21

predict_n 100
标签 4
依据市值top 20/10/5选股

标准化
量: d / mid_vol 
中间价: (d / mid_price) / 0.001

batch_size=128

过滤时间 09:30 - 14:57

测试 按照标的读取阈值，训练模型
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


def yfunc(threshold_data, y_len, y):
    if y_len == 3:
        a, b = threshold_data

        if y > b:
            return 0
        elif y < a:
            return 1
        else:
            return 2

    elif y_len == 2:
        if y > threshold_data[0]:
            return 0
        else:
            return 1
    raise Exception('y_len 必须为 2 或 3')

class test(test_base):

    @classmethod
    def title_base(cls):
        return f'train_depth_each_symbol'

    def __init__(self, *args, target_type=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.y_n = 3

        # 读取训练数据类别阈值
        # 513050 1 (-0.1666666666666483, 0.11666666666676484) 1.445
        # 518880 1 (-0.20000000000042206, 0.20000000000042206) 0.976
        
        # 513050 2 (-0.3333333333332966, 0.2500000000000835) 0.835
        # 513180 2 (-0.150000000000039, 0.08333333333332416) 0.92
        # 518880 2 (-0.36666666666551606, 0.3666666666672924) 0.936
        # 513050 3 (-0.4449999999999177, 0.34999999999996145) 0.938
        # 513180 3 (-0.25999999999998247, 0.170000000000059) 1.036
        # 513330 3 (-0.1100000000000545, 0.04499999999996174) 1.07
        # 518880 3 (-0.4949999999999122, 0.4949999999999122) 1.007
        thresholds = cal_symbol_y_idx_thresholds(os.path.join(self.data_folder, 'train'), self.y_n)

        vars = []
        classify_idx = 0
        for predict_n in [3, 30, 60, 100]:
            # 检查 classify_idx 是否存在阈值
            if classify_idx in thresholds:
                for code in [
                    '513050',
                    '513330',
                    '518880',
                    '159941',
                    '513180'
                ]:

                    # 检查 code 是否存在阈值
                    if code not in thresholds[classify_idx]:
                        continue
                    
                    # 同一个训练使用4个随机种子，最终取均值
                    for seed in range(4):
                        vars.append((predict_n, classify_idx, seed, code, thresholds[classify_idx][code]))
            classify_idx+=1

        predict_n, classify_idx, seed, code, threshold_data = vars[self.idx]

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
            'std_mode': '中间标准化'
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            batch_size=64*2, epochs=epochs,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=functools.partial(yfunc, threshold_data, self.y_n),

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
        d1, d2, d3, d4 = [21, 20, 10, 3]
        return m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)

    def get_transform(self, device):
        return transform_stable(device, self.para, 103, num_rows=21)

if '__main__' == __name__:
    t1, t2, t3, t4 = [100, 30, 10, 1]
    d1, d2, d3, d4 = [21, 20, 10, 3]
    model = m_bin_ctabl(d2, d1, t1, t2, d3, t3, d4, t4)
    print(f"模型参数量: {model_params_num(model)}")

    input_folder = r'/kaggle/input'

    data_folder_name = os.listdir(input_folder)[0]
    data_folder = os.path.join(input_folder, data_folder_name)
    # data_folder = r'D:\L2_DATA_T0_ETF\train_data\depth_clear_keep_dup'

    run(
        test, 
        # findbest_lr=True,
        amp='fp16',
        mode='cache_data',
        data_folder=data_folder,

        # debug=True,
        # idx=0
    )