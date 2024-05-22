import pytz
import os
import pickle
import numpy as np
import pandas as pd
import torch
import random
import datetime

from .train_param import params, logger, data_parm2str, data_str2parm

tz_beijing = pytz.timezone('Asia/Shanghai')

# 随机选择 max_mask_num 的行数
# 按照 mask_prob 的概率进行遮盖
# tensor 为原始的数据，没有切片 目前应该shape[1] == 100
def random_mask_row(tensor, begin, end, mask_prob=0.5, max_mask_num=5):
    need_length = end-begin
    assert need_length+max_mask_num <= tensor.shape[1]

    # 随机选择 max_mask_num 行
    rows = random.sample(range(need_length), max_mask_num)

    # 选择随机的行删除
    rows_mask = torch.rand(max_mask_num) < mask_prob
    del_count = torch.sum(rows_mask).item()

    # print(f"删除行数: {del_count}")
    if del_count == 0:
        # 不需要删除
        return tensor[:, begin:end, :].clone()

    print(f'del_count: {del_count}')
    mask = torch.zeros(need_length+del_count, dtype=torch.bool)
    mask[[i+del_count for i in rows]] = rows_mask

    # 需要删除
    # 在行起始位置补充
    begin -= del_count

    # 切片
    data = tensor[:, begin:end, :].clone()

    # 删除行
    return data[:, ~mask, :]

# 定义随机遮挡函数
def random_mask(tensor, mask_prob=1e-4):
    mask = torch.rand(tensor.size()) < mask_prob
    tensor.masked_fill_(mask, 0)
    return tensor

# 定义随机缩放函数
def random_scale(tensor, scale_prob=0.005, min_scale=0.95, max_scale=1.05):
    mask = torch.rand(tensor.size()) < scale_prob
    scale = torch.rand(1)*(max_scale-min_scale)+min_scale
    tensor.masked_fill_(mask, tensor*scale)
    return tensor

class ResumeSample():
    """
    支持恢复的采样器
    随机/顺序采样
    """

    def __init__(self, length=0, shuffle=True):
        # 随机产生种子
        self.shuffle = shuffle
        self.step = random.randint(0, 100)
        self.seed = random.randint(0, 100)
        self.size = length
        self.idx = 0
        self.data = []
        self._loop = False

    def state_dict(self):
        return {'step': self.step, 'seed': self.seed, 'shuffle': self.shuffle, 'size': self.size, 'idx': self.idx, 'loop': self._loop}

    def load_state_dict(self, state_dict):
        self.step = state_dict['step']
        self.seed = state_dict['seed']
        self.shuffle = state_dict['shuffle']
        self.size = state_dict['size']
        self.idx = state_dict['idx']
        self._loop = state_dict['loop']

        # 使用原种子
        random.seed(self.seed)
        self._init_data()

        return self

    def _init_data(self):
        if self.shuffle:

            self.data = random.sample(range(self.size), self.size)
        else:
            self.data = list(range(self.size))

    def __iter__(self):
        if not self._loop:
            # 更新种子
            self.seed += self.step
            random.seed(self.seed)
            self._init_data()

            self.idx = 0
            self._loop = True

        return self

    def __next__(self):
        if self.idx >= self.size:
            self._loop = False
            self.idx = 0
            raise StopIteration

        v = self.data[self.idx]
        self.idx += 1
        return v

    def __len__(self):
        return self.size

class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, raw_data, x, y, mean_std, ids=[], regress_y_idx=-1, classify_y_idx=-1,classify_func=None, train=True):
        """Initialization"""

        # 原始数据
        # self.data = torch.tensor(np.array(raw_data), dtype=torch.float)
        self.data = torch.from_numpy(raw_data.values)
        self.data = torch.unsqueeze(self.data, 0)  # 增加一个通道维度

        # 针对回归数据集, y可能为一个列表
        # regress_y_idx
        if isinstance(y[0], list):
            if regress_y_idx != -1:
                y = [i[regress_y_idx] for i in y]
                # y 可能为nan
                idxs = [i for i in range(len(y)) if not np.isnan(y[i])]
                # 过滤nan
                y = [y[i] for i in idxs]
                x = [x[i] for i in idxs]
                mean_std = [mean_std[i] for i in idxs]
                ids = [ids[i] for i in idxs]
            elif classify_y_idx!=1:
                y = [i[regress_y_idx] for i in y]
                if None is classify_func:
                    raise "pls set classify_func to split class"
                y = [classify_func(i) for i in y]

                # 训练集 数据平衡
                if train:
                    labels = set(y)
                    sy = pd.Series(y)
                    min_num = sy.value_counts().min()
                    logger.debug(f'min_num: {min_num}')

                    idx = []
                    for label in labels:
                        origin_idx = sy[sy == label].index

                        if len(origin_idx) > min_num:
                            idx += np.random.choice(origin_idx,
                                                    min_num, replace=False).tolist()
                        else:
                            idx += origin_idx.tolist()

                    # 排序
                    idx.sort()

                    logger.debug(f'reindex')
                    ids = [ids[i] for i in idx]
                    x = [x[i] for i in idx]
                    y = [y[i] for i in idx]
                    mean_std = [mean_std[i] for i in idx]
            else:
                raise "regress_y_idx/classify_y_idx no set"

        # pred_5_pass_40_y_1_bd_2024-04-08_dr_8@2@2_th_72_s_2_t_samepaper.7z
        self.time_length = int(params.data_set.split('_')[3])

        # id
        self.ids = ids

        # 数据长度
        self.length = len(x)

        # x 切片索引
        self.x_idx = x
        # x = torch.tensor(np.array(x), dtype=torch.float)

        # y
        # 标签onehot 编码
        # self.y = torch.tensor(pd.get_dummies(np.array(y)).values, dtype=torch.int64)
        self.y = torch.tensor(np.array(y), dtype=torch.int64)

        # 标准化数据
        self.mean_std = mean_std

        # 区分价量列
        self.price_cols = [i*2 for i in range(20)] + [42, 45]
        self.vol_cols = [i*2+1 for i in range(20)]

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        # 切片范围
        a, b = self.x_idx[index]

        # 截断数据 b向上截取
        # a=3, b=6
        # 3, 4, 5
        # self.time_length=2
        # 4, 5
        # a -> 4
        ab_length = b-a
        if ab_length > self.time_length:
            a += (ab_length-self.time_length)

        # 获取切片
        if train and params.random_mask_row>0:
            # 随机删除行，保持行数不变
            x = random_mask_row(x, a, b, params.random_mask_row)
        else:
            x = self.data[:, a:b, :].clone()

        # 获取均值方差
        mean_std = torch.tensor(
            self.mean_std[index], dtype=torch.float)

        # mid_price
        mid = (float(x[0, -1, 0]) + float(x[0, -1, 2])) / 2

        # 价格标准化
        x[0, :, self.price_cols] /= mid
        # x[0, :, self.price_cols] -= mean_std[:, 0]
        # x[0, :, self.price_cols] /= mean_std[:, 1]
        x[0, :, :] -= mean_std[:, 0]
        x[0, :, :] /= mean_std[:, 1]

        # 随机mask
        if train and params.random_mask>0:
            x = random_mask(x, params.random_mask)

        # 随机缩放
        if train and params.random_scale>0:
            x = random_scale(x, params.random_scale)

        # return x, (self.y[index], self.ids[index])
        return x, self.y[index]

def re_blance_sample(ids, price_mean_std, test_x, test_y, test_raw):

    # 索引数组
    idx = np.arange(len(test_x))
    need_reindex = False

    # 标签平衡
    logger.debug('标签平衡')
    need_reindex = True

    labels = set(test_y)
    sy = pd.Series(test_y)
    min_num = sy.value_counts().min()

    idx = []
    for label in labels:
        origin_idx = sy[sy == label].index

        if len(origin_idx) > min_num:
            idx += np.random.choice(origin_idx, min_num,
                                    replace=False).tolist()
        else:
            idx += origin_idx.tolist()

    # 重新索引
    if need_reindex:
        ids = [ids[i] for i in idx]
        test_x = [test_x[i] for i in idx]
        test_y = [test_y[i] for i in idx]
        price_mean_std = [price_mean_std[i] for i in idx]

    return ids, price_mean_std, test_x, test_y, test_raw

def read_data(_type, reblance=False, max_num=10000, head_n=0, pct=100, need_id=False):
    # # 读取测试数据
    # price_mean_std, x, y, raw = pickle.load(open(os.path.join(data_path, f'{_type}.pkl'), 'rb'))

    data_path = params.data_folder

    # 数据集参数
    target_parm = params.data_parm

    # 获取数据分段
    files = []
    if target_parm['y_n'] > 1:
        # 分类数据集
        for file in os.listdir(data_path):
            if _type in file:
                files.append(file)
        files.sort()
    else:
        # 回归数据集
        # 起始时间
        date = target_parm['begin_date']
        dt = tz_beijing.localize(
            datetime.datetime.strptime(date, '%Y-%m-%d'))

        # 初始化个部分的 begin end
        _rate_sum = sum(target_parm['data_rate'])
        idx = 0 if _type=='train' else 1 if _type=='val' else 2

        begin_hour = 0
        for i in range(idx):
            begin_hour = int(target_parm['total_hours'] * (target_parm['data_rate'][i] / _rate_sum))
        begin_dt = dt + datetime.timedelta(hours=begin_hour)

        rate = target_parm['data_rate'][idx] / _rate_sum
        hours = int(target_parm['total_hours'] * rate)# 使用时长

        for i in range(hours // 2):
            _dt = begin_dt + datetime.timedelta(hours=i*2)
            file = f'{datetime.datetime.strftime(_dt, "%Y%m%d_%H")}.pkl'
            files.append(file)

    logger.debug(f'{files}')

    # 读取分段合并
    ids, mean_std, x, y, raw = [], [], [], [], pd.DataFrame()
    diff_length = 0
    count = 0
    for file in files:
        count += 1
        if count > max_num:
            break
        _id, _mean_std, _x, _y, _raw = pickle.load(
            open(os.path.join(data_path, file), 'rb'))
        ids += _id
        mean_std += _mean_std
        y += _y
        x += [(i[0] + diff_length, i[1] + diff_length) for i in _x]
        raw = pd.concat([raw, _raw], axis=0, ignore_index=True)
        diff_length += len(_raw)

    if head_n == 0 and pct < 100 and pct > 0:
        head_n = int(len(x) * (pct / 100))

    if head_n > 0:
        logger.debug(f"控制样本数量 -> {head_n} / {len(x)}")
        raw = raw.iloc[:head_n, :]
        to_del_idx = [i for i in range(len(x)) if x[i][-1] > head_n]

        x = [x[i] for i in range(len(x)) if i not in to_del_idx]
        y = [y[i] for i in range(len(y)) if i not in to_del_idx]
        mean_std = [mean_std[i]
                    for i in range(len(mean_std)) if i not in to_del_idx]
        ids = [ids[i] for i in range(len(ids)) if i not in to_del_idx]

    if reblance:
        logger.debug(f"样本均衡")
        ids, mean_std, x, y, raw = re_blance_sample(ids, mean_std, x, y, raw)

    logger.debug(f"nan值样本数量 {raw.isna().sum().sum()}")
    logger.debug(f'\n标签分布\n{pd.Series(y).value_counts()}')

    if not need_id:
        ids = []
    dataset_test = Dataset(raw, x, y, mean_std, ids, params.regress_y_idx, params.classify_y_idx, params.classify_func, train=_type == 'train')
    del ids, x, y, raw

    data_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=params.batch_size if not (params.amp and _type == 'train') else int(
        params.batch_size*params.amp_ratio), sampler=ResumeSample(len(dataset_test), shuffle=_type == 'train'), num_workers=params.workers, pin_memory=True if params.workers>0 else False)
    del dataset_test

    return data_loader

if __name__ == "__main__":
    data = read_data(r'D:\code\featrue_data\notebook\20240413_滚动标准化', 'test')
    print(len(data))