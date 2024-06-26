import pytz, time
import os, math
import pickle
import numpy as np
import pandas as pd
import torch
import random
import datetime
from tqdm import tqdm
from collections.abc import Iterable
from pympler import asizeof
# import gc
from dl_helper.train_param import logger, data_parm2str, data_str2parm
from dl_helper.tool import report_memory_usage, check_nan

tz_beijing = pytz.timezone('Asia/Shanghai')

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    # logger.debug('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    # logger.debug('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # logger.debug('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def convert_float16_2_32(df):
    for col in df.columns:
        if df[col].dtype == 'float16':
            df[col] = df[col].astype('float32')

    return df

# 具有随机shape 影响速度，暂时弃用
# 随机选择 max_mask_num 的行数
# 按照 mask_prob 的概率进行遮盖
# tensor 为原始的数据，没有切片 目前应该shape[1] == 105
def random_mask_row_0(tensor, begin, end, mask_prob=0.5, max_mask_num=5):
    need_length = end-begin
    assert need_length+max_mask_num <= tensor.shape[1], f"need_length:{need_length} max_mask_num:{max_mask_num} tensor.shape:{tensor.shape}"

    # 实际的 begin，end
    length = tensor.shape[1]
    begin, end = length-need_length , length

    # 随机选择 max_mask_num 行
    rows = random.sample(range(need_length), max_mask_num)

    # 选择随机的行删除
    rows_mask = torch.rand(max_mask_num) < mask_prob
    del_count = torch.sum(rows_mask).item()

    # print(f"删除行数: {del_count}")
    if del_count == 0:
        # 不需要删除
        return tensor[:, begin:end, :]

    # print(f'del_count: {del_count}')
    mask = torch.zeros(need_length+del_count, dtype=torch.bool)
    mask[[i+del_count for i in rows]] = rows_mask

    # 需要删除
    # 在行起始位置补充
    begin -= del_count

    # 切片
    data = tensor[:, begin:end, :]

    # 删除行
    return data[:, ~mask, :]

# 按照 mask_prob 的概率随机遮盖时间点全部数据 -> 0
def random_mask_row(tensor, mask_prob=1e-4):
    mask = torch.rand(tensor.size()[1]) < mask_prob
    tensor[:, mask, :] = 0
    return tensor

# 定义随机遮挡函数
def random_mask(tensor, mask_prob=1e-4):
    mask = torch.rand(tensor.size()) < mask_prob
    return tensor * ~mask

# 定义随机缩放函数
# 只对vol作用
def random_scale_0(tensor, vol_cols, scale_prob=0.005, min_scale=0.97, max_scale=1.03):
    mask = torch.zeros(tensor.size(), dtype=torch.bool)
    # 只用vol_cols
    mask[:, :, vol_cols] = torch.rand(tensor.size()[0], tensor.size()[1], len(vol_cols)) < scale_prob
    
    scale_num = mask.sum().item()
    if scale_num == 0:
        return tensor

    scale = torch.rand(scale_num)*(max_scale-min_scale)+min_scale
    tensor[mask] *= scale

    return tensor

# 定义随机缩放函数
# 只对vol作用
def random_scale_1(tensor, vol_cols, scale_prob=0.005, min_scale=0.97, max_scale=1.03):
    # 矩阵算法
    mask = torch.zeros(tensor.size(), dtype=torch.bool)
    # 只用vol_cols
    mask[:, :, vol_cols] = torch.rand(tensor.size()[0], tensor.size()[1], len(vol_cols)) < scale_prob
    scale_pct_change = torch.rand(tensor.size())*(max_scale-min_scale)+min_scale-1# 变化率
    scale_pct_change *= mask

    tensor *= (1 + scale_pct_change)
    return tensor

# 定义随机缩放函数
# 只对vol作用
def random_scale(tensor, vol_cols, scale_prob=0.005, min_scale=0.97, max_scale=1.03):
    # 矩阵算法
    mask = torch.zeros(tensor.size(), dtype=torch.bool)
    
    # 只用vol_cols
    mask[:, :, vol_cols] = torch.where(torch.rand(tensor.size()[0], tensor.size()[1], len(vol_cols)) < scale_prob, True, False)

    # 只用vol_cols
    # mask[:, :, vol_cols] = torch.rand(tensor.size()[0], tensor.size()[1], len(vol_cols)) < scale_prob
    # scale_pct_change = torch.rand(tensor.size())*(max_scale-min_scale)+min_scale-1# 变化率
    # scale_pct_change *= mask

    # tensor *= (1 + scale_pct_change)
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
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        if not state_dict:
            return self

        self.__dict__.update(state_dict)

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

    def __init__(self, params, data_map, need_split_data_set, classify=False, train=True, log=False):
        """Initialization"""
        self.log = log
        self.params = params

        # 原始数据
        # report_memory_usage()

        # 区分价量列
        self.price_cols = [i*2 for i in range(20)]
        self.vol_cols = [i*2+1 for i in range(20)]

        if not need_split_data_set:
            # if self.log:
            #     logger.debug("使用全部数据")
            if params.use_trade:
                self.price_cols += [42, 45]
                self.vol_cols += [40, 41, 43, 44]

        else:
            # 使用部分截取
            if (params.use_pk and params.use_trade):
                raise Exception('no use')

            elif params.use_pk:
                # if self.log:
                #     logger.debug("只使用盘口数据")
                pass

            elif params.use_trade:
                # if self.log:
                #     logger.debug("只使用交易数据")
                self.price_cols = [2, 5]
                self.vol_cols = [0, 1, 3, 4]

        self.data = torch.from_numpy(data_map['raw'].values)
        del data_map['raw']

        self.mean_std = data_map['mean_std']
        del data_map['mean_std']

        # logger.debug('del data_map > raw / mean_std')
        # report_memory_usage()

        self.data = torch.unsqueeze(self.data, 0)  # 增加一个通道维度

        # 训练数据集
        self.train = train

        if classify:
            # 分类训练集 数据平衡
            if train:
                labels = set(data_map['y'])
                sy = pd.Series(data_map['y'])
                min_num = sy.value_counts().min()

                # if self.log:
                #     logger.debug(f'min_num: {min_num}')

                # report_memory_usage()

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

                # if self.log:
                #     logger.debug(f'reindex')

                data_map['ids'] = [data_map['ids'][i] for i in idx] if data_map['ids'] else data_map['ids']
                data_map['x'] = [data_map['x'][i] for i in idx]
                data_map['y'] = [data_map['y'][i] for i in idx]
                self.mean_std = [self.mean_std[i] for i in idx]
        # else:
        #     # 回归数据集
        #     # y 可能为nan
        #     idxs = [i for i in range(len(data_map['y'])) if not np.isnan(data_map['y'][i])]
        #     # 过滤nan
        #     data_map['y'] = [data_map['y'][i] for i in idxs]
        #     data_map['x'] = [data_map['x'][i] for i in idxs]
        #     self.mean_std = [self.mean_std[i] for i in idxs]
        #     data_map['ids'] = [data_map['ids'][i] for i in idxs] if data_map['ids'] else ids

        # report_memory_usage()

        # pred_5_pass_40_y_1_bd_2024-04-08_dr_8@2@2_th_72_s_2_t_samepaper.7z
        self.time_length = int(params.data_set.split('_')[3])

        # id
        self.ids = data_map['ids']
        del data_map['ids']
        
        # 数据长度
        self.length = len(data_map['x'])

        # x 切片索引
        self.x_idx = data_map['x']
        del data_map['x']
        # x = torch.tensor(np.array(x), dtype=torch.float)

        # 最大musk时间个数
        self.max_mask_num = 5

        # y
        # 标签onehot 编码
        # self.y = torch.tensor(pd.get_dummies(np.array(y)).values, dtype=torch.int64)
        self.y = torch.tensor(np.array(data_map['y']), dtype=torch.int64 if params.classify else torch.float)
        
        data_map.clear()

        self.need_split_data_set = need_split_data_set

        self.input_shape = self.__getitem__(0)[0].shape
        # 增加一个batch维度
        self.input_shape = (1,) + self.input_shape

        # if self.log:
        #     logger.debug(f'数据集初始化完毕')
        # report_memory_usage()

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        # #############################
        # # 测试用
        # #############################
        # return torch.randn(40, 100), 0
        # #############################
        # #############################
        
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

        #############################
        # 1. 部分截取
        #############################
        if self.need_split_data_set:
            # 使用部分截取 xa, xb
            xa, xb = 0, 46
            if self.params.use_pk and self.params.use_trade:
                raise Exception('no use')
            elif self.params.use_pk:
                xb = 40
            elif self.params.use_trade:
                xa = 40

            # 截取mean_std
            mean_std = torch.tensor(self.mean_std[index][xa:xb], dtype=torch.float)
        else:
            mean_std = torch.tensor(self.mean_std[index], dtype=torch.float)

        #############################
        # 2. 全部使用，在读取数据的部分作截取操作
        #############################

        # 获取切片x
        x = self.data[:, a:b, :].clone()
        if self.train and self.params.random_mask_row>0:
            # 随机遮蔽行
            x = random_mask_row(x, self.params.random_mask_row)
        check_nan(x)

        #############################
        #############################
        # mid_price
        mid = (x[0, -1, 0] + x[0, -1, 2]) / 2

        # 价格标准化
        x[0, :, self.price_cols] /= mid
        x[0, :, :] -= mean_std[:, 0]
        x[0, :, :] /= mean_std[:, 1]

        # 随机缩放
        if self.train and self.params.random_scale>0:
            x = random_scale(x, self.vol_cols, self.params.random_scale)
        check_nan(x)

        # 随机mask
        if self.train and self.params.random_mask>0:
            x = random_mask(x, self.params.random_mask)
        check_nan(x)

        # #############################
        # # 测试用
        # #############################
        # # x = self.data[:, a:b, :].clone()
        # x = self.data[:, a:b, :]
        # #############################
        # #############################

        # return x, (self.y[index], self.ids[index])
        if self.params.cnn:
            # x:[channel, pass_n, feature] -> [1, 100, 40/6/46]
            return x, self.y[index]
        else:
            # x:[feature, pass_n] -> [40/6/46, 100]
            return x[0].permute(1, 0), self.y[index]

def re_blance_sample(ids, price_mean_std, test_x, test_y, test_raw):

    # 索引数组
    idx = np.arange(len(test_x))
    need_reindex = False

    # 标签平衡
    # logger.debug('标签平衡')
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

class cache():
    def __init__(self, obj, name):
        self.file = f'cache_{name}.pkl'
        self.data = obj
        self.cost = 0
    def cache(self):
        t0 = time.time()
        pickle.dump(self.data, open(self.file, 'wb'))
        del self.data
        self.data = None
        self.cost = time.time() - t0
    def load(self):
        t0 = time.time()
        self.data = pickle.load(open(self.file, 'rb'))
        self.cost = time.time() - t0


def load_data(params, file, diff_length, data_map, log=False):
    # report_memory_usage('begin')

    ids,mean_std, x, y, raw = pickle.load(open(file, 'rb'))

    # 每3个降采样
    # 更长时间范围的样本数据
    if params.down_freq > 1:
        idxs = [i for i in range(0, len(x), 3)]
        ids = [ids[i] for i in idxs]
        mean_std = [mean_std[i] for i in idxs]
        x = [x[i] for i in idxs]
        y = [y[i] for i in idxs]

    # TODO a股数据内存占用过大
    # ids,mean_std, x, y, raw   : load 占用的内存：5.290GB
    # ids,mean_std, x, _, _     : load 占用的内存：2.596GB
    # ids,_, _, _, _            ：load 占用的内存：1.635GB
    # _,_, _, _, _              ：load 占用的内存：1.406GB
    # _,_, _, _, _ = pickle.load(open(file, 'rb'))
    # gc.collect()

    # total_size = (asizeof.asizeof((ids,mean_std, x, y)) + raw.memory_usage(index=True, deep=True).sum())  / (1024**3)

    length = 0

    # 判断是否需要截取操作
    # raw数据feature == 46: 包含 深度/交易 数据，则需要根据 params.use_pk/params.use_trade 对数据作截取操作
    # feature == 46 且 params.use_pk/params.use_trade 不全为true
    need_split_data_set = len(mean_std[0]) == 46 and not (params.use_pk and params.use_trade)
    # if log:
    #     logger.debug(f'need_split_data_set: {need_split_data_set}')

    # 校正参数
    if len(mean_std[0]) == 40:
        params.use_pk = True
        params.use_trade = False

    ###################################################
    # 1. 不做截取操作 在dataset中截取
    ###################################################
    if not need_split_data_set:
        raw = reduce_mem_usage(raw)
        data_map['raw'] = pd.concat([data_map['raw'], raw], axis=0, ignore_index=True)
        length = len(raw)
        # report_memory_usage('concat raw')

    data_map['mean_std'] += mean_std
    # report_memory_usage('concat mean_std')
    ###################################################
    # 2. 截取操作
    ###################################################
    if need_split_data_set:
        xa, xb = 0, 46
        if params.use_pk and params.use_trade:
            raise Exception('no use')
        elif params.use_pk:
            xb = 40
        elif params.use_trade:
            xa = 40

        raw2 = raw.iloc[:, xa:xb].copy()
        raw2 = reduce_mem_usage(raw2)
        data_map['raw']  = pd.concat([data_map['raw'], raw2], axis=0, ignore_index=True)
        length = len(raw2)

    ###################################################
    ###################################################

    # 预处理标签
    y_idx = -1
    if params.regress_y_idx != -1:
        # if log:
        #     logger.debug(f"回归标签列表处理 使用标签idx:{params.regress_y_idx}")
        y_idx = params.regress_y_idx
        
    elif params.classify_y_idx!=1:
        # if log:
        #     logger.debug(f"分类标签列表处理 使用标签idx:{params.classify_y_idx}")
        y_idx = params.classify_y_idx

    # 多个可迭代
    if isinstance(y_idx, Iterable):
        y = [[i[j] for j in y_idx] for i in y]
    else:
        y = [i[y_idx] for i in y]

    # 预处理
    if not params.y_func is None:
        y = [params.y_func(i) for i in y]

    data_map['ids'] += ids
    data_map['y'] += y
    data_map['x'] += [(i[0] + diff_length, i[1] + diff_length) for i in x]
    diff_length += length
    # report_memory_usage('concat other')

    return diff_length, need_split_data_set

def read_data(_type, params, max_num=10000, head_n=0, pct=100, need_id=False, log=False, data_sample_getter_func=None):
    data_path = params.data_folder

    # 数据集参数
    target_parm = params.data_parm

    # 获取数据分段
    files = []
    data_set_files = sorted([i for i in os.listdir(data_path)])

    # 判断数据名类型
    _type_in_dataname = False
    for file in data_set_files:
        if _type in file:
            _type_in_dataname = True
            break

    if _type_in_dataname:
        # 按照数据类型读取数据集
        for file in data_set_files:
            if _type in file:
                files.append(file)
        files.sort()
    else:
        # 按照日期读取回归数据集
        begin_date = ''
        totals = 0
        if len(data_set_files[0]) == 12:
            # a股数据集 20240313.pkl
            begin_date = target_parm['begin_date'].replace('-', '') + '.pkl'
            totals = target_parm['total_hours'] // 24
        else:
            # 数字货币数据集 20240427_10.pkl
            begin_date = target_parm['begin_date'].replace('-', '') + '_00' + '.pkl'
            totals = target_parm['total_hours'] // 2

        files = data_set_files[data_set_files.index(begin_date):]

        # 初始化个部分的 begin end
        _rate_sum = sum(target_parm['data_rate'])
        idx = 0 if _type=='train' else 1 if _type=='val' else 2

        # 起始索引，以begin_date为0索引
        begin_idx = 0
        for i in range(idx):
            begin_idx += int(totals * (target_parm['data_rate'][i] / _rate_sum))
        end_idx = int(totals * (target_parm['data_rate'][idx] / _rate_sum)) + begin_idx

        files = files[begin_idx:end_idx]

    # if log:
    #     logger.debug(f'{files}')

    # 读取分段合并
    diff_length = 0
    count = 0
    need_split_data_set = False

    # ids, mean_std, x, y, raw = [], [], [], [], pd.DataFrame()
    # data_map['ids'] = []
    # data_map['mean_std'] = []
    # data_map['x'] = []
    # data_map['y'] = []
    # data_map['raw'] = pd.DataFrame()
    # 最终数据
    data_map = {
        'ids': [],
        'mean_std': [],
        'x': [],
        'y': [],
        'raw': pd.DataFrame()
    }

    for file in files:
        count += 1
        if count > max_num:
            break

        diff_length, need_split_data_set = load_data(params, os.path.join(data_path, file), diff_length, data_map)
        # report_memory_usage()

    if head_n == 0 and pct < 100 and pct > 0:
        head_n = int(len(x) * (pct / 100))

    if head_n > 0:
        # if log:
        #     logger.debug(f"控制样本数量 -> {head_n} / {len(x)}")
        data_map['raw'] = data_map['raw'].iloc[:head_n, :]
        to_del_idx = [i for i in range(len(data_map['x'])) if data_map['x'][i][-1] > head_n]

        data_map['x'] = [data_map['x'][i] for i in range(len(data_map['x'])) if i not in to_del_idx]
        data_map['y'] = [data_map['y'][i] for i in range(len(data_map['y'])) if i not in to_del_idx]
        data_map['mean_std'] = [data_map['mean_std'][i] for i in range(len(data_map['mean_std'])) if i not in to_del_idx]
        data_map['ids'] = [data_map['ids'][i] for i in range(len(data_map['ids'])) if i not in to_del_idx]

    if not need_id:
        data_map['ids'].clear()

    # if log:
    #     logger.debug(f"恢复成 float32")
    data_map['raw'] = convert_float16_2_32(data_map['raw'])
    # report_memory_usage()

    # 检查数值异常
    assert data_map['raw'].isna().any().any()==False and np.isinf(data_map['raw']).any().any()==False, '数值异常'
    
    # # fake
    # num_classes = 3
    # num_samples = 272955
    # data = torch.randn(num_samples, 40, 100)
    # # data = torch.randn(num_samples, 3, 64, 64)
    # target = torch.randint(0, num_classes, (num_samples,))
    # dataset_test = torch.utils.data.TensorDataset(data, target)
    dataset_test = Dataset(params, data_map, need_split_data_set, params.classify, train=_type == 'train', log=log)
    # if log:
    #     if params.classify:
    #         logger.debug(f'\n标签分布\n{pd.Series(dataset_test.y).value_counts()}')
    #     else:
    #         try:
    #             logger.debug(f'\n标签分布\n{pd.Series(dataset_test.y).describe()}')
    #         except:
    #             _df = pd.DataFrame(dataset_test.y)
    #             for col in list(_df):
    #                 logger.debug(f'\n标签 {col} 分布\n{_df[col].describe()}')

    train_sampler = None
    if not None is data_sample_getter_func:
        train_sampler = data_sample_getter_func(dataset_test)

    data_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=params.batch_size, sampler=train_sampler, num_workers=params.workers, pin_memory=True if params.workers>0 else False,drop_last=True)
    del dataset_test

    return data_loader

if __name__ == "__main__":
    data = read_data(r'D:\code\featrue_data\notebook\20240413_滚动标准化', 'test')
    print(len(data))