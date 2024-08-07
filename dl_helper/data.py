import pytz, time
import os, math
import pickle
import numpy as np
import pandas as pd
import threading
import queue, copy

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import random
import datetime
from tqdm import tqdm
from collections.abc import Iterable
# from pympler import asizeof
# import gc
from dl_helper.train_param import logger, data_parm2str, data_str2parm
from dl_helper.tool import report_memory_usage, check_nan

from py_ext.tool import log, debug

# from accelerate.data_loader import DataLoaderStateMixin


tz_beijing = pytz.timezone('Asia/Shanghai')

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2

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
    # logger.# debug('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # logger.# debug('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def convert_float16_2_32(df):
    for col in df.columns:
        if df[col].dtype == 'float16':
            df[col] = df[col].astype('float32')

    return df

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

class DataLoaderDevice(DataLoader):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        
    def __iter__(self):
        for batch in super().__iter__():
            if isinstance(batch, torch.Tensor):
                yield batch.to(self.device)
            elif isinstance(batch, list):
                yield [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
            else:
                raise Exception(f'未支持的类型: {type(batch)}')

class Dataset_0(torch.utils.data.Dataset): 
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
            #     logger.# debug("使用全部数据")
            if params.use_trade:
                self.price_cols += [42, 45]
                self.vol_cols += [40, 41, 43, 44]

        else:
            # 使用部分截取
            if (params.use_pk and params.use_trade):
                raise Exception('no use')

            elif params.use_pk:
                # if self.log:
                #     logger.# debug("只使用盘口数据")
                pass

            elif params.use_trade:
                # if self.log:
                #     logger.# debug("只使用交易数据")
                self.price_cols = [2, 5]
                self.vol_cols = [0, 1, 3, 4]

        self.data = torch.from_numpy(data_map['raw'].values)
        del data_map['raw']

        self.mean_std = data_map['mean_std']
        del data_map['mean_std']

        # logger.# debug('del data_map > raw / mean_std')
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
                #     logger.# debug(f'min_num: {min_num}')

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
                #     logger.# debug(f'reindex')

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
        #     logger.# debug(f'数据集初始化完毕')
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
        if self.train and self.params.random_mask_row>0:
            # 随机遮蔽行
            x = random_mask_row(self.data[:, self.x_idx[index][0]:b, :].clone(), self.time_length)
        else:
            x = self.data[:, a:b, :].clone()

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
            x = random_scale(x, self.params.random_scale)
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

        # print(x.device)
        # raise

        # return x, (self.y[index], self.ids[index])
        if self.params.cnn:
            # x:[channel, pass_n, feature] -> [1, 100, 40/6/46]
            return x, self.y[index]
        else:
            # x:[feature, pass_n] -> [40/6/46, 100]
            return x[0].permute(1, 0), self.y[index]

class Dataset_cahce(torch.utils.data.Dataset): 
    """
    不会主动load数据
    """
    def __init__(self, params, _type, device=None):
        """Initialization"""
        self.params = params
        self.type = _type# 数据类型 train/val/test
        self.device = device

        self.use_data_id = []

        # 数据集参数
        self.target_parm = data_str2parm(params.data_set)
        self.pass_n = self.target_parm['pass_n']

        # 当前数据类型的所有可读取数据文件列表
        self.files = []
        self.read_files()

        # 读取一个文件，判断是否需要拆分数据集
        _,mean_std, _, _, _ = pickle.load(open(os.path.join(self.params.data_folder, self.files[0]), 'rb'))
        self.need_split_data_set = len(mean_std[0]) == 46 and not (params.use_pk and params.use_trade)

        # 区分价量列
        self.price_cols = [i*2 for i in range(20)]
        self.vol_cols = [i*2+1 for i in range(20)]

        if not self.need_split_data_set:
            if params.use_trade:
                self.price_cols += [42, 45]
                self.vol_cols += [40, 41, 43, 44]

        else:
            # 使用部分截取
            if (params.use_pk and params.use_trade):
                raise Exception('no use')

            elif params.use_pk:
                pass

            elif params.use_trade:
                self.price_cols = [2, 5]
                self.vol_cols = [0, 1, 3, 4]

        # pred_5_pass_40_y_1_bd_2024-04-08_dr_8@2@2_th_72_s_2_t_samepaper.7z
        self.time_length = int(params.data_set.split('_')[3])

        # 读取数据线程是否需要停止
        self.producer_thread_stop = False

        self.q = queue.Queue(maxsize=1)
        # debug(f'{self.type} Dataset_cahce init done')
        
        # # 如果是val/test，直接读取数据
        # if _type in ['val', 'test']:
        #     # 从文件中读取 data_map
        #     data_map = self._parse_data_map(self.files)
        #     # 整理初始化 data_map
        #     self._load_data_map(data_map)
        #     self.q = None
        # else:
        #     # 存放 data_mape 
        #     self.q = queue.Queue(maxsize=1)

    def read_files(self):
        # 根据数据类型 整理 待读取的数据文件列表、
        data_path = self.params.data_folder

        # 针对按照文件夹（train/val/test）分配的数据集
        folder_data_path = os.path.join(data_path, self.type)
        # debug(f"folder_data_path: {folder_data_path}")
        if os.path.exists(folder_data_path):
            log(f'使用文件夹方式读取数据 {self.type}: {folder_data_path}')
            self.files = sorted([i for i in os.listdir(folder_data_path)])
            self.files = [f'{self.type}/{i}' for i in self.files]
            return

        # data_folder 路径下没有分配的文件夹
        data_set_files = sorted([i for i in os.listdir(data_path)])

        # 判断数据名类型
        _type_in_dataname = False
        for file in data_set_files:
            if self.type in file:
                _type_in_dataname = True
                break

        if _type_in_dataname:
            # 按照数据类型读取数据集
            for file in data_set_files:
                if self.type in file:
                    self.files.append(file)
            self.files.sort()
        else:
            # 按照日期读取回归数据集
            begin_date = ''
            totals = 0
            if len(data_set_files[0]) == 12:
                # a股数据集 20240313.pkl
                begin_date = self.target_parm['begin_date'].replace('-', '') + '.pkl'
                totals = self.target_parm['total_hours'] // 24
            else:
                # 数字货币数据集 20240427_10.pkl
                begin_date = self.target_parm['begin_date'].replace('-', '') + '_00' + '.pkl'
                totals = self.target_parm['total_hours'] // 2

            self.files = data_set_files[data_set_files.index(begin_date):]

            # 初始化各个部分的 begin end
            _rate_sum = sum(self.target_parm['data_rate'])
            idx = 0 if self.type=='train' else 1 if self.type=='val' else 2

            # 起始索引，以begin_date为0索引
            begin_idx = 0
            for i in range(idx):
                begin_idx += int(totals * (self.target_parm['data_rate'][i] / _rate_sum))
            end_idx = int(totals * (self.target_parm['data_rate'][idx] / _rate_sum)) + begin_idx

            self.files = self.files[begin_idx:end_idx]

    def init_data_thread_start(self, mini_epoch_file_indices, mini_dataset_length, mini_epoch, world_size, rank):
        # debug(f'{self.type} init_data_thread_start {rank}')
        self.producer_thread_stop = False
        producer_thread = threading.Thread(target=self._init_data, args=(mini_epoch_file_indices, mini_dataset_length, mini_epoch, world_size, rank))
        producer_thread.start()

    def init_data_thread_close(self):
        # debug(f"{self.type} init_data_thread_close")
        self.producer_thread_stop = True

    def load_data(self):
        """从 队列中 加载数据"""
        data_map = self.q.get()
        # debug(f'{self.type} get mini_epoch data_map, ramin:{self.q.qsize()} full:{self.q.full()}')
        self._load_data_map(data_map)

    def _init_data(self, mini_epoch_file_indices, mini_dataset_length, mini_epoch, world_size, rank):
        """多进程 初始化数据 放在 队列中"""

        # 从 mini_epoch_file_indices 截取 mini_dataset_length 个文件序号
        # 作为本次迭代 mini_epoch 使用的文件序号
        log(f"{self.type} init_data begin")

        for i in range(mini_epoch):
            file_indices = mini_epoch_file_indices[:mini_dataset_length]
            mini_epoch_file_indices = mini_epoch_file_indices[mini_dataset_length:]
            files = [self.files[i] for i in file_indices]
            # debug(f"{self.type} 读取文件 1: {files}")

            # 每个设备负责的实际数据idx，会被真实的load进内存
            each_files_num = len(files) // world_size 
            assert each_files_num > 0, f'each device wait read files: 0'

            offset = each_files_num * rank
            # 根据偏移分片 初始化 dataset 数据，而非全部数据
            files = files[offset:offset+each_files_num]
            log(f"{self.type} rank:{rank} 读取文件: {files}")

            data_map = self._parse_data_map(files)
            # debug(f"{self.type} parse_data_map done")
            stop = False
            while 1:
                try:
                    self.q.put(data_map, timeout=5)
                    break
                except:
                    # 检查是否需要暂停
                    if self.producer_thread_stop:
                        log(f"{self.type} producer_thread_stop:{self.producer_thread_stop}")
                        stop = True
                        break
                # debug(f'{self.type} {id(self)} put retry producer_thread_stop:{self.producer_thread_stop}')

            if stop:
                break

            # debug(f'{self.type} put {i} mini_epoch data_map, ramin:{self.q.qsize()} full:{self.q.full()}')

        log(f"{self.type} read data done")

    def _parse_data_map(self, file_name_list):
        # 1.0 读取原始数据
        data_path = self.params.data_folder

        # 获取数据分段
        files = [i for i in file_name_list if i in self.files]

        # 读取分段合并
        diff_length = 0

        # 最终数据
        data_map = {
            'ids': [],
            'mean_std': [],
            'x': [],
            'y': [],
            'raw': None
        }

        for file in files:
            diff_length, _ = load_data(self.target_parm, self.params, os.path.join(data_path, file), diff_length, data_map, self.device)
            # report_memory_usage()

        # 检查数值异常
        if isinstance(data_map['raw'], pd.DataFrame):
            assert data_map['raw'].isna().any().any()==False and np.isinf(data_map['raw']).any().any()==False, '数值异常'
        else:
            has_nan = torch.isnan(data_map['raw']).any()
            has_inf = torch.isinf(data_map['raw']).any()
            assert not has_nan and not has_inf, '数值异常'

        # 2.0 数据初始化
        # data_map['data'] = torch.from_numpy(data_map['raw'].values)
        # del data_map['raw']

        # 分类训练集 数据平衡
        if self.params.classify:
            if self.type == 'train':
                labels = set(data_map['y'])
                sy = pd.Series(data_map['y'])
                min_num = sy.value_counts().min()

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

                data_map['ids'] = [data_map['ids'][i] for i in idx] if data_map['ids'] else data_map['ids']
                data_map['x'] = [data_map['x'][i] for i in idx]
                data_map['y'] = [data_map['y'][i] for i in idx]
                data_map['mean_std'] = [data_map['mean_std'][i] for i in idx]
        
        # 标签类型
        data_map['y'] = torch.tensor(np.array(data_map['y']), dtype=torch.int64 if self.params.classify else torch.float)

        return data_map

    def _load_data_map(self, data_map):
        if isinstance(data_map['raw'] , pd.DataFrame):
            self.data = torch.from_numpy(data_map['raw'].values)
        else:
            self.data = data_map['raw']
        del data_map['raw']

        self.mean_std = data_map['mean_std']
        del data_map['mean_std']

        # id
        self.ids = data_map['ids']
        del data_map['ids']
        
        # 数据长度
        self.length = len(data_map['x'])

        # x 切片索引
        self.x_idx = data_map['x']
        del data_map['x']
        # x = torch.tensor(np.array(x), dtype=torch.float)

        # y
        self.y = data_map['y']
        del data_map['y']
        
        data_map.clear()

        self.use_data_id = []

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data
        x, y, mean_std
        """
        # 切片范围
        a, b = self.x_idx[index]
        # data_pass_n = b-a
        # data_diff = self.pass_n - data_pass_n
        # if data_diff>0:
        #     a+=data_diff

        x = self.data[a:b, :]
        if (x == 0.0).all().item():
            # 记录异常
            log(f'[x all 0.0] index:{index} id:{self.ids[index]}')
            # raise ValueError

        self.use_data_id.append(self.ids[index])

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

        return x, self.y[index], mean_std

class Dataset_cahce_folder(torch.utils.data.Dataset):
    def read_files(self):
        # 文件夹读取训练/验证/测试 数据
        data_path = os.path.join(self.params.data_folder, self.type)
        self.files = sorted([i for i in os.listdir(data_path)])

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
            #     logger.# debug("使用全部数据")
            if params.use_trade:
                self.price_cols += [42, 45]
                self.vol_cols += [40, 41, 43, 44]

        else:
            # 使用部分截取
            if (params.use_pk and params.use_trade):
                raise Exception('no use')

            elif params.use_pk:
                # if self.log:
                #     logger.# debug("只使用盘口数据")
                pass

            elif params.use_trade:
                # if self.log:
                #     logger.# debug("只使用交易数据")
                self.price_cols = [2, 5]
                self.vol_cols = [0, 1, 3, 4]

        if isinstance(data_map['raw'] , pd.DataFrame):
            self.data = torch.from_numpy(data_map['raw'].values)
        else:
            self.data = data_map['raw']
        del data_map['raw']

        self.mean_std = data_map['mean_std']
        del data_map['mean_std']

        # logger.# debug('del data_map > raw / mean_std')
        # report_memory_usage()

        # 训练数据集
        self.train = train

        if classify:
            # 分类训练集 数据平衡
            if train:
                labels = set(data_map['y'])
                sy = pd.Series(data_map['y'])
                min_num = sy.value_counts().min()

                # if self.log:
                #     logger.# debug(f'min_num: {min_num}')

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
                #     logger.# debug(f'reindex')

                data_map['ids'] = [data_map['ids'][i] for i in idx] if data_map['ids'] else data_map['ids']
                data_map['x'] = [data_map['x'][i] for i in idx]
                data_map['y'] = [data_map['y'][i] for i in idx]
                self.mean_std = [self.mean_std[i] for i in idx]

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

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data
        x, y, mean_std
        """
        # 切片范围
        a, b = self.x_idx[index]
        x = self.data[a:b, :]

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

        return x, self.y[index], mean_std

def find_nearest_mini_dataset_length(a, b, world_size):
    if a % b == 0 and b%world_size == 0:
        return b

    if a % world_size != 0:
        a = (a // world_size) * world_size
    
    # 求++
    if (b > a):
        return a
    else:
        c = b
        while(a % c != 0 or c%world_size!=0):
            if c > a: 
                c = a
                break
            c+=1

    # 求--
    d = b
    while(a % d != 0 or d%world_size!=0):
        d-=1
        
    if (c-b) < (b-d):
        return c
    else:
        return d

class DistributedSampler(Sampler):
    def __init__(self, dataset, accelerator, shuffle=False, mini_dataset_length=16):
        """
        mini_dataset_length:
            每次分片加载数据集的长度
            每个epoch会分成mini_epoch组的数据集，每组数据集长度为mini_dataset_length，暂时舍弃多余不能整除的数据
        """
        # assert isinstance(dataset, Dataset_cahce), f'only support {Dataset_cahce}, get {type(dataset)}'
        self.dataset = dataset
        self.shuffle = shuffle

        self.accelerator = accelerator
        self.world_size = accelerator.num_processes
        self.rank = accelerator.process_index
        # debug(f'{self.dataset.type} begin find_nearest_mini_dataset_length:{len(self.dataset.files), mini_dataset_length, self.world_size}')

        # # for debug
        # mini_dataset_length = 2

        # 验证/测试 数据暂时全部load： mini_dataset_length = len(self.dataset.files)
        _mini_dataset_length = find_nearest_mini_dataset_length(len(self.dataset.files), mini_dataset_length, self.world_size)
        # debug(f'{self.dataset.type} _mini_dataset_length:{_mini_dataset_length}')
        self.mini_dataset_length = _mini_dataset_length  if dataset.type == 'train' else len(self.dataset.files)
        # debug(f'{self.dataset.type} self.mini_dataset_length:{self.mini_dataset_length}')
        assert self.mini_dataset_length > 0, f'mini_dataset_length must > 0, get {self.mini_dataset_length}'

        self.mini_epoch = len(self.dataset.files) // self.mini_dataset_length
        # debug(f'{self.dataset.type} self.mini_epoch:{self.mini_epoch}')
        self.mini_epoch_indices_ramain = self.mini_epoch
        if self.shuffle:
            mini_epoch_file_indices = list(torch.randperm(self.mini_epoch * self.mini_dataset_length))
        else:
            mini_epoch_file_indices = list(torch.arange(self.mini_epoch * self.mini_dataset_length))
        log(f'{self.dataset.type} mini_epoch: {self.mini_epoch}, files: {len(self.dataset.files)}, mini_dataset_length: {self.mini_dataset_length}')

        self.dataset.init_data_thread_start(mini_epoch_file_indices, self.mini_dataset_length, self.mini_epoch, self.world_size, self.rank)

    def data_loader_close(self):
        self.mini_epoch_indices_ramain = 0
        log(f'{self.dataset.type} {id(self.dataset)} self.dataset.producer_thread_stop:{self.dataset.producer_thread_stop} -> close')
        self.dataset.init_data_thread_close()
        log(f'{self.dataset.type} {id(self.dataset)} self.dataset.producer_thread_stop:{self.dataset.producer_thread_stop}')

    def __iter__(self):
        # 如果 mini_epoch_file_indices 为0，需要重新生成，说明该epoch训练结束
        if self.mini_epoch_indices_ramain == 0:
            self.mini_epoch_indices_ramain = self.mini_epoch
            if self.shuffle:
                mini_epoch_file_indices = list(torch.randperm(self.mini_epoch * self.mini_dataset_length))
                # debug(f'{self.dataset.type} new mini_epoch_file_indices')
            else:
                mini_epoch_file_indices = list(torch.arange(self.mini_epoch * self.mini_dataset_length))
            self.dataset.init_data_thread_start(mini_epoch_file_indices, self.mini_dataset_length, self.mini_epoch, self.world_size, self.rank)

        self.mini_epoch_indices_ramain -= 1
        self.dataset.load_data()

        # 同步数据长度
        # data_length = torch.tensor([len(self.dataset)], device=self.accelerator.device)
        data_length = torch.tensor(len(self.dataset), device=self.accelerator.device)
        self.accelerator.wait_for_everyone()
        data_length = self.accelerator.gather_for_metrics(data_length)
        data_length = torch.min(data_length)
        log(f'{self.dataset.type} data_length: {data_length}')

        if self.shuffle:
            indices = list(torch.randperm(data_length))
        else:
            indices = list(range(data_length))

        return iter(indices)

   
def re_blance_sample(ids, price_mean_std, test_x, test_y, test_raw):

    # 索引数组
    idx = np.arange(len(test_x))
    need_reindex = False

    # 标签平衡
    # logger.# debug('标签平衡')
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

def load_data(target_parm, params, file, diff_length, data_map, device=None, log=False):
    # report_memory_usage('begin')

    ids,mean_std, x, y, raw = pickle.load(open(file, 'rb'))

    reindex = False

    # 每3个降采样
    # 更长时间范围的样本数据
    if params.down_freq > 1:
        idxs = [i for i in range(0, len(x), params.down_freq)]
        reindex = True
    else:
        idxs = list(range(0, len(x)))
        
    # 过滤掉不需要的symbol
    symbols = target_parm['symbols'].split('@')
    if symbols not in [
        ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT'],
        ['成交量 >= 100w']
    ]:
        symbols = [i.lower() for i in symbols]
        # id: btcusdt_1710289478588
        idxs = [i for i in idxs if ids[i].split('_')[0] in symbols]
        reindex = True

    # 重新取样
    if reindex:
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
    #     logger.# debug(f'need_split_data_set: {need_split_data_set}')

    # 校正参数
    if len(mean_std[0]) == 40:
        params.use_pk = True
        params.use_trade = False

    ###################################################
    # 1. 不做截取操作 在dataset中截取
    ###################################################
    if not need_split_data_set:
        if None is device:
            raw = reduce_mem_usage(raw)
            if None is data_map['raw']:
                data_map['raw'] = raw
            else:
                data_map['raw'] = pd.concat([data_map['raw'], raw], axis=0, ignore_index=True)
        else:
            # 直接放在 device 中
            # raw = torch.from_numpy(raw.values).to(device).float()
            raw = torch.tensor(raw.values, dtype=torch.float32).to(device)
            if None is data_map['raw']:
                data_map['raw'] = raw
            else:
                data_map['raw'] = torch.cat([data_map['raw'], raw], axis=0)

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

        if None is device:
            raw2 = raw.iloc[:, xa:xb].copy()
            raw2 = reduce_mem_usage(raw2)
            if None is data_map['raw']:
                data_map['raw'] = raw2
            else:
                data_map['raw']  = pd.concat([data_map['raw'], raw2], axis=0, ignore_index=True)
        else:
            # 直接放在 device 中
            # raw2 = torch.from_numpy(raw.iloc[:, xa:xb].values).to(device)
            raw2 = torch.tensor(raw.iloc[:, xa:xb].values, dtype=torch.float32).to(device)
            if None is data_map['raw']:
                data_map['raw'] = raw2
            else:
                data_map['raw'] = torch.cat([data_map['raw'], raw2], axis=0)

        length = len(raw2)

    ###################################################
    ###################################################

    # 预处理标签
    y_idx = -1
    if params.regress_y_idx != -1:
        # if log:
        #     logger.# debug(f"回归标签列表处理 使用标签idx:{params.regress_y_idx}")
        y_idx = params.regress_y_idx
        
    elif params.classify_y_idx!=1:
        # if log:
        #     logger.# debug(f"分类标签列表处理 使用标签idx:{params.classify_y_idx}")
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

def read_data(_type, params, device=None, max_num=10000, need_id=False, log=False, data_sample_getter_func=None):
    data_path = params.data_folder

    # 数据集参数
    target_parm = data_str2parm(params.data_set)

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
    #     logger.# debug(f'{files}')

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
        'raw': None
    }

    for file in files:
        count += 1
        if count > max_num:
            break

        diff_length, need_split_data_set = load_data(target_parm, params, os.path.join(data_path, file), diff_length, data_map, device)
        # report_memory_usage()

    if not need_id:
        data_map['ids'].clear()

    if None is device:
        data_map['raw'] = convert_float16_2_32(data_map['raw'])
        # 检查数值异常
        assert data_map['raw'].isna().any().any()==False and np.isinf(data_map['raw']).any().any()==False, '数值异常'
    else:
        # 检查数值异常
        has_nan = torch.isnan(data_map['raw']).any()
        has_inf = torch.isinf(data_map['raw']).any()
        assert not has_nan and not has_inf, '数值异常'
    
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
    #         logger.# debug(f'\n标签分布\n{pd.Series(dataset_test.y).value_counts()}')
    #     else:
    #         try:
    #             logger.# debug(f'\n标签分布\n{pd.Series(dataset_test.y).describe()}')
    #         except:
    #             _df = pd.DataFrame(dataset_test.y)
    #             for col in list(_df):
    #                 logger.# debug(f'\n标签 {col} 分布\n{_df[col].describe()}')

    train_sampler = None
    if not None is data_sample_getter_func:
        train_sampler = data_sample_getter_func(dataset_test, _type)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_test, 
        batch_size=params.batch_size, 
        sampler=train_sampler, 
        shuffle=False if not None is train_sampler else True if _type == 'train' else False,
        num_workers=params.workers, 
        pin_memory=True if params.workers>0 else False,drop_last=True)
    del dataset_test

    return data_loader

if __name__ == "__main__":
    data = read_data(r'D:\code\featrue_data\notebook\20240413_滚动标准化', 'test')
    print(len(data))