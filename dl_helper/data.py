import pytz, time
import os, math
import pickle
import numpy as np
import pandas as pd
import threading
import queue, copy, sys
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

class Dataset_cahce(torch.utils.data.Dataset): 
    """
    不会主动load数据
    """
    def __init__(self, params, _type, device=None, predict_output=False):
        """Initialization"""
        self.params = params
        self.type = _type# 数据类型 train/val/test
        self.device = device
        self.predict_output = predict_output

        self.use_data_id = []

        # 数据集参数
        self.target_parm = data_str2parm(params.data_set)
        self.pass_n = self.target_parm['pass_n']

        # 当前数据类型的所有可读取数据文件列表
        self.files = []
        self.read_files()

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
        print(data_path)

        if not os.path.exists(data_path):
            return

        # 针对按照文件夹（train/val/test）分配的数据集
        folder_data_path = os.path.join(data_path, self.type)
        # debug(f"folder_data_path: {folder_data_path}")
        if os.path.exists(folder_data_path):
            log(f'使用文件夹方式读取数据 {self.type}: {folder_data_path}')
            self.files = sorted([i for i in os.listdir(folder_data_path)])
            self.files = [f'{self.type}/{i}' for i in self.files]
        else:
            # data_folder 路径下没有分配的文件夹
            # 获取所有的文件
            data_set_files = sorted([i for i in os.listdir(data_path)])
            print(data_set_files)

            # 判断数据名类型
            _type_in_dataname = False
            for file in data_set_files:
                if self.type in file:
                    _type_in_dataname = True
                    break

            if _type_in_dataname:
                # 文件包含 数据类型 
                # 按照数据类型读取数据集
                for file in data_set_files:
                    if self.type in file:
                        self.files.append(file)
                self.files.sort()

            elif self.params.k_fold_k > 0:
                # k折交叉验证
                each_num = sum(self.params.k_fold_ratio)
                total_num = each_num + (self.params.k_fold_k - 1)*self.params.k_fold_ratio[2]
                
                idx = 0 if self.type=='train' else 1 if self.type=='val' else 2
                each_1 = len(data_set_files) // total_num
                diff = self.params.k_fold_idx * (self.params.k_fold_ratio[2] * each_1)
                data_length = each_1 * self.params.k_fold_ratio[idx]

                if self.type in ['train', 'val']:
                    # 创建随机数生成器对象
                    rng = random.Random(self.params.k_fold_idx)
                    # 从原始列表中随机抽取
                    data_set_files = data_set_files[diff: diff + each_1 * sum(self.params.k_fold_ratio[:2])]
                    rng.shuffle(data_set_files)
                    begin = sum(self.params.k_fold_ratio[:idx]) * each_1
                    self.files = data_set_files[begin: begin + data_length]

                else:
                    diff += sum(self.params.k_fold_ratio[:idx]) * each_1
                    self.files = data_set_files[diff: diff + data_length]

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

        # 排序
        self.files.sort()

        if self.params.test:
            log('测试模式,只使用前5个数据文件')
            self.files = self.files[:5]

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
            if i == mini_epoch-1:
                # 最后一个 mini_epoch 会读取全部文件
                file_indices = mini_epoch_file_indices
                mini_epoch_file_indices = []
            else:
                # 正常每次读取 mini_dataset_length 个文件
                file_indices = mini_epoch_file_indices[:mini_dataset_length]
                mini_epoch_file_indices = mini_epoch_file_indices[mini_dataset_length:]

            files = [self.files[i] for i in file_indices]
            # debug(f"{self.type} 读取文件 1: {files}")

            # 每个设备负责的实际数据idx，会被真实的load进内存
            each_files_num = len(files) // world_size
            each_files_num = len(files) if each_files_num == 0 else each_files_num
            assert each_files_num > 0, f'each device wait read files: 0'

            offset = each_files_num * rank if len(files) > each_files_num else 0
            # 根据偏移分片 初始化 dataset 数据，而非全部数据
            # 若为测试集,加载全部文件,在内部再进行分发
            if self.type != 'test':
                files = files[offset:offset+each_files_num] if rank != world_size-1 else files[offset:]# 最后一个rank会loadoffset后的所有数据

            log(f"{self.type} rank:{rank} 读取文件: {files}")

            data_map = self._parse_data_map(files, world_size, rank)
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
                # debug(f'{self.type} {id(self)} put retry producer_thread_stop:{self.producer_thread_stop} qsize:{self.q.qsize()}')

            if stop:
                break

            # debug(f'{self.type} put {i} mini_epoch data_map, ramin:{self.q.qsize()} full:{self.q.full()}')

        log(f"{self.type} read data done")

    def _parse_data_map(self, file_name_list, world_size, rank):
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
            diff_length = load_data(self.target_parm, self.params, os.path.join(data_path, file), diff_length, data_map, self.device)
            # report_memory_usage()

        # 数据集内部分发
        if (file_name_list == self.files and world_size >1) or (self.type == "test"):
            log('数据集内部分发设备')
            _each_length = len(data_map['ids']) // world_size
            diff = rank * _each_length
            # 最后一个rank会loadoffset后的所有数据
            data_map['ids'] = data_map['ids'][diff:_each_length + diff] if rank != world_size-1 else data_map['ids'][diff:]
            data_map['mean_std'] = data_map['mean_std'][diff:_each_length + diff] if rank != world_size-1 else data_map['mean_std'][diff:]
            data_map['x'] = data_map['x'][diff:_each_length + diff] if rank != world_size-1 else data_map['x'][diff:]
            data_map['y'] = data_map['y'][diff:_each_length + diff] if rank != world_size-1 else data_map['y'][diff:]

        # 分类训练集 数据平衡(若用于预测输出，则不做数据平衡)
        if self.params.classify and not self.predict_output:
        # 测试用
        # if self.params.classify and 0:
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

    def dup_more(self, length):
        if length > self.length:
            # 数据不足, 需要补齐
            need_num = length - self.length
            self.mean_std += self.mean_std[-need_num:]
            self.ids += self.ids[-need_num:]
            self.x_idx += self.x_idx[-need_num:]
            self.y = torch.cat((self.y, self.y[-need_num:]))


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
        # if (x == 0.0).all().item():
        #     # 记录异常
        #     log(f'[x all 0.0] index:{index} id:{self.ids[index]}')
        #     # raise ValueError

        self.use_data_id.append(self.ids[index])

        mean_std = torch.tensor(self.mean_std[index], dtype=torch.float)

        # # 全0 异常
        # assert not torch.all(mean_std == 0).item(), 'mean_std all 0.0'

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
            每个epoch会分成mini_epoch组的数据集，每组数据集长度为 mini_dataset_length, 最后一个 mini_epoch 会加载剩余的全部数据（无法被mini_dataset_length整除的部分）
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

        if self.dataset.files:
            # 验证/测试 数据暂时全部load： mini_dataset_length = len(self.dataset.files)
            _mini_dataset_length = find_nearest_mini_dataset_length(len(self.dataset.files), mini_dataset_length, self.world_size)
            # debug(f'{self.dataset.type} _mini_dataset_length:{_mini_dataset_length}')
            self.mini_dataset_length = _mini_dataset_length if dataset.type == 'train' else len(self.dataset.files)
            self.mini_dataset_length = self.mini_dataset_length if self.mini_dataset_length > 0 else len(self.dataset.files)
            # debug(f'{self.dataset.type} self.mini_dataset_length:{self.mini_dataset_length}')
            assert self.mini_dataset_length > 0, f'mini_dataset_length must > 0, get {self.mini_dataset_length}'

            self.mini_epoch = len(self.dataset.files) // self.mini_dataset_length
            mini_epoch_file_indices = self._init_mini_epoch_data()

            log(f'{self.dataset.type} mini_epoch: {self.mini_epoch}, files: {len(self.dataset.files)}, mini_dataset_length: {self.mini_dataset_length}')

            self.dataset.init_data_thread_start(mini_epoch_file_indices, self.mini_dataset_length, self.mini_epoch, self.world_size, self.rank)

    def _init_mini_epoch_data(self):
        # 初始化 数据索引
        log(f'{self.dataset.type} {self.rank} {id(self.dataset)} -> init')
        self.mini_epoch_indices_ramain = self.mini_epoch
        if self.shuffle:
            mini_epoch_file_indices = list(torch.randperm(len(self.dataset.files)))
        else:
            mini_epoch_file_indices = list(torch.arange(len(self.dataset.files)))
        return mini_epoch_file_indices

    def data_loader_close(self):
        self.mini_epoch_indices_ramain = 0
        log(f'{self.dataset.type} {id(self.dataset)} self.dataset.producer_thread_stop:{self.dataset.producer_thread_stop} -> close')
        self.dataset.init_data_thread_close()
        log(f'{self.dataset.type} {id(self.dataset)} self.dataset.producer_thread_stop:{self.dataset.producer_thread_stop}')

    def __iter__(self):
        # 如果 mini_epoch_file_indices 为0，需要重新生成，说明该epoch训练结束
        if self.mini_epoch_indices_ramain == 0:
            mini_epoch_file_indices = self._init_mini_epoch_data()
            self.dataset.init_data_thread_start(mini_epoch_file_indices, self.mini_dataset_length, self.mini_epoch, self.world_size, self.rank)

        self.mini_epoch_indices_ramain -= 1
        self.dataset.load_data()

        # 同步数据长度
        data_length = torch.tensor(len(self.dataset), device=self.accelerator.device)
        self.accelerator.wait_for_everyone()
        data_length = self.accelerator.gather_for_metrics(data_length)
        data_length = torch.max(data_length)# 同步最大值, 数据量小的一方用多余的数据补齐
        # 补齐数据
        self.dataset.dup_more(data_length)
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

    # fix 在某个时点上所有数据都为0的情况，导致模型出现nan的bug
    all_cols = list(raw)
    if 'OBC买10量' in all_cols and 'OSC卖10量' in all_cols:
        # 订单数据
        order_cols = [i for i in all_cols if i.startswith('O')]
        order_raw = raw.loc[:, order_cols]
        raw.loc[(order_raw == 0).all(axis=1), ['OBC买10量', 'OSC卖10量']] = 1
    if '卖10量' in all_cols and '卖10价' not in all_cols:
        # 深度数据
        depth_cols = ['卖10量',
            '卖9量',
            '卖8量',
            '卖7量',
            '卖6量',
            '卖5量',
            '卖4量',
            '卖3量',
            '卖2量',
            '卖1量',
            '买1量',
            '买2量',
            '买3量',
            '买4量',
            '买5量',
            '买6量',
            '买7量',
            '买8量',
            '买9量',
            '买10量']
        depth_raw = raw.loc[:, depth_cols]
        wait_fix_index = depth_raw[(depth_raw == 0).all(axis=1)].index.to_list()
        if wait_fix_index and wait_fix_index[0] == 0:
            # 若第一个数据就为0，填充 卖10量/买10量 为1，最小化影响
            raw.loc[0, '卖10量'] = 1
            raw.loc[0, '买10量'] = 1
            # 去掉第一个记录
            wait_fix_index = wait_fix_index[1:]

        raw.loc[wait_fix_index, depth_cols] = np.nan# 先用nan填充，方便后续处理
        for col in depth_cols:
            raw[col].fillna(method='ffill', inplace=True)
    if 'DB卖1量' in all_cols and 'DS买1量' in all_cols: 
        # 成交数据
        deal_cols = [i for i in all_cols if i.startswith('D')]
        deal_raw = raw.loc[:, deal_cols]
        raw.loc[(deal_raw == 0).all(axis=1), ['DB卖1量', 'DS买1量']] = 1

    # 40档位价量数据nan处理
    if raw.shape[1] in [40, 44]:
        # 价格nan填充, 使用上一个档位数据 +-0.001 进行填充
        for i in range(2, 11):
            # 买价
            raw[f'买{i}价'].fillna(raw[f'买{i-1}价'] - 0.001, inplace=True)

            # 卖价
            raw[f'卖{i}价'].fillna(raw[f'卖{i-1}价'] + 0.001, inplace=True)

        # 量nan，用0填充
        vol_cols = [i for i in list(raw)[:40] if '价' not in i]
        raw[vol_cols] = raw[vol_cols].fillna(0)

    # # 异常全0检查
    # for i, _mean_std in enumerate(mean_std):
    #     if _mean_std[0][0] == 0:
    #         raise Exception(f'{i} {_mean_std}')# 159567_1705629192 - 159567_1705645164

    reindex = False

    # 每3个降采样
    # 更长时间范围的样本数据
    if params.down_freq > 1:
        idxs = [i for i in range(0, len(x), params.down_freq)]
        reindex = True
    else:
        idxs = list(range(0, len(x)))
        
    # 过滤掉不需要的symbol
    if not None is target_parm:
        symbols = target_parm['symbols'].split('@')
        if (symbols not in [['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']]) and ('成交量 >=' not in symbols[0]) and ('成交额 >=' not in symbols[0]) and ('fi2010' != symbols[0]) and ('top' not in symbols[0]):
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

    length = 0

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

    # 预处理标签
    y_idx = -1
    if params.regress_y_idx != -1:
        # if log:
        #     logger.# debug(f"回归标签列表处理 使用标签idx:{params.regress_y_idx}")
        y_idx = params.regress_y_idx
        
    elif params.classify_y_idx!= -1:
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

    return diff_length

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

        diff_length = load_data(target_parm, params, os.path.join(data_path, file), diff_length, data_map, device)
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
    dataset_test = Dataset(params, data_map, params.classify, train=_type == 'train', log=log)
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
    # data = read_data(r'D:\code\featrue_data\notebook\20240413_滚动标准化', 'test')
    # print(len(data))

    print(find_nearest_mini_dataset_length(1, 16, 2))