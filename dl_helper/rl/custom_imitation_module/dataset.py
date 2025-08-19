import os, gc, time, datetime, shutil
import pandas as pd
import pytz 
import pickle
import threading
import numpy as np
import psutil
from typing import List, Union, Dict, Any
import torch
from torch.utils.data import Dataset
import random, copy
from tqdm import tqdm

from dl_helper.rl.rl_env.lob_trade.lob_const import DATA_FOLDER, USE_CODES, MAX_SEC_BEFORE_CLOSE
from dl_helper.rl.rl_env.lob_trade.lob_data_helper import fix_raw_data

from dl_helper.rl.custom_imitation_module.rollout import KEYS
from dl_helper.tool import in_windows
from dl_helper.rl.rl_utils import date2days, days2date

from py_ext.tool import log
from py_ext.wechat import send_wx, wx

class IndexMapper:
    """
    高效的字典索引映射类，用于快速查找全局索引对应的键和真实索引
    
    这个类在初始化时预处理数据，构建索引映射，
    之后的查找操作可以在O(log n)时间内完成，其中n是字典中的键数量
    
    输入字典的值直接为列表长度（整数）
    """
    
    def __init__(self, data_dict):
        """
        初始化索引映射器
        
        参数:
        data_dict: 字典，其中值为整数，表示原列表长度
        """
        self.data_dict = data_dict
        self.sorted_keys = sorted(data_dict.keys())
        
        # 预计算每个键对应列表的起始索引和长度
        self.prefix_sums = []
        self.list_lengths = []
        current_sum = 0
        
        for key in self.sorted_keys:
            length = data_dict[key]  # 直接使用值作为长度
            self.prefix_sums.append(current_sum)
            self.list_lengths.append(length)
            current_sum += length
            
        self.total_length = current_sum
    
    def get_key_and_index(self, idx):
        """
        根据全局索引获取对应的键和真实索引
        
        参数:
        idx: 全局索引
        
        返回:
        (key, list_idx): 元素所在的键和在列表中的真实索引
        如果索引超出范围，返回(None, None)
        """
        # 检查索引是否在有效范围内
        if idx < 0 or idx >= self.total_length:
            return None, None
        
        # 二分查找 - O(log n)时间复杂度
        left, right = 0, len(self.sorted_keys) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            start_idx = self.prefix_sums[mid]
            end_idx = start_idx + self.list_lengths[mid] - 1
            
            if start_idx <= idx <= end_idx:
                key = self.sorted_keys[mid]
                list_idx = idx - start_idx
                return key, list_idx
            elif idx < start_idx:
                right = mid - 1
            else:  # idx > end_idx
                left = mid + 1
        
        # 正常情况下不会到达此处
        return None, None
    
    def get_total_length(self):
        """返回所有列表元素的总数"""
        return self.total_length
    
    def update_data_dict(self, new_data_dict):
        """
        更新数据字典并重新构建索引
        
        参数:
        new_data_dict: 新的数据字典，值为整数表示列表长度
        """
        self.__init__(new_data_dict)

class LobTrajectoryDataset(Dataset):
    def __init__(
        self, 
        data_folder:str='', 
        data_config={}, 
        data_dict: Dict[int, Dict[str, np.ndarray]]=None, 
        input_zero:bool=False, 
        sample_num_limit:int=None, 
        data_type:str='train',
        std:bool=True,
        base_data_folder=DATA_FOLDER,
        split_rng:np.random.Generator=np.random.default_rng(),
    ):
        """
        data_config:
            {
                'his_len': 100,# 每个样本的 历史数据长度
                'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
            }

        data_dict: 
            {
                0/symbol: {
                    'obs': np.ndarray,
                    'acts': np.ndarray,
                },
                1/symbol: {
                    'obs': np.ndarray,
                    'acts': np.ndarray,
                },
            }

        std: 是否标准化，默认True
            若为False,返回 (x, y, (std_data_mean, std_data_std))
            若为True,返回 (x, y), x为标准化后的数据

        split_rng: 随机数生成器，用于分割数据集, 默认使用np.random.default_rng()

        键为 symbol_id/symbol, 值为字典, 字典的键为特征名, 值为 numpy 数组
        """
        # 是否是训练集
        # 控制样本均衡
        self.data_type = data_type

        # 是否标准化
        self.std = std

        # 随机数生成器
        self.split_rng = split_rng

        if data_dict:
            # 先储存到tmp 目录，在读取
            data_folder = os.path.join(os.getenv('TEMP'), 'lob_data')
            if os.path.exists(data_folder):
                shutil.rmtree(data_folder)
            os.makedirs(data_folder)
            pickle.dump(data_dict, open(os.path.join(data_folder, 'data_dict.pkl'), 'wb'))

        data_dict = self._load_data_dict(data_folder)
        fix_data_dict = {}
        # 修改键为 symbol > symbol_id
        for k in list(data_dict.keys()):
            if isinstance(k, str):
                new_key = USE_CODES.index(k)
                fix_data_dict[new_key] = data_dict[k]
            else:
                fix_data_dict[k] = data_dict[k]
        self.data_dict = fix_data_dict

        # 如果设置了样本数限制，尽可能的均衡symbol采样
        self.sample_num_limit = sample_num_limit
        if self.sample_num_limit:
            _new_data_dict = {}
            keys = list(self.data_dict.keys())
            num_keys = len(keys)
            if num_keys > 0:
                # 计算每个 key 的基本采样数和剩余样本
                base_samples_per_key = self.sample_num_limit // num_keys
                extra_samples = self.sample_num_limit % num_keys
                total_used = 0
                
                # 第一次分配，尽量均匀
                for i, k in enumerate(keys):
                    v = self.data_dict[k]
                    # 分配基本采样数，额外样本按顺序分配给前 extra_samples 个 key
                    use_num = min(base_samples_per_key + (1 if i < extra_samples else 0), len(v['obs']))
                    if use_num > 0:
                        _new_data_dict[k] = {
                            'obs': v['obs'][:use_num],
                            'acts': v['acts'][:use_num],
                        }
                        total_used += use_num
                
                # 如果总采样数不足 sample_num_limit，重新分配剩余样本
                remaining_samples = self.sample_num_limit - total_used
                while remaining_samples > 0:
                    allocated = 0
                    for k in keys:
                        if k in _new_data_dict:
                            v = self.data_dict[k]
                            current_used = len(_new_data_dict[k]['obs'])
                            # 尝试为该 key 额外分配一个样本
                            if current_used < len(v['obs']):
                                _new_data_dict[k]['obs'].append(v['obs'][current_used])
                                _new_data_dict[k]['acts'].append(v['acts'][current_used])
                                allocated += 1
                                total_used += 1
                                remaining_samples -= 1
                                if remaining_samples == 0:
                                    break
                    # 如果没有分配到任何样本，退出循环以避免死循环
                    if allocated == 0:
                        break
                        
            self.data_dict = _new_data_dict

        self.mapper = IndexMapper({i: len(v['obs']) for i, v in self.data_dict.items()})
        self.length = self.mapper.get_total_length()
        self.his_len = data_config.get('his_len', None)
        self.need_cols = data_config.get('need_cols', None)
        self.need_cols_idx = []
        self.vol_cols_idx = []
        self.input_zero = input_zero

        # data_dict 包含的 symbols
        self.use_symbols = [USE_CODES[i] for i in list(self.data_dict.keys())]

        # 创建symbol到id的映射字典，避免重复查找
        self.symbol_to_id = {sym: i for i, sym in enumerate(USE_CODES) if i in self.data_dict}

        # 缓存对应标的的订单簿数据
        # {symbol_id: {date: _all_raw_data}}
        self.all_data = {}
        self.raw_data = {}
        for file in os.listdir(base_data_folder):
            if not file.endswith('.pkl'):
                continue
            file_path = os.path.join(base_data_folder, file)
            # _ids, _mean_std, _x, _all_raw_data = pickle.load(open(file_path, 'rb'))
            datas = pickle.load(open(file_path, 'rb'))
            _ids, _mean_std, _x, _all_raw_data = datas[:4]
            # 转换数据类型为float32
            for col in _all_raw_data.iloc[:, :-3].columns:
                _all_raw_data[col] = _all_raw_data[col].astype('float32')
            # 距离市场关闭的秒数
            date = file[:8]
            dt = datetime.datetime.strptime(f'{date} 15:00:00', '%Y%m%d %H:%M:%S')
            days = int(date2days(date))
            dt = pytz.timezone('Asia/Shanghai').localize(dt)
            close_ts = int(dt.timestamp())
            before_market_close_sec = ((close_ts - np.array([int(i.split('_')[1]) for i in _ids])) / MAX_SEC_BEFORE_CLOSE).astype(np.float32)

            # 列过滤
            if not self.need_cols:
                # 默认保留所有列(除 时间 列)
                self.need_cols = [i for i in _all_raw_data.columns.tolist() if i != '时间']
                self.vol_cols_idx = [i for i in range(len(self.need_cols)) if '量' in self.need_cols[i] and 'BASE' in self.need_cols[i]]
            if not self.need_cols_idx:
                self.need_cols_idx = [_all_raw_data.columns.get_loc(col) for col in self.need_cols]
            # 只保留需要的列
            _all_raw_data = _all_raw_data.loc[:, self.need_cols]

            # fix raw data
            _all_raw_data = fix_raw_data(_all_raw_data)
            # 储存日raw_data
            self.raw_data[days] = _all_raw_data.values
            # 区分标的
            symbols = np.array([i.split('_')[0] for i in _ids])
            unique_symbols = [i for i in np.unique(symbols) if i in self.use_symbols]
            # 获取所有标的的起止索引
            for symbol in unique_symbols:
                symbol_mask = symbols == symbol
                symbol_indices = np.where(symbol_mask)[0]
                a = symbol_indices[0]
                b = symbol_indices[-1]
                symbol_id = self.symbol_to_id[symbol]
                symbol_mean_std = _mean_std[a:b+1]
                symbol_x = _x[a:b+1]
                symbol_before_market_close_sec = before_market_close_sec[a:b+1]
                x_a, x_b = symbol_x[0][0], symbol_x[-1][1]

                if symbol_id not in self.all_data:
                    self.all_data[symbol_id] = {}

                # 一天内的 mean_std 是相同的，只取第一个
                try:
                    ms = pd.DataFrame(symbol_mean_std[0]['all_std']['all'], dtype=np.float32).iloc[self.need_cols_idx, :].values
                except:
                    ms = pd.DataFrame(symbol_mean_std[0]['price_vol_each']['robust'], dtype=np.float32).iloc[self.need_cols_idx, :].values
                
                # 截断极值
                try:
                    clip_threshold = symbol_mean_std[0]['all_std']['vol_clip_threshold']
                    # 只截断极大值，保留下限
                    self.raw_data[days][x_a:x_b, self.vol_cols_idx] = np.minimum(
                        self.raw_data[days][x_a:x_b, self.vol_cols_idx], clip_threshold
                    )
                except:
                    pass

                # 提前标准化数据
                if self.std:
                    # 直接标准化标的的数据
                    self.raw_data[days][x_a:x_b] -= ms[:, 0]
                    self.raw_data[days][x_a:x_b] /= ms[:, 1]

                # 构建查找表，加速后续检索
                close_sec_to_idx = {sec: i for i, sec in enumerate(symbol_before_market_close_sec)}

                # 保存 其他数据
                self.all_data[symbol_id][days] = {
                    'x': symbol_x,
                    'close_sec_to_idx': close_sec_to_idx,
                }

                if not self.std:
                    self.all_data[symbol_id][days]['ms'] = ms

        print(f"[{self.data_type}] 样本数量: {self.length}")

    def _load_data_dict(self, data_folder:str):
        """
        - 每个标的，相同持仓条件(持仓/空仓)下act类别(未来持仓/空仓)数量均衡     -> 避免模型偏向某一种act类别
        """
        file_paths = []
        for root, dirs, _files in os.walk(data_folder):
            for _file in _files:
                if _file.endswith('.pkl'):
                    file_paths.append(os.path.join(root, _file))
        file_paths.sort()

        if len(file_paths) == 0:
            raise ValueError(f"没有找到任何 pkl 文件")

        elif len(file_paths) >= 3:
            # 多文件情况, 分割 训练/验证/测试集
            if len(file_paths) >= 60:
                # 最后20个日期文件作为 val/ test
                if self.data_type == 'train':
                    file_paths = file_paths[:-20]
                elif self.data_type == 'val':
                    file_paths = file_paths[-20:-10]
                elif self.data_type == 'test':
                    file_paths = file_paths[-10:]
            else:
                if self.data_type == 'train':
                    file_paths = file_paths[:-2]
                elif self.data_type == 'val':
                    file_paths = [file_paths[-2]]
                elif self.data_type == 'test':
                    file_paths = [file_paths[-1]]

        # 首先遍历一遍所有文件，获取每个键值和子键值的总长度
        key_shape = {}  # 记录数据结构
        key_lengths = {}    # 记录每个键值的总长度

        # 统计各个标的下 样本 obs[-2] == 0 / 1 下， acts 的 0/1 数量
        key_type_sample_idxs = {}
        key_total_cur_length = {}
        
        # 第一遍遍历：了解数据结构和计算总长度
        for file_path in file_paths:
            _data_dict = pickle.load(open(file_path, 'rb'))
            
            for key, value in _data_dict.items():
                if key not in key_lengths:
                    # 初始化
                    key_lengths[key] = 0
                    key_shape[key] = {}
                    for k, v in value.items():
                        key_shape[key][k] = v.shape
                    key_type_sample_idxs[key] = {
                        'obs_0_act_0': np.array([], dtype=np.int32),
                        'obs_0_act_1': np.array([], dtype=np.int32),
                        'obs_1_act_0': np.array([], dtype=np.int32),
                        'obs_1_act_1': np.array([], dtype=np.int32),
                    }
                    key_total_cur_length[key] = 0

                # 累加样本长度
                key_lengths[key] += value['obs'].shape[0]

                # 统计样本 obs[-2] == 0 / 1 下， acts 的 0/1 数量
                obs_feature = value['obs'][:, -2]
                act = value['acts']
                obs_0_mask = obs_feature == 0
                obs_1_mask = obs_feature == 1

                # 获取各类样本的索引
                # 若类型的样本不存在，返回空数组
                try:
                    obs_0_act_0_idx = np.where((act == 0) & obs_0_mask)[0]
                    obs_0_act_0_idx = (obs_0_act_0_idx + key_total_cur_length[key]) if len(obs_0_act_0_idx) > 0 else np.array([], dtype=np.int32)
                    obs_0_act_1_idx = np.where((act == 1) & obs_0_mask)[0]
                    obs_0_act_1_idx = (obs_0_act_1_idx + key_total_cur_length[key]) if len(obs_0_act_1_idx) > 0 else np.array([], dtype=np.int32)
                    obs_1_act_0_idx = np.where((act == 0) & obs_1_mask)[0]
                    obs_1_act_0_idx = (obs_1_act_0_idx + key_total_cur_length[key]) if len(obs_1_act_0_idx) > 0 else np.array([], dtype=np.int32)
                    obs_1_act_1_idx = np.where((act == 1) & obs_1_mask)[0]
                    obs_1_act_1_idx = (obs_1_act_1_idx + key_total_cur_length[key]) if len(obs_1_act_1_idx) > 0 else np.array([], dtype=np.int32)
                    key_total_cur_length[key] += value['obs'].shape[0]
                except Exception as e:
                    print(f'{file_path} 数据异常')
                    raise e

                # 合并索引
                key_type_sample_idxs[key]['obs_0_act_0'] = np.concatenate([key_type_sample_idxs[key]['obs_0_act_0'], obs_0_act_0_idx])
                key_type_sample_idxs[key]['obs_0_act_1'] = np.concatenate([key_type_sample_idxs[key]['obs_0_act_1'], obs_0_act_1_idx])
                key_type_sample_idxs[key]['obs_1_act_0'] = np.concatenate([key_type_sample_idxs[key]['obs_1_act_0'], obs_1_act_0_idx])
                key_type_sample_idxs[key]['obs_1_act_1'] = np.concatenate([key_type_sample_idxs[key]['obs_1_act_1'], obs_1_act_1_idx])

                assert value['obs'].shape[0] == value['acts'].shape[0], f"obs/acts 数据长度不一致: {value['obs'].shape[0]} != {value['acts'].shape[0]}"
        
        # 计算均衡后的样本数
        if self.data_type == 'train':
            for key in key_type_sample_idxs:
                total_samples = key_lengths[key]
                # pos 0/1 分类的最小样本数
                pos_0_min_sampel_length = min(key_type_sample_idxs[key]['obs_0_act_0'].shape[0], key_type_sample_idxs[key]['obs_0_act_1'].shape[0])
                pos_1_min_sampel_length = min(key_type_sample_idxs[key]['obs_1_act_0'].shape[0], key_type_sample_idxs[key]['obs_1_act_1'].shape[0])
                print(f'{key} 空仓使用样本数量: {pos_0_min_sampel_length*2}')
                print(f'{key} 持仓使用样本数量: {pos_1_min_sampel_length*2}')
                # 随机打乱
                self.split_rng.shuffle(key_type_sample_idxs[key]['obs_0_act_0'])
                self.split_rng.shuffle(key_type_sample_idxs[key]['obs_0_act_1'])
                self.split_rng.shuffle(key_type_sample_idxs[key]['obs_1_act_0'])
                self.split_rng.shuffle(key_type_sample_idxs[key]['obs_1_act_1'])
                # 截取最小样本数
                key_type_sample_idxs[key]['obs_0_act_0'] = key_type_sample_idxs[key]['obs_0_act_0'][:pos_0_min_sampel_length]
                key_type_sample_idxs[key]['obs_0_act_1'] = key_type_sample_idxs[key]['obs_0_act_1'][:pos_0_min_sampel_length]
                key_type_sample_idxs[key]['obs_1_act_0'] = key_type_sample_idxs[key]['obs_1_act_0'][:pos_1_min_sampel_length]
                key_type_sample_idxs[key]['obs_1_act_1'] = key_type_sample_idxs[key]['obs_1_act_1'][:pos_1_min_sampel_length]
                # 再排序，用于最后的读取
                key_type_sample_idxs[key]['obs_0_act_0'].sort()
                key_type_sample_idxs[key]['obs_0_act_1'].sort()
                key_type_sample_idxs[key]['obs_1_act_0'].sort()
                key_type_sample_idxs[key]['obs_1_act_1'].sort()
                # 修正样本数
                key_lengths[key] = (pos_0_min_sampel_length + pos_1_min_sampel_length) * 2
                print(f'{key} 样本总数: {key_lengths[key]}')
                
        # 创建最终数据字典并预分配空间
        data_dict = {}
        for key, length in key_lengths.items():
            data_dict[key] = {}
            for k in ['obs', 'acts']:
                # 为每个子键值预分配空间
                data_dict[key][k] = np.empty([key_lengths[key]] + list(key_shape[key][k][1:]), dtype=np.float32)
        
        # 第二遍遍历：填充数据
        position = {key: {k: 0 for k in sub_keys} for key, sub_keys in key_shape.items()}
        
        key_total_cur_length = {}
        for file_path in file_paths:
            _data_dict = pickle.load(open(file_path, 'rb'))
            
            for key, value in _data_dict.items():
                # 待填充数据长度
                length = value['obs'].shape[0]

                # 初始化 key_total_cur_length
                if key not in key_total_cur_length:
                    key_total_cur_length[key] = 0

                # 训练集需要均衡
                if self.data_type == 'train':
                    # 截取范围 key_total_cur_length[key]开始的 length 个索引
                    begin = key_total_cur_length[key]
                    end = begin + length

                    # 合并所有索引
                    keep_sample_idx = np.concatenate([
                        key_type_sample_idxs[key]['obs_0_act_0'],
                        key_type_sample_idxs[key]['obs_0_act_1'], 
                        key_type_sample_idxs[key]['obs_1_act_0'],
                        key_type_sample_idxs[key]['obs_1_act_1']
                    ])

                    cur_file_use_idx = keep_sample_idx[(keep_sample_idx >= begin) & (keep_sample_idx < end)]
                    # 截取数据
                    local_idx = cur_file_use_idx - begin
                    value['obs'] = value['obs'][local_idx]
                    value['acts'] = value['acts'][local_idx]

                # 更新位置
                key_total_cur_length[key] += length

                # 更新 length
                length = value['obs'].shape[0]

                for k in ['obs', 'acts']:
                    # 当前位置
                    current_pos = position[key][k]

                    # 拷贝数据到预分配的数组中
                    data_dict[key][k][current_pos:current_pos + length] = value[k]
                    
                    # 更新位置
                    position[key][k] += length

        # 检查是否样本均衡
        if self.data_type == 'train':
            for key in data_dict:
                # 获取样本数
                obs_feature = data_dict[key]['obs'][:, -2]
                act = data_dict[key]['acts']
                obs_0_mask = obs_feature == 0
                obs_1_mask = obs_feature == 1
                # 获取各类样本数量
                obs_0_act_0_num = len(np.where((act == 0) & obs_0_mask)[0])
                obs_0_act_1_num = len(np.where((act == 1) & obs_0_mask)[0])
                obs_1_act_0_num = len(np.where((act == 0) & obs_1_mask)[0])
                obs_1_act_1_num = len(np.where((act == 1) & obs_1_mask)[0])
                # 检查是否均衡
                assert obs_0_act_0_num == obs_0_act_1_num and obs_1_act_0_num == obs_1_act_1_num, \
                    f"样本不均衡: {obs_0_act_0_num} != {obs_0_act_1_num} or {obs_1_act_0_num} != {obs_1_act_1_num}"

        # 输出样本各个类别的数量
        class_nums = [0, 0]
        for key in data_dict:
            act = data_dict[key]['acts']
            for i in range(2):
                class_nums[i] += len(np.where(act == i)[0])
        print(f"样本类别数量: {class_nums}")        

        return data_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        1. 根据 idx 获取对应的 symbol_id 和 list_idx
        2. 根据 symbol_id 和 list_idx 获取对应 obs/acts
        3. 根据 obs ([before_market_close_sec, pos, days]) 获取对应的订单簿obs数据
        4. 拼接 final_obs = 订单簿obs + symbol_id + obs
        5. 返回 final_obs(x), act(y)
        """
        # 获取对应 obs/acts
        key, list_idx = self.mapper.get_key_and_index(idx)
        obs = self.data_dict[key]['obs'][list_idx]
        act = self.data_dict[key]['acts'][list_idx]

        if self.input_zero and hasattr(self, 'obs_shape'):
            # 直接返回空白的obs
            return np.zeros(self.obs_shape, dtype=np.float32), act

        # 获取订单簿数据
        before_market_close_sec = obs[0]
        days = int(obs[2])
        symbol_id = int(key)
        _data_dict = self.all_data[symbol_id][days]
        # 当前数据的 ms
        ms = _data_dict.get('ms', None)
        # 使用预计算的查找表直接获取索引
        latest_idx = _data_dict['close_sec_to_idx'][before_market_close_sec]
        # 获取订单簿数据
        x_a, x_b = _data_dict['x'][latest_idx]
        raw_data = self.raw_data[days]
        if self.his_len and x_b - x_a >= self.his_len:
            # 如果数据足够长，直接取最后his_len个
            order_book_obs = raw_data[x_b-self.his_len:x_b]
        else:
            order_book_obs = raw_data[x_a:x_b]
            
        # 打平
        order_book_obs = order_book_obs.reshape(-1)

        # 拼接 final_obs = 订单簿obs + symbol_id + obs
        # 预分配数组大小，避免多次拼接
        if not hasattr(self, 'final_obs_len'):
            self.final_obs_len = len(order_book_obs) + 1 + len(obs)
        final_obs = np.empty(self.final_obs_len, dtype=np.float32)
        final_obs[:len(order_book_obs)] = order_book_obs
        final_obs[len(order_book_obs)] = np.float32(symbol_id)
        final_obs[len(order_book_obs)+1:] = obs

        if self.input_zero and not hasattr(self, 'obs_shape'):
            # 记录 obs_shape
            self.obs_shape = final_obs.shape
            if self.std:
                return np.zeros(self.obs_shape, dtype=np.float32), act
            else:
                return np.zeros(self.obs_shape, dtype=np.float32), act, ms

        if self.std:
            # 返回 final_obs(x), act(y)
            return final_obs, act
        else:
            return final_obs, act, ms
    
def test_trajectory_dataset():
    data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data_20250518'
    dataset = LobTrajectoryDataset(data_folder=data_folder)
    print(len(dataset))

def test_lob_trajectory_dataset_num_limit():
    data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data'
    dataset = LobTrajectoryDataset(data_folder=data_folder, sample_num_limit=5)
    print(len(dataset))
    for i in range(len(dataset)):
        d = dataset[i]

def test_lob_trajectory_dataset_dataloader():
    data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data'
    dataset = LobTrajectoryDataset(data_folder=data_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=10, 
        shuffle=True
    )
    for batch in dataloader:
        print(batch)

def test_lob_trajectory_dataset_multi_file():
    data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data'
    dataset = LobTrajectoryDataset(data_folder=data_folder)
    for i in range(len(dataset))[:10]:
        d = dataset[i]

def test_lob_trajectory_dataset(shuffle=True):
    file = r"D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data\transitions.pkl"
    data_dict = pickle.load(open(file, 'rb'))

    data_config = {
        'his_len': 100,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
    }

    dataset = LobTrajectoryDataset(data_config=data_config, data_dict=data_dict)
    print(len(dataset))

    t = time.time()
    if shuffle:
        idx = np.random.permutation(len(dataset))
    else:
        idx = np.arange(len(dataset))
    for i in tqdm(idx):
        d = dataset[i]
    print(f'耗时: {time.time() - t:.2f} 秒')

def test_lob_trajectory_dataset_correct():
    # 检查 dataset 获取的数据 与 env 获取的数据是否一致
    data_config = {
        'his_len': 100,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
    }

    # 实例化 env
    from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
    env_config ={
        'data_type': 'train',# 训练/测试
        'his_len': data_config['his_len'],# 每个样本的 历史数据长度
        'need_cols': data_config['need_cols'],
        'train_folder': r'C:\Users\lh\Desktop\temp',
        'train_title': 'temp',
    }
    data_std = True
    env = LOB_trade_env(env_config, data_std=data_std)
    transitions = {}
    real_transitions = {}
    for i in range(10):
        # obs, info = env.reset(123)
        obs, info = env.reset()
        # 初始化 
        symbol_id = int(obs[-4])
        if symbol_id not in transitions:
            transitions[symbol_id] = {i:[] for i in ['obs', 'acts']}
            real_transitions[symbol_id] = {'obs':[]}
        act = env.action_space.sample()
        # step填充
        done = False
        while not done:
            # 填充数据
            transitions[symbol_id]['obs'].append(obs[-3:])
            if not data_std:
                transitions[symbol_id]['obs'][-1][-3] /= MAX_SEC_BEFORE_CLOSE
            transitions[symbol_id]['acts'].append(act)
            real_transitions[symbol_id]['obs'].append(obs)
            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            act = env.action_space.sample()

    # 实例化 dataset
    dataset = LobTrajectoryDataset(data_config=data_config, data_dict=transitions)
    print(len(dataset))

    mapper = dataset.mapper

    # 检查 dataset 获取的数据 与 env 获取的数据是否一致
    for i in range(len(dataset)):
        dataset_obs, _ = dataset[i]
        symbol_id, list_idx = mapper.get_key_and_index(i)
        real_obs = real_transitions[symbol_id]['obs'][list_idx]
        assert np.array_equal(dataset_obs, real_obs)

if __name__ == "__main__":
    test_trajectory_dataset()
    # test_lob_trajectory_dataset()
    # test_lob_trajectory_dataset_correct()
    # test_lob_trajectory_dataset_multi_file()
    # test_lob_trajectory_dataset_dataloader()
    # test_lob_trajectory_dataset_num_limit()