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
        self.input_zero = input_zero

        # data_dict 包含的 symbols
        self.use_symbols = [USE_CODES[i] for i in list(self.data_dict.keys())]

        # 创建symbol到id的映射字典，避免重复查找
        self.symbol_to_id = {sym: i for i, sym in enumerate(USE_CODES) if i in self.data_dict}

        # 缓存对应标的的订单簿数据
        # {symbol_id: {date: _all_raw_data}}
        self.all_data = {}
        self.raw_data = {}
        for file in os.listdir(DATA_FOLDER):
            if not file.endswith('.pkl'):
                continue
            file_path = os.path.join(DATA_FOLDER, file)
            _ids, _mean_std, _x, _all_raw_data = pickle.load(open(file_path, 'rb'))
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

    def _load_data_dict(self, data_folder: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        从数据文件夹加载、合并并均衡化样本数据。

        - 每个标的的训练样本数量均衡                                        -> 保证每个标的都得到充分的训练
        - 每个标的不同持仓条件下的训练样本数量均衡                            -> 保证每种持仓状态下都得到充分的训练
        - 每个标的，相同持仓条件(持仓/空仓)下act类别(未来持仓/空仓)数量均衡     -> 避免模型偏向某一种act类别

        该方法采用内存高效的两遍式处理策略：
        1. 第一遍：扫描所有文件，收集元数据和每个类别样本的全局索引，不加载实际数据。
        2. 计算均衡点：基于第一遍收集的信息，计算出一个全局统一的、最严格的样本数，
           以实现标的间、持仓状态间、行动间的“三级均衡”。
        3. 第二遍：根据计算出的均衡索引，高效地从文件中加载所需数据到预分配的内存中。

        Args:
            data_folder (str): 包含 .pkl 数据文件的文件夹路径。

        Returns:
            Dict[str, Dict[str, np.ndarray]]: 处理和均衡化后的数据字典。
        """
        file_paths = []
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.pkl'):
                    file_paths.append(os.path.join(root, file))
        file_paths.sort()

        if not file_paths:
            raise ValueError(f"在目录 {data_folder} 中没有找到任何 .pkl 文件")

        # 根据数据集类型（训练/验证/测试）分割文件
        if len(file_paths) >= 3:
            if len(file_paths) >= 60:
                if self.data_type == 'train':
                    file_paths = file_paths[:-40]
                elif self.data_type == 'val':
                    file_paths = file_paths[-40:-20]
                elif self.data_type == 'test':
                    file_paths = file_paths[-20:]
            else:
                if self.data_type == 'train':
                    file_paths = file_paths[:-2]
                elif self.data_type == 'val':
                    file_paths = [file_paths[-2]]
                elif self.data_type == 'test':
                    file_paths = [file_paths[-1]]

        # --- 第一遍：收集元数据和样本索引 ---
        key_shape = {}          # 存储数据结构 e.g., {'symbol': {'obs': (len, 3), 'acts': (len,)}}
        key_category_indices = {} # 存储每个标的下四种分类的全局索引
        key_total_length = {}   # 记录每个标的在合并前累积的样本长度

        print(f"[{self.data_type}] Pass 1/2: 正在扫描 {len(file_paths)} 个文件以收集元数据...")
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                _data_dict = pickle.load(f)

            for key, value in _data_dict.items():
                if key not in key_category_indices:
                    key_shape[key] = {k: v.shape for k, v in value.items()}
                    key_total_length[key] = 0
                    key_category_indices[key] = {
                        'obs_0_act_0': [], 'obs_0_act_1': [],
                        'obs_1_act_0': [], 'obs_1_act_1': [],
                    }

                current_offset = key_total_length[key]
                obs_feature = value['obs'][:, -2]
                acts = value['acts']
                
                # 计算各类样本在当前文件中的局部索引
                obs_0_mask = (obs_feature == 0)
                obs_1_mask = (obs_feature == 1)
                
                # 将局部索引转换为全局索引并追加
                key_category_indices[key]['obs_0_act_0'].extend(
                    np.where((acts == 0) & obs_0_mask)[0] + current_offset
                )
                key_category_indices[key]['obs_0_act_1'].extend(
                    np.where((acts == 1) & obs_0_mask)[0] + current_offset
                )
                key_category_indices[key]['obs_1_act_0'].extend(
                    np.where((acts == 0) & obs_1_mask)[0] + current_offset
                )
                key_category_indices[key]['obs_1_act_1'].extend(
                    np.where((acts == 1) & obs_1_mask)[0] + current_offset
                )
                
                key_total_length[key] += len(value['obs'])

        # --- 计算全局均衡点 (仅对训练集) ---
        final_indices_to_keep = {}
        final_key_lengths = {}

        if self.data_type == 'train':
            print(f"[{self.data_type}] 正在计算全局样本均衡点...")
            # 找到所有标的、所有类别中样本数量的最小值，作为全局均衡的目标
            min_samples_per_category = float('inf')
            
            # 必须保证参与均衡的标的在所有四种类别下都有样本
            valid_keys_for_balancing = []
            for key, categories in key_category_indices.items():
                if all(len(indices) > 0 for indices in categories.values()):
                    min_samples_per_category = min(min_samples_per_category, 
                                                   *[len(indices) for indices in categories.values()])
                    valid_keys_for_balancing.append(key)
                else:
                    print(f"警告: 标的 '{key}' 因缺少某些类别的样本，将不参与训练集的均衡。")
            
            if not valid_keys_for_balancing:
                raise ValueError("没有一个标的拥有全部四种样本类别，无法进行均衡训练。")
            
            if min_samples_per_category == float('inf') or min_samples_per_category == 0:
                raise ValueError("计算出的最小样本数为0，无法创建训练集。请检查数据。")

            print(f"[{self.data_type}] 全局均衡目标：每个标的、每个类别取 {min_samples_per_category} 个样本。")

            # 为每个有效的标的，随机抽取并合并索引
            for key in valid_keys_for_balancing:
                all_indices_for_key = []
                for category, indices in key_category_indices[key].items():
                    # 随机打乱索引并截取
                    self.split_rng.shuffle(indices)
                    all_indices_for_key.extend(indices[:min_samples_per_category])
                
                # 排序索引，以便在第二遍加载时进行高效查找
                all_indices_for_key.sort()
                final_indices_to_keep[key] = np.array(all_indices_for_key, dtype=np.int64)
                final_key_lengths[key] = len(all_indices_for_key)
        else:
            # 对于验证集和测试集，不进行均衡，使用所有数据
            for key, categories in key_category_indices.items():
                all_indices_for_key = np.concatenate(list(categories.values()))
                all_indices_for_key.sort()
                final_indices_to_keep[key] = np.array(all_indices_for_key, dtype=np.int64)
                final_key_lengths[key] = len(all_indices_for_key)
        
        # --- 预分配内存 ---
        data_dict = {}
        for key, length in final_key_lengths.items():
            if length == 0: continue
            data_dict[key] = {}
            for k, shape_info in key_shape[key].items():
                # e.g., shape_info for 'obs' is (original_len, 3)
                # new shape is (final_len, 3)
                final_shape = [length] + list(shape_info[1:])
                data_dict[key][k] = np.empty(final_shape, dtype=np.float32)

        # --- 第二遍：填充数据 ---
        print(f"[{self.data_type}] Pass 2/2: 正在加载均衡后的数据...")
        position = {key: 0 for key in data_dict.keys()}
        key_total_length = {key: 0 for key in key_shape.keys()} # 重置累积长度计数器

        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                _data_dict = pickle.load(f)

            for key, value in _data_dict.items():
                if key not in data_dict:  # 如果此标的因均衡被排除，则跳过
                    continue

                file_chunk_len = len(value['obs'])
                # 当前文件块对应的全局索引范围
                global_start_idx = key_total_length[key]
                global_end_idx = global_start_idx + file_chunk_len

                # 从已排序的最终索引列表中，高效查找属于当前文件块的索引
                # `searchsorted` 可以在有序数组中进行二分查找，非常快
                indices_to_load_from_final_list_start = np.searchsorted(
                    final_indices_to_keep[key], global_start_idx, side='left'
                )
                indices_to_load_from_final_list_end = np.searchsorted(
                    final_indices_to_keep[key], global_end_idx, side='right'
                )
                
                # 获取这些全局索引值
                global_indices_in_this_chunk = final_indices_to_keep[key][
                    indices_to_load_from_final_list_start:indices_to_load_from_final_list_end
                ]

                if len(global_indices_in_this_chunk) > 0:
                    # 将全局索引转换为相对于当前文件块的局部索引
                    local_indices = global_indices_in_this_chunk - global_start_idx
                    
                    # 待填充的数据量
                    num_to_fill = len(local_indices)
                    current_pos = position[key]

                    # 从文件数据中提取所需样本并填充到预分配的数组中
                    for k in ['obs', 'acts']:
                        data_dict[key][k][current_pos : current_pos + num_to_fill] = value[k][local_indices]
                    
                    position[key] += num_to_fill

                key_total_length[key] += file_chunk_len

        # --- 最终检查 (仅对训练集) ---
        if self.data_type == 'train':
            print(f"[{self.data_type}] 正在验证样本均衡结果...")
            for key in data_dict:
                obs_feature = data_dict[key]['obs'][:, -2]
                acts = data_dict[key]['acts']
                
                counts = {
                    'obs_0_act_0': np.sum((obs_feature == 0) & (acts == 0)),
                    'obs_0_act_1': np.sum((obs_feature == 0) & (acts == 1)),
                    'obs_1_act_0': np.sum((obs_feature == 1) & (acts == 0)),
                    'obs_1_act_1': np.sum((obs_feature == 1) & (acts == 1)),
                }
                
                # 所有类别的样本数都应该等于我们计算出的全局均衡点
                first_count = next(iter(counts.values()))
                assert all(c == first_count for c in counts.values()), \
                    f"标的 '{key}' 训练集样本均衡失败! 类别数量: {counts}"

                print(f"[{self.data_type}] 标的 '{key}' 样本共 {sum(counts.values())} 条")
                for k, v in counts.items():
                    print(f"    {k}: {v} 条")

            print(f"[{self.data_type}] 样本均衡验证通过。")
            
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