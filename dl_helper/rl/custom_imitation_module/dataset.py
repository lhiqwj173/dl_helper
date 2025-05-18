import os, gc, time, datetime
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
    def __init__(self, data_folder:str='', data_config={}, data_dict: Dict[int, Dict[str, np.ndarray]]=None, input_zero:bool=False, sample_num_limit:int=None):
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

        键为 symbol_id/symbol, 值为字典, 字典的键为特征名, 值为 numpy 数组
        """
        if data_dict:
            fix_data_dict = {}
            # 修改键为 symbol > symbol_id
            for k in list(data_dict.keys()):
                if isinstance(k, str):
                    new_key = USE_CODES.index(k)
                    fix_data_dict[new_key] = data_dict[k]
                else:
                    fix_data_dict[k] = data_dict[k]

            self.data_dict = fix_data_dict
        else:
            self.data_dict = self._load_data_dict(data_folder)

        # 如果设置了样本数限制，按照顺序采样
        self.sample_num_limit = sample_num_limit
        if self.sample_num_limit:
            wait_nums = self.sample_num_limit
            _new_data_dict = {}
            while wait_nums > 0:
                for k, v in self.data_dict.items():
                    use_num = min(wait_nums, len(v['obs']))
                    if use_num > 0:
                        _new_data_dict[k] = {
                            'obs': v['obs'][:use_num],
                            'acts': v['acts'][:use_num],
                        }
                        wait_nums -= use_num
                    if wait_nums == 0:
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
            if self.need_cols:
                if not self.need_cols_idx:
                    self.need_cols_idx = [_all_raw_data.columns.get_loc(col) for col in self.need_cols]
                # 只保留需要的列
                _all_raw_data = _all_raw_data.loc[:, self.need_cols]
            else:
                _all_raw_data = _all_raw_data.iloc[:, :-6]
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
                if self.need_cols:
                    ms = pd.DataFrame(symbol_mean_std[0]['price_vol_each']['robust'], dtype=np.float32).iloc[self.need_cols_idx, :].values
                else:
                    ms = pd.DataFrame(symbol_mean_std[0]['price_vol_each']['robust'], dtype=np.float32).values

                # 直接标准化标的的数据
                self.raw_data[days][x_a:x_b] -= ms[:, 0]
                self.raw_data[days][x_a:x_b] /= ms[:, 1]

                # 构建查找表，加速后续检索
                close_sec_to_idx = {sec: i for i, sec in enumerate(symbol_before_market_close_sec)}

                # 保存 其他数据
                self.all_data[symbol_id][days] = {
                    'x': symbol_x,
                    'close_sec_to_idx': close_sec_to_idx,
                    # # for debug
                    # 'ms': ms,
                }

    def _load_data_dict(self, data_folder:str):
        file_paths = []
        for root, dirs, _files in os.walk(data_folder):
            for _file in _files:
                if _file.endswith('.pkl'):
                    file_paths.append(os.path.join(root, _file))

        if len(file_paths) == 0:
            raise ValueError(f"没有找到任何 pkl 文件")

        elif len(file_paths) == 1:
            data_dict = pickle.load(open(file_paths[0], 'rb'))

        else:
            # 多文件情况
            # 首先遍历一遍所有文件，获取每个键值和子键值的总长度
            key_shape = {}  # 记录数据结构
            key_lengths = {}    # 记录每个键值的总长度
            
            # 第一遍遍历：了解数据结构和计算总长度
            for file_path in file_paths:
                _data_dict = pickle.load(open(file_path, 'rb'))
                
                for key, value in _data_dict.items():
                    if key not in key_lengths:
                        key_lengths[key] = 0
                        key_shape[key] = {}
                        for k, v in value.items():
                            key_shape[key][k] = v.shape
                    
                    key_lengths[key] += value['obs'].shape[0]  # 累加长度

                    for k, v in value.items():
                        assert value['obs'].shape[0] == v.shape[0], "数据长度不一致"
            
            # 创建最终数据字典并预分配空间
            data_dict = {}
            for key, length in key_lengths.items():
                data_dict[key] = {}
                for k in ['obs', 'acts']:
                    # 为每个子键值预分配空间
                    data_dict[key][k] = np.empty([key_lengths[key]] + list(key_shape[key][k][1:]), dtype=np.float32)
            
            # 第二遍遍历：填充数据
            position = {key: {k: 0 for k in sub_keys} for key, sub_keys in key_shape.items()}
            
            for file_path in file_paths:
                _data_dict = pickle.load(open(file_path, 'rb'))
                
                for key, value in _data_dict.items():
                    for k, v in value.items():
                        # 当前位置
                        current_pos = position[key][k]
                        length = v.shape[0]
                        
                        # 拷贝数据到预分配的数组中
                        data_dict[key][k][current_pos:current_pos + length] = v
                        
                        # 更新位置
                        position[key][k] += length

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
        # 在载入数据时就已经标准化了
        # # 标准化
        # order_book_obs -= _data_dict['mean_std'][:, 0]
        # order_book_obs /= _data_dict['mean_std'][:, 1]
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
            return np.zeros(self.obs_shape, dtype=np.float32), act

        # 返回 final_obs(x), act(y)
        return final_obs, act
    
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
    # test_trajectory_dataset()
    # test_lob_trajectory_dataset()
    # test_lob_trajectory_dataset_correct()
    # test_lob_trajectory_dataset_multi_file()
    # test_lob_trajectory_dataset_dataloader()
    test_lob_trajectory_dataset_num_limit()