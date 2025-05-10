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

class TrajectoryDataset(Dataset):
    """
    轨迹数据集类，用于加载和管理强化学习轨迹数据。
    遵循 PyTorch Dataset 接口，支持随机文件加载和内存管理。
    """
    
    def __init__(
        self, 
        input_folders: Union[str, List[str]], 
        each_load_batch_file_num: int = 3,  # 每批次加载的文件数量
        pre_load_batch_num: int = 3,        # 预加载的批次数量
        shuffle: bool = True,               # 是否打乱数据
        cache: bool = False,                # 是否只缓存数据
        use_his_len: int = None             # 对数据进行预先切片 TODO: 未实现
    ):
        """
        初始化轨迹数据集。
        
        参数:
            input_folders: 包含数据文件的文件夹路径或路径列表
            each_load_batch_file_num: 每批次加载的文件数量
            pre_load_batch_num: 预加载的批次数量
            shuffle: 是否在每个epoch开始时打乱数据
            cache: 是否只缓存数据
            use_his_len: 对数据进行预先切片
        """
        raise NotImplementedError("TrajectoryDataset 已废弃，请使用 LobTrajectoryDataset")
        self.input_folders = [input_folders] if isinstance(input_folders, str) else input_folders
        self.shuffle = shuffle
        self.each_load_batch_file_num = each_load_batch_file_num
        self.pre_load_batch_num = pre_load_batch_num
        self.use_his_len = use_his_len
            
        # 初始化缓存和数据结构
        self.data_length = 0            # 数据集样本总长度
        self.file_metadata_cache = {}   # 文件元数据缓存
        self.all_files = []             # 所有可用数据文件
        self.pending_files = []         # 待加载的文件列表
        self.pre_load_data_list = []    # 预加载的数据, 储存 (data_dict, current_index_map)
        self.data_dict = {}             # 当前加载的数据
        self.current_index_map = []     # 当前加载数据的索引映射
        self.current_index_min = -1     # 当前加载数据的最小索引
        self.current_index_max = -1     # 当前加载数据的最大索引
        self.epoch = 0                  # 当前训练的epoch

        # 前一个 idx
        self.last_idx = -1
        
        # 扫描并缓存所有文件的元数据
        self._scan_files()

        if cache:
            import sys
            sys.exit()
        
        # 初始化数据加载
        self._init_data_loading()

        # 加载线程启停标志
        self.load_thread_stop = False

        # pending_load_data_num 可用或正在加载的批次数量
        self.pending_load_data_num = 0

        # 当前加载的批次起始索引
        self.current_idx = 0

        # 线程锁
        self.load_thread_lock = threading.Lock()
        # 启动3个加载线程
        self.load_threads = [threading.Thread(target=self._load_thread) for _ in range(3)]
        for thread in self.load_threads:
            thread.start()

        # 等待 pre_load_data_list 加载完成
        while len(self.pre_load_data_list) < self.pre_load_batch_num:
            time.sleep(0.01)
        
    def _scan_files(self):
        """扫描所有文件夹并缓存文件元数据"""
        log(f"扫描数据文件夹: {self.input_folders}")

        metadata_file = rf'/kaggle/input/file-metadata-cache/file_metadata_cache' if not in_windows() else r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data\file_metadata_cache'
        
        # 收集所有.pkl文件
        files = []
        for folder in self.input_folders:
            for root, _, filenames in os.walk(folder):
                for fname in filenames:
                    if fname.endswith('.pkl'):
                        files.append(os.path.join(root, fname))
        
        log(f"找到{len(files)}个数据文件")

        if os.path.exists(metadata_file):
            log(f'元数据文件: {metadata_file}')
            with open(metadata_file, 'rb') as f:
                self.file_metadata_cache = pickle.load(f)
        
        # 提取出 broken_files
        if 'broken_files' not in self.file_metadata_cache:
            self.file_metadata_cache['broken_files'] = []
        broken_files = self.file_metadata_cache['broken_files']

        # 遍历元数据中不存在的文件，获取元数据
        fail_count = 0
        new_edit = False
        for file_path in files:
            if (file_path in self.file_metadata_cache) or (file_path in broken_files):
                continue

            # 需要读取文件，获取元数据
            try:
                log(f'读取文件元数据: {file_path}')
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # 提取并存储元数据
                metadata = {}
                est_memory = 0
                for key in KEYS:
                    try:
                        _data = getattr(data, key)
                        metadata[key] = {
                            'shape': _data.shape,
                            'dtype': _data.dtype
                        }
                        est_memory += _data.nbytes
                    except AttributeError:
                        log(f"警告: 文件{file_path}中没有找到键{key}")
                        fail_count += 1
                        break
                
                # 只有当所有键都存在时才添加到缓存
                if len(metadata) == len(KEYS):
                    metadata['est_memory'] = est_memory
                    metadata['length'] = data.acts.shape[0]  # 所有数组第一维相同
                    self.file_metadata_cache[file_path] = metadata
                    new_edit = True
                del data  # 释放内存

            except Exception as e:
                log(f"读取文件失败: {file_path}, 错误: {e}")
                new_edit = True
                fail_count += 1
                broken_files.append(file_path)

        # 删除不在files中的缓存数据
        files_set = set(files)
        del_count = 0
        for cached_file in list(self.file_metadata_cache.keys()):
            if 'broken_files' == cached_file:
                continue
            if cached_file not in files_set:
                del self.file_metadata_cache[cached_file]
                del_count += 1
                new_edit = True
        
        # 删除不在files中的broken_files
        for broken_file in broken_files[:]:
            if broken_file not in files_set:
                broken_files.remove(broken_file)
                del_count += 1
                new_edit = True

        if del_count:
            log(f"从缓存中删除了{del_count}个不存在的文件:")

        # 缓存到文件
        if new_edit:
            pickle.dump(self.file_metadata_cache, open('file_metadata_cache', 'wb'))
            wx.send_file('file_metadata_cache')
        
        # 剔除掉 broken_files
        self.file_metadata_cache.pop('broken_files', None)

        if fail_count:
            msg = f'警告: {fail_count}个文件无法使用'
            log(msg)
            send_wx(msg)

        self.all_files = list(self.file_metadata_cache.keys())
        for metadata in self.file_metadata_cache.values():
            self.data_length += metadata['length']

        if not self.all_files:
            raise RuntimeError("没有找到有效的数据文件")
            
        log(f"成功缓存 {len(self.all_files)} 个文件的元数据, 共 {self.data_length} 个样本")
        
        # 初始索引映射会在_init_data_loading中计算
    
    def _init_data_loading(self):
        """初始化数据加载，准备第一批文件"""
        # 复制文件列表以便可以修改
        self.pending_files = self.all_files.copy()
        
        # 如果启用shuffle，重新打乱文件顺序
        if self.shuffle:
            random.shuffle(self.pending_files)

        for file_path in self.pending_files:
            log(f"[pending_files]: {file_path}, 长度: {self.file_metadata_cache[file_path]['length']}")
            
        # 重新计算每个文件的起始索引
        self.current_idx = 0
    
    def _load_thread(self):
        while not self.load_thread_stop:
            need_load = False
            with self.load_thread_lock:
                if self.pending_load_data_num < self.pre_load_batch_num:
                    need_load = True
                    self.pending_load_data_num += 1

            if need_load:
                log(f'准备加载批次文件数据，系统剩余内存: {psutil.virtual_memory().available / (1024**3):.2f} GB')
                data_dict, current_index_map_min, current_index_map_max, current_index_map = self._load_file_data()
                with self.load_thread_lock:
                    log(f'batch_begin_idx: {self.current_idx}, current_index_map_min: {current_index_map_min}, current_index_map_max: {current_index_map_max}')
                    _current_index_map_max = current_index_map_max
                    current_index_map_min += self.current_idx
                    current_index_map_max += self.current_idx
                    self.current_idx += _current_index_map_max + 1
                    self.pre_load_data_list.append((data_dict, current_index_map_min, current_index_map_max, current_index_map))

                    # 判断是否新的epoch
                    assert self.current_idx <= self.data_length, f"current_idx: {self.current_idx}, data_length: {self.data_length}"
                    if self.current_idx == self.data_length:
                        self.epoch += 1
                        log(f"Epoch {self.epoch} 结束，重新初始化数据加载")
                        self._init_data_loading()

                log(f'批次文件数据加载完成，系统剩余内存: {psutil.virtual_memory().available / (1024**3):.2f} GB')
            else:
                time.sleep(0.001)

    def _load_file_data(self):
        """
        加载下一批文件，确保内存使用在限制范围内
        返回是否成功加载了新数据
        """
        log(f'加载数据')

        while True:
            with self.load_thread_lock:
                if self.pending_files:
                    # 每次load固定数量的文件
                    selected_files = self.pending_files[:self.each_load_batch_file_num]
                    self.pending_files = self.pending_files[self.each_load_batch_file_num:]
                    break
            time.sleep(0.01)

        # 根据内存限制选择文件
        total_memory = 0
        
        for i, file_path in enumerate(selected_files):
            est_memory = self.file_metadata_cache[file_path]['est_memory']
            total_memory += est_memory
            
        assert selected_files, "没有选择任何文件"
        log(f"选择加载 {len(selected_files)} 个文件，总内存：{total_memory / (1024**3):.2f} GB")
        
        # 初始化形状和类型字典
        first_file = selected_files[0]
        shape_dict = {
            key: [0] + list(self.file_metadata_cache[first_file][key]['shape'][1:])
            for key in KEYS
        }
        type_dict = {
            key: self.file_metadata_cache[first_file][key]['dtype']
            for key in KEYS
        }
        
        # 计算总形状
        total_samples = 0
        for file_path in selected_files:
            file_length = self.file_metadata_cache[file_path]['length']
            total_samples += file_length
            for key in KEYS:
                shape_dict[key][0] += file_length
                
        # 创建大数组存储所有数据
        data_dict = {
            key: np.empty(shape_dict[key], dtype=type_dict[key])
            for key in KEYS
        }
        
        # 初始化索引映射数组和本地索引映射（用于随机打乱）
        current_index_map = []
        
        # 加载并拷贝数据到大数组
        start = 0
        idx = 0
        for file_path in selected_files:
            log(f"加载文件: {file_path}")
            with open(file_path, 'rb') as f:
                transitions = pickle.load(f)
                
            file_length = self.file_metadata_cache[file_path]['length']
            end = start + file_length
            
            # 复制数据到大数组
            for key in KEYS:
                data = getattr(transitions, key)
                data_dict[key][start:end] = data[:file_length]
                
            for i in range(file_length):
                current_index_map.append(idx)
                idx += 1
                
            start = end
            del transitions  # 释放内存

        # 将列表转换为numpy数组，便于后续操作
        current_index_map = np.array(current_index_map)

        # 最大最小值
        current_index_map_min = current_index_map.min()
        current_index_map_max = current_index_map.max()
        
        # 如果启用随机打乱，则打乱当前加载的数据
        if self.shuffle:
            # 随机排列
            np.random.shuffle(current_index_map)

        return data_dict, current_index_map_min, current_index_map_max, current_index_map
  
    def stop(self):
        self.load_thread_stop = True
        for thread in self.load_threads:
            thread.join()

    def __len__(self):
        """返回数据集的总长度（所有文件的样本总数）"""
        return self.data_length
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据项，按顺序访问
        
        参数:
            idx: 全局索引
            
        返回:
            包含所有特征键的字典
        """
        # 加载数据
        if not self.data_dict:
            t = 0
            while True:
                with self.load_thread_lock:  # 获取锁
                    if self.pre_load_data_list:  # 检查数据是否已加载
                        break  # 数据已加载，退出循环

                # 如果数据未加载，继续等待
                t = time.time() if t == 0 else t
                log(f'等待加载数据')
                time.sleep(0.1)

            log(f'更新迭代数据')
            with self.load_thread_lock:
                self.data_dict, self.current_index_min, self.current_index_max, self.current_index_map = self.pre_load_data_list.pop(0)
                self.pending_load_data_num -= 1

            log(f'self.current_index_min: {self.current_index_min}, self.current_index_max: {self.current_index_max}')
            if t:
                cost = time.time() - t
                msg = f'加载数据耗时: {cost:.2f} 秒'
                log(msg)
                send_wx(msg)

        # 检查
        assert idx >= 0 and idx < self.data_length, f"索引 {idx} 超出范围 [0, {self.data_length-1}]"
        assert self.current_index_min <= idx <= self.current_index_max, f"索引 {idx} 必须在范围 [{self.current_index_min}, {self.current_index_max}] 内"
        assert self.current_index_map.size, "数据未加载"
        assert idx == 1 + self.last_idx, "Dataloader 必须按顺序访问数据, 设置 shuffle=False"
        self.last_idx = idx if idx < self.data_length - 1 else -1

        # 获取 idx 对应当前加载数据中的索引
        local_idx = idx - self.current_index_min

        if local_idx == 0:
            log(f'local_idx: {local_idx}, idx: {idx}, self.current_index_min: {self.current_index_min}, self.current_index_max: {self.current_index_max}')

        # self.current_index_map 中对应的数据索引
        data_idx = self.current_index_map[local_idx]
        
        # 如果找到了索引，直接返回数据
        res = {key: self.data_dict[key][data_idx] for key in KEYS}

        # 判断是否需要加载新的批次文件
        if idx == self.current_index_max:
            log(f'迭代数据结束，需要更新数据')
            self.on_batch_end()

        return res
    
    def on_batch_end(self):
        del self.data_dict
        self.data_dict = {}

    def on_epoch_end(self):
        log(f'epoch结束，迭代数据结束，需要更新数据')
        # 前一个 idx
        self.last_idx = -1
        self.on_batch_end()

def test_trajectory_dataset():
    from dl_helper.tool import in_windows

    if not in_windows():
        data_folder = [
            rf'/kaggle/input/pre-trained-policy-2/',# kaggle 命名失误
            rf'/kaggle/input/lob-bc-train-data-filted-3/',
            rf'/kaggle/input/lob-bc-train-data-filted-4/'
        ]
    else:
        data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data'

    dataset = TrajectoryDataset(
        input_folders=data_folder,
        each_load_batch_file_num=1,
        pre_load_batch_num=2,
    )
    print(len(dataset))

    t = time.time()
    # epoch = 5
    # for i in range(epoch):
    #     for i in range(len(dataset)):
    #         d = dataset[i]

    dataset.stop()
    print(f'耗时: {time.time() - t:.2f} 秒')

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
    def __init__(self, data_folder:str='', data_config=None, data_dict: Dict[int, Dict[str, np.ndarray]]=None):
        """
        data_config:
            {
                'his_len': 100,# 每个样本的 历史数据长度
                'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
            }

        data_dict: 
            {
                0: {
                    'obs': np.ndarray,
                    'acts': np.ndarray,
                    'dones': np.ndarray,
                },
                1: {
                    'obs': np.ndarray,
                    'acts': np.ndarray,
                    'dones': np.ndarray,
                },
            }

        键为 symbol_id, 值为字典, 字典的键为特征名, 值为 numpy 数组
        """
        if data_dict:
            self.data_dict = data_dict
        else:
            self.data_dict = self._load_data_dict(data_folder)

        self.mapper = IndexMapper({i: len(v['obs']) for i, v in self.data_dict.items()})
        self.length = self.mapper.get_total_length()
        self.his_len = data_config.get('his_len', None)
        self.need_cols = data_config.get('need_cols', None)
        assert self.need_cols is not None, "need_cols 不能为空"
        assert self.his_len is not None, "his_len 不能为空"
        self.need_cols_idx = []

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
                for k in ['obs', 'acts', 'dones']:
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
        2. 根据 symbol_id 和 list_idx 获取对应 obs/acts/dones
        3. 根据 obs ([before_market_close_sec, pos, days]) 获取对应的订单簿obs数据
        4. 拼接 final_obs = 订单簿obs + symbol_id + obs
        5. 返回 final_obs(x), act(y)
        """
        # 获取对应 obs/acts/dones
        key, list_idx = self.mapper.get_key_and_index(idx)
        obs = self.data_dict[key]['obs'][list_idx]
        act = self.data_dict[key]['acts'][list_idx]

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
        if x_b - x_a >= self.his_len:
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

        # 返回 final_obs(x), act(y)
        return final_obs, act
    
def test_lob_trajectory_dataset_multi_file():
    data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data'
    data_dict = LobTrajectoryDataset(data_folder=data_folder)
    print(data_dict.keys())

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
            transitions[symbol_id] = {i:[] for i in ['obs', 'acts', 'dones']}
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
            transitions[symbol_id]['dones'].append(0)
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
    test_lob_trajectory_dataset_multi_file()