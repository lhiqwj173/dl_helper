import os, gc, time
import pickle
import threading
import numpy as np
import psutil
from typing import List, Union, Dict, Any
import torch
from torch.utils.data import Dataset
import random, copy

from dl_helper.rl.custom_imitation_module.rollout import KEYS
from dl_helper.tool import in_windows

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

if __name__ == "__main__":
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
