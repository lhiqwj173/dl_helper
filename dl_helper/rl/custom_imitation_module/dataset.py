import os
import pickle
import numpy as np
import psutil
from typing import List, Union, Dict, Any
import torch
from torch.utils.data import Dataset
import random

from dl_helper.rl.custom_imitation_module.rollout import KEYS

from py_ext.tool import log
from py_ext.wechat import send_wx

class TrajectoryDataset(Dataset):
    """
    轨迹数据集类，用于加载和管理强化学习轨迹数据。
    遵循 PyTorch Dataset 接口，支持随机文件加载和内存管理。
    """
    
    def __init__(
        self, 
        input_folders: Union[str, List[str]], 
        keep_run_size: int = 5,         # 预留缓冲区大小（GB）
        shuffle: bool = True            # 是否打乱数据
    ):
        """
        初始化轨迹数据集。
        
        参数:
            input_folders: 包含数据文件的文件夹路径或路径列表
            keep_run_size: 系统预留内存缓冲区大小（GB）
            shuffle: 是否在每个epoch开始时打乱数据
        """
        self.input_folders = [input_folders] if isinstance(input_folders, str) else input_folders
        self.keep_run_size = keep_run_size * (1024**3)  # 转换为字节
        self.shuffle = shuffle
        
        # 使用可用系统内存减去缓冲区
        self.memory_limit = psutil.virtual_memory().available - self.keep_run_size
            
        # 初始化缓存和数据结构
        self.data_length = 0            # 数据集样本总长度
        self.file_metadata_cache = {}   # 文件元数据缓存
        self.all_files = []             # 所有可用数据文件
        self.pending_files = []         # 待加载的文件列表
        self.loaded_files = []          # 当前已加载的文件
        self.data_dict = {}             # 当前加载的数据
        self.current_index_map = []     # 当前加载数据的索引映射
        self.current_index_min = -1     # 当前加载数据的最小索引
        self.current_index_max = -1     # 当前加载数据的最大索引
        self.start_indices = {}         # 每个文件在全局索引中的起始位置
        self.epoch = 0                  # 当前训练的epoch

        # 前一个 idx
        self.last_idx = -1
        
        # 扫描并缓存所有文件的元数据
        self._scan_files()
        
        # 初始化数据加载
        self._init_data_loading()
        
    def _scan_files(self):
        """扫描所有文件夹并缓存文件元数据"""
        log(f"扫描数据文件夹: {self.input_folders}")
        
        # 收集所有.pkl文件
        files = []
        for folder in self.input_folders:
            for root, _, filenames in os.walk(folder):
                for fname in filenames:
                    if fname.endswith('.pkl'):
                        files.append(os.path.join(root, fname))
        
        log(f"找到{len(files)}个数据文件")
        
        # 读取并缓存每个文件的元数据
        fail_count = 0
        for file_path in files:
            try:
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
                    self.all_files.append(file_path)
                    self.data_length += metadata['length']
                
                del data  # 释放内存
                
            except Exception as e:
                log(f"读取文件失败: {file_path}, 错误: {e}")
                fail_count += 1
        
        if fail_count:
            msg = f'警告: {fail_count}个文件无法使用'
            log(msg)
            send_wx(msg)

        if not self.all_files:
            raise RuntimeError("没有找到有效的数据文件")
            
        log(f"成功缓存 {len(self.all_files)} 个文件的元数据, 共 {self.data_length} 个样本")
        
        # 初始索引映射会在_init_data_loading中计算
    
    def _init_data_loading(self):
        """初始化数据加载，准备第一批文件"""
        # 复制文件列表以便可以修改
        self.pending_files = self.all_files.copy()
        self.last_idx = -1  
        
        # 如果启用shuffle，重新打乱文件顺序
        if self.shuffle:
            random.shuffle(self.pending_files)

        for file_path in self.pending_files:
            log(f"[pending_files]: {file_path}, 长度: {self.file_metadata_cache[file_path]['length']}")
            
        # 重新计算每个文件的起始索引
        current_idx = 0
        self.start_indices = {}
        for file_path in self.pending_files:
            self.start_indices[file_path] = current_idx
            current_idx += self.file_metadata_cache[file_path]['length']
        
        # 加载第一批数据
        self._load_next_batch()
    
    def _load_next_batch(self):
        """
        加载下一批文件，确保内存使用在限制范围内
        返回是否成功加载了新数据
        """
        # 如果没有待加载的文件，则返回False
        if not self.pending_files:
            return False
            
        # 清理之前加载的数据以释放内存
        del self.data_dict
        self.data_dict = {}
        self.current_index_map = []
        self.loaded_files = []
        
        # 根据内存限制选择文件
        selected_files = []
        total_memory = 0
        
        for i, file_path in enumerate(self.pending_files[:]):
            est_memory = self.file_metadata_cache[file_path]['est_memory']
            
            # 检查是否超出内存限制
            # 至少加载一个文件
            if i > 0 :
                if total_memory + est_memory > self.memory_limit:
                    log(f"停止加载：已达到内存上限 {self.memory_limit / (1024**3):.2f} GB")
                    break
                
            total_memory += est_memory
            selected_files.append(file_path)
            # 从待加载列表中移除已选择的文件
            self.pending_files.remove(file_path)
            
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
        self.data_dict = {
            key: np.zeros(shape_dict[key], dtype=type_dict[key])
            for key in KEYS
        }
        
        # 初始化索引映射数组和本地索引映射（用于随机打乱）
        self.current_index_map = []
        # local_indices = np.arange(total_samples)
        
        # 加载并拷贝数据到大数组
        start = 0
        for file_path in selected_files:
            log(f"加载文件: {file_path}")
            try:
                with open(file_path, 'rb') as f:
                    transitions = pickle.load(f)
                    
                file_length = self.file_metadata_cache[file_path]['length']
                end = start + file_length
                
                # 复制数据到大数组
                for key in KEYS:
                    data = getattr(transitions, key)
                    self.data_dict[key][start:end] = data[:file_length]
                    
                # 记录全局索引范围
                global_start_idx = self.start_indices[file_path]
                for i in range(file_length):
                    self.current_index_map.append(global_start_idx + i)
                    
                start = end
                self.loaded_files.append(file_path)
                del transitions  # 释放内存
                
            except Exception as e:
                log(f"加载文件失败: {file_path}, 错误: {e}")
        
        # 将列表转换为numpy数组，便于后续操作
        self.current_index_map = np.array(self.current_index_map)

        # 更新当前加载数据的最小和最大索引
        self.current_index_min = self.current_index_map[0]
        self.current_index_max = self.current_index_map[-1]
        
        # 如果启用随机打乱，则打乱当前加载的数据
        if self.shuffle:
            # 随机排列
            np.random.shuffle(self.current_index_map)

        return True
    
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
        # 检查
        assert idx >= 0 and idx < self.data_length, f"索引 {idx} 超出范围 [0, {self.data_length-1}]"
        assert self.current_index_min <= idx <= self.current_index_max, f"索引 {idx} 必须在范围 [{self.current_index_min}, {self.current_index_max}] 内"
        assert self.current_index_map.size, "数据未加载"
        assert idx == 1 + self.last_idx, "Dataloader 必须按顺序访问数据, 设置 shuffle=False"
        self.last_idx = idx

        # 获取 idx 对应当前加载数据中的索引
        local_idx = idx - self.current_index_min
        
        # 如果找到了索引，直接返回数据
        res = {key: self.data_dict[key][local_idx] for key in KEYS}
        
        # 判断是否需要加载新的批次文件
        if idx == self.current_index_max:
            if not self._load_next_batch():
                # 加载失败 > epoch 结束
                self.on_epoch_end()

        return res

    def on_epoch_end(self):
        """在每个epoch结束时调用，重新初始化数据加载"""
        self.epoch += 1
        log(f"Epoch {self.epoch} 结束，重新初始化数据加载")
        self._init_data_loading()

if __name__ == "__main__":
    dataset = TrajectoryDataset(
        input_folders=[r"D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data"],
        keep_run_size=5,
    )
    print(len(dataset))

    epoch = 3
    for i in range(epoch):
        for i in range(len(dataset)):
            d = dataset[i]
