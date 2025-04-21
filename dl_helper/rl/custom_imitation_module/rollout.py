import os
import pickle
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
from imitation.data import types
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL

from py_ext.wechat import wx

KEYS = ["obs", "next_obs", "acts", "dones", "infos"]


def balance_rollout(r):

    obs = r.obs
    acts = r.acts

    # 查找 obs[-2](pos) == 1 且 act==ACTION_SELL 的数量
    cond = (obs[:-1, -2] == 1) & (acts==ACTION_SELL)
    pos_sell_indices = np.where(cond)[0]
    # print(f"持仓卖出: {len(pos_sell_indices)} / {len(acts)} : {pos_sell_indices}")

    # 查找 obs[-2](pos) == 1 且 act==ACTION_BUY 的数量
    cond = (obs[:-1, -2] == 1) & (acts==ACTION_BUY)
    pos_buy_indices = np.where(cond)[0]
    # print(f"持仓买入: {len(pos_buy_indices)} / {len(acts)}")
    # 随机 pos_sell_indices 相同的个数
    pos_buy_indices = np.random.choice(pos_buy_indices, size=len(pos_sell_indices), replace=False) if len(pos_buy_indices) else []
    # print(f"持仓买入(随机): {len(pos_buy_indices)} / {len(acts)} : {pos_buy_indices}")

    # 查找 obs[-2](pos) == 0 且 act==ACTION_BUY 的数量
    cond = (obs[:-1, -2] == 0) & (acts==ACTION_BUY)
    blank_buy_indices = np.where(cond)[0]
    # print(f"空仓买入: {len(blank_buy_indices)} / {len(acts)} : {blank_buy_indices}")

    # 查找 obs[-2](pos) == 0 且 act==ACTION_SELL 的数量
    cond = (obs[:-1, -2] == 0) & (acts==ACTION_SELL)
    blank_sell_indices = np.where(cond)[0]
    # print(f"空仓卖出: {len(blank_sell_indices)} / {len(acts)}")
    # 随机 blank_buy_indices 相同的个数
    blank_sell_indices = np.random.choice(blank_sell_indices, size=len(blank_buy_indices), replace=False) if len(blank_sell_indices) else []
    # print(f"空仓卖出(随机): {len(blank_sell_indices)} / {len(acts)} : {blank_sell_indices}")

    # 合并所有的索引
    wait_concat_list = [i for i in [pos_sell_indices, pos_buy_indices, blank_buy_indices, blank_sell_indices] if len(i)]
    all_indices = np.concatenate(wait_concat_list) if wait_concat_list else np.array([], dtype=int)
    all_indices = np.sort(all_indices)
    # print(f'取用索引: {all_indices}')

    # 修改 rollout
    _obs = r.obs[all_indices]
    _next_obs = r.obs[all_indices + 1]
    _acts = r.acts[all_indices]
    _infos = r.infos[all_indices] if r.infos is not None else None
    _rews = r.rews[all_indices]

    # print(f'obs: {_obs}')
    # print(f'next_obs: {_next_obs}')
    # print(f'acts: {_acts}')
    # print(f'infos: {_infos}')
    # print(f'rews: {_rews}')

    return _obs, _next_obs, _acts, _infos, _rews

def flatten_trajectories(
    cat_parts,
) -> types.Transitions:
    """
    合并 cat_parts 中的数据
    cat_parts
        {
            "obs": [[obs1, obs2, ...], [obs1, obs2, ...], ...],
            "next_obs": [[next_obs1, next_obs2, ...], [next_obs1, next_obs2, ...], ...],
            "acts": [[act1, act2, ...], [act1, act2, ...], ...],
            "dones": [[done1, done2, ...], [done1, done2, ...], ...],
            "infos": [[info1, info2, ...], [info1, info2, ...], ...],
        }
    """
    cat_parts = {
        key: types.concatenate_maybe_dictobs(part_list)
        for key, part_list in cat_parts.items()
    }

    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.Transitions(**cat_parts)

class rollouts_filter:

    def __init__(self) -> None:
        # mypy struggles without Any annotation here.
        # The necessary constraints are enforced above.
        self.parts: Mapping[str, List[Any]] = {key: [] for key in KEYS}
        self._length = 0

    def add_rollouts(self, trajectories: Iterable[types.Trajectory]):
        def all_of_type(key, desired_type):
            return all(
                isinstance(getattr(traj, key), desired_type) for traj in trajectories
            )

        assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
        assert all_of_type("acts", np.ndarray)

        for i, traj in enumerate(trajectories):
            # print(i)
            try:
                _obs, _next_obs, _acts, _infos, _rews = balance_rollout(traj)
            except Exception as e:
                import pickle
                pickle.dump(traj, open(f'balance_rollout_error.pkl', 'wb'))
                wx.send_file(f'balance_rollout_error.pkl')
                continue

            if len(_obs) == 0:
                continue
            else:
                self._length += len(_obs)

            self.parts["acts"].append(_acts)
            self.parts["obs"].append(_obs)
            self.parts["next_obs"].append(_next_obs)

            dones = np.zeros(len(_acts), dtype=bool)
            dones[-1] = traj.terminal
            self.parts["dones"].append(dones)

            if _infos is None:
                infos = np.array([{}] * len(_acts))
            else:
                infos = _infos
            self.parts["infos"].append(infos)

    def __len__(self):
        return self._length
    
    def flatten_trajectories(
        self,
        cat_parts=None,
    ) -> types.Transitions:
        """
        合并 cat_parts 中的数据
        cat_parts
            {
                "obs": [[obs1, obs2, ...], [obs1, obs2, ...], ...],
                "next_obs": [[next_obs1, next_obs2, ...], [next_obs1, next_obs2, ...], ...],
                "acts": [[act1, act2, ...], [act1, act2, ...], ...],
                "dones": [[done1, done2, ...], [done1, done2, ...], ...],
                "infos": [[info1, info2, ...], [info1, info2, ...], ...],
            }
        """
        if cat_parts is None:
            cat_parts = self.parts

        return flatten_trajectories(cat_parts)


def combing_trajectories(trajectories: Iterable[types.Transitions]):
    """
    合并 trajectories 中的数据
    """
    parts: Mapping[str, List[Any]] = {key: [] for key in KEYS}

    for traj in trajectories:
        parts["obs"].append(traj.obs)
        parts["next_obs"].append(traj.next_obs)
        parts["acts"].append(traj.acts)
        parts["dones"].append(traj.dones)
        parts["infos"].append(traj.infos)

    return flatten_trajectories(parts)


def load_trajectories_0(input_folder: str, load_file_num=None, max_memory_gb: float = 28.0):
    """
    提前创建内存加载 trajectories，支持最大内存限制（单位 GB）。
    :param input_folder: 包含多个数据文件夹路径的文件夹
    :param load_file_num: 加载的文件数量，None 表示尽可能多加载
    :param max_memory_gb: 最大内存限制，单位 GB
    """
    max_memory_bytes = max_memory_gb * 1024**3

    # 获取 input_folder 下的所有数据文件夹路径
    data_folder_list = [os.path.join(input_folder, i) for i in os.listdir(input_folder)]
    # 必须是文件夹
    data_folder_list = [i for i in data_folder_list if os.path.isdir(i)]

    # 收集所有文件夹中的 .pkl 文件
    files = []
    for data_folder in data_folder_list:
        folder_files = [os.path.join(data_folder, i) for i in os.listdir(data_folder) if i.endswith('.pkl')]
        files.extend(folder_files)

    # 打乱文件顺序，避免每次加载相同文件
    np.random.shuffle(files)
    
    if load_file_num is None:
        load_file_num = len(files)

    # 初始化形状和类型字典
    shape_dict: Mapping[str, List[Any]] = {key: None for key in KEYS}
    type_dict: Mapping[str, List[Any]] = {key: None for key in KEYS}
    selected_files = []
    total_memory = 0

    # 遍历文件，估算内存并记录形状/类型
    for file in files[:load_file_num]:
        with open(file, 'rb') as f:
            _transitions = pickle.load(f)
        
        est_memory = 0
        for key in KEYS:
            _data = getattr(_transitions, key)
            est_memory += _data.nbytes  # 计算 NumPy 数据占用的内存

        # 检查是否超出内存限制
        if total_memory + est_memory > max_memory_bytes:
            print(f"停止加载：已达到内存上限 {max_memory_gb}GB")
            break

        total_memory += est_memory
        selected_files.append(file)

        # 更新数据形状和类型
        for key in KEYS:
            _data = getattr(_transitions, key)
            if shape_dict[key] is None:
                shape_dict[key] = list(_data.shape)
                type_dict[key] = _data.dtype
            else:
                shape_dict[key][0] += _data.shape[0]
        del _transitions

    # 创建大数组存储所有数据
    all_data_dict = {
        key: np.zeros(shape_dict[key], dtype=type_dict[key])
        for key in KEYS
    }

    # 拷贝数据到大数组
    start = 0
    for file in selected_files:
        _transitions = pickle.load(open(file, 'rb'))
        end = start + _transitions.obs.shape[0]
        for key in KEYS:
            _data = getattr(_transitions, key)
            all_data_dict[key][start:end] = _data
        start = end
        del _transitions

    # 封装成 Transitions 对象
    transitions = types.Transitions(**all_data_dict)

    return transitions

# 全局缓存字典，用于存储文件元数据
file_metadata_cache = {}

def initialize_cache(input_folder: str):
    """
    初始化全局缓存，存储每个文件的元数据。
    
    :param input_folder: 包含多个数据文件夹路径的文件夹
    """
    global file_metadata_cache
    if file_metadata_cache:  # 如果缓存已存在，直接返回
        return

    # 获取 input_folder 下的所有数据文件夹路径
    data_folder_list = [os.path.join(input_folder, i) for i in os.listdir(input_folder)]
    data_folder_list = [i for i in data_folder_list if os.path.isdir(i)]

    # 收集所有文件夹中的 .pkl 文件
    files = []
    for data_folder in data_folder_list:
        folder_files = [os.path.join(data_folder, i) for i in os.listdir(data_folder) if i.endswith('.pkl')]
        files.extend(folder_files)

    # 遍历文件，计算并存储元数据
    for file in files:
        with open(file, 'rb') as f:
            _transitions = pickle.load(f)
        
        metadata = {}
        est_memory = 0
        for key in KEYS:
            _data = getattr(_transitions, key)
            metadata[key] = {
                'shape': _data.shape,
                'dtype': _data.dtype
            }
            est_memory += _data.nbytes  # 计算内存使用量
        metadata['est_memory'] = est_memory
        file_metadata_cache[file] = metadata
        del _transitions  # 释放临时变量内存

def load_trajectories(input_folder: str, load_file_num=None, max_memory_gb: float = 28.0):
    """
    提前创建内存加载 trajectories，支持最大内存限制（单位 GB）。
    
    :param input_folder: 包含多个数据文件夹路径的文件夹
    :param load_file_num: 加载的文件数量，None 表示尽可能多加载
    :param max_memory_gb: 最大内存限制，单位 GB
    :return: Transitions 对象，包含加载的数据
    """
    global file_metadata_cache
    max_memory_bytes = max_memory_gb * 1024**3  # 转换为字节

    # 初始化缓存（如果尚未初始化）
    initialize_cache(input_folder)

    # 获取所有文件并随机打乱顺序
    files = list(file_metadata_cache.keys())
    np.random.shuffle(files)

    if load_file_num is None:
        load_file_num = len(files)

    # 根据内存限制选择文件
    selected_files = []
    total_memory = 0
    for file in files[:load_file_num]:
        est_memory = file_metadata_cache[file]['est_memory']
        if total_memory + est_memory > max_memory_bytes:
            print(f"停止加载：已达到内存上限 {max_memory_gb}GB")
            break
        total_memory += est_memory
        selected_files.append(file)
        
    print(f"选择加载 {len(selected_files)} 个文件，总内存：{total_memory / (1024**3):.2f} GB")
    # 初始化形状和类型字典
    shape_dict = {
        key: [0] + list(file_metadata_cache[selected_files[0]][key]['shape'][1:])
        for key in KEYS
    }
    type_dict = {
        key: file_metadata_cache[selected_files[0]][key]['dtype']
        for key in KEYS
    }

    # 计算总形状
    for file in selected_files:
        for key in KEYS:
            shape_dict[key][0] += file_metadata_cache[file][key]['shape'][0]

    # 创建大数组存储所有数据
    all_data_dict = {
        key: np.zeros(shape_dict[key], dtype=type_dict[key])
        for key in KEYS
    }

    # 加载并拷贝数据到大数组
    start = 0
    for file in selected_files:
        _transitions = pickle.load(open(file, 'rb'))
        end = start + file_metadata_cache[file][KEYS[0]]['shape'][0]
        for key in KEYS:
            _data = getattr(_transitions, key)
            all_data_dict[key][start:end] = _data
        start = end
        del _transitions  # 释放临时变量内存

    # 封装成 Transitions 对象
    transitions = types.Transitions(**all_data_dict)

    return transitions

if __name__ == '__main__':
    folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data'

    trajectories = load_trajectories(folder)
