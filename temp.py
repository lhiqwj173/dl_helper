import numpy as np
import pickle
import psutil
import gc
KEYS = ["obs", "next_obs", "acts", "dones", "infos"]
TEST_REST_GB = 5

def log(msg):
    print(msg)

def calculate_sample_size_bytes(sample):
    total = 0
    log("各字段内存占用（字节）：")
    for k, v in sample.items():
        size = v.nbytes
        log(f"  {k}: {size} B")
        total += size
    log(f"=> 单条样本总计: {total} B\n")
    return total

def get_max_rows(sample_size_bytes, reserved_gb=TEST_REST_GB):
    """
    参数：
        sample_size_bytes: 单条样本占用字节数
        reserved_gb: 保留的系统内存，单位GB（避免内存占满）
    返回：
        可分配的最大样本行数
    """
    available = psutil.virtual_memory().available
    reserved_bytes = reserved_gb * 1024 ** 3
    usable = max(available - reserved_bytes, 0)
    max_rows = usable // sample_size_bytes
    log(f"当前系统可用内存: {available / 1024**2:.2f} MB")
    log(f"预留内存: {reserved_gb} GB")
    log(f"可用于分配的内存: {usable / 1024**2:.2f} MB")
    log(f"最多可分配样本条数: {max_rows}\n")
    return int(max_rows)

def initialize_dataset(spec, num_rows):
    data = {}
    for key, (shape, dtype) in spec.items():
        new_shape = (num_rows, *shape[1:])  # 替换首维为行数
        if dtype == object:
            arr = np.empty(new_shape, dtype=dtype)  # object用empty
            arr.fill('') 
        else:
            arr = np.zeros(new_shape, dtype=dtype)
            arr.fill(1.0) 
        data[key] = arr
        print(f"初始化 {key}，形状: {arr.shape}，类型: {arr.dtype}")
    return data

class Test:
    def __init__(self):
        self.transitions_dict = None
        self.capacity = 0
        self.cur_idx = 0
        self.full = False

    def _copy_data(self, transitions):
        t_length = transitions.acts.shape[0]  # 待写入数据长度

        log(f'capacity: {self.capacity}, t_length: {t_length}, cur_idx: {self.cur_idx}, full: {self.full}')
        # 写入数据
        # 情况1：写入数据比容量大 → 只保留最后 capacity 条
        if t_length > self.capacity:
            offset = t_length - self.capacity
            for key in KEYS:
                data = getattr(transitions, key)
                # transitions_data = data[offset:]
                # self.transitions_dict[key][:] = transitions_data
                np.copyto(
                    self.transitions_dict[key],
                    data[offset:]
                )
                del data
            self.cur_idx = 0
            self.full = True
        else:
            # 正常或跨越写入
            begin = self.cur_idx
            # 第一段：从当前 cur_idx 到缓冲区尾部
            first_part_len = min(self.capacity - begin, t_length)
            # 第二段（如果需要）：从头开始继续写
            second_part_len = t_length - first_part_len
            for key in KEYS:
                data = getattr(transitions, key)
                # 使用 np.copyto 避免生成切片临时数组
                np.copyto(
                    self.transitions_dict[key][begin:begin+first_part_len],
                    data[:first_part_len]
                )
                if second_part_len > 0:
                    np.copyto(
                        self.transitions_dict[key][0:second_part_len],
                        data[first_part_len:]
                    )
                    self.full = True
                del data
            # 更新状态
            self.cur_idx = (begin + t_length) % self.capacity

        return t_length
    
    def _init_transitions_dict(self, transitions):
        # 计算单条数据的占用大小
        single_data_dict = {}
        for key in KEYS:
            d = getattr(transitions, key)
            single_data_dict[key] = [[1] + list(d.shape[1:]), d.dtype]
        # 根据单行数据，计算最大行数
        sample = {key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in single_data_dict.items()}
        sample_size = calculate_sample_size_bytes(sample)
        max_rows = get_max_rows(sample_size)
        # 初始化数据集
        self.transitions_dict = initialize_dataset(single_data_dict, max_rows)
        self.capacity = self.transitions_dict[KEYS[0]].shape[0]  # 缓冲区容量


    def _handle_demo_path(self, path):
        log(f'load demo: {path}')

        ava_mem = psutil.virtual_memory().available
        log(f"[before demo load] 系统可用内存: {ava_mem / (1024**3):.2f} GB")

        # # 载入 transitions
        transitions = pickle.load(open(str(path), 'rb'))
        # ava_mem2 = psutil.virtual_memory().available
        # log(f"[pickle.load] 系统可用内存: {ava_mem2 / (1024**3):.2f} GB({(ava_mem2 - ava_mem) / (1024**3):.2f} GB)")

        # # 检查初始化
        # if self.transitions_dict is None:
        #     self._init_transitions_dict(transitions)
        #     log(f"[init_transitions_dict] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

        # # 拷贝数据
        # t_length = self._copy_data(transitions)

        # # 释放内存
        # del transitions

        # gc.collect()
        # for g in gc.garbage:
        #     log(f'garbage: {g}')

        # debug_growth()

        log(f"[after demo done] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        # return t_length
    
class Test:
    def _handle_demo_path(self, path):
        log(f'load demo: {path}')

        ava_mem = psutil.virtual_memory().available
        log(f"[before demo load] 系统可用内存: {ava_mem / (1024**3):.2f} GB")

        # 载入 transitions
        # transitions = pickle.load(open(str(path), 'rb'))
        with open(str(path), 'rb') as f:
            transitions = pickle.load(f)

        ava_mem2 = psutil.virtual_memory().available
        log(f"[pickle.load] 系统可用内存: {ava_mem2 / (1024**3):.2f} GB({(ava_mem2 - ava_mem) / (1024**3):.2f} GB)")

        # 释放内存
        del transitions
        gc.collect()

        log(f"[after demo done] 系统可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")

if __name__ == '__main__':
    file = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data\bc_train_data_0\0.pkl'
    test = Test()

    ava_mem = psutil.virtual_memory().available

    for i in range(20):
        test._handle_demo_path(file)

    log(f"[0] 系统可用内存: {ava_mem / (1024**3):.2f} GB")
    ava_mem2 = psutil.virtual_memory().available
    log(f"[1] 系统可用内存: {ava_mem2 / (1024**3):.2f} GB({(ava_mem2 - ava_mem) / (1024**3):.2f} GB)")
