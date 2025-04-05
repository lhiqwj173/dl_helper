import numpy as np

def random_his_window(raw_data: np.ndarray, his_len: int, max_random_num: int = None, random_prob: float = 0.1):
    """
    raw_data 为更大的历史窗口数据
    随机在 raw_data[-his_len: ] 范围内删除历史切片(行),
    返回的数据历史长度(行)大于等于his_len, 之后需要截断his_len个长度使用

    若有修改，会返回一个新的数据，否则返回原始数据

    Args:
        raw_data: np.ndarray, 原始数据
        his_len: int, 历史长度
        max_random_num: int, 最大随机删除行数, 默认为 his_len * 0.3
        random_prob: float, 随机删除行数的概率, 默认为 0.1

    Returns:
        np.ndarray: 修改后的数据数组，行数 >= his_len
        
    Raises:
        ValueError: 如果 raw_data 的行数 < his_len
    """
    # 获取 raw_data 的行数
    N = raw_data.shape[0]
    
    # 检查输入是否满足要求
    if N < his_len:
        raise ValueError("raw_data must have at least his_len rows")
    
    data = raw_data
    
    # 如果 max_random_num 未指定，则默认为 his_len
    if max_random_num is None:
        max_random_num = int(his_len * 0.3)
    
    # 确保 max_random_num 不超过 N - his_len
    # 因为最后输出的数据长度至少为 his_len
    max_random_num = min(max_random_num, N - his_len)
    
    # 使用二项分布根据 random_prob 生成要删除的行数 m
    m = np.random.binomial(his_len, random_prob)
    
    # 限制 m 不超过 max_random_num
    m = min(m, max_random_num)
    
    # 如果 m > 0，则进行删除
    if m > 0:
        # 在最后 his_len 行中随机选择 m 个不同的行索引
        to_delete = np.random.choice(his_len, m, replace=False)
        to_delete = len(data) - 1 - to_delete
        
        # 从数据中删除这些行
        data = np.delete(data, to_delete, axis=0)
    
    return data

def gaussian_noise_vol(shape, limit=50):
    """
    返回符合正态分布的小幅随机噪声(int)，模拟市场中的微小波动(limit以内)。
    """
    noise = np.random.normal(0, limit/3, size=shape)
    noise = np.clip(np.round(noise).astype(int), -limit, limit)
    return noise


if __name__ == "__main__":
    # raw_data = np.arange(30).reshape(15, 2)
    # print(len(raw_data))
    # print(raw_data[-10:])
    # print('')

    # his_len = 10
    # result = random_his_window(raw_data, his_len)
    # print(len(result))
    # print(result[-10:])

    print(gaussian_noise_vol((10, 2)))
