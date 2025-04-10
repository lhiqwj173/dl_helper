import numpy as np

def random_his_window(raw_data: np.ndarray, his_len: int, max_random_num: int = None, excluding_last_num:int = 3, random_prob: float = 0.1, rng=None):
    """
    raw_data 为更大的历史窗口数据
    随机在 raw_data[-his_len: ] 范围内删除历史切片(行),
    返回的数据历史长度(行)大于等于his_len, 之后需要截断his_len个长度使用

    若有修改，会返回一个新的数据，否则返回原始数据

    Args:
        raw_data: np.ndarray, 原始数据
        his_len: int, 历史长度
        max_random_num: int, 最大随机删除行数, 默认为 his_len * 0.3
        excluding_last_num: int, 不删除最后几行, 默认为 3
        random_prob: float, 随机删除行数的概率, 默认为 0.1

    Returns:
        (bool: 是否修改了数据, np.ndarray: 修改后的数据数组，行数 >= his_len)
        
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
    if rng is None:
        m = np.random.binomial(his_len - excluding_last_num, random_prob)
    else:
        m = rng.binomial(his_len - excluding_last_num, random_prob)
    
    # 限制 m 不超过 max_random_num
    m = min(m, max_random_num)
    
    # 是否修改了数据
    if_modified = False
    # 如果 m > 0，则进行删除
    if m > 0:
        if_modified = True

        # 随机选择 m 个不同的行索引
        if rng is None:
            to_delete = np.random.choice(his_len - excluding_last_num, m, replace=False)
        else:
            to_delete = rng.choice(his_len - excluding_last_num, m, replace=False)

        # 从数据中删除这些行
        data = np.delete(data, to_delete, axis=0)
    
    return if_modified, data

def gaussian_noise_vol(shape, limit=50, random_prob=0.3, rng=None):
    """
    返回符合正态分布的小幅随机噪声(int)，模拟市场中的微小波动(limit以内)。

    Args:
        shape: tuple, 噪声的形状
        limit: int, 噪声的限制
        random_prob: float, 添加噪声的概率
        rng: np.random.RandomState, 随机数生成器
    """
    if rng is None:
        noise = np.random.normal(0, limit/3, size=shape)
    else:
        noise = rng.normal(0, limit/3, size=shape)
    noise = np.clip(np.round(noise).astype(int), -limit, limit)

    # 根据 random_prob 随机赋值0
    # 覆盖比例
    mask_ratio = 1 - random_prob
    # 生成随机掩码
    if rng is None:
        mask = np.random.random(shape) < mask_ratio
    else:
        mask = rng.random(shape) < mask_ratio
    # 根据掩码赋值0
    noise = np.where(mask, 0, noise)

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
