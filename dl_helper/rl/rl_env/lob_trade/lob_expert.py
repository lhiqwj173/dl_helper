"""
专家策略

通过透视未来数据, 给出最佳的交易决策
最大化收益率
"""
import tempfile
import os, pickle, shutil
import inspect
import datetime
import random
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dl_helper.train_param import in_kaggle
from dl_helper.tool import max_profit_reachable, plot_trades
from dl_helper.rl.rl_env.lob_trade.lob_const import MEAN_SEC_BEFORE_CLOSE, STD_SEC_BEFORE_CLOSE, MAX_SEC_BEFORE_CLOSE
from dl_helper.rl.rl_env.lob_trade.lob_const import USE_CODES, STD_REWARD
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL
from dl_helper.rl.rl_env.lob_trade.lob_const import LOCAL_DATA_FOLDER, KAGGLE_DATA_FOLDER, DATA_FOLDER
from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_utils import date2days, days2date
from dl_helper.tool import find_not_stable_sign, _extend_sell_save_start, blank_logout, _extend_profit_start, calculate_profit, calculate_sell_save, reset_profit_sell_save, clear_folder, process_lob_data_extended_sell_save, filte_no_move, fix_profit_sell_save
from dl_helper.tool import report_memory_usage

from py_ext.tool import log, share_tensor, export_df_to_image_dft

def has_ever_decreased_since_start(prices: pd.Series) -> pd.Series:
    """
    检查价格序列从起点到当前点是否曾发生过任何价格下跌。

    价格下跌定义为 t 时刻的价格小于 t-1 时刻的价格。

    Args:
        prices (pd.Series): 输入的价格序列，索引应为时间顺序。

    Returns:
        pd.Series: 一个布尔类型的序列，与输入等长。
                   如果从序列开始到当前位置（包含）的任何地方发生过价格下跌，
                   则值为 True，否则为 False。
    """
    # 1. 计算相邻价格之间的差异。
    #    price[t] - price[t-1]。第一个元素将是 NaN。
    price_changes = prices.diff()

    # 2. 检查差异是否为负。负差异表示价格下跌。
    #    `price_changes < 0` 会生成一个布尔序列。
    #    由于 NaN < 0 的结果是 False，所以第一个元素会被正确处理为 False。
    is_decrease = price_changes < 0

    # 3. 使用累积最大值 (cummax) 来传播第一个 True。
    #    一旦 is_decrease 中出现 True（即第一次下跌），
    #    has_decreased 序列中此后的所有值都将变为 True。
    has_decreased = is_decrease.cummax()

    return has_decreased

def has_ever_increased_since_start(prices: pd.Series) -> pd.Series:
    """
    检查价格序列从起点到当前点是否曾发生过任何价格上涨。

    价格上涨定义为 t 时刻的价格大于 t-1 时刻的价格。

    Args:
        prices (pd.Series): 输入的价格序列，索引应为时间顺序。

    Returns:
        pd.Series: 一个布尔类型的序列，与输入等长。
                   如果从序列开始到当前位置（包含）的任何地方发生过价格上涨，
                   则值为 True，否则为 False。
    """
    # 1. 计算相邻价格之间的差异。
    price_changes = prices.diff()

    # 2. 检查差异是否为正。正差异表示价格上涨。
    #    `price_changes > 0` 会生成一个布尔序列。
    #    由于 NaN > 0 的结果是 False，所以第一个元素会被正确处理为 False。
    is_increase = price_changes > 0

    # 3. 使用累积最大值 (cummax) 来传播第一个 True。
    #    一旦 is_increase 中出现 True（即第一次上涨），
    #    has_increased 序列中此后的所有值都将变为 True。
    has_increased = is_increase.cummax()

    return has_increased

def extend_profit_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据特定逻辑，向前延长 profit > 0 的区间的起始点。

    函数流程:
    1. 识别出所有 profit 从非正数变为正数的“起始点”。
    2. 遍历每一个识别出的“起始点”：
       a. 记录该点的 profit 值和 mid_price (p0)。
       b. 从该点的前一个位置开始，向索引减小的方向（向前）回溯搜索。
       c. 在回溯过程中，如果遇到某点的 mid_price < p0，则认为找到了一个
          潜在的更早的起点，更新 p0 为这个更低的 mid_price，并继续向前搜索。
       d. 回溯在以下任一情况发生时停止：
          - 遇到某点的 mid_price >= p0。
          - 遇到某点的 no_move_len_raw > 50。
    3. 如果在回溯过程中找到了任何更早的潜在起点，则将新旧起点之间的
       profit 值全部更新为原始“起始点”的 profit 值。

    Args:
        df (pd.DataFrame): 
            输入的DataFrame，必须包含以下列:
            - 'profit': 利润值，浮点数或整数。
            - 'no_move_len_raw': 价格无变动持续长度，整数。
            - 'mid_price': 中间价，浮点数。

    Returns:
        pd.DataFrame: 
            返回一个新的DataFrame，其中 'profit' 列已根据上述逻辑被修改。
            原始输入的DataFrame不会被改变。
    """
    # 为保证函数纯净性（不修改原始输入），我们在一开始就创建一个副本
    df_out = df.copy()

    # --- 步骤 1: 找到所有 profit > 0 的起始点 ---
    # 一个点是起始点，当且仅当它的 profit > 0 且它前一个点的 profit <= 0。
    # 我们使用 shift(1) 来访问前一个点的值。fillna(0) 用于处理第一行的情况。
    is_start_point = (df_out['profit'] > 0) & (df_out['profit'].shift(1).fillna(0) <= 0)
    
    # 获取所有起始点的行索引
    # .index 是Pandas的索引对象，可以直接进行布尔索引
    start_indices = df_out.index[is_start_point].tolist()

    # 如果没有找到任何起始点，则无需进行任何操作，直接返回副本
    if not start_indices:
        return df_out

    # 为了在循环中实现高效读取，提前将所需列转换为NumPy数组
    profits = df_out['profit'].values
    mid_prices = df_out['mid_price'].values
    # 替换成 no_move_len_pct
    # no_move_lens = df_out['no_move_len_raw'].values
    no_move_lens = df_out['no_move_len_pct'].values

    # --- 步骤 2: 遍历每个起点，并向前回溯搜索 ---
    for start_idx in start_indices:
        _extend_profit_start(df_out, start_idx, profits, mid_prices, no_move_lens)

    return df_out

def _plot_df_delay(b, e, idx, df, _type_name='profit', extra_name='', logout=blank_logout):
    """
    绘制带有高亮分段区域和边界垂直线的 DataFrame 图表。

    Args:
        b (int): 主要绘图范围的起始索引。
        e (int): 主要绘图范围的结束索引。
        idx (int): 需要高亮的索引。
        df (pd.DataFrame): 包含价格数据的 DataFrame。
        _type_name (str, optional): 日志文件名的类型前缀。默认为 'profit'。
        logout (function, optional): 用于记录日志和获取文件路径的函数。
    """
    # 准备绘图所需的数据
    plot_df = df.loc[b: e, ['mid_price', 'BASE买1价', 'BASE卖1价']].copy()
    
    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(15, 6)) # 使用 subplots 以便更好地控制
    
    # 绘制价格曲线
    ax.plot(plot_df.index, plot_df['mid_price'], color='C0', label='mid_price')
    ax.plot(plot_df.index, plot_df['BASE买1价'], color='C1', alpha=0.3, label='BASE买1价')
    ax.plot(plot_df.index, plot_df['BASE卖1价'], color='C2', alpha=0.3, label='BASE卖1价')

    ax.axvspan(idx, e, color='#b3e5fc', alpha=0.4, zorder=0)  # 淡蓝色填充

    # 初始化一个集合，用于存储所有自定义的X轴刻度
    custom_x_ticks = set()

    # 将 b 和 e 也添加到 X 轴刻度中，确保边界点被显示
    custom_x_ticks.add(idx)

    # 设置X轴的刻度为我们收集到的边界点
    ax.set_xticks(sorted(list(custom_x_ticks)))
    ax.tick_params(axis='x', rotation=45, labelsize=10) # 调整标签大小和旋转角度

    # 添加图表标题和坐标轴标签
    ax.set_title(f'价格走势与分析区间 ({_type_name})', fontsize=16)
    ax.set_xlabel('数据索引', fontsize=12)
    ax.set_ylabel('价格', fontsize=12)
    
    # 显示图例
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6) # 添加网格线，方便查看

    # 调整布局，防止标签被截断
    plt.tight_layout()
    
    # 保存图表
    plot_file_path = logout(
        title=f'{_type_name}_{extra_name}_plot' if extra_name else f'{_type_name}_plot', 
        plot=True)
    if plot_file_path is not None:
        plt.savefig(plot_file_path, dpi=150) # 提高保存图片的分辨率
    plt.close(fig) # 关闭图形，释放内存

def delay_profit_start(df: pd.DataFrame, logout=blank_logout) -> pd.DataFrame:
    """
    向后推迟 profit 的起始点。

    此函数通过识别利润（profit）从非正转为正的时刻，并根据特定的价格（mid_price_raw）
    行为模式，尝试将这个利润周期的起始点向后推移。

    函数流程:
    1. 识别出所有 profit 从非正数变为正数的“起始点”。
    2. 遍历每一个识别出的“起始点”：
       a. 记录该点的 mid_price_raw (p0) 和索引 (original_start_idx)。
       b. 从该点的后一个位置开始，向索引增大的方向（向后）搜索。
       c. 在搜索过程中，如果遇到 mid_price_raw == p0，并且在旧的起点和当前点之间，
          存在唯一一个满足 p0 < mid_price_raw < (p0 + 0.0012) 的点, 
          且下一个时点的 mid_price_raw == 当前点，
          则将潜在的新起点更新为当前点，并从这个新位置继续向后搜索。
       d. 搜索在以下任一情况发生时停止：
          - 遇到连续两次 mid_price_raw > p0。
          - 遇到 profit < 0。
    3. 如果在搜索过程中找到了任何更晚的潜在起点，则将原始起点与最终找到的新起点
       之间的 profit 值全部更新为 0。

    Args:
        df (pd.DataFrame): 包含 'profit' 和 'mid_price_raw' 列的 DataFrame。
                           索引应为唯一且有序的。

    Returns:
        pd.DataFrame: 一个新的 DataFrame，其中部分 profit 区间的起始点可能被推迟
                      （即区段开始的 profit 值被设为 0）。

    Notes:
        - 函数会返回一个新的 DataFrame 副本，不会修改原始传入的 DataFrame。
        - 价格比较的阈值 0.0012 是硬编码的，根据具体需求可以进行调整。
    """
    # 为避免修改原始数据，我们在一开始就创建一个副本
    df_copy = df.copy()

    # --- 步骤 1: 识别所有 profit 的“起始点” ---
    # 条件：当前行的 profit > 0 且上一行的 profit <= 0
    # 使用 .shift(1) 获取上一行的值，fillna(0) 处理第一行可能出现的 NaN
    profit_starts_mask = (df_copy['profit'] > 0) & (df_copy['profit'].shift(1).fillna(0) <= 0)
    
    # 获取这些起始点的索引
    start_indices = df_copy[profit_starts_mask].index

    # --- 步骤 2: 遍历每一个“起始点”并向后搜索 ---
    for original_start_idx in start_indices:
        # 2a. 记录初始状态
        p0 = df_copy.loc[original_start_idx, 'mid_price_raw']
        
        # latest_valid_start_idx 用于追踪在搜索中找到的最后一个有效的新起点
        # 初始时，它就是原始起点
        latest_valid_start_idx = original_start_idx
        
        # search_anchor_idx 是下一次检查“唯一中间价”的起始边界
        search_anchor_idx = original_start_idx

        # 为了安全和高效地进行循环，我们获取原始起点的整数位置
        start_pos = df_copy.index.get_loc(original_start_idx)

        # 2b. 从原始起点的下一个位置开始向后搜索
        # 遍历 DataFrame 剩余部分的索引
        for current_pos in range(start_pos + 1, len(df_copy) - 1):
            current_idx = df_copy.index[current_pos]
            current_price = df_copy.loc[current_idx, 'mid_price_raw']
            current_profit = df_copy.loc[current_idx, 'profit']

            # 2d. 检查搜索停止条件
            # 条件1: profit 变为负数
            if current_profit < 0:
                break
            
            # 条件2: 连续两次 mid_price > p0
            # 我们需要检查当前点和前一点的价格
            prev_price = df_copy.loc[df_copy.index[current_pos - 1], 'mid_price_raw']
            if current_price > p0 and prev_price > p0:
                break

            # 2c. 检查是否可以更新起点
            # 条件：当前价格等于 p0
            if np.isclose(current_price, p0):
                # 在 search_anchor_idx 和 current_idx 之间寻找满足条件的中间价
                # 切片范围：从上一个锚点之后到当前点之前
                search_slice_start = df_copy.index[df_copy.index.get_loc(search_anchor_idx) + 1]
                search_slice_end = df_copy.index[df_copy.index.get_loc(current_idx) - 1]
                
                # 如果切片为空（即两个点相邻），则跳过
                if search_slice_start > search_slice_end:
                    continue

                intermediate_prices = df_copy.loc[search_slice_start:search_slice_end, 'mid_price_raw']
                
                # 检查是否存在“唯一”的中间价
                # (p0 < price < p0 + 0.0012)
                intermediate_condition_count = (
                    (intermediate_prices > p0) & (intermediate_prices < (p0 + 0.0012))
                ).sum()

                if intermediate_condition_count == 1 and df_copy.loc[current_idx + 1, 'mid_price_raw'] == current_price:
                    # 如果条件满足，更新“最晚有效起点”为当前点
                    latest_valid_start_idx = current_idx
                    # 同时，将下一次搜索的锚点也更新为当前点
                    search_anchor_idx = current_idx

        # --- 步骤 3: 如果找到了更晚的起点，则更新 profit 值为 0 ---
        if latest_valid_start_idx != original_start_idx:
            extra_len = max(30, latest_valid_start_idx - original_start_idx)
            e = original_start_idx + extra_len
            b = original_start_idx - extra_len
            # 绘制原始图片
            _plot_df_delay(b, e, original_start_idx, df, _type_name='profit', extra_name='0', logout=logout)

            # 绘制调整后的图片
            _plot_df_delay(b, e, latest_valid_start_idx, df, _type_name='profit', extra_name='1', logout=logout)

            # 使用 .loc 进行基于标签的索引，将原始起点到新起点（不含）之间的 profit 设为 0
            df_copy.loc[original_start_idx:latest_valid_start_idx-1, 'profit'] = 0
            
    return df_copy

def extend_sell_save_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据特定逻辑，向前延长 sell_save > 0 的区间的起始点。

    函数流程:
    1. 识别出所有 sell_save 从非正数变为正数的“起始点”。
    2. 遍历每一个识别出的“起始点”：
       a. 记录该点的 sell_save 值和 mid_price (p0)。
       b. 从该点的前一个位置开始，向索引减小的方向（向前）回溯搜索。
       c. 在回溯过程中，如果遇到某点的 mid_price > p0，则认为找到了一个
          潜在的更早的起点，更新 p0 为这个更高的 mid_price，并继续向前搜索。
       d. 回溯在以下任一情况发生时停止：
          - 遇到某点的 mid_price < p0。
          - 遇到某点的 no_move_len_raw > 50。
    3. 如果在回溯过程中找到了任何更早的潜在起点 (new_idx)，则将从 new_idx
       到原始起点前一个位置的 sell_save 值，全部更新为原始“起始点”的 sell_save 值。

    Args:
        df (pd.DataFrame): 
            输入的DataFrame，必须包含以下列:
            - 'sell_save': 卖出节省值，浮点数或整数。
            - 'no_move_len_raw': 价格无变动持续长度，整数。
            - 'mid_price': 中间价，浮点数。

    Returns:
        pd.DataFrame: 
            返回一个新的DataFrame，其中 'sell_save' 列已根据上述逻辑被修改。
            原始输入的DataFrame不会被改变。
    """
    # 创建输入DataFrame的副本，以避免修改原始数据
    df_out = df.copy()

    # --- 步骤 1: 找到所有 sell_save > 0 的起始点 ---
    # 起始点的定义：当前 sell_save > 0 且前一个点的 sell_save <= 0
    is_start_point = (df_out['sell_save'] > 0) & (df_out['sell_save'].shift(1).fillna(0) <= 0)
    
    # 获取所有起始点的行索引
    start_indices = df_out.index[is_start_point].tolist()

    # 如果没有起始点，则直接返回副本
    if not start_indices:
        return df_out

    # 为提高循环效率，将列转换为NumPy数组
    sell_saves = df_out['sell_save'].values
    mid_prices = df_out['mid_price'].values
    # 替换成 no_move_len_pct
    # no_move_lens = df_out['no_move_len_raw'].values
    no_move_lens = df_out['no_move_len_pct'].values

    # --- 步骤 2: 遍历每个起点，并向前回溯搜索 ---
    for start_idx in start_indices:
        _extend_sell_save_start(df_out, start_idx, sell_saves, mid_prices, no_move_lens)

    return df_out

def delay_sell_save_start(df: pd.DataFrame, logout=blank_logout) -> pd.DataFrame:
    """
    向后推迟 sell_save 的起始点。

    此函数通过识别 sell_save 从非正转为正的时刻，并根据特定的价格（mid_price_raw）
    行为模式，尝试将这个 sell_save 周期的起始点向后推移。

    函数流程:
    1. 识别出所有 sell_save 从非正数变为正数的“起始点”。
    2. 遍历每一个识别出的“起始点”：
       a. 记录该点的 mid_price_raw (p0) 和索引 (original_start_idx)。
       b. 从该点的后一个位置开始，向索引增大的方向（向后）搜索。
       c. 在搜索过程中，如果遇到 mid_price_raw == p0，并且在旧的起点和当前点之间，
          存在唯一一个满足 p0 > mid_price_raw > (p0 - 0.0012) 的点, 
          且下一个时点的 mid_price_raw == 当前点，
          则将潜在的新起点更新为当前点，并从这个新位置继续向后搜索。
       d. 搜索在以下任一情况发生时停止：
          - 遇到连续两次 mid_price_raw < p0。
          - 遇到 sell_save < 0。
    3. 如果在搜索过程中找到了任何更晚的潜在起点，则将原始起点与最终找到的新起点
       之间的 sell_save 值全部更新为 0。

    Args:
        df (pd.DataFrame): 包含 'sell_save' 和 'mid_price_raw' 列的 DataFrame。
                           索引应为唯一且有序的。

    Returns:
        pd.DataFrame: 一个新的 DataFrame，其中部分 sell_save 区间的起始点可能被推迟
                      （即区段开始的 sell_save 值被设为 0）。

    Notes:
        - 函数会返回一个新的 DataFrame 副本，不会修改原始传入的 DataFrame。
        - 价格比较的阈值 0.0012 是硬编码的，根据具体需求可以进行调整。
    """
    # 为避免修改原始数据，我们在一开始就创建一个副本
    df_copy = df.copy()

    # --- 步骤 1: 识别所有 sell_save 的“起始点” ---
    # 条件：当前行的 sell_save > 0 且上一行的 sell_save <= 0
    # 使用 .shift(1) 获取上一行的值，fillna(0) 处理第一行可能出现的 NaN
    sell_save_starts_mask = (df_copy['sell_save'] > 0) & (df_copy['sell_save'].shift(1).fillna(0) <= 0)
    
    # 获取这些起始点的索引
    start_indices = df_copy[sell_save_starts_mask].index

    # --- 步骤 2: 遍历每一个“起始点”并向后搜索 ---
    for original_start_idx in start_indices:
        # 2a. 记录初始状态
        p0 = df_copy.loc[original_start_idx, 'mid_price_raw']
        
        # latest_valid_start_idx 用于追踪在搜索中找到的最后一个有效的新起点
        # 初始时，它就是原始起点
        latest_valid_start_idx = original_start_idx
        
        # search_anchor_idx 是下一次检查“唯一中间价”的起始边界
        search_anchor_idx = original_start_idx

        # 为了安全和高效地进行循环，我们获取原始起点的整数位置
        start_pos = df_copy.index.get_loc(original_start_idx)

        # 2b. 从原始起点的下一个位置开始向后搜索
        # 遍历 DataFrame 剩余部分的索引
        for current_pos in range(start_pos + 1, len(df_copy) - 1):
            current_idx = df_copy.index[current_pos]
            current_price = df_copy.loc[current_idx, 'mid_price_raw']
            current_sell_save = df_copy.loc[current_idx, 'sell_save']

            # 2d. 检查搜索停止条件
            # 条件1: sell_save 变为负数
            if current_sell_save < 0:
                break
            
            # 条件2: 连续两次 mid_price_raw < p0
            # 我们需要检查当前点和前一点的价格
            prev_price = df_copy.loc[df_copy.index[current_pos - 1], 'mid_price_raw']
            if current_price < p0 and prev_price < p0:
                break

            # 2c. 检查是否可以更新起点
            # 条件：当前价格等于 p0
            if np.isclose(current_price, p0):
                # 在 search_anchor_idx 和 current_idx 之间寻找满足条件的中间价
                # 切片范围：从上一个锚点之后到当前点之前
                
                # 获取 search_anchor_idx 的位置
                search_anchor_pos = df_copy.index.get_loc(search_anchor_idx)
                
                # 确保切片是有效的，即 search_anchor_pos + 1 < current_pos
                if search_anchor_pos + 1 >= current_pos:
                    continue # 如果没有中间点，则跳过

                # 定义中间价搜索的起始和结束位置
                intermediate_start_pos = search_anchor_pos + 1
                intermediate_end_pos = current_pos - 1 # 不包含 current_idx

                # 获取中间价格的 Series
                intermediate_prices = df_copy.iloc[intermediate_start_pos:intermediate_end_pos + 1, df_copy.columns.get_loc('mid_price_raw')]
                
                # 检查是否存在“唯一”的中间价
                # (p0 > price > p0 - 0.0012)
                intermediate_condition_count = (
                    (intermediate_prices < p0) & (intermediate_prices > (p0 - 0.0012))
                ).sum()

                if intermediate_condition_count == 1 and df_copy.loc[current_idx + 1, 'mid_price_raw'] == current_price:
                    # 如果条件满足，更新“最晚有效起点”为当前点
                    latest_valid_start_idx = current_idx
                    # 同时，将下一次搜索的锚点也更新为当前点
                    search_anchor_idx = current_idx

        # --- 步骤 3: 如果找到了更晚的起点，则更新 sell_save 值为 0 ---
        if latest_valid_start_idx != original_start_idx:
            extra_len = max(30, latest_valid_start_idx - original_start_idx)
            e = original_start_idx + extra_len
            b = original_start_idx - extra_len
            # 绘制原始图片
            _plot_df_delay(b, e, original_start_idx, df, _type_name='sell_save', extra_name='0', logout=logout)

            # 绘制调整后的图片
            _plot_df_delay(b, e, latest_valid_start_idx, df, _type_name='sell_save', extra_name='1', logout=logout)

            df_copy.loc[original_start_idx:latest_valid_start_idx-1, 'sell_save'] = 0
            
    return df_copy

def delay_start_platform(df: pd.DataFrame, logout=blank_logout) -> pd.DataFrame:
    """
    推迟 profit 和 sell_save 列中正数平台的起始点。

    该函数会识别出 'profit' 和 'sell_save' 列中所有从非正数变为正数的
    “起始点”。对于每一个起始点所在的连续正数平台（即数值相同的一段数据），
    如果该平台的长度大于2，函数会将其前 n-2 个点的值置为0，仅保留平台
    末尾的两个点。

    Args:
        df (pd.DataFrame): 包含 'profit' 和 'sell_save' 列的DataFrame。
                           DataFrame 必须有一个单调递增的默认整数索引。

    Returns:
        pd.DataFrame: 经过处理后的新 DataFrame。
    """
    # 创建一个副本以避免修改原始 DataFrame，这是良好的编程实践
    df_copy = df.copy()

    # 需要处理的目标列
    target_cols = ['profit', 'sell_save']

    for col in target_cols:
        if col not in df_copy.columns:
            print(f"警告: 列 '{col}' 不在 DataFrame 中，将跳过。")
            continue

        s = df_copy[col]

        # --- 步骤 1: 识别所有 profit/sell_save 从非正数变为正数的“起始点” ---
        # is_positive: 当前值是否为正
        is_positive = s > 0
        # was_non_positive: 上一个值是否为非正（<= 0）
        # .shift(1) 获取上一行的值，fill_value=0 用于处理第一行（我们视其前面为0）
        was_non_positive = s.shift(1, fill_value=0) <= 0
        
        # “起始点”的条件是：当前为正，且前一个值为非正
        start_point_mask = is_positive & was_non_positive
        start_indices = df_copy.index[start_point_mask]

        # 如果没有找到任何起始点，则无需对该列进行任何操作
        if start_indices.empty:
            continue

        # --- 步骤 2 & 3: 高效处理每个平台 ---
        # 为了高效地找到每个平台的范围，我们使用一个技巧：
        # 当一个值与前一个值不同时，标记为 True。
        # 然后使用 .cumsum() 为每个连续值的块（平台）创建一个唯一的组ID。
        block_ids = (s != s.shift(1)).cumsum()

        # 遍历每一个识别出的“起始点”
        for start_idx in start_indices:
            # 获取该起始点所在平台的ID
            platform_id = block_ids.loc[start_idx]
            
            # 找到该平台包含的所有行的索引
            platform_mask = (block_ids == platform_id)
            platform_indices = df_copy.index[platform_mask]
            
            # 如果平台的长度大于2，我们需要修改它
            if len(platform_indices) > 2:
                # 确定需要被置为0的索引：即平台的前 n-2 个元素
                indices_to_zero = platform_indices[:-2]
                
                # 将这些位置的值赋为0
                df_copy.loc[indices_to_zero, col] = 0
                
    return df_copy

class LobExpert_file():
    """
    专家策略
    通过 文件 准备数据
    """
    def __init__(self, env=None, rng=None, pre_cache=False, data_folder=DATA_FOLDER, cache_debug=False, logout=True):
        """
        pre_cache: 是否缓存数据 
        logout: 是否生成日志（图片），用于debug
        """
        self._env = env
        self.rng = rng

        # 日志
        self.log_folder = os.path.join(tempfile.gettempdir(), 'lob_expert_log')
        self.log_item_name = 'test'
        os.makedirs(self.log_folder, exist_ok=True)
        self.logout = logout

        # 是否缓存数据 TODO
        self.pre_cache = pre_cache
        # 缓存数据 {date: {symbol: lob_data}}
        self.cache_data = {}

        # 数据文件夹
        self.data_folder = data_folder

        # 是否写入文件，用于debug
        self.cache_debug = cache_debug

        self.all_file_paths = []
        self.all_file_names = []
        for root, dirs, _files in os.walk(self.data_folder):
            for _file in _files:
                if _file.endswith('.pkl'):
                    self.all_file_paths.append(os.path.join(root, _file))
                    self.all_file_names.append(_file)

        if self.pre_cache:
            log('cache all expert data')
            self.cache_all()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property   
    def action_space(self):
        return self._env.action_space

    def set_rng(self, rng):
        self.rng = rng

    def cache_all(self):
        """
        缓存所有数据
        """
        for root, dirs, _files in os.walk(self.data_folder):
            for _file in _files:
                if _file.endswith('.pkl'):
                    _file_path = os.path.join(root, _file)
                    date = _file.split('.')[0]
                    log(f'准备数据: {date}')
                    self.prepare_train_data_file(date2days(date), _data_file_path=_file_path)

        log(f'cache_all done, cache_data: {len(self.cache_data)} dates')

    def _logout(self, *args, title='', df=None, plot=False, **kwargs):
        if self.logout:

            if df is not None:
                # 输出图片
                try:
                    export_df_to_image_dft(df, os.path.join(self.log_folder, self.log_item_name, f"{title}.png"))
                except Exception as e:
                    pass

            elif plot:
                return os.path.join(self.log_folder, self.log_item_name, f"{title}.png")

            else:
                # 输出日志
                # 获取调用者的文件名和行号
                frame = inspect.currentframe().f_back
                filename = os.path.basename(frame.f_code.co_filename)
                lineno = frame.f_lineno
                
                # 构建日志消息
                # msg = f'{datetime.datetime.now()}: '
                msg = f'[{filename}:{lineno}] '
                msg += ' '.join([str(i) for i in args])
                msg += ' '.join([f'{k}={v}' for k, v in kwargs.items()])
                msg += '\n'

                with open(os.path.join(self.log_folder, self.log_item_name, f"{self.log_item_name}.txt"), 'a') as f:
                    f.write(msg)

    def _logout_switch_file(self, item_name):
        if self.logout:
            # 检查当前的log_item_name是否存在且为空，若是则删除文件夹
            current_log_path = os.path.join(self.log_folder, self.log_item_name)
            if os.path.isdir(current_log_path) and not os.listdir(current_log_path):
                log(f'clear_empty_folder: {current_log_path}')
                os.rmdir(current_log_path)
            self.log_item_name = item_name
            os.makedirs(os.path.join(self.log_folder, self.log_item_name), exist_ok=True)

    def _prepare_data(self, begin_idx, end_idx, x, before_market_close_sec, dtype):
        # 清空日志
        if self.logout:
            clear_folder(self.log_folder)

        # 截取范围
        x = x[begin_idx:end_idx]
        lob_data_begin = x[0][0]
        lob_data_end = x[-1][1]
        lob_data = self.full_lob_data.iloc[lob_data_begin: lob_data_end].copy()

        # 只保留 'BASE买1价', 'BASE卖1价'
        lob_data = lob_data[['BASE买1价', 'BASE卖1价']]

        # 距离市场关闭的秒数
        sample_idxs = [i[1]-1 for i in x]
        lob_data['before_market_close_sec'] = np.nan
        lob_data.loc[sample_idxs,'before_market_close_sec'] = [i for i in before_market_close_sec[begin_idx:end_idx]]
        lob_data['before_market_close_sec'] /= MAX_SEC_BEFORE_CLOSE

        lob_data = lob_data.reset_index(drop=True)

        if dtype == np.float32:
            lob_data['before_market_close_sec'] = lob_data['before_market_close_sec'].astype(np.float32)

        # 区分上午下午
        # 11:30:00 - 13:00:00
        am_close_sec = np.float64(12600 / MAX_SEC_BEFORE_CLOSE)
        pm_begin_sec = np.float64(7200 / MAX_SEC_BEFORE_CLOSE)
        # 处理 before_market_close_sec nan
        # 使用 其后第一个非nan + 1/MAX_SEC_BEFORE_CLOSE, 来填充
        filled = lob_data['before_market_close_sec'].bfill()
        mask = lob_data['before_market_close_sec'].isna()
        lob_data['before_market_close_sec'] = np.where(mask, filled + 1/MAX_SEC_BEFORE_CLOSE, lob_data['before_market_close_sec'])
        am = lob_data.loc[lob_data['before_market_close_sec'] >= am_close_sec]
        pm = lob_data.loc[lob_data['before_market_close_sec'] <= pm_begin_sec]

        lob_data['valley_peak'] = np.nan
        lob_data['action'] = np.nan
        for _lob_data in [am, pm]:
            # 第一个数据的索引
            idx_1st = _lob_data.index[0]

            # 计算潜在收益
            trades, total_log_return, _valleys, _peaks = max_profit_reachable(
                # 去掉第一个, 第一个数据无法成交
                _lob_data['BASE买1价'].iloc[1:], 
                _lob_data['BASE卖1价'].iloc[1:], 
                rep_select='last',
                rng=self.rng,
            )# 增加随机泛化
            # plot_trades((lob_data['BASE买1价']+lob_data['BASE卖1价'])/2, trades, valleys, peaks)
            # 需要 +1
            _valleys = [i+1 + idx_1st for i in _valleys]
            _peaks = [i+1 + idx_1st for i in _peaks]

            # 添加到 lob_data 中
            lob_data.loc[_valleys, 'valley_peak'] = 0
            lob_data.loc[_peaks, 'valley_peak'] = 1

            # b/s/h
            # 无需提前一个k线，发出信号
            # trades 中的索引0实际是 lob_data 中的索引1
            # 沿用 索引0 就已经提前了一个k线
            buy_idx = [i[0] + idx_1st for i in trades]
            sell_idx = [i[1] + idx_1st for i in trades]
            lob_data.loc[buy_idx, 'action'] = ACTION_BUY
            lob_data.loc[sell_idx, 'action'] = ACTION_SELL

        # 设置 env 的潜在收益数据，用于可视化
        # 恢复到 full_lob_data 中 
        self.full_lob_data['action'] = np.nan
        self.full_lob_data['valley_peak'] = np.nan
        self.full_lob_data.iloc[lob_data_begin: lob_data_end, -2:] = lob_data.loc[:, ['action', 'valley_peak']].values

        # 区分上午下午填充
        am_cond = lob_data['before_market_close_sec'] >= am_close_sec
        lob_data.loc[am_cond, 'action'] = lob_data.loc[am_cond, 'action'].ffill()
        lob_data.loc[am_cond, 'action'] = lob_data.loc[am_cond, 'action'].fillna(ACTION_SELL)
        pm_cond = lob_data['before_market_close_sec'] <= pm_begin_sec
        lob_data.loc[pm_cond, 'action'] = lob_data.loc[pm_cond, 'action'].ffill()
        lob_data.loc[pm_cond, 'action'] = lob_data.loc[pm_cond, 'action'].fillna(ACTION_SELL)

        # 计算 action==0 时点买入的收益
        am_res = calculate_profit(lob_data.loc[am_cond, :].copy())
        pm_res = calculate_profit(lob_data.loc[pm_cond, :].copy().reset_index(drop=True))
        lob_data['profit'] = np.nan
        lob_data.loc[am_cond, 'profit'] = am_res['profit'].values
        lob_data.loc[pm_cond, 'profit'] = pm_res['profit'].values

        # 计算 action==1 时点卖出节省的收益
        am_res = calculate_sell_save(am_res)
        pm_res = calculate_sell_save(pm_res)
        lob_data['sell_save'] = np.nan
        lob_data.loc[am_cond, 'sell_save'] = am_res['sell_save'].values
        lob_data.loc[pm_cond, 'sell_save'] = pm_res['sell_save'].values

        # 保存 profit / sell_save
        lob_data['raw_sell_save'] = lob_data['sell_save']
        lob_data['raw_profit'] = lob_data['profit']

        # 第一个 profit > 0/ sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
        # self._logout_switch_file('reset_profit_sell_save')
        lob_data = reset_profit_sell_save(lob_data)

        # no_move filter
        self._logout_switch_file('filte_no_move')
        lob_data = filte_no_move(lob_data, logout=self._logout)

        # fix profit / sell_save
        self._logout_switch_file('fix_profit_sell_save')
        lob_data = fix_profit_sell_save(lob_data, logout=self._logout)

        # 第一个 profit > 0/ sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
        self._logout_switch_file('reset_profit_sell_save2')
        lob_data = reset_profit_sell_save(lob_data, logout=self._logout)

        # 推迟 profit start
        self._logout_switch_file('delay_start')
        lob_data = delay_profit_start(lob_data, logout=self._logout)

        # 推迟 sell_save start
        lob_data = delay_sell_save_start(lob_data, logout=self._logout)

        # 推迟平台起点
        self._logout_switch_file('delay_start_platform')
        lob_data = delay_start_platform(lob_data, logout=self._logout)
        
        if self.cache_debug:
            try:
                # 写入文件
                _file_path = os.path.join(tempfile.gettempdir(), 'lob_data.csv')
                lob_data.to_csv(_file_path, encoding='gbk')
            except Exception as e:
                log(f'cache_debug error: {e}')

        # 最终数据 action, before_market_close_sec, profit, sell_save, no_move_len
        lob_data = lob_data.loc[:, ['action', 'before_market_close_sec', 'profit', 'sell_save', 'BASE买1价', 'BASE卖1价']]
        return lob_data

    def prepare_train_data_file(self, date_key, symbol_key=[], dtype=np.float32, _data_file_path=''):
        """
        通过 文件 准备数据
        """
        if date_key not in self.cache_data:
            self.cache_data[date_key] = {}

        date = days2date(date_key)
        if not isinstance(symbol_key, list):
            symbol_key = [symbol_key]
        symbols = [USE_CODES[i] for i in symbol_key]

        # 读取数据
        if _data_file_path == '':
            _date_file = f'{date}.pkl'
            _idx = self.all_file_names.index(_date_file)
            _data_file_path = self.all_file_paths[_idx]
        ids, mean_std, x, self.full_lob_data = pickle.load(open(_data_file_path, 'rb'))

        # 距离市场关闭的秒数
        dt = datetime.datetime.strptime(f'{date} 15:00:00', '%Y%m%d %H:%M:%S')
        dt = pytz.timezone('Asia/Shanghai').localize(dt)
        close_ts = int(dt.timestamp())
        before_market_close_sec = np.array([int(i.split('_')[1]) for i in ids])
        before_market_close_sec = close_ts - before_market_close_sec

        # 按照标的读取样本索引范围 a,b
        _symbols = np.array([i.split('_')[0] for i in ids])

        # 若没有指定标的, 则使用所有标的
        if len(symbols) == 0:
            symbols = list(set(_symbols))

        # 获取所有标的的起止索引
        for idx, s in enumerate(symbols):
            symbol_mask = _symbols == s
            symbol_indices = np.where(symbol_mask)[0]
            begin_idx = symbol_indices[0]
            end_idx = symbol_indices[-1] + 1

            log(f'准备数据: {date} {s}')
            lob_data = self._prepare_data(begin_idx, end_idx, x, before_market_close_sec, dtype)
            self.cache_data[date_key][USE_CODES.index(s)] = lob_data

    def add_potential_data_to_env(self, env):
        if self.need_add_potential_data_to_env:
            env.add_potential_data(self.full_lob_data.loc[:, ['action', 'valley_peak']])
            self.need_add_potential_data_to_env = False

    def check_need_prepare_data(self, obs):
        """
        检查是否需要准备数据
        返回 obs 对应的 date_key, symbol_key
        """
        if len(obs.shape) == 1:
            obs_date = obs[-1]
            obs_symbol = obs[-4]
        elif len(obs.shape) == 2:
            assert obs.shape[0] == 1
            obs_date = obs[0][-1]
            obs_symbol = obs[0][-4]
        else:
            raise ValueError(f'obs.shape: {obs.shape}')
        
        # 如果不在缓存数据中，需要准备数据
        date_key = int(obs_date)
        symbol_key = int(obs_symbol)
        if date_key not in self.cache_data or symbol_key not in self.cache_data[date_key]:
            if not self.pre_cache:
                # 不预缓存数据
                # 清理之前的缓存，可以减少内存占用
                report_memory_usage('before clear cache_data')
                self.cache_data.clear()
                report_memory_usage('after clear cache_data')

            self.prepare_train_data_file(date_key, symbol_key, dtype=obs.dtype)
            self.need_add_potential_data_to_env = True

        return date_key, symbol_key

    @staticmethod
    def _get_action(obs, lob_data):
        """
        获取专家动作
        obs 单个样本
        """
        # 距离市场关闭的秒数 / pos
        before_market_close_sec = obs[-3]
        pos = obs[-2]
        
        # 查找 action
        # 向后多取 future_act_num 个数据
        future_act_num = 10
        data = lob_data[(lob_data['before_market_close_sec'] <= before_market_close_sec) & (lob_data['before_market_close_sec'] >= (before_market_close_sec - 0.1))].iloc[:future_act_num]
        assert len(data) > 0, f'len(data): {len(data)}'# 至少有一个数据

        # 是否马上收盘/休盘 （30s）
        noon_need_close = np.float32(12630 / MAX_SEC_BEFORE_CLOSE) >= before_market_close_sec and np.float32(12565 / MAX_SEC_BEFORE_CLOSE) < before_market_close_sec
        pm_need_close = np.float32(30 / MAX_SEC_BEFORE_CLOSE) >= before_market_close_sec
        if noon_need_close or pm_need_close:
            res = ACTION_SELL
        else:
            if pos == 0:
                # 当前空仓
                # 若未来 future_act_num 个数据中, 有买入动作[且]买入收益为正[且]价格与当前一致（若当前存在收益值，潜在收益一致）, 则买入
                if len(data[\
                    # 有买入动作
                    (data['action']==ACTION_BUY) & \
                        # 潜在收益为正
                        (data['profit'] > 0) & \

                            # # 价格与当前一致
                            # (data['BASE卖1价'] == data['BASE卖1价'].iloc[0]) & \
                            # (data['BASE买1价'] == data['BASE买1价'].iloc[0]) & \
                            #     # 与 第一行数据之间没有发生 BASE卖1价 的下跌(小于第一行 BASE卖1价 的个数为0)
                            #     # 1. 不允许任何的变化
                            #     ((data['BASE卖1价'] != data['BASE卖1价'].iloc[0]).cumsum() == 0) & \
                            #     ((data['BASE买1价'] != data['BASE买1价'].iloc[0]).cumsum() == 0)

                            # 价格与当前一致
                            (data['BASE卖1价'] >= data['BASE卖1价'].iloc[0]) & \
                            (data['BASE买1价'] >= data['BASE买1价'].iloc[0]) & \
                                # 与 第一行数据之间没有发生 任何下跌
                                # 2. 不允许下跌
                                (~has_ever_decreased_since_start(data['BASE卖1价'])) & \
                                (~has_ever_decreased_since_start(data['BASE买1价']))

                                    ]) > 0:
                    res = ACTION_BUY
                else:
                    res = ACTION_SELL
            else:
                # 当前有持仓
                # 若未来 future_act_num 个数据中, 有卖出动作[且]卖出收益为正[且]价格与当前一致（潜在收益一致）, 则卖出
                if len(data[\
                    (data['action']==ACTION_SELL) & \
                        # 潜在收益为正
                        (data['sell_save'] > 0) & \

                            # (data['BASE买1价'] == data['BASE买1价'].iloc[0]) & \
                            # (data['BASE卖1价'] == data['BASE卖1价'].iloc[0]) & \
                            #     # 与 第一行数据之间没有发生 BASE买1价 的上涨(小于第一行 BASE买1价 的个数为0)
                            #     ((data['BASE卖1价'] != data['BASE卖1价'].iloc[0]).cumsum() == 0) & \
                            #     ((data['BASE买1价'] != data['BASE买1价'].iloc[0]).cumsum() == 0)

                            (data['BASE买1价'] <= data['BASE买1价'].iloc[0]) & \
                            (data['BASE卖1价'] <= data['BASE卖1价'].iloc[0]) & \
                                # 与 第一行数据之间没有发生 任何上涨
                                (~has_ever_increased_since_start(data['BASE买1价'])) & \
                                (~has_ever_increased_since_start(data['BASE卖1价']))

                                    ]) > 0:
                    res = ACTION_SELL
                else:
                    res = ACTION_BUY

        return res
    
    def get_action(self, obs):
        """
        获取专家动作
        obs 允许多个样本
        """
        if not self.pre_cache:
            # 若不缓存数据
            # 只允许一次处理一个样本
            if len(obs.shape) == 2:
                assert obs.shape[0] == 1

        # 获取动作
        if len(obs.shape) == 1:
            date_key, symbol_key = self.check_need_prepare_data(obs)
            return self._get_action(obs, self.cache_data[date_key][symbol_key])
        elif len(obs.shape) == 2:
            rets = []
            for i in obs:
                date_key, symbol_key = self.check_need_prepare_data(i)
                rets.append(self._get_action(i, self.cache_data[date_key][symbol_key]))
            return np.array(rets)
        else:
            raise ValueError(f'obs.shape: {obs.shape}')

    def __call__(self, obs, state, dones):
        return self.get_action(obs), None
    
    def predict(
        self,
        observation,
        state = None,
        episode_start = None,
        deterministic = False,
    ):
        return self.get_action(observation), None


def test_expert():
    date = '20240521'
    code = '513050'
    max_drawdown_threshold = 0.005

    env = LOB_trade_env({
        # 'data_type': 'val',# 训练/测试
        'data_type': 'train',# 训练/测试
        'his_len': 10,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        'use_symbols': [code],

        'train_folder': r'C:\Users\lh\Desktop\temp\lob_env',
        'train_title': 'test',

        'max_drawdown_threshold': max_drawdown_threshold,

        # 'no_position': BlankRewardStrategy,

    },
    debug_date=[date],
    )

    obs, info = env.reset()
    obs2, reward, terminated, truncated, info = env.step(1)
    batch = np.stack([obs, obs2], axis=0) 

    expert = LobExpert_file(pre_cache=True)
    action = expert.get_action(batch)
    print(action)

def play_lob_data_with_expert(render=True):
    import time

    debug_obs_date = np.float32(12448.0)
    debug_obs_time = np.float32(0.65676767)
    debug_obs_time = 14400 , 12900
    debug_obs_time = random.uniform(14400/MAX_SEC_BEFORE_CLOSE, 12900/MAX_SEC_BEFORE_CLOSE)
    init_pos = 1

    code = '513050'
    env = LOB_trade_env({
        # 'data_type': 'val',# 训练/测试
        'data_type': 'train',# 训练/测试
        'his_len': 30,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        # 'use_symbols': [code],
        'render_freq': 1,

        'train_folder': r'C:\Users\lh\Desktop\temp\lob_env',
        'train_title': 'test',

        'render_mode': 'human' if render else 'none',
    },
    # debug_obs_date=debug_obs_date,
    # debug_obs_time=debug_obs_time,
    # debug_init_pos = init_pos,
    # dump_bid_ask_accnet=True,
    )

    expert = LobExpert_file(pre_cache=False if render else True)

    rounds = 5
    rounds = 1
    for i in range(rounds):
        print('reset')
        seed = random.randint(0, 1000000)
        # seed = 17442
        print(f'seed: {seed}')
        obs, info = env.reset(seed)
        expert.set_rng(env.np_random)

        if render:
            env.render()

        act = 1
        need_close = False
        while not need_close:
            act = expert.get_action(obs)
            if render:
                expert.add_potential_data_to_env(env)

            obs, reward, terminated, truncated, info = env.step(act)
            if render:
                env.render()
            need_close = terminated or truncated
            # if render:
            #     time.sleep(0.1)
            
        log(f'seed: {seed}')
        if rounds > 1:
            keep_play = input('keep play? (y)')
            if keep_play == 'y':
                continue
            else:
                break

    input('all done, press enter to close')
    env.close()

def eval_expert():
    from stable_baselines3.common.evaluation import evaluate_policy

    code = '513050'
    env = LOB_trade_env({
        # 'data_type': 'val',# 训练/测试
        'data_type': 'train',# 训练/测试
        'his_len': 100,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        'use_symbols': [code],

        # 'render_mode': 'human',

        'train_folder': r'C:\Users\lh\Desktop\temp\lob_env',
        'train_title': 'test',
    },
    )

    expert = LobExpert_file()

    reward, _ = evaluate_policy(
        expert,
        env,
        n_eval_episodes=1,
    )
    print(f"Reward after training: {reward}")

def play_lob_data_by_button():
    env = LOB_trade_env({
        'data_type': 'train',# 训练/测试
        'his_len': 30,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        'train_folder': r'C:\Users\lh\Desktop\temp\play_lob_data_by_button',
        'train_title': r'C:\Users\lh\Desktop\temp\play_lob_data_by_button',

        # 不使用数据增强
        'use_random_his_window': False,# 是否使用随机历史窗口
        'use_gaussian_noise_vol': False,# 是否使用高斯噪声
        'use_spread_add_small_limit_order': False,# 是否使用价差添加小单

        'render_mode': 'human',
        'human_play': True,
    },
    # data_std=False,
    # debug_date=['20240521'],
    )

    expert = LobExpert_file()

    print('reset')
    seed = random.randint(0, 1000000)
    seed = 603045
    obs, info = env.reset(seed=seed)

    act = env.render()

    need_close = False
    while not need_close:
        # 只是为了参考
        expert.get_action(obs)
        expert.add_potential_data_to_env(env)

        obs, reward, terminated, truncated, info = env.step(act)
        act = env.render()
        need_close = terminated or truncated
        
    env.close()
    input(f'all done, seed: {seed}')

if __name__ == '__main__':
    # test_expert()

    import time
    t = time.time()
    play_lob_data_with_expert(True)
    print(time.time() - t)

    # eval_expert()

    # play_lob_data_by_button()

    # dump_file = r"D:\code\dl_helper\get_action.pkl"
    # data = pickle.load(open(dump_file, 'rb'))
    # obs, lob_data, valleys, peaks = data
    # action = LobExpert._get_action(obs, lob_data)
    # print(action)
