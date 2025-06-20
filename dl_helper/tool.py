import psutil, pickle, torch, os, time
import numpy as np
from tqdm import tqdm
import platform
import time, math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import asyncio
from multiprocessing.queues import Queue
import multiprocessing
from asyncio import Queue as AsyncQueue
from queue import Empty
import threading

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import imgkit
import datetime

from py_ext.wechat import wx
from py_ext.tool import debug, log, get_log_file, init_logger
from dl_helper.train_param import logger, match_num_processes
if match_num_processes() ==8:
    import torch_xla.core.xla_model as xm

import torch
from torchstat import stat
from torchinfo import summary
from torch.nn.utils import parameters_to_vector
import df2img
import requests, socket
from py_ext.alist import alist

UPLOAD_INTERVAL = 300  # 5分钟 = 300秒
MAX_FLAT_RATIO = 0.2  # 平段占比最大值

def in_windows():
    return platform.system() == 'Windows'

def init_logger_by_ip(train_title=''):
    try:
        ip = requests.get('https://api.ipify.org').text
    except:
        # 如果获取外网IP失败,使用内网IP作为备选
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    init_logger(f'{ip}' if train_title == '' else f'{train_title}@{ip}', level='INFO', timestamp=False, enqueue=True, 
                home=os.path.expanduser("~") if (in_windows()) or (not os.path.exists(r'/kaggle/working')) else r'/kaggle/working',
                )
    log(f'init_logger: {get_log_file()}')


def print_directory_tree(folder_path, prefix="", level=0, log_func=print):
    """
    以树形结构打印文件夹中的所有文件和子文件夹。
    
    Args:
        folder_path (str): 要列出的文件夹路径
        prefix (str): 用于缩进的前缀字符串
        level (int): 当前递归层级
    """
    try:
        folder = Path(folder_path)
        if not folder.exists():
            log_func("错误：文件夹不存在！")
            return
        if not folder.is_dir():
            log_func("错误：这不是一个文件夹！")
            return

        # 获取所有文件和子文件夹，按名称排序
        items = sorted(folder.iterdir(), key=lambda x: x.name)
        for index, item in enumerate(items):
            is_last = index == len(items) - 1  # 是否是最后一个项目
            # 构造当前层级的缩进和连接符
            connector = "└── " if is_last else "├── "

            # 如果是文件，添加最后修改时间；如果是文件夹，仅显示名称
            if item.is_file():
                mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                time_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{prefix}{connector}{item.name} ({time_str})")
            else:
                print(f"{prefix}{connector}{item.name}")

            # 如果是子文件夹，递归调用
            if item.is_dir():
                # 下一层级的缩进
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_directory_tree(item, next_prefix, level + 1, log_func=log_func)

    except Exception as e:
        log_func(f"发生错误：{str(e)}")

def upload_log_file(train_title):
    """上传日志文件到alist"""
    # 获取日志文件
    log_file = get_log_file()
    if log_file and os.path.exists(log_file):
        # 上传到alist
        client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
        # 上传文件夹
        upload_folder = f'/rl_learning_process/{train_title}/logs/'
        client.mkdir(upload_folder)
        client.upload(log_file, upload_folder)
        log(f'Upload log file {log_file} \n\t-> alist {upload_folder}')
def keep_upload_log_file(train_title):

    """保持上传日志文件"""
    log(f'keep_upload_log_file start: {get_log_file()}')
    while True:
        time.sleep(UPLOAD_INTERVAL)
        upload_log_file(train_title)

class AsyncProcessEventReader:
    """
    异步进程事件转发器，将进程事件转发到异步事件
    """
    def __init__(self, process_event):
        self.process_event = process_event
        self._loop = None
        self._thread = None
        self._stop_flag = False
        self._event = asyncio.Event()

    def _reader_thread(self):
        """后台读取线程，负责监听进程事件并触发异步事件"""
        while not self._stop_flag:
            try:
                if self.process_event.wait(timeout=0.1):
                    log(f'process_event active')
                    self.process_event.clear()
                    # 进程事件被触发时，设置异步事件

                    asyncio.run_coroutine_threadsafe(
                        self._set_event(),
                        self._loop
                    )
            except Exception as e:
                print(f"Reader thread error: {e}")
                time.sleep(0.1)

    async def _set_event(self):
        """设置异步事件"""
        log(f'async event set')
        self._event.set()
        self._event.clear()

    def start(self, loop=None):
        """启动事件转发器"""
        if self._thread is not None:
            return
            
        if None is loop:
            self._loop = asyncio.get_event_loop()
        else:
            self._loop = loop
        log(f'AsyncProcessEventReader start, loop: {self._loop}')

        self._stop_flag = False
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

    def stop(self):
        """停止事件转发器"""
        if self._thread is None:
            return
            
        self._stop_flag = True
        self._thread.join()
        self._thread = None

    def __del__(self):
        """析构时确保停止转发器"""
        self.stop()

    async def wait(self):
        """等待事件发生"""
        await self._event.wait()

    def is_set(self) -> bool:
        """检查事件是否已触发"""
        return self._event.is_set()

class AsyncProcessQueueReader:
    """
    异步进程队列读取器，使用单个专用线程
    """
    def __init__(self, queue: Queue):
        self.queue = queue
        self._loop = None
        self._thread = None
        self._running = False
        log(f'AsyncProcessQueueReader queue maxsize: {self.queue._maxsize}')
        self.async_queue = AsyncQueue(self.queue._maxsize)

        self._stop = False

    def _reader_thread(self):
        """后台读取线程"""
        while not self._stop:
            try:
                # 使用较短的超时以便能够响应停止信号
                item = self.queue.get(timeout=0.1)
                # 使用线程安全的方式将任务加入事件循环
                asyncio.run_coroutine_threadsafe(
                    self.async_queue.put(item), 
                    self._loop
                )
            except Empty:
                continue
            except Exception as e:
                print(f"Reader thread error: {e}")
                # 出错时短暂等待后继续
                time.sleep(0.1)

    def start(self, loop=None):
        """启动读取器"""
        if self._thread is not None:
            return
        if None is loop:
            self._loop = asyncio.get_event_loop()
        else:
            self._loop = loop
        self._stop = False
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()



    def _stop(self):
        """停止读取器"""
        if self._thread is None:
            return
            
        self._stop = True
        self._thread.join()
        self._thread = None

    def __del__(self):
        """析构函数中停止读取器"""
        self._stop()

    async def get(self):
        """获取队列中的数据"""
        return await self.async_queue.get()

def keep_only_latest_files(folder, num=50):
    """
    只保留文件夹中的最新修改的50个文件
    """
    files = os.listdir(folder)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    for file in files[:-num]:
        os.remove(os.path.join(folder, file))

def remove_old_env_output_files(save_folder, num=50):
    for folder in os.listdir(save_folder):
        folder = os.path.join(save_folder, folder)
        if os.path.isdir(folder):
            keep_only_latest_files(folder, num)


def calc_sharpe_ratio(returns, risk_free_rate=0.02, num_per_year=250, annualize=True):
    """计算年化夏普比率
    不会做returns长度/nan检查
    Args:
        returns: 对数收益率序列
        risk_free_rate: 无风险利率(年化), 默认 0.02
        num_per_year: 一年的收益率周期个数, 默认returns为日周期 num_per_year=250
        annualize: 是否年化, 默认True
    Returns:
        float: 夏普比率（年化）
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values

    # 将年化无风险利率转换为收益率周期的无风险利率
    period_risk_free = risk_free_rate / num_per_year if annualize else risk_free_rate

    excess_returns = returns - period_risk_free
    std = excess_returns.std()
    if std == 0:
        return 0

    # 年化处理
    if annualize:
        mean_excess = excess_returns.mean() * num_per_year
        annualized_std = std * np.sqrt(num_per_year)
        return mean_excess / annualized_std
    else:
        return excess_returns.mean() / std

def calc_sortino_ratio(returns, risk_free_rate=0.02, num_per_year=250, annualize=True):
    """计算年化索提诺比率
    不会做returns长度/nan检查
    Args:
        returns: 对数收益率序列
        risk_free_rate: 无风险利率(年化), 默认 0.02
        num_per_year: 一年的收益率周期个数, 默认returns为日周期 num_per_year=250
        annualize: 是否年化, 默认True
    Returns:
        float: 索提诺比率（年化）
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values

    # 将年化无风险利率转换为收益率周期的无风险利率
    period_risk_free = risk_free_rate / num_per_year if annualize else risk_free_rate

    excess_returns = returns - period_risk_free
    # 只考虑负收益的标准差
    downside_returns = excess_returns[excess_returns < 0]
        
    # 正常情况下的计算
    if len(downside_returns) == 0:
        down_std = 0
    else:
        down_std = downside_returns.std()

    # 年化处理
    if annualize:
        mean_excess = excess_returns.mean() * num_per_year
        annualized_down_std = down_std * np.sqrt(num_per_year)
        return mean_excess / (max(annualized_down_std, 1e-5))
    else:
        return excess_returns.mean() / max(down_std, 1e-5)

def calc_drawdown(net, tick_size=0.001):
    """计算最大回撤和回撤对应的最小刻度数量
    不会做net长度/nan检查
    Args:
        net: 净值序列（直接输入净值，而不是收益率）
        tick_size: 最小刻度大小，默认0.001rmb
    Returns:
        float: 最大回撤(负值)
        int: 回撤对应的最小刻度数量
    """
    if isinstance(net, (pd.Series, pd.DataFrame)):
        net = net.values
    running_max = np.maximum.accumulate(net)
    
    # 计算最大回撤
    drawdown_ratio = net/running_max - 1
    max_drawdown = np.min(drawdown_ratio)
    
    # 计算回撤对应的最小刻度数量
    drawdown_price = running_max - net  # 计算价差而不是变动率
    max_drawdown_ticks = round(abs(np.max(drawdown_price)) / tick_size)
    
    return max_drawdown, max_drawdown_ticks

def calc_drawup_ticks(net, tick_size=0.001, count_level=2):
    """计算净值序列从阶段低点向上变动的最大刻度数量
    Args:
        net: 净值序列（直接输入净值，而不是收益率）
        tick_size: 最小刻度大小, 默认0.001rmb
        count_level: 计算新高count_level个tick_size的数量
    Returns:
        int: 上涨对应的最小刻度数量, int: 新高count_level个tick_size的数量
    """
    if isinstance(net, (pd.Series, pd.DataFrame)):
        net = net.values
    running_min = np.minimum.accumulate(net)
    
    # 计算价差而不是变动率
    drawup_price = net - running_min
    max_drawup_ticks = round(abs(np.max(drawup_price)) / tick_size)
    
    return max_drawup_ticks, np.sum(np.diff((drawup_price > count_level*tick_size) - 0) > 0)

def calc_return(returns, num_per_year=250, annualize=True):
    """计算年化总对数收益率
    不会做returns长度/nan检查
    Args:
        returns: 对数收益率序列
        num_per_year: 一年的收益率周期个数, 默认returns为日周期 num_per_year=250
        annualize: 是否年化处理, 默认True
    Returns:
        float: 年化/非年化总对数收益率
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    total_return = np.sum(returns)

    if annualize:
        # 年化处理
        period_count = len(returns)
        years = period_count / num_per_year
        # 年化对数收益率
        return total_return / years
    else:
        return total_return

def calc_volatility(returns, num_per_year=250, annualize=True):
    """计算波动率
    不会做returns长度/nan检查
    Args:
        returns: 对数收益率序列
        num_per_year: 一年的收益率周期个数, 默认returns为日周期 num_per_year=250
        annualize: 是否年化处理, 默认True
    Returns:
        float: 年化/非年化波动率
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    std = returns.std()
    if annualize:
        return std * np.sqrt(num_per_year)  # 根据序列长度标准化
    else:
        return std

def _cal_left_right_diff(prices):
    # 计算相邻差值
    left_diff = prices[1:-1] - prices[:-2]   # p[i] - p[i-1]
    right_diff = prices[2:] - prices[1:-1]   # p[i+1] - p[i]
    return left_diff, right_diff

def _smooth_price_series(prices):
    """
    更优化的版本，完全向量化实现
    prices: ndarray 价格序列
    Returns:
        smoothed: ndarray 平滑后的价格序列
    """
    n = len(prices)
    
    if n < 3:
        return prices.copy(), [], []
    
    smoothed = prices.copy()

    # 先处理峰值/谷值相邻的情况
    left_diff, right_diff = _cal_left_right_diff(smoothed)
    # 找到峰值和谷值
    peak_mask = (left_diff > 0) & (right_diff < 0)
    valley_mask = (left_diff < 0) & (right_diff > 0)
    
    # 1. 创建mask标识峰值和谷值的位置（注意索引偏移）
    pv_mask = np.zeros(n, dtype=bool)
    peak_indices = np.where(peak_mask)[0] + 1  # 峰值索引（偏移+1）
    valley_indices = np.where(valley_mask)[0] + 1  # 谷值索引（偏移+1）
    pv_mask[peak_indices] = True  # 标记峰值
    pv_mask[valley_indices] = True  # 标记谷值
    
    # 2. 找到连续的峰值/谷值块
    # 使用差分法找到连续块的起始和结束位置
    diff_pv = np.diff(pv_mask.astype(int), prepend=0, append=0)  # 补齐边界
    start_indices = np.where(diff_pv == 1)[0]  # 连续块开始
    end_indices = np.where(diff_pv == -1)[0]  # 连续块结束
    
    # 3. 只处理真正的连续块（至少包含两个点）
    for start, end in zip(start_indices, end_indices):
        if end - start > 1:  # 确保是连续块（至少两个 True）
            # 获取块的起始和结束值
            start_val = smoothed[start - 1]
            end_val = smoothed[end]
            # 计算线性插值
            t = np.arange(end - start + 2) / (end - start + 1)
            interpolated = start_val + (end_val - start_val) * t[1:-1]
            # 替换连续块中的值
            smoothed[start:end] = interpolated

    # ---- 峰值 ----
    # 计算相邻差值
    left_diff, right_diff = _cal_left_right_diff(smoothed)
    # 找到峰值
    peak_mask = (left_diff > 0) & (right_diff < 0)
    # 获取索引
    peak_indices = np.where(peak_mask)[0] + 1
    # 向量化处理峰值
    if len(peak_indices) > 0:
        # 峰值替换为相邻两点的最大值
        left_vals = smoothed[peak_indices - 1]
        right_vals = smoothed[peak_indices + 1]
        smoothed[peak_indices] = np.maximum(left_vals, right_vals)

    # ---- 谷值 ----
    left_diff, right_diff = _cal_left_right_diff(smoothed)
    valley_mask = (left_diff < 0) & (right_diff > 0)
    valley_indices = np.where(valley_mask)[0] + 1
    
    if len(valley_indices) > 0:
        # 谷值替换为相邻两点的最小值
        left_vals = smoothed[valley_indices - 1]
        right_vals = smoothed[valley_indices + 1]
        smoothed[valley_indices] = np.minimum(left_vals, right_vals)
    
    # 最后检查
    # 不应该再存在 峰值/谷值
    left_diff, right_diff = _cal_left_right_diff(smoothed)
    peak_mask = (left_diff > 0) & (right_diff < 0)
    valley_mask = (left_diff < 0) & (right_diff > 0)
    if np.any(peak_mask) or np.any(valley_mask):
        raise ValueError("存在峰值/谷值")

    return smoothed

def _find_plateaus(prices):
    """
    识别价格序列中的平台期。
    
    参数:
    prices (np.array): 价格序列
    
    返回:
    list: 平台期列表，每个元素为 (start, end, value)
    """
    plateaus = []
    n = len(prices)
    i = 0
    while i < n:
        start = i
        value = prices[i]
        while i + 1 < n and prices[i + 1] == value:
            i += 1
        end = i
        plateaus.append((start, end, value))
        i += 1
    return plateaus

def _identify_peaks_valleys(plateaus, rep_select, rng=None):
    """
    根据平台期识别波峰和波谷。
    
    参数:
    plateaus (list): 平台期列表
    rep_select: 波峰波谷的选取方式
        'mid' 表示中间点
        'random' 表示随机点
        'first' 表示第一个点
        'last' 表示最后一个点

    返回:
    tuple: (peaks, valleys)，分别为波峰和波谷的代表点列表
    """
    peaks = []
    valleys = []

    # 记录每个平台期的点数
    peaks_num_points = []
    valleys_num_points = []

    n = len(plateaus)
    
    for i in range(n):
        start, end, value = plateaus[i]
        num_points = end - start + 1

        if rep_select == 'mid':
            rep = (start + end) // 2  # 选择中间点作为代表
        elif rep_select == 'random':
            if start == end:
                rep = start
            else:
                if rng is None:
                    rep = np.random.randint(start, end)
                else:
                    rep = rng.integers(start, end)
        elif rep_select == 'first':
            rep = start
        elif rep_select == 'last':
            rep = end

        # 如果平台期的点数 > 1
        # 则 rep 应该为 start + 1 之后的点
        if num_points > 1:
            rep = max(start + 1, rep)

        if i == 0:  # 第一个平台期
            if n > 1 and value < plateaus[1][2]:
                valleys.append(rep)
                valleys_num_points.append(num_points)
            elif n > 1 and value > plateaus[1][2]:
                peaks.append(rep)
                peaks_num_points.append(num_points)
        elif i == n - 1:  # 最后一个平台期
            if value < plateaus[i - 1][2]:
                valleys.append(rep)
                valleys_num_points.append(num_points)
            elif value > plateaus[i - 1][2]:
                peaks.append(rep)
                peaks_num_points.append(num_points)
        else:  # 中间平台期
            prev_value = plateaus[i - 1][2]
            next_value = plateaus[i + 1][2]
            if value > prev_value and value > next_value:
                peaks.append(rep)
                peaks_num_points.append(num_points)
            elif value < prev_value and value < next_value:
                valleys.append(rep)
                valleys_num_points.append(num_points)
    return peaks, valleys, peaks_num_points, valleys_num_points

def check_stationary_interval_vectorized(mid, pre_t2, t2, min_length=100):
    # 转换为 NumPy 数组（如果 mid 不是 NumPy 数组）
    mid = np.asarray(mid)
    # 提取指定区间
    mid_range = mid[pre_t2:t2+1]
    
    # 计算相邻元素是否相等
    is_same = np.concatenate(([False], mid_range[1:] == mid_range[:-1]))
    
    # 计算连续相同值的长度
    # 使用 np.where 找到变化点
    change_points = np.where(~is_same)[0]
    # 如果没有变化点，整个区间都是相同的
    if len(change_points) == 0:
        return len(mid_range) > min_length
    
    # 计算每个连续段的长度
    lengths = np.diff(np.concatenate(([0], change_points, [len(mid_range)])))
    
    # 检查是否存在长度 > min_length 的区间
    return np.any(lengths > min_length)

def _find_max_profitable_trades(bid, ask, mid, peaks, valleys, peaks_num_points, valleys_num_points, fee=5e-5, profit_threshold=0.0, profit_fee_times=0):
    """
    寻找所有可盈利的交易对，最大化对数收益率总和，考虑更低波谷的潜在更大利润。

    参数:
    bid (np.array): bid 价格序列
    ask (np.array): ask 价格序列
    valleys (list): 波谷位置列表
    peaks (list): 波峰位置列表
    valleys_num_points (list): 波谷点数列表
    peaks_num_points (list): 波峰点数列表
    fee (float): 交易费率，默认为 5e-5
    profit_threshold (float): 最小利润，默认为 0.0
    profit_fee_times (int): 最小利润率与交易费率的倍数，默认为 0

    返回:
    trades (list): 所有可盈利的交易对列表，每个元素为 (valley, peak)
    total_log_return (float): 所有可盈利交易的对数收益率总和
    """
    trades = []
    total_log_return = 0
    valley_idx = 0
    peak_idx = 0

    def cal_profit(t1, t2):
        buy_cost = ask[t1] * (1 + fee)
        sell_income = bid[t2] * (1 - fee)
        return np.log(sell_income / buy_cost), sell_income - buy_cost

    # 控制边界范围
    if valleys:
        valleys[-1] = min(valleys[-1], len(bid) - 1)
    if peaks:
        peaks[-1] = min(peaks[-1], len(ask) - 1)

    # 记录备选的波谷（相同的价格数值）
    valley_backup = []
    find_latest_valley = None
    # 前一个交易对的数据
    pre_valley_backup = []
    pre_find_latest_valley = None

    pre_t2 = 0# 上一个波峰t
    while valley_idx < len(valleys) and peak_idx < len(peaks):
        t1 = valleys[valley_idx]
        t2 = peaks[peak_idx]

        if t1 >= t2:
            # 波谷在波峰之后，跳到下一个波峰
            peak_idx += 1
            continue

        if t1 + 1 == t2:
            if t2 == len(mid) - 1:
                # 波峰波谷连接，且是最后一个点
                # 直接跳出
                break
            raise Exception(f't1 + 1 == t2')

            # 波谷与波峰连续，没有留有成交空间
            # 1. 尝试使用 t2 + 1 作为 t2
            # 需要检查 mid[t2+ 1] > mid[t1]
            if t2 + 1 < len(mid) and mid[t2 + 1] > mid[t1]:
                peaks[peak_idx] = t2 + 1
                t2 = t2 + 1
            
            # 2. 尝试使用 t1 - 1 作为 t1
            # 需要检查 mid[t1 - 1] < mid[t2] 且 t1-1 > pre_t2
            elif mid[t1 - 1] < mid[t2] and t1 - 1 > pre_t2:
                valleys[valley_idx] = t1 - 1
                t1 = t1 - 1

            # 3. 跳到下一个波峰
            else:
                peak_idx += 1
                continue

        if t1 < pre_t2:
            # 新的波谷肯定药在前一个波峰之后
            valley_idx += 1
            continue

        if mid[t2] < mid[t1]:
            # 波峰不高于波谷，测试下一个波谷
            valley_idx += 1
            continue
        
        if valley_backup and ask[valley_backup[-1][0]] != ask[t1]:
            # 不是同一个批次，需要清空
            valley_backup = []
            find_latest_valley = None

        # 当前波峰波谷中的其他波谷,
        # 选择一个最小值
        _valley_idx = valley_idx + 1
        while _valley_idx < len(valleys) and valleys[_valley_idx] < t2:
            # 时间药满足小于 t2
            _valley_t = valleys[_valley_idx]
            if ask[_valley_t] < ask[t1] or mid[_valley_t] < mid[t1]:
                t1 = _valley_t
                valley_idx = _valley_idx
                valley_backup = []
                find_latest_valley = None

            elif ask[_valley_t] > ask[t1]:
                pass

            elif mid[_valley_t] == mid[t1]:
                # 如果ask相等，检查 相同点之间的距离 / 总交易距离 的占比，
                # 如果占比大于 MAX_FLAT_RATIO，则用新的点替换之前的波谷
                # 用于控制 平整 的占比不能过大(买入后很久才开始上涨)
                no_up_distance = _valley_t - t1
                total_distance = t2 - t1
                if no_up_distance / total_distance >  MAX_FLAT_RATIO:
                    valley_backup.append((t1, valley_idx))
                    find_latest_valley = _valley_t
                    t1 = _valley_t
                    valley_idx = _valley_idx
                else:
                    # 记录 平整 的最后一个点
                    find_latest_valley = _valley_t

            _valley_idx += 1

        # 若 (当前波谷ask >= 上一个波峰bid) 或 (当前波谷与上一个波峰连续，没有留有成交空间)
        # 且 当前波峰bid >= 上一个波峰bid, 则修改上一个波峰为当前波峰
        # 上一个波峰卖出，当前波谷买入 平添交易费/损失持仓量  
        if pre_t2 > 0 and ask[t1] >= bid[pre_t2] and \
            (
                # 当前波峰更高
                ((bid[t2] > bid[pre_t2]) and mid[t2] > mid[pre_t2]) or \
                # 当前波峰是多点，且数值相等，切换到当前波峰会更优
                ((peaks_num_points[peak_idx] > 1) and ((bid[t2] == bid[pre_t2]) and mid[t2] == mid[pre_t2]))
            ):

            # print(f'前一个波峰卖出损失')
            trades[-1] = (trades[-1][0], t2)

            # 遍历计算 平整 的占比
            for _t1, _valley_idx in pre_valley_backup:
                _find_latest_valley = pre_find_latest_valley if pre_find_latest_valley else trades[-1][0]
                no_up_distance = _find_latest_valley - _t1
                total_distance = t2 - _t1
                if no_up_distance / total_distance <= MAX_FLAT_RATIO:
                    # 更新 trades[-1]
                    trades[-1] = (_t1, t2)
                    break

            valley_backup = []
            pre_t2 = t2
            peak_idx += 1
            valley_idx += 1
            continue

        buy_cost = ask[t1] * (1 + fee)
        sell_income = bid[t2] * (1 - fee)
        # 计算利润：卖出收入减去买入成本
        profit = sell_income - buy_cost
        if profit >= profit_threshold and profit/ask[t1] >= profit_fee_times * fee:
            # # 检查 t1,t2 范围内的 mid 是否都在 mid[t1], mid[t2] 之间
            # mid_range = mid[t1:t2+1]
            # mid_min = mid[t1]
            # mid_max = mid[t2]
            # if not ((mid_range >= mid_min) & (mid_range <= mid_max)).all():
            #     # 若存在超出范围的点,跳过当前交易对
            #     peak_idx += 1
            #     continue

            # 确认当前交易
            trades.append((t1, t2))
            pre_t2 = t2
            valley_idx += 1
            peak_idx += 1
            pre_valley_backup = valley_backup
            pre_find_latest_valley = find_latest_valley
            valley_backup = []
            find_latest_valley = None
        else:
            # 不盈利
            # 1. 对比前一个交易对的波峰，若当前的波峰更高，则修改上一个交易对的波峰为当前波峰
            # 2. 若当前的波峰更低，则尝试下一个波峰
            if pre_t2 > 0 and \
                (
                    # 当前波峰更高
                    ((bid[t2] > bid[pre_t2]) and mid[t2] > mid[pre_t2]) or \
                    # 当前波峰是多点，且数值相等，切换到当前波峰会更优
                    ((peaks_num_points[peak_idx] > 1) and ((bid[t2] == bid[pre_t2]) and mid[t2] == mid[pre_t2]))
                 ):

                # print(f'当前波峰波谷不盈利，合并到上一个交易对')
                # 修改上一个交易对的波峰为当前波峰
                trades[-1] = (trades[-1][0], t2)

                # 遍历计算 平整 的占比
                for _t1, _valley_idx in pre_valley_backup:
                    _find_latest_valley = pre_find_latest_valley if pre_find_latest_valley else trades[-1][0]
                    no_up_distance = _find_latest_valley - _t1
                    total_distance = t2 - _t1
                    if no_up_distance / total_distance <= MAX_FLAT_RATIO:
                        # 更新 trades[-1]
                        trades[-1] = (_t1, t2)
                        break

                valley_backup = []
                pre_t2 = t2
                valley_idx += 1
                peak_idx += 1
            else:
                # 尝试下一个波峰
                peak_idx += 1

    # 计算总对数收益率
    slope = []
    for t1, t2 in trades:
        profit, diff = cal_profit(t1, t2)
        total_log_return += profit
        slope.append((profit/(t2 - t1), diff, t1, t2))

    # print(f'slope')
    # for p, d, t1, t2 in slope:
    #     print(f'{p}, {d}, {t1}, {t2}, {t2 - t1}')
    return trades, total_log_return

def max_profit_reachable(bid, ask, rep_select='mid', rng=None):
    """
    计算bid/ask序列中 潜在的最大利润
    rep_select: 波峰波谷的选取方式
        'mid' 表示中间点
        'random' 表示随机点
        'first' 表示第一个点
        'last' 表示最后一个点

    返回 
    trades: 交易对列表，每个元素为 (t1, t2) 表示交易的起点和终点
    total_log_return: 所有可盈利交易的对数收益率总和
    valleys: 波谷位置列表
    peaks: 波峰位置列表
    """
    # 检查 bid/ask, 转化为 numpy
    if not isinstance(bid, np.ndarray):
        bid = np.array(bid)
    if not isinstance(ask, np.ndarray):
        ask = np.array(ask)

    # 计算 mid-price
    mid = (bid + ask) / 2

    # 平滑价格序列
    mid = _smooth_price_series(mid)

    # 识别平台期
    plateaus = _find_plateaus(mid)

    # 识别波峰和波谷
    peaks, valleys, peaks_num_points, valleys_num_points = _identify_peaks_valleys(plateaus, rep_select, rng)

    # 匹配交易对
    # 计算可盈利交易的对数收益率总和
    trades,total_log_return = _find_max_profitable_trades(bid, ask, mid, peaks, valleys, peaks_num_points, valleys_num_points)
    return trades, total_log_return, valleys, peaks

def plot_trades(mid, trades, valleys, peaks):
    """
    使用 Plotly 可视化 mid-price 序列和交易机会，支持交互缩放。
    
    参数:
    mid (np.array): mid-price 序列
    trades (list): 交易对列表，每个元素为 (t1, t2) 表示交易的起点和终点
    valleys (list): 波谷位置列表
    peaks (list): 波峰位置列表
    """
    import plotly.graph_objects as go
    
    # 创建时间轴
    time = list(range(len(mid)))
    
    # 创建交互式图表
    import plotly.graph_objects as go
    fig = go.Figure()
    
    # 绘制 mid-price 曲线
    fig.add_trace(go.Scatter(x=time, y=mid, mode='lines', name='Mid Price', line=dict(color='blue')))
    
    # 添加波谷点
    fig.add_trace(go.Scatter(x=valleys, y=[mid[v] for v in valleys], mode='markers', name='Valleys',
                             marker=dict(color='red', size=10)))
    
    # 添加波峰点
    fig.add_trace(go.Scatter(x=peaks, y=[mid[p] for p in peaks], mode='markers', name='Peaks',
                             marker=dict(color='green', size=10)))
    
    # 添加交易对的连接线
    for t1, t2 in trades:
        fig.add_trace(go.Scatter(x=[t1, t2], y=[mid[t1], mid[t2]], mode='lines',
                                 line=dict(color='green', dash='dash'), name=f'Trade {t1}-{t2}'))
    
    # 设置图表布局，启用缩放
    fig.update_layout(
        title='Mid Price with Trading Opportunities',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='closest',  # 鼠标悬停时显示最近的数据点信息
        template='plotly_white',  # 使用白色背景模板
        dragmode='zoom',  # 设置拖动模式为缩放
        xaxis_rangeslider_visible=False  # 隐藏范围滑块，确保鼠标滚轮缩放
    )
    
    # 启用鼠标滚轮缩放
    fig.update_xaxes(fixedrange=False)  # 允许 x 轴缩放
    fig.update_yaxes(fixedrange=False)  # 允许 y 轴缩放
    
    # 在浏览器中显示图表
    fig.show(renderer='browser')

def plot_trades_plt(mid, trades, valleys, peaks, bid=None, ask=None, figsize=(10, 4), plot_vps=True, begin_diff=0):
    """
    使用 Matplotlib 可视化 mid-price 序列和交易机会。
    
    参数:
    mid (np.array): mid-price 序列
    trades (list): 交易对列表，每个元素为 (t1, t2) 表示交易的起点和终点
    valleys (list): 波谷位置列表
    peaks (list): 波峰位置列表
    """
    # 创建时间轴
    time = list(range(len(mid)))
    
    # 根据 begin_diff 调整时间轴
    time = [t + begin_diff for t in time]
    trades = [(t1 + begin_diff, t2 + begin_diff) for t1, t2 in trades]
    valleys = [v + begin_diff for v in valleys]
    peaks = [p + begin_diff for p in peaks]

    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制 mid-price 曲线
    plt.plot(time, mid, label='Mid Price', color='blue')
    if not bid is None:
        plt.plot(time, bid, label='Bid', color='red', alpha=0.5)
    if not ask is None:
        plt.plot(time, ask, label='Ask', color='green', alpha=0.5)

    # 注交易对涵盖的波峰波谷点
    trade_points = set()
    for t1, t2 in trades:
        trade_points.add(t1)
        trade_points.add(t2)

    # 添加波谷点并标注索引号
    for i, valley in enumerate(valleys):
        if plot_vps or valley in trade_points:
            plt.scatter(valley, mid[valley-begin_diff], color='red', label='Valleys' if i == 0 else None, s=50)
            plt.text(valley, mid[valley-begin_diff]-0.0001, f'{valley}', fontsize=10, ha='center', va='top', color='red')

    # 添加波峰点并标注索引号
    for i, peak in enumerate(peaks):
        if plot_vps or peak in trade_points:
            plt.scatter(peak, mid[peak-begin_diff], color='green', label='Peaks' if i == 0 else None, s=50)
            plt.text(peak, mid[peak-begin_diff]+0.0001, f'{peak}', fontsize=10, ha='center', va='bottom', color='green')

    # 添加交易对的连接线
    for t1, t2 in trades:
        plt.plot([t1, t2], [mid[t1-begin_diff], mid[t2-begin_diff]], color='green', linestyle='--', label=f'Trade {t1}-{t2}')
    
    # 设置图表标题和标签
    plt.title('Mid Price with Trading Opportunities')
    plt.xlabel('Time')
    plt.ylabel('Price')
    
    # 显示图例
    plt.legend()
    
    # 显示网格
    plt.grid(True)
    
    # 显示图表
    plt.show()

def calculate_profit(df, fee=5e-5):
    """
    完全向量化的高效实现
    参数:
    df: 包含 'BASE买1价', 'BASE卖1价', 'action' 列的DataFrame
    
    返回:
    添加了 'profit' 列的DataFrame
    所有action==0的行, 用BASE卖1价买入，下一个action==1时用BASE买1价卖出，的收益
    """
    # 初始化profit列为0
    df['profit'] = 0.0
    
    # 创建信号变化标记，找到买入卖出配对
    # 1. 标记每个信号的索引位置
    buy_idx = np.where(df['action'] == 0)[0]
    sell_idx = np.where(df['action'] == 1)[0]
    
    if len(buy_idx) == 0 or len(sell_idx) == 0:
        return df
    
    # 2. 使用searchsorted一次性找到所有买入位置对应的下一个卖出位置
    # searchsorted找到每个buy_idx在sell_idx中应该插入的位置
    # 这个位置就是第一个大于buy_idx的sell_idx的索引
    next_sell_indices = np.searchsorted(sell_idx, buy_idx, side='right')
    
    # 3. 过滤掉没有对应卖出信号的买入位置
    valid_mask = next_sell_indices < len(sell_idx)
    valid_buy_positions = buy_idx[valid_mask]
    valid_sell_positions = sell_idx[next_sell_indices[valid_mask]]

    # 4. 检查与 valid_sell_positions+1 处连续且BASE买1价相同的个数
    # 若个数 == 1, 则将 valid_sell_positions 向后推延/先前提前
    if len(valid_sell_positions) > 0:
        # 获取 valid_sell_positions+1 处的 BASE买1价
        sell_price = df.loc[valid_sell_positions + 1, 'BASE买1价'].values
        # 获取 valid_sell_positions 处的 BASE买1价
        pre_price = df.loc[valid_sell_positions, 'BASE买1价'].values
        pre_same_price_mask = pre_price == sell_price # 与前一个数据的价格相同，说明连续且数值相同的数据至少2个
        # 初始化 same_price_mask 为全 False，长度与 valid_sell_positions 一致
        next_same_price_mask = np.zeros_like(pre_same_price_mask, dtype=bool)
        # 获取 valid_sell_positions+2 处的 BASE买1价（可能越界，需检查）
        next_indices = valid_sell_positions + 2
        next_indices = next_indices[next_indices < len(df)]  # 确保不越界

        # 检查向前是否越界
        valid_mask = valid_sell_positions > 0  # 确保不会向前越界到负数索引
        if len(next_indices) > 0:
            next_price = df.loc[next_indices, 'BASE买1价'].values
            # 向量化的价格比较，检查 sell_price 和 next_price 是否相等
            same_price_mask = sell_price[:len(next_price)] == next_price # 与后一个数据的价格相同，说明连续且数值相同的数据至少2个
            # 将结果放回对应位置
            next_same_price_mask[:len(same_price_mask)] = same_price_mask
            # 个数 == 1, 则将 valid_sell_positions += 1， 向后推延
            # 两个mask都不满足 > 个数 == 1
            need_push_mask = (~next_same_price_mask) & (~pre_same_price_mask)

            # next_price >= pre_price 时 或 valid_mask不合法时，向后推延 (+1)
            forward_mask = need_push_mask[:len(next_price)] & ((next_price >= pre_price[:len(next_price)]) | ~valid_mask[:len(next_price)])
            # 初始化最终的 mask
            final_mask = np.zeros_like(pre_same_price_mask, dtype=bool)
            final_mask[:len(next_price)] = forward_mask
            valid_sell_positions[final_mask] += 1
            
            # next_price < pre_price 时，向前推延 (-1)
            adjust_mask = need_push_mask & valid_mask# 只有向前推的需要 valid_mask 过滤
            backward_mask = (next_price < pre_price[:len(next_price)]) & adjust_mask[:len(next_price)]
            final_mask[:len(next_price)] = backward_mask
            valid_sell_positions[final_mask] -= 1

        else:
            # 只判断 pre_same_price_mask
            need_push_mask = ~pre_same_price_mask
            # 向后的数据不存在，只能向前
            adjust_mask = need_push_mask & valid_mask
            valid_sell_positions[adjust_mask] -= 1

    # 5. 批量计算所有利润
    if len(valid_buy_positions) > 0:
        # 获取价格数组（避免逐个loc访问）
        buy_prices = df.loc[valid_buy_positions+1, 'BASE卖1价'].values
        sell_prices = df.loc[valid_sell_positions+1, 'BASE买1价'].values
        
        # 向量化计算利润
        buy_cost = buy_prices * (1 + fee)
        sell_income = sell_prices * (1 - fee)
        profits = np.log(sell_income / buy_cost)
        
        # 一次性更新利润
        df.loc[valid_buy_positions, 'profit'] = profits
    
    return df

def calculate_sell_save(df, fee=5e-5):
    """
    真正的完全向量化实现，不使用任何循环：
    对于所有 action==1 的行:
    1. 在下一行的 BASE买1价 卖出
    2. 在下一个 action==0 后的下一行的 BASE卖1价 买入
    3. 计算对数收益，考虑交易费用 fee=5e-5
    """
    # 最后一行添加一个买入点，
    # 价格为最后一个action=0至末尾区间的最低价
    # 找到最后一个 action=0 的索引
    last_zero_index = df[df['action'] == 0].index[-1] if not df[df['action'] == 0].empty else 0
    # 获取从最后一个 action=0 到末尾的最低价
    min_bid = df['BASE买1价'][last_zero_index:].min()
    min_ask = df['BASE卖1价'][last_zero_index:].min()
    new_row = {
        'BASE买1价': min_bid, 
        'BASE卖1价': min_ask, 
        'before_market_close_sec': 0.0, 
        'valley_peak': 0, 
        'action': 0, 
        'profit': 0.0, 
    }
    # 使用 loc 添加 2 行
    df.loc[len(df)] = new_row
    df.loc[len(df)] = new_row

    # 原地添加 sell_save 列并初始化为0
    df['sell_save'] = 0.0
    
    # 获取关键信息
    n = len(df)
    
    # 检查是否存在足够的数据
    if n <= 2:
        return df
    
    # 创建表示位置的数组
    positions = np.arange(n)
    
    # 获取action的布尔掩码
    sell_mask = df['action'].values == 1
    buy_mask = df['action'].values == 0
    
    # 没有卖出点则直接返回
    if not np.any(sell_mask):
        return df
    
    # 找到所有卖出和买入的位置
    sell_positions = positions[sell_mask]
    buy_positions = positions[buy_mask]
    
    # 如果没有买入点，直接返回
    if len(buy_positions) == 0:
        return df
    
    # 使用另一种方法找到每个卖出点之后的最近买入点
    # 为每个卖出点创建一个掩码，标识其后的买入点
    next_buy_positions = np.zeros(len(sell_positions), dtype=np.int64)
    next_buy_found = np.zeros(len(sell_positions), dtype=bool)
    
    # 为每个卖出点创建掩码数组，找出之后的买入点
    for i, sell_pos in enumerate(sell_positions):
        # 找出所有在卖出点之后的买入点
        mask = buy_positions > sell_pos
        if np.any(mask):
            # 获取最近的买入点
            next_buy_positions[i] = np.min(buy_positions[mask])
            next_buy_found[i] = True

    # 只保留找到了下一个买入点的卖出点
    valid_sell_positions = sell_positions[next_buy_found]
    valid_next_buy_positions = next_buy_positions[next_buy_found]
    
    # 确保卖出点和买入点后都有行可用于获取价格
    valid_mask = (valid_sell_positions + 1 < n) & (valid_next_buy_positions + 1 < n)
    final_sell_positions = valid_sell_positions[valid_mask]
    final_buy_positions = valid_next_buy_positions[valid_mask]
    
    # 如果没有有效匹配，直接返回
    if len(final_sell_positions) == 0:
        return df

    # 检查与 final_buy_positions+1 处连续且BASE卖1价相同的个数
    # 若个数 == 1, 则将 final_buy_positions 向后推延/先前提前
    if len(final_buy_positions) > 0:
        # 获取 final_buy_positions+1 处的 BASE卖1价
        buy_price = df['BASE卖1价'].values[final_buy_positions + 1]
        # 获取 final_buy_positions 处的 BASE卖1价
        pre_price = df['BASE卖1价'].values[final_buy_positions]
        pre_same_price_mask = pre_price == buy_price  # 与前一个数据的价格相同，说明连续且数值相同的数据至少2个

        # 初始化 next_same_price_mask 为全 False，长度与 final_buy_positions 一致
        next_same_price_mask = np.zeros_like(pre_same_price_mask, dtype=bool)
        # 获取 final_buy_positions+2 处的 BASE卖1价（可能越界，需检查）
        next_indices = final_buy_positions + 2
        next_indices = next_indices[next_indices < len(df)]  # 确保不越界

        # 检查向前是否越界
        valid_mask = final_buy_positions > 0  # 确保不会向前越界到负数索引
        if len(next_indices) > 0:
            next_price = df['BASE卖1价'].values[next_indices]
            # 向量化的价格比较，检查 buy_price 和 next_price 是否相等
            same_price_mask = buy_price[:len(next_price)] == next_price  # 与后一个数据的价格相同
            # 将结果放回对应位置
            next_same_price_mask[:len(same_price_mask)] = same_price_mask
            # 个数 == 1 时，需调整 final_buy_positions
            # 两个 mask 都不满足 > 个数 == 1
            need_push_mask = (~next_same_price_mask) & (~pre_same_price_mask)

            # next_price <= pre_price 时 或 valid_mask 不合法时，向后推延 (+1)
            forward_mask = need_push_mask[:len(next_price)] & ((next_price <= pre_price[:len(next_price)]) | ~valid_mask[:len(next_price)])
            # 初始化最终的 mask
            final_mask = np.zeros_like(pre_same_price_mask, dtype=bool)
            final_mask[:len(next_price)] = forward_mask
            final_buy_positions[final_mask] += 1

            # next_price > pre_price 时，向前推延 (-1)
            adjust_mask = need_push_mask & valid_mask  # 只有向前推的需要 valid_mask 过滤
            backward_mask = (next_price > pre_price[:len(next_price)]) & adjust_mask[:len(next_price)]
            final_mask[:len(next_price)] = backward_mask
            final_buy_positions[final_mask] -= 1
        else:
            # 只判断 pre_same_price_mask
            need_push_mask = ~pre_same_price_mask
            # 向后的数据不存在，只能向前
            adjust_mask = need_push_mask & valid_mask
            final_buy_positions[adjust_mask] -= 1
    
    # 获取卖出价格（下一行的BASE买1价）和买入价格（下一行的BASE卖1价）
    sell_prices = df['BASE买1价'].values[final_sell_positions + 1]
    buy_prices = df['BASE卖1价'].values[final_buy_positions + 1]
    
    # 计算对数收益
    sell_after_fee = sell_prices * (1 - fee)
    buy_after_fee = buy_prices * (1 + fee)
    log_returns = np.log(sell_after_fee / buy_after_fee)
    
    # 更新DataFrame中的sell_save列
    df.loc[df.index[final_sell_positions], 'sell_save'] = log_returns
    
    # 删除最后一行
    df = df.iloc[:-2]

    return df

def filte_no_move_profit(df, no_move_threshold=50):
    """
    去除超过阈值价格没有波动的连续块
    功能一：连续 no move 超过阈值个，空仓情况下不进行买入（profit=0）
            会保留最后一个 no move 的买入动作，因为之后价格开始变化（信号变动有效）
    功能二（修改）：去除剩余的 profit>0 且时点价格波动小于 0.0012 的连续块
            找到超过阈值的每组第一行之前连续的 profit>0 范围，
            向前至 profit<=0 或前一个超过阈值的组的末尾，
            若范围内 BASE卖1价 的 min/max 之差 < 0.0012，则将这个范围内的 profit 置为 0
    """
    # 检测 mid_price 是否变化（与前一行比较）
    mid = df['BASE卖1价'] + df['BASE买1价']
    changes = mid.ne(mid.shift(1))  # 标记价格变化的位置
    # 使用 cumsum() 创建分组，相同值的连续段属于同一组
    groups = changes.cumsum()
    # 计算每个组的长度
    group_lengths = df.groupby(groups).size()
    # 将每个组的长度映射回原始 DataFrame
    no_move_len = groups.map(group_lengths)
    # 标记每组的最后一行
    is_last_in_group = mid.ne(mid.shift(-1)) | df.index.isin([len(df)-1])
    # 将每组最后一个值的 no_move_len 设为 0
    no_move_len[is_last_in_group] = 0
    
    # 找到超过阈值的组
    over_threshold_groups = group_lengths[group_lengths > no_move_threshold].index
    
    # 获取所有超过阈值的组的末尾索引
    group_end_indices = {}
    for group in over_threshold_groups:
        group_end_idx = df[groups == group].index[-1]
        group_end_indices[group] = group_end_idx
    
    # 处理功能二：查找连续 profit>0 范围并检查价格波动
    for group in over_threshold_groups:
        # 找到该组的起始索引
        group_start_idx = df[groups == group].index[0]
        
        # 向前查找连续 profit>0 的范围
        current_idx = group_start_idx - 1
        profit_positive_start = group_start_idx  # 默认设为组开始位置
        
        # 获取前一个超过阈值的组的末尾索引（如果存在）
        prev_group_end_idx = max(
            [end_idx for g, end_idx in group_end_indices.items() if end_idx < group_start_idx],
            default=-1
        )
        prev_group_end_idx -=1
        
        # 向前查找，直到 action==1 或到达前一个超过阈值的组的末尾
        while current_idx > prev_group_end_idx and current_idx >= 0 and df.iloc[current_idx]['action'] == 0:
            profit_positive_start = current_idx
            current_idx -= 1
            
        # 如果找到了 profit>0 的范围
        if profit_positive_start < group_start_idx:
            # 检查 BASE卖1价 的 min/max 之差
            price_range = df.loc[profit_positive_start:group_start_idx-1, 'BASE卖1价']
            if price_range.max() - price_range.min() < 0.0012:
                # 将范围内 profit 置为 0
                df.loc[profit_positive_start:group_start_idx-1, 'profit'] = 0
    
    # 功能一：超过阈值的组的 profit 置为 0
    df.loc[no_move_len > no_move_threshold, 'profit'] = 0
    
    return df

def filte_no_move_0(df, no_move_threshold=50):
    """
    去除超过阈值价格没有波动的连续块，处理 profit 和 sell_save 两列
    功能一（profit）：连续 no move 超过阈值个，空仓情况下不进行买入（profit=0）
                     会保留最后一个 no move 的买入动作，因为之后价格开始变化（信号变动有效）
    功能二（profit）：去除剩余的 profit>0 且时点价格波动小于 0.0012 的连续块
                     找到超过阈值的每组第一行之前连续的 profit>0 范围，
                     向前至 profit<=0 或前一个超过阈值的组的末尾，
                     若范围内 BASE卖1价 的 min/max 之差 < 0.0012，则将这个范围内的 profit 置为 0

    功能一（sell_save）：连续 no move 超过阈值个，持仓情况下不进行卖出（sell_save=0）
                        会保留最后一个 no move 的卖出动作，因为之后价格开始变化（信号变动有效）
    功能二（sell_save）：去除剩余的 sell_save>0 且时点价格波动小于 0.0012 的连续块
                        找到超过阈值的每组第一行之前连续的 sell_save>0 范围，
                        向前至 sell_save<=0 或前一个超过阈值的组的末尾，
                        若范围内 BASE买1价 的 min/max 之差 < 0.0012，则将这个范围内的 sell_save 置为 0

    # TODO 改成 
    # action==0 中间若有多段 profit>0, 需要对每段重新计算 profit
    # action==1 中间若有多段 sell_save>0, 需要对每段重新计算 sell_save
    """
    # 检测 mid_price 是否变化（与前一行比较）
    mid = df['BASE卖1价'] + df['BASE买1价']
    changes = mid.ne(mid.shift(1))  # 标记价格变化的位置
    # 使用 cumsum() 创建分组，相同值的连续段属于同一组
    groups = changes.cumsum()
    # 计算每个组的长度
    group_lengths = df.groupby(groups).size()
    # 将每个组的长度映射回原始 DataFrame
    no_move_len = groups.map(group_lengths)
    # 标记每组的最后一行
    is_last_in_group = mid.ne(mid.shift(-1)) | df.index.isin([len(df)-1])
    # 将每组最后一个值的 no_move_len 设为 0
    no_move_len[is_last_in_group] = 0
    
    # 找到超过阈值的组
    over_threshold_groups = group_lengths[group_lengths > no_move_threshold].index
    
    # 获取所有超过阈值的组的末尾索引
    group_end_indices = {}
    for group in over_threshold_groups:
        group_end_idx = df[groups == group].index[-1]
        group_end_indices[group] = group_end_idx
    
    def find_last_max_mid_value(range_data):
        range_mid = (range_data['BASE卖1价'] + range_data['BASE买1价']) / 2
        mid_set = range_mid.drop_duplicates().sort_values(ascending=False)
        # 遍历 mid_set，连续的最大值的最后一个
        max_idx = -1
        max_mid_value = 0
        range_mid = range_mid.values
        for max_v in mid_set:
            _idx = np.where(range_mid == max_v)[0]
            for _i in _idx[::-1]:
                if range_mid[_i] == range_mid[_i-1]:
                    max_idx = _i
                    max_mid_value = max_v
                    break
            if max_mid_value != 0:
                break
        return max_idx, max_mid_value
    
    def find_last_min_mid_value(range_data):
        """
        找到最后一个正值索引 last_idx，
        last_idx 之前的数据，若正值保留，若非正值用向前最近的正值替换
        """
        range_mid = (range_data['BASE卖1价'] + range_data['BASE买1价']) / 2
        mid_set = range_mid.drop_duplicates().sort_values()
        # 遍历 mid_set，连续的最小值的最后一个
        min_idx = -1
        min_mid_value = 0
        range_mid = range_mid.values
        for min_v in mid_set:
            _idx = np.where(range_mid == min_v)[0]
            for _i in _idx[::-1]:
                if range_mid[_i] == range_mid[_i-1]:
                    min_idx = _i
                    min_mid_value = min_v
                    break
            if min_mid_value != 0:
                break
        return min_idx, min_mid_value
        
    def fill_with_last_positive(arr):
        arr = arr.copy()
        pos_indices = np.where(arr > 0)[0]
        if len(pos_indices) == 0:
            return arr

        last_idx = pos_indices[-1]
        prefix = arr[:last_idx]

        # 标记正值的位置，其它为 False
        is_pos = prefix > 0
        # 创建索引数组，用于跟踪每个位置上一个正值的索引
        last_pos_idx = np.where(is_pos, np.arange(len(prefix)), -1)
        last_pos_idx = np.maximum.accumulate(last_pos_idx)

        # 替换 <= 0 的位置
        replace_mask = (prefix <= 0) & (last_pos_idx != -1)
        arr[:last_idx][replace_mask] = prefix[last_pos_idx[replace_mask]]

        return arr

    # 处理功能二：查找连续 profit>0 和 sell_save>0 的范围并检查价格波动
    for group in over_threshold_groups:
        # 找到该组的起始索引
        group_start_idx = df[groups == group].index[0]
        
        # 获取前一个超过阈值的组的末尾索引（如果存在）
        prev_group_end_idx = max(
            [end_idx for g, end_idx in group_end_indices.items() if end_idx < group_start_idx],
            default=-1
        )
        prev_group_end_idx -= 1
        
        # --- 处理 profit ---
        # 向前查找连续 profit>0 的范围
        current_idx = group_start_idx - 1
        profit_positive_start = group_start_idx  # 默认设为组开始位置
        while current_idx > prev_group_end_idx and current_idx >= 0 and df.iloc[current_idx]['action'] == 0:
            profit_positive_start = current_idx
            current_idx -= 1
        # 如果找到了 profit>0 的范围
        if profit_positive_start < group_start_idx:
            # 检查 BASE卖1价 的 min/max 之差
            range_data = df.loc[profit_positive_start:group_start_idx+1, :]
            price_range = range_data['BASE卖1价']
            if price_range.max() - price_range.min() < 0.0012:
                # 将范围内 profit 置为 0
                df.loc[profit_positive_start:group_start_idx+1, 'profit'] = 0
            else:
                # 检查 这个区间是否也满足可盈利 TODO
                # 按照区间最后一个最高价格作为卖出价（必须是连续的，数量>1）
                max_idx, max_mid_value = find_last_max_mid_value(range_data)
                # 时间大于等于卖出价时刻-1的 profit 置为 0
                df.loc[range_data.iloc[max_idx-1:].index, 'profit'] = 0
                # 时间早于卖出时刻-1的 profit 重新计算: 下一个时刻卖1价买入成交， 卖出价时刻的买1价卖出成交，计算 profit
                buy_cost = range_data.iloc[1:max_idx]['BASE卖1价'] * (1 + 5e-5)
                sell_gain = range_data.iloc[max_idx]['BASE买1价'] * (1 - 5e-5)
                profit = np.log(sell_gain / buy_cost)
                profit = fill_with_last_positive(profit.values)
                df.loc[range_data.iloc[:max_idx-1].index, 'profit'] = profit

        # --- 处理 sell_save ---
        # 向前查找连续 sell_save>0 的范围
        current_idx = group_start_idx - 1
        sell_save_positive_start = group_start_idx  # 默认设为组开始位置
        while current_idx > prev_group_end_idx and current_idx >= 0 and df.iloc[current_idx]['action'] == 1:
            sell_save_positive_start = current_idx
            current_idx -= 1
        # 如果找到了 sell_save>0 的范围
        if sell_save_positive_start < group_start_idx:
            # 检查 BASE买1价 的 min/max 之差
            range_data = df.iloc[sell_save_positive_start:group_start_idx+1, :]
            print(sell_save_positive_start, group_start_idx, len(df))
            range_data.to_csv(r"C:\Users\lh\Desktop\temp\range_data.csv", encoding='gbk')
            price_range = range_data['BASE买1价']
            if price_range.max() - price_range.min() < 0.0012:
                # 将范围内 sell_save 置为 0
                df.loc[sell_save_positive_start:group_start_idx+1, 'sell_save'] = 0
            else:
                # 检查 这个区间是否也满足可节省 TODO
                # 按照区间最后一个最低价格作为买入价（必须是连续的，数量>1）
                min_idx, min_mid_value = find_last_min_mid_value(range_data)
                # 时间大于等于买入价时刻-1的 sell_save 置为 0
                df.loc[range_data.iloc[min_idx-1:].index, 'sell_save'] = 0
                # 时间早于买入时刻-1的 sell_save 重新计算: 下一个时刻买1价卖出成交， 买入价时刻的卖1价买入成交，计算 sell_save
                buy_cost = range_data.iloc[min_idx]['BASE卖1价'] * (1 + 5e-5)
                sell_gain = range_data.iloc[1:min_idx]['BASE买1价'] * (1 - 5e-5)
                sell_save = np.log(sell_gain / buy_cost)
                sell_save = fill_with_last_positive(sell_save.values)
                df.loc[range_data.iloc[:min_idx-1].index, 'sell_save'] = sell_save

    
    # 功能一：超过阈值的组的 profit 和 sell_save 置为 0
    df.loc[no_move_len > no_move_threshold, ['profit', 'sell_save']] = 0
    
    return df

def filte_no_move_1(df, no_move_threshold=50):
    """
    去除超过阈值价格没有波动的连续块，处理 profit 和 sell_save 两列
    功能一（profit）：连续 无新高 超过阈值个，空仓情况下不进行买入（profit=0）
                     会保留最后一个 无新高 的买入动作，因为之后价格开始变化（信号变动有效）
    功能二（profit）：去除剩余的 profit>0 且时点价格波动小于 0.0012 的连续块
                     找到超过阈值的每组第一行之前连续的 profit>0 范围，
                     向前至 profit<=0 或前一个超过阈值的组的末尾，
                     若范围内 BASE卖1价 的 min/max 之差 < 0.0012，则将这个范围内的 profit 置为 0

    功能一（sell_save）：连续 无新低 超过阈值个，持仓情况下不进行卖出（sell_save=0）
                        会保留最后一个 无新低 的卖出动作，因为之后价格开始变化（信号变动有效）
    功能二（sell_save）：去除剩余的 sell_save>0 且时点价格波动小于 0.0012 的连续块
                        找到超过阈值的每组第一行之前连续的 sell_save>0 范围，
                        向前至 sell_save<=0 或前一个超过阈值的组的末尾，
                        若范围内 BASE买1价 的 min/max 之差 < 0.0012，则将这个范围内的 sell_save 置为 0
    """
    # 检测 mid_price 是否变化（与前一行比较）
    mid = df['BASE卖1价'] + df['BASE买1价']
    mid = _smooth_price_series(mid)

    new_window = 10

    # 处理 profit：检测“无新高”的连续段
    is_new_high = mid > mid.rolling(window=new_window, min_periods=1).max().shift(1).fillna(mid.iloc[0])  # 判断是否创新高
    groups_profit = is_new_high.cumsum()  # 创建分组，每创新高则组号+1
    group_lengths_profit = df.groupby(groups_profit).size()  # 计算每个组的长度
    no_move_len_profit = groups_profit.map(group_lengths_profit)  # 映射组长度到每行
    is_last_in_group_profit = (groups_profit.shift(-1) > groups_profit) | (df.index == len(df)-1)  # 标记每组最后一行
    # 将每组最后一个值的 no_move_len 设为 0
    no_move_len_profit[is_last_in_group_profit] = 0
    # 找到超过阈值的组
    over_threshold_groups_profit = group_lengths_profit[group_lengths_profit > no_move_threshold].index
    # 获取所有超过阈值的组的末尾索引
    group_end_indices_profit = {}
    for group in over_threshold_groups_profit:
        group_end_idx = df[groups_profit == group].index[-1]
        group_end_indices_profit[group] = group_end_idx
    
    # 处理 sell_save：检测“无新低”的连续段
    is_new_low = mid < mid.rolling(window=new_window, min_periods=1).min().shift(1).fillna(mid.iloc[0])  # 判断是否创新低
    groups_sell_save = is_new_low.cumsum()  # 创建分组，每创新低则组号+1
    group_lengths_sell_save = df.groupby(groups_sell_save).size()  # 计算每个组的长度
    no_move_len_sell_save = groups_sell_save.map(group_lengths_sell_save)  # 映射组长度到每行
    is_last_in_group_sell_save = (groups_sell_save.shift(-1) > groups_sell_save) | (df.index == len(df)-1)  # 标记每组最后一行
    # 将每组最后一个值的 no_move_len 设为 0
    no_move_len_sell_save[is_last_in_group_sell_save] = 0
    # 找到超过阈值的组
    over_threshold_groups_sell_save = group_lengths_sell_save[group_lengths_sell_save > no_move_threshold].index
    # 获取所有超过阈值的组的末尾索引
    group_end_indices_sell_save = {}
    for group in over_threshold_groups_sell_save:
        group_end_idx = df[groups_sell_save == group].index[-1]
        group_end_indices_sell_save[group] = group_end_idx

    def find_last_max_mid_value(range_data):
        range_mid = (range_data['BASE卖1价'] + range_data['BASE买1价']) / 2
        mid_set = range_mid.drop_duplicates().sort_values(ascending=False)
        # 遍历 mid_set，连续的最大值的最后一个
        max_idx = -1
        max_mid_value = 0
        range_mid = range_mid.values
        for max_v in mid_set:
            _idx = np.where(range_mid == max_v)[0]
            for _i in _idx[::-1]:
                if range_mid[_i] == range_mid[_i-1]:
                    max_idx = _i
                    max_mid_value = max_v
                    break
            if max_mid_value != 0:
                break
        return max_idx, max_mid_value
    
    def find_last_min_mid_value(range_data):
        """
        找到最后一个正值索引 last_idx，
        last_idx 之前的数据，若正值保留，若非正值用向前最近的正值替换
        """
        range_mid = (range_data['BASE卖1价'] + range_data['BASE买1价']) / 2
        mid_set = range_mid.drop_duplicates().sort_values()
        # 遍历 mid_set，连续的最小值的最后一个
        min_idx = -1
        min_mid_value = 0
        range_mid = range_mid.values
        for min_v in mid_set:
            _idx = np.where(range_mid == min_v)[0]
            for _i in _idx[::-1]:
                if range_mid[_i] == range_mid[_i-1]:
                    min_idx = _i
                    min_mid_value = min_v
                    break
            if min_mid_value != 0:
                break
        return min_idx, min_mid_value
        
    def fill_with_last_positive(arr):
        arr = arr.copy()
        pos_indices = np.where(arr > 0)[0]
        if len(pos_indices) == 0:
            return arr

        last_idx = pos_indices[-1]
        prefix = arr[:last_idx]

        # 标记正值的位置，其它为 False
        is_pos = prefix > 0
        # 创建索引数组，用于跟踪每个位置上一个正值的索引
        last_pos_idx = np.where(is_pos, np.arange(len(prefix)), -1)
        last_pos_idx = np.maximum.accumulate(last_pos_idx)

        # 替换 <= 0 的位置
        replace_mask = (prefix <= 0) & (last_pos_idx != -1)
        arr[:last_idx][replace_mask] = prefix[last_pos_idx[replace_mask]]

        return arr

    # 处理功能二：查找连续 profit>0 的范围并检查价格波动
    for group in over_threshold_groups_profit:
        # 找到该组的起始索引
        group_start_idx = df[groups_profit == group].index[0]
        
        # 获取前一个超过阈值的组的末尾索引（如果存在）
        prev_group_end_idx = max(
            [end_idx for g, end_idx in group_end_indices_profit.items() if end_idx < group_start_idx],
            default=-1
        )
        prev_group_end_idx -= 1
        
        # --- 处理 profit ---
        # 向前查找连续 profit>0 的范围
        current_idx = group_start_idx - 1
        profit_positive_start = group_start_idx  # 默认设为组开始位置
        while current_idx > prev_group_end_idx and current_idx >= 0 and df.iloc[current_idx]['action'] == 0:
            profit_positive_start = current_idx
            current_idx -= 1
        # 如果找到了 profit>0 的范围
        if profit_positive_start < group_start_idx:
            # 检查 BASE卖1价 的 min/max 之差
            range_data = df.loc[profit_positive_start:group_start_idx+1, :]
            price_range = range_data['BASE卖1价']
            if price_range.max() - price_range.min() < 0.0012:
                # 将范围内 profit 置为 0
                df.loc[profit_positive_start:group_start_idx+1, 'profit'] = 0
            else:
                # 检查 这个区间是否也满足可盈利 TODO
                # 按照区间最后一个最高价格作为卖出价（必须是连续的，数量>1）
                max_idx, max_mid_value = find_last_max_mid_value(range_data)
                # 时间大于等于卖出价时刻-1的 profit 置为 0
                df.loc[range_data.iloc[max_idx-1:].index, 'profit'] = 0
                # 时间早于卖出时刻-1的 profit 重新计算: 下一个时刻卖1价买入成交， 卖出价时刻的买1价卖出成交，计算 profit
                buy_cost = range_data.iloc[1:max_idx]['BASE卖1价'] * (1 + 5e-5)
                sell_gain = range_data.iloc[max_idx]['BASE买1价'] * (1 - 5e-5)
                profit = np.log(sell_gain / buy_cost)
                profit = fill_with_last_positive(profit.values)
                df.loc[range_data.iloc[:max_idx-1].index, 'profit'] = profit
                
    # 处理功能二：查找连续 sell_save>0 的范围并检查价格波动
    for group in over_threshold_groups_sell_save:
        # 找到该组的起始索引
        group_start_idx = df[groups_sell_save == group].index[0]
        
        # 获取前一个超过阈值的组的末尾索引（如果存在）
        prev_group_end_idx = max(
            [end_idx for g, end_idx in group_end_indices_sell_save.items() if end_idx < group_start_idx],
            default=-1
        )
        prev_group_end_idx -= 1
        
        # --- 处理 sell_save ---
        # 向前查找连续 sell_save>0 的范围
        current_idx = group_start_idx - 1
        sell_save_positive_start = group_start_idx  # 默认设为组开始位置
        while current_idx > prev_group_end_idx and current_idx >= 0 and df.iloc[current_idx]['action'] == 1:
            sell_save_positive_start = current_idx
            current_idx -= 1
        # 如果找到了 sell_save>0 的范围
        if sell_save_positive_start < group_start_idx:
            # 检查 BASE买1价 的 min/max 之差
            range_data = df.iloc[sell_save_positive_start:group_start_idx+1, :]
            price_range = range_data['BASE买1价']
            if price_range.max() - price_range.min() < 0.0012:
                # 将范围内 sell_save 置为 0
                df.loc[sell_save_positive_start:group_start_idx+1, 'sell_save'] = 0
            else:
                # 检查 这个区间是否也满足可节省 TODO
                # 按照区间最后一个最低价格作为买入价（必须是连续的，数量>1）
                min_idx, min_mid_value = find_last_min_mid_value(range_data)
                # 时间大于等于买入价时刻-1的 sell_save 置为 0
                df.loc[range_data.iloc[min_idx-1:].index, 'sell_save'] = 0
                # 时间早于买入时刻-1的 sell_save 重新计算: 下一个时刻买1价卖出成交， 买入价时刻的卖1价买入成交，计算 sell_save
                buy_cost = range_data.iloc[min_idx]['BASE卖1价'] * (1 + 5e-5)
                sell_gain = range_data.iloc[1:min_idx]['BASE买1价'] * (1 - 5e-5)
                sell_save = np.log(sell_gain / buy_cost)
                sell_save = fill_with_last_positive(sell_save.values)
                df.loc[range_data.iloc[:min_idx-1].index, 'sell_save'] = sell_save
    
    # 功能一：超过阈值的组的 profit 置为 0
    df.loc[no_move_len_profit > no_move_threshold, 'profit'] = 0
    # 功能一：超过阈值的组的 sell_save 置为 0
    df.loc[no_move_len_sell_save > no_move_threshold, 'sell_save'] = 0

    return df

def filte_no_move(df, no_move_threshold=50):
    """
    去除超过阈值价格没有波动的连续块，处理 profit 和 sell_save 两列
    功能一（profit）：连续 no move 超过阈值个，空仓情况下不进行买入（profit=0）
                     会保留最后一个 no move 的买入动作，因为之后价格开始变化（信号变动有效）

    功能一（sell_save）：连续 no move 超过阈值个，持仓情况下不进行卖出（sell_save=0）
                        会保留最后一个 no move 的卖出动作，因为之后价格开始变化（信号变动有效）
    """
    # 检测 mid_price 是否变化（与前一行比较）
    mid = df['BASE卖1价'] + df['BASE买1价']
    mid = pd.Series(_smooth_price_series(mid.values))

    changes = mid.ne(mid.shift(1))  # 标记价格变化的位置
    # 使用 cumsum() 创建分组，相同值的连续段属于同一组
    groups = changes.cumsum()
    # 计算每个组的长度
    group_lengths = df.groupby(groups).size()
    # 将每个组的长度映射回原始 DataFrame
    no_move_len = groups.map(group_lengths)

    # # 标记每组的最后一行
    # is_last_in_group = mid.ne(mid.shift(-1)) | df.index.isin([len(df)-1])
    # # 将每组最后一个值的 no_move_len 设为 0
    # no_move_len[is_last_in_group] = 0

    # 标记每组的最后两行
    is_last_two_in_group = mid.ne(mid.shift(-1)) | mid.ne(mid.shift(-2)) | df.index.isin([len(df)-1, len(df)-2])
    # 将每组最后两个值的 no_move_len 设为 0
    no_move_len[is_last_two_in_group] = 0
    
    # 功能一：超过阈值的组的 profit 和 sell_save 置为 0
    df.loc[no_move_len > no_move_threshold, ['profit', 'sell_save']] = 0
    
    return df

def fix_profit_sell_save(df):
    """ 
    df 列字段: action, profit, sell_save, BASE卖1价, BASE买1价
    action==0 中间若有多段 profit>0, 需要对每段重新计算 profit
    action==1 中间若有多段 sell_save>0, 需要对每段重新计算 sell_save
    """
    def find_last_max_b1_value(range_data):
        range_b1 = range_data['BASE买1价']
        b1_set = range_b1.drop_duplicates().sort_values(ascending=False)
        # 遍历 mid_set，连续的最大值的最后一个
        max_idx = -1
        max_b1_value = 0
        range_b1 = range_b1.values
        for max_v in b1_set:
            _idx = np.where(range_b1 == max_v)[0]
            for _i in _idx[::-1]:
                if _i == 0:
                    # 第一个值了
                    break
                if range_b1[_i] == range_b1[_i-1]:
                    max_idx = _i
                    max_b1_value = max_v
                    break
            if max_b1_value != 0:
                break
        return max_idx, max_b1_value
    
    def find_last_min_a1_value(range_data):
        range_a1 = range_data['BASE卖1价']
        a1_set = range_a1.drop_duplicates().sort_values()
        # 遍历 mid_set，连续的最小值的最后一个
        min_idx = -1
        min_a1_value = 0
        range_a1 = range_a1.values
        for min_v in a1_set:
            _idx = np.where(range_a1 == min_v)[0]
            for _i in _idx[::-1]:
                if _i == 0:
                    # 第一个值了
                    break
                if range_a1[_i] == range_a1[_i-1]:
                    min_idx = _i
                    min_a1_value = min_v
                    break
            if min_a1_value != 0:
                break
        return min_idx, min_a1_value
        
    def find_equal_begin_last_mid_idx(range_data):
        """
        寻找与第一个值相等的最后一个索引(至少连续的2个)
        用于上涨段中找到上涨的起点
        """
        range_mid = (range_data['BASE卖1价'] + range_data['BASE买1价']) / 2
        begin_value = range_mid.iloc[0]
        last_idx = 0
        min_mid_value = begin_value
        range_mid = range_mid.values
        _idx = np.where(range_mid == min_mid_value)[0]
        for _i in _idx[::-1]:
            if _i == 0:
                # 第一个值了
                break

            # # 至少连续的2个
            # if range_mid[_i] == range_mid[_i-1]:
            #     last_idx = _i
            #     break

            # 1个即可满足
            last_idx = _i
            break

        return last_idx

    def find_equal_end_first_mid_idx(range_data):
        """
        寻找与最后一个值相等的第一个索引(至少连续的2个)
        用于上涨段中找到上涨的终点
        """
        range_mid = (range_data['BASE卖1价'] + range_data['BASE买1价']) / 2
        end_value = range_mid.iloc[-1]
        first_idx = len(range_mid) - 1
        max_mid_value = end_value
        range_mid = range_mid.values
        _idx = np.where(range_mid == max_mid_value)[0]
        for _i in _idx:
            if _i == len(range_mid) - 1:
                break

            # # 至少连续的2个
            # if range_mid[_i] == range_mid[_i+1]:
            #     first_idx = _i+1
            #     break

            # 1个即可满足
            first_idx = _i
            break
        return first_idx

    def fill_with_last_positive(arr):
        arr = arr.copy()
        pos_indices = np.where(arr > 0)[0]
        if len(pos_indices) == 0:
            return arr

        last_idx = pos_indices[-1]
        prefix = arr[:last_idx]

        # 标记正值的位置，其它为 False
        is_pos = prefix > 0
        # 创建索引数组，用于跟踪每个位置上一个正值的索引
        last_pos_idx = np.where(is_pos, np.arange(len(prefix)), -1)
        last_pos_idx = np.maximum.accumulate(last_pos_idx)

        # 替换 <= 0 的位置
        replace_mask = (prefix <= 0) & (last_pos_idx != -1)
        arr[:last_idx][replace_mask] = prefix[last_pos_idx[replace_mask]]

        return arr

    def zero_positive_after_nonpositive(arr):
        arr = arr.copy()  # 创建数组副本，避免修改原数组
        # 查找第一个非正数（<= 0）的索引
        non_positive_indices = np.where(arr <= 0)[0]
        
        if len(non_positive_indices) == 0:  # 如果没有非正数
            return arr  # 返回原数组
        
        # 获取第一个非正数的索引
        first_non_positive = non_positive_indices[0]
        
        # 将第一个非正数之后的所有正数替换为0
        if first_non_positive < len(arr):
            arr[first_non_positive + 1:][arr[first_non_positive + 1:] > 0] = 0
        
        return arr

    def fix_profit(df, begin, end):
        # 检查 BASE卖1价 的 min/max 之差
        range_data = df.loc[begin:end, :]
        price_range = range_data['BASE卖1价']
        if price_range.max() - price_range.min() < 0.0012:
            # 将范围内 profit 置为 0
            df.loc[begin:end, 'profit'] = 0
        else:
            # 检查 这个区间是否也满足可盈利

            # 限制平段占比
            # 上涨段中找到上涨的起点
            up_begin_idx = find_equal_begin_last_mid_idx(range_data)
            up_end_idx = find_equal_end_first_mid_idx(range_data)
            if up_begin_idx != 0:
                # 检查平段在全部范围的占比（限制平段占比）
                total = up_end_idx + 1
                flat = up_begin_idx
                if total >= 5:# 小于5的话 flat 可能会被限制为 0，所以不限制
                    if flat / total > MAX_FLAT_RATIO:
                        max_flat_length = MAX_FLAT_RATIO * (up_end_idx + 1 - up_begin_idx) / (1 - MAX_FLAT_RATIO)
                        max_flat_length = math.floor(max_flat_length)
                        max_flat_idx = up_begin_idx - max_flat_length
                        # max_flat_idx 之前的 profit 置为 0
                        to_0_indexs = range_data.iloc[:max_flat_idx].index
                        df.loc[to_0_indexs, 'profit'] = 0
                        # 更新 range_data
                        range_data = df.loc[to_0_indexs[-1]+1:end, :]
                
            # 按照区间最后一个最高价格作为卖出价（必须是连续的，数量>1）
            max_idx, max_mid_value = find_last_max_b1_value(range_data)
            # 时间大于等于卖出价时刻-1的 profit 置为 0
            df.loc[range_data.iloc[max_idx-1:].index, 'profit'] = 0
            # 时间早于卖出时刻-1的 profit 重新计算: 下一个时刻卖1价买入成交， 卖出价时刻的买1价卖出成交，计算 profit
            buy_cost = range_data.iloc[1:max_idx]['BASE卖1价'] * (1 + 5e-5)
            sell_gain = range_data.iloc[max_idx]['BASE买1价'] * (1 - 5e-5)
            profit = np.log(sell_gain / buy_cost)
            # profit = fill_with_last_positive(profit.values)
            profit = zero_positive_after_nonpositive(profit.values)
            df.loc[range_data.iloc[:max_idx-1].index, 'profit'] = profit
            
        # 第一个 profit > 0/ sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
        range_data = reset_profit(df.loc[begin:end, :].copy())
        df.loc[begin:end, 'profit'] = range_data['profit']

    def fix_sell_save(df, begin, end):
        range_data = df.iloc[begin:end, :]
        price_range = range_data['BASE买1价']
        if price_range.max() - price_range.min() < 0.0012:
            # 将范围内 sell_save 置为 0
            df.loc[begin:end, 'sell_save'] = 0
        else:
            # 检查 这个区间是否也满足可节省 TODO
            # 按照区间最后一个最低价格作为买入价（必须是连续的，数量>1）
            min_idx, min_mid_value = find_last_min_a1_value(range_data)
            # 时间大于等于买入价时刻-1的 sell_save 置为 0
            df.loc[range_data.iloc[min_idx-1:].index, 'sell_save'] = 0
            # 时间早于买入时刻-1的 sell_save 重新计算: 下一个时刻买1价卖出成交， 买入价时刻的卖1价买入成交，计算 sell_save
            buy_cost = range_data.iloc[min_idx]['BASE卖1价'] * (1 + 5e-5)
            sell_gain = range_data.iloc[1:min_idx]['BASE买1价'] * (1 - 5e-5)
            sell_save = np.log(sell_gain / buy_cost)
            # sell_save = fill_with_last_positive(sell_save.values)
            sell_save = zero_positive_after_nonpositive(sell_save.values)
            df.loc[range_data.iloc[:min_idx-1].index, 'sell_save'] = sell_save

        # 第一个 profit > 0/ sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
        range_data = reset_sell_save(df.loc[begin:end, :].copy())
        df.loc[begin:end, 'sell_save'] = range_data['sell_save']

    def find_segments(condition_series):
        cond = condition_series.values
        edges = np.diff(np.concatenate([[0], cond.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        return list(zip(starts, ends))

    def clear_dup(segments):
        unique_segments = []
        for seg in segments:
            if seg not in unique_segments:
                unique_segments.append(seg)
        return unique_segments

    # 向量化处理 profit 段
    act_segs = find_segments(df['action'] == 0)
    for b, e in act_segs:
        _act_0_data = df.loc[b: e]
        # 查找 profit>0 的 b, e
        profit_segs = find_segments(_act_0_data['profit'] > 0)
        _act_0_data_begin_idx = _act_0_data.iloc[0].name
        # 删除之间只有一个间隔的分组（合并成一个大的分组）
        for idx, (_b, _e) in enumerate(profit_segs[:-1]):
            if profit_segs[idx+1][0] - _e == 2:
                _new = (profit_segs[idx][0], profit_segs[idx+1][1])
                # 向前回溯修改
                for _idx in range(idx-1, -1, -1):
                    if profit_segs[idx] == profit_segs[_idx]:
                        profit_segs[_idx] = _new
                profit_segs[idx] = _new
                profit_segs[idx+1] = _new
                # 修正 profit 为前一个正值
                df.loc[_act_0_data_begin_idx+_e+1, 'profit'] = df.loc[_act_0_data_begin_idx+_e, 'profit']
        # 去重
        profit_segs = clear_dup(profit_segs)
        if len(profit_segs) > 1:
            # 需要对除了最后一段的 profit 进行修正
            for idx, (_b, _e) in enumerate(profit_segs[:-1]):
                _e = profit_segs[idx+1][0] - 1
                real_b, real_e = _act_0_data_begin_idx + _b, _act_0_data_begin_idx + _e
                fix_profit(df, real_b, real_e)

    # 向量化处理 sell_save 段
    act_segs = find_segments(df['action'] == 1)
    for b, e in act_segs:
        _act_1_data = df.loc[b: e]
        # 查找 sell_save>0 的 b, e
        sell_save_segs = find_segments(_act_1_data['sell_save'] > 0)
        _act_1_data_begin_idx = _act_1_data.iloc[0].name
        # 删除之间只有一个间隔的分组（合并成一个大的分组）
        for idx, (_b, _e) in enumerate(sell_save_segs[:-1]):
            if sell_save_segs[idx+1][0] - _e == 2:
                _new = (sell_save_segs[idx][0], sell_save_segs[idx+1][1])
                # 向前回溯修改
                for _idx in range(idx-1, -1, -1):
                    if sell_save_segs[idx] == sell_save_segs[_idx]:
                        sell_save_segs[_idx] = _new
                sell_save_segs[idx] = _new
                sell_save_segs[idx+1] = _new
                # 修正 profit 为前一个正值
                df.loc[_act_1_data_begin_idx+_e+1, 'sell_save'] = df.loc[_act_1_data_begin_idx+_e, 'profit']
        # 去重
        sell_save_segs = clear_dup(sell_save_segs)
        if len(sell_save_segs) > 1:
            # 需要对除了最后一段的 sell_save 进行修正
            for idx, (_b, _e) in enumerate(sell_save_segs[:-1]):
                _e = sell_save_segs[idx+1][0] - 1
                real_b, real_e = _act_1_data_begin_idx + _b, _act_1_data_begin_idx + _e
                fix_sell_save(df, real_b, real_e)
    
    return df

def reset_profit(lob_data):
    """
    剔除掉动作之后下一个点的价格变化带来的优势 (成交价格带来的优势不允许利用)
    第一个profit>0信号点，下一个时点价格不能下跌
    第一个profit<0信号点，下一个时点价格不能上涨
    """
    # 计算前一行和下一行的值
    prev_profit = lob_data['profit'].shift(1)
    next_sell_price = lob_data['BASE卖1价'].shift(-1)

    # 剔除 profit
    # 条件1：profit > 0, 前一行 profit <= 0, 且下一个 BASE卖1价 < 当前 BASE卖1价
    profit_cond1 = (lob_data['profit'] > 0) & (prev_profit <= 0) & (next_sell_price < lob_data['BASE卖1价'])
    # 将满足条件1的行的 profit 置为 0
    lob_data.loc[profit_cond1, 'profit'] = 0
    # 条件2：profit <= 0, 前一行 profit > 0, 且下一个 BASE卖1价 > 当前 BASE卖1价
    prev_profit = lob_data['profit'].shift(1) #需要重新更新, 可能被 cond1 修改
    profit_cond2 = (lob_data['profit'] <= 0) & (prev_profit > 0) & (next_sell_price > lob_data['BASE卖1价'])
    # 将前一行的profit值赋给当前行
    lob_data.loc[profit_cond2, 'profit'] = prev_profit.loc[profit_cond2]

    return lob_data


def reset_sell_save(lob_data):
    """
    剔除掉动作之后下一个点的价格变化带来的优势 (成交价格带来的优势不允许利用)
    第一个sell_save>0信号点，下一个时点价格不能上涨
    第一个sell_save<0信号点，下一个时点价格不能下跌
    """ 
    # 计算前一行和下一行的值
    prev_sell_save = lob_data['sell_save'].shift(1)
    next_buy_price = lob_data['BASE买1价'].shift(-1)    

    # 剔除 sell_save
    # 条件1：sell_save > 0, 前一行 sell_save <= 0, 且下一个 BASE买1价 > 当前 BASE买1价
    sell_save_cond1 = (lob_data['sell_save'] > 0) & (prev_sell_save <= 0) & (next_buy_price > lob_data['BASE买1价'])
    # 将满足条件1的行的 sell_save 置为 0
    lob_data.loc[sell_save_cond1, 'sell_save'] = 0
    # 条件2：sell_save <= 0, 前一行 sell_save > 0, 且下一个 BASE买1价 < 当前 BASE买1价
    prev_sell_save = lob_data['sell_save'].shift(1) #需要重新更新, 可能被 cond1 修改
    sell_save_cond2 = (lob_data['sell_save'] <= 0) & (prev_sell_save > 0) & (next_buy_price < lob_data['BASE买1价'])    
    # 将前一行的sell_save值赋给当前行
    lob_data.loc[sell_save_cond2, 'sell_save'] = prev_sell_save.loc[sell_save_cond2]

    return lob_data

def reset_profit_sell_save(lob_data):
    """
    剔除掉动作之后下一个点的价格变化带来的优势 (成交价格带来的优势不允许利用)
    第一个profit>0信号点，下一个时点价格不能下跌
    第一个profit<0信号点，下一个时点价格不能上涨
    第一个sell_save>0信号点，下一个时点价格不能上涨
    第一个sell_save<0信号点，下一个时点价格不能下跌
    """
    lob_data = reset_profit(lob_data)
    lob_data = reset_sell_save(lob_data)
    return lob_data

def remove_spikes_vectorized(arr):
    """
    使用向量化操作处理ndarray序列，检测针刺（某点>0且前后点<=0）并赋值为0。
    
    参数:
        arr: 输入的ndarray序列
        
    返回:
        new_arr: 处理后的ndarray序列
    """
    # 复制输入数组，避免修改原始数据
    new_arr = arr.copy()
    
    # 使用向量化操作检测针刺
    # 条件：当前点>0 且 前点<=0 且 后点<=0
    spikes = (arr[1:-1] > 0) & (arr[:-2] <= 0) & (arr[2:] <= 0)
    
    # 将检测到的针刺点赋值为0（注意索引偏移）
    new_arr[1:-1][spikes] = 0
    
    return new_arr

def process_lob_data_extended_0(df):
    """
    处理lob_data DataFrame，根据规则调整'profit'和'sell_save'列。
    
    规则:
    1. 对于profit<=0连续块，若之间无action=1，只保留最后一个与action=1相连的块（若无则全不保留），其余替换为之前最近的profit>0值；若无profit>0值，替换为0。
    2. 对于sell_save<=0连续块，若之间无action=0，只保留最后一个与action=0相连的块（若无则全不保留），其余替换为之前最近的sell_save>0值；若无sell_save>0值，替换为0。
    
    参数:
    df (pd.DataFrame): 包含'profit', 'sell_save'和'action'列的DataFrame，'profit'和'sell_save'为浮点数，'action'为0或1，且'profit'和'sell_save'不会同时非零。
    
    返回:
    pd.DataFrame: 处理后的DataFrame。
    """

    # --- 处理 profit<=0 连续块 ---
    # 需要特别忽略针刺 profit>0 的情况
    # p点 profit>0, 而前后都是 profit<=0 的情况，则将 p 点 profit 置为 0
    profit = pd.Series(remove_spikes_vectorized(df['profit'].values))
    df['last_pos_profit'] = profit.where(profit > 0).ffill().fillna(0)
    df['is_neg_profit'] = (profit <= 0) & (df['action'] == 0)# 同时需要action=0
    df['group_profit'] = (df['action'] == 1).cumsum()
    df['block_profit'] = df.groupby('group_profit')['is_neg_profit'].transform(
        lambda x: (x != x.shift()).cumsum()
    )
    df['next_action'] = df['action'].shift(-1).fillna(0)
    
    # 标记最后一个profit<=0块是否与action=1相连
    df['is_last_block_with_action1'] = False
    last_blocks = df[df['is_neg_profit']].groupby('group_profit')['block_profit'].max()
    for group, last_block in last_blocks.items():
        last_block_rows = df[(df['group_profit'] == group) & (df['block_profit'] == last_block) & (df['is_neg_profit'])]
        if not last_block_rows.empty:
            last_row_idx = last_block_rows.index[-1]
            if df.loc[last_row_idx, 'next_action'] == 1:
                df.loc[last_block_rows.index, 'is_last_block_with_action1'] = True
    
    mask_profit = df['is_neg_profit'] & (~df['is_last_block_with_action1'])
    df.loc[mask_profit, 'profit'] = df.loc[mask_profit, 'last_pos_profit']

    # --- 处理 sell_save<=0 连续块 ---
    # 需要特别忽略针刺 sell_save>0 的情况
    sell_save = pd.Series(remove_spikes_vectorized(df['sell_save'].values))
    df['last_pos_sell_save'] = sell_save.where(sell_save > 0).ffill().fillna(0)
    df['is_neg_sell_save'] = (sell_save <= 0) & (df['action'] == 1)# 同时需要action=1
    df['group_sell_save'] = (df['action'] == 0).cumsum()
    df['block_sell_save'] = df.groupby('group_sell_save')['is_neg_sell_save'].transform(
        lambda x: (x != x.shift()).cumsum()
    )
    df['is_last_block_with_action0'] = False
    last_blocks = df[df['is_neg_sell_save']].groupby('group_sell_save')['block_sell_save'].max()
    for group, last_block in last_blocks.items():
        last_block_rows = df[(df['group_sell_save'] == group) & (df['block_sell_save'] == last_block) & (df['is_neg_sell_save'])]
        if not last_block_rows.empty:
            last_row_idx = last_block_rows.index[-1]
            if df.loc[last_row_idx, 'next_action'] == 0:
                df.loc[last_block_rows.index, 'is_last_block_with_action0'] = True
    
    mask_sell_save = df['is_neg_sell_save'] & (~df['is_last_block_with_action0'])
    df.loc[mask_sell_save, 'sell_save'] = df.loc[mask_sell_save, 'last_pos_sell_save']

    # 清理辅助列
    df.drop(columns=[
        'last_pos_profit', 'group_profit', 'is_neg_profit', 'block_profit', 
        'next_action', 'is_last_block_with_action1',
        'last_pos_sell_save', 'group_sell_save', 'is_neg_sell_save', 'block_sell_save', 
        'is_last_block_with_action0'
    ], inplace=True)

    return df

def process_lob_data_extended(df):
    """
    处理lob_data DataFrame，根据规则调整'profit'和'sell_save'列。
    
    规则:
    1. action=0的连续块中，若有大于等于1个的(profit<=0连续块)，第一个(profit<=0连续块)之后的profit都改成0 。
    2. action=1的连续块中，若有大于等于1个的(sell_save<=0连续块)，第一个(sell_save<=0连续块)之后的sell_save都改成0 。
    
    参数:
    df (pd.DataFrame): 包含'profit', 'sell_save'和'action'列的DataFrame，'profit'和'sell_save'为浮点数，'action'为0或1，且'profit'和'sell_save'不会同时非零。
    
    返回:
    pd.DataFrame: 处理后的DataFrame。
    """
    df = df.copy()

    # --- 处理 action=0 的连续块（针对 profit） ---
    df['group0'] = (df['action'] != 0).cumsum()
    for group, group_df in df[df['action'] == 0].groupby('group0'):
        # 找出profit<=0的连续块起始位置
        is_neg = group_df['profit'] <= 0
        if is_neg.any():
            block_id = (is_neg != is_neg.shift()).cumsum()
            block_start_indices = is_neg[is_neg].groupby(block_id).apply(lambda x: x.index[0])
            first_neg_start = block_start_indices.iloc[0]
            # 将该起始点之后所有 profit 设为 0
            mask = group_df.index >= first_neg_start
            df.loc[group_df.index[mask], 'profit'] = 0

    # --- 处理 action=1 的连续块（针对 sell_save） ---
    df['group1'] = (df['action'] != 1).cumsum()
    for group, group_df in df[df['action'] == 1].groupby('group1'):
        is_neg = group_df['sell_save'] <= 0
        if is_neg.any():
            block_id = (is_neg != is_neg.shift()).cumsum()
            block_start_indices = is_neg[is_neg].groupby(block_id).apply(lambda x: x.index[0])
            first_neg_start = block_start_indices.iloc[0]
            # 将该起始点之后所有 sell_save 设为 0
            mask = group_df.index >= first_neg_start
            df.loc[group_df.index[mask], 'sell_save'] = 0

    df.drop(columns=['group0', 'group1'], inplace=True)
    return df

def process_lob_data_extended_sell_save(df):
    """
    处理lob_data DataFrame，根据规则调整'profit'和'sell_save'列。
    
    规则:
    1. 对于sell_save<=0连续块，若之间无action=0，只保留最后一个与action=0相连的块（若无则全不保留），其余替换为之前最近的sell_save>0值；若无sell_save>0值，替换为0。
    
    参数:
    df (pd.DataFrame): 包含'profit', 'sell_save'和'action'列的DataFrame，'profit'和'sell_save'为浮点数，'action'为0或1，且'profit'和'sell_save'不会同时非零。
    
    返回:
    pd.DataFrame: 处理后的DataFrame。
    """

    df['next_action'] = df['action'].shift(-1).fillna(0)

    # --- 处理 sell_save<=0 连续块 ---
    # 需要特别忽略针刺 sell_save>0 的情况
    sell_save = pd.Series(remove_spikes_vectorized(df['sell_save'].values))
    df['last_pos_sell_save'] = sell_save.where(sell_save > 0).ffill().fillna(0)
    df['is_neg_sell_save'] = (sell_save <= 0) & (df['action'] == 1)# 同时需要action=1
    df['group_sell_save'] = (df['action'] == 0).cumsum()
    df['block_sell_save'] = df.groupby('group_sell_save')['is_neg_sell_save'].transform(
        lambda x: (x != x.shift()).cumsum()
    )
    df['is_last_block_with_action0'] = False
    last_blocks = df[df['is_neg_sell_save']].groupby('group_sell_save')['block_sell_save'].max()
    for group, last_block in last_blocks.items():
        last_block_rows = df[(df['group_sell_save'] == group) & (df['block_sell_save'] == last_block) & (df['is_neg_sell_save'])]
        if not last_block_rows.empty:
            last_row_idx = last_block_rows.index[-1]
            if df.loc[last_row_idx, 'next_action'] == 0:
                df.loc[last_block_rows.index, 'is_last_block_with_action0'] = True
    
    mask_sell_save = df['is_neg_sell_save'] & (~df['is_last_block_with_action0'])
    df.loc[mask_sell_save, 'sell_save'] = df.loc[mask_sell_save, 'last_pos_sell_save']

    # 清理辅助列
    df.drop(columns=[
        'next_action',
        'last_pos_sell_save', 'group_sell_save', 'is_neg_sell_save', 'block_sell_save', 
        'is_last_block_with_action0'
    ], inplace=True)

    return df


def adjust_class_weights_df(predict_df):
    # timestamp,target,0,1,2
    min_class_count = predict_df['target'].value_counts().min()
    
    sampled_indices = []
    for class_label in predict_df['target'].unique():
        indices = predict_df.index[predict_df['target'] == class_label]
        sampled_indices.extend(np.random.choice(indices, min_class_count, replace=False))
    
    adjusted_df = predict_df.loc[sampled_indices].reset_index(drop=True)
    return adjusted_df

def adjust_class_weights_numpy(y_true, y_pred):
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    min_class_count = np.min(class_counts)
    
    sampled_indices = []
    for class_label in unique_classes:
        indices = np.where(y_true == class_label)[0]
        sampled_indices.extend(np.random.choice(indices, min_class_count, replace=False))
    
    adjusted_y_true = y_true[sampled_indices]
    adjusted_y_pred = y_pred[sampled_indices]
    
    return adjusted_y_true, adjusted_y_pred

def adjust_class_weights_torch(y_true, y_pred):
    unique_classes, class_counts = torch.unique(y_true, return_counts=True)
    min_class_count = torch.min(class_counts)
    
    sampled_indices = []
    for class_label in unique_classes:
        indices = (y_true == class_label).nonzero().squeeze()
        sampled_indices.extend(torch.tensor(np.random.choice(indices, min_class_count, replace=False)), device=y_true.device)
    
    adjusted_y_true = y_true[sampled_indices]
    adjusted_y_pred = y_pred[sampled_indices]
    
    return adjusted_y_true, adjusted_y_pred
    
def hex_to_rgb(hex_color):
    # 将十六进制颜色转换为RGB
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def output_leaderboard_html(df, out_file):
    # # 直接从CSV文件读取数据
    # df = pd.read_csv(csv_path)
    
    # 定义颜色
    color_good = hex_to_rgb('#93ff93')
    color_bad = hex_to_rgb('#ff4444') 
    
    # 定义需要着色的数值列
    numeric_cols = ['score_test', 'score_val', 'pred_time_test', 'pred_time_val', 
                   'fit_time', 'pred_time_test_marginal', 'pred_time_val_marginal', 
                   'fit_time_marginal']
    
    html = '''
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #000000; padding: 8px; text-align: left; }
            th { background-color: #e1e1e1; }  /* 淡灰色背景 */
        </style>
    </head>
    <body>
    '''
    
    # 创建表格
    table = '<table>'
    
    # 添加表头
    table += '<tr>'
    for col in df.columns:
        table += f'<th>{col}</th>'
    table += '</tr>'
    
    # 添加数据行
    for _, row in df.iterrows():
        table += '<tr>'
        for col in df.columns:
            value = row[col]
            if col in numeric_cols:
                # 对于数值列，计算相对位置并设置颜色
                min_val = df[col].min()
                max_val = df[col].max()
                if col in ['score_test', 'score_val']:  # 这些指标越大越好
                    norm_val = (value - min_val) / (max_val - min_val)
                    # 在两个颜色之间进行插值
                    red = int(color_good[0] + (color_bad[0] - color_good[0]) * (1-norm_val))
                    green = int(color_good[1] + (color_bad[1] - color_good[1]) * (1-norm_val))
                    blue = int(color_good[2] + (color_bad[2] - color_good[2]) * (1-norm_val))
                else:  # 其他指标越小越好
                    norm_val = (value - min_val) / (max_val - min_val)
                    red = int(color_good[0] + (color_bad[0] - color_good[0]) * norm_val)
                    green = int(color_good[1] + (color_bad[1] - color_good[1]) * norm_val)
                    blue = int(color_good[2] + (color_bad[2] - color_good[2]) * norm_val)
                color = f'rgb({red}, {green}, {blue})'
                table += f'<td style="background-color: {color}">{value:.4f}</td>'
            else:
                table += f'<td>{value}</td>'
        table += '</tr>'
    
    table += '</table>'
    html += table + '</body></html>'
    
    # 保存为HTML
    with open(out_file, 'w') as f:
        f.write(html)
    
    # imgkit.from_file(out_png_file.replace('jpg', 'html'), out_png_file)

def save_df_pic(title, df, fig_size=(500, 140)):
    fig = df2img.plot_dataframe(df, row_fill_color=("#ffffff", "#f2f2f2"), fig_size=fig_size)
    df2img.save_dataframe(fig=fig, filename=title)

def cal_symbol_y_idx_thresholds(train_folder, y_len=3):
    """
    返回
    {
        y_idx:{
            code1: (-0.1666666666666483, 0.11666666666676484),
            code2: (-0.1666666666666483, 0.11666666666676484),
        }
    }
    """

    ys = {}
    thresholds = {}

    train_files = os.listdir(train_folder)
    _,_, _, y, _ = pickle.load(open(os.path.join(train_folder, train_files[0]), 'rb'))
    y_idxs = [i for i in range(len(y[0]))]

    for file in tqdm(train_files):
        file = os.path.join(train_folder, file)
        ids,_, _, y, _ = pickle.load(open(file, 'rb'))

        for y_idx in y_idxs:
            if y_idx not in ys:
                ys[y_idx] = {}
                thresholds[y_idx] = {}

            for i, _id in enumerate(ids):
                code, _ = _id.split('_')
                if code not in ys[y_idx]:
                    ys[y_idx][code] = []
                ys[y_idx][code].append(y[i][y_idx])

    for y_idx in ys:
        for code in ys[y_idx]:
            # 计算 33% 和 66% 分位数
            threshold = (np.percentile(ys[y_idx][code], 33), np.percentile(ys[y_idx][code], 66)) if y_len == 3 else (np.percentile(ys[y_idx][code], 50),) if y_len == 2 else []

            if y_len == 3:
                c_0 = len([i for i in ys[y_idx][code] if i <= threshold[0]])
                c_1 = len([i for i in ys[y_idx][code] if i > threshold[0] and i <= threshold[1]])
                c_2 = len([i for i in ys[y_idx][code] if i > threshold[1]])
                _max = max(c_0, c_1, c_2)
                _min = min(c_0, c_1, c_2)
            elif y_len == 2:
                c_0 = len([i for i in ys[y_idx][code] if i <= threshold[0]])
                c_1 = len([i for i in ys[y_idx][code] if i > threshold[0]])
                _max = max(c_0, c_1)
                _min = min(c_0, c_1)
            else:
                raise Exception('y_len 必须为 2 或 3')

            max_diff = _max - _min
            max_diff_pct = max_diff / len(ys[y_idx][code]) * 100
            if max_diff_pct > 3:
                continue

            thresholds[y_idx][code] = threshold
            print(code, y_idx, thresholds[y_idx][code], round(max_diff_pct, 3))

    # 删除值长度为0的键
    thresholds = {key: value for key, value in thresholds.items() if len(value) != 0}
    return thresholds

def cal_symbol_class_pct(train_folder, y_len=3, thresholds=[-0.5, 0.5], min_class_pct=0.1):
    res = {}
    ys = {}

    train_files = os.listdir(train_folder)
    _,_, _, y, _ = pickle.load(open(os.path.join(train_folder, train_files[0]), 'rb'))
    y_idxs = [i for i in range(len(y[0]))]

    for file in tqdm(train_files):
        file = os.path.join(train_folder, file)
        ids,_, _, y, _ = pickle.load(open(file, 'rb'))

        for y_idx in y_idxs:
            if y_idx not in ys:
                ys[y_idx] = {}
                res[y_idx] = []

            for i, _id in enumerate(ids):
                code, _ = _id.split('_')
                if code not in ys[y_idx]:
                    ys[y_idx][code] = []
                ys[y_idx][code].append(y[i][y_idx])

    for y_idx in ys:
        for code in ys[y_idx]:
            pct = []
            _y = ys[y_idx][code]
            _thresholds = [i for i in thresholds]
            # 添加一个无穷大
            _thresholds.append(float('inf'))
            for i in range(y_len):
                _y2 = [x for x in _y if x >= _thresholds[0]]
                num = len(_y) - len(_y2)
                pct.append(100*num/len(ys[y_idx][code]))
                _thresholds = _thresholds[1:]
                _y = _y2

            if min(pct) > min_class_pct * 100:
                print(f"{y_idx} {code} {[f'{i:.2f}%' for i in pct]}")
                res[y_idx].append(code)

    # 删除值长度为0的键
    res = {key: value for key, value in res.items() if len(value) != 0}
    return res

def model_params_num(model):
    # 将所有参数转换为一个向量
    vector = parameters_to_vector(model.parameters())
    # 获取参数数量
    num_params = vector.numel()
    return num_params

def check_gradients(
    model,
    gradient_threshold_min=1e-5,
    gradient_threshold_max=1e10,
    if_raise = False,
    plot_histogram=False
):
    """
    检查模型每一层的梯度范数，记录信息并可选抛出异常。
    
    Args:
        model: 深度学习模型（继承自 torch.nn.Module）
        gradient_threshold_min: 梯度消失的阈值
        gradient_threshold_max: 梯度爆炸的阈值
        if_raise: 是否抛出异常
        plot_histogram: 是否绘制梯度范数直方图
    
    Returns:
        total_grad_norm: 全局梯度范数
    """
    grad_norms = defaultdict(float)
    total_grad_norm = 0.0
    grad_count = 0

    # 收集梯度信息
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad
        # 检查 NaN 或 Inf
        # 一定会抛出异常
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            raise RuntimeError(f"NaN or Inf detected in gradient of {name}")

        # 计算梯度范数
        grad_norm = torch.norm(grad).item()
        grad_norms[name] = grad_norm
        total_grad_norm += grad_norm ** 2
        grad_count += 1

    # 计算全局梯度范数
    total_grad_norm = (total_grad_norm ** 0.5) if grad_count > 0 else 0.0

    # 检查全局梯度是否异常
    not_normal = False
    if total_grad_norm < gradient_threshold_min:
        not_normal = True
        log(f"Potential global gradient vanishing: total norm = {total_grad_norm:.6e}")
    elif total_grad_norm > gradient_threshold_max:
        not_normal = True
        log(f"Potential global gradient explosion: total norm = {total_grad_norm:.6e}")

    # 逐层检查（仅在全局异常时详细分析）
    if not_normal:
        for name, grad_norm in grad_norms.items():
            if grad_norm < gradient_threshold_min:
                if if_raise:
                    raise RuntimeError(f"Gradient vanishing in {name}: norm = {grad_norm:.6e}")
                else:
                    log(f"Gradient vanishing in {name}: norm = {grad_norm:.6e}")
            if grad_norm > gradient_threshold_max:
                if if_raise:
                    raise RuntimeError(f"Gradient explosion in {name}: norm = {grad_norm:.6e}")
                else:
                    log(f"Gradient explosion in {name}: norm = {grad_norm:.6e}")

    # 可视化梯度分布
    if plot_histogram:
        plt.figure(figsize=(10, 6))
        plt.hist(list(grad_norms.values()), bins=30, log=True)
        plt.title("Gradient Norm Histogram")
        plt.xlabel("Gradient Norm")
        plt.ylabel("Frequency (Log Scale)")
        plt.savefig("gradient_norm_histogram.png")
        plt.close()
        log("Gradient norm histogram saved as 'gradient_norm_histogram.png'")

    return total_grad_norm

def check_nan(output):
    """
    检查 output 中是否存在 NaN 或 inf
    存在则返回 True，否则返回 False
    """
    has_nan = torch.isnan(output).any(dim=-1)  # 检查是否有 NaN
    has_inf = torch.isinf(output).any(dim=-1)  # 检查是否有 inf
    
    if has_nan.any() or has_inf.any():
        return True
    else:
        return False

def stop_all_python_processes():
    current_pid = os.getpid()

    # 获取当前正在运行的所有进程
    all_processes = psutil.process_iter()

    # 遍历所有进程并停止 Python 进程
    for process in all_processes:
        try:
            process_info = process.as_dict(attrs=['pid', 'name'])
            pid = process_info['pid']
            name = process_info['name']

            # 如果进程是 Python 进程且不是当前进程，则终止该进程
            if name.lower() == 'python' and pid != current_pid:
                process.terminate()
                print(f"Terminated Python process: {pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 忽略无法访问或不存在的进程
            pass

def report_memory_usage(msg='', log_func=print):
    memory_usage = psutil.virtual_memory()
    log_func(f"{msg} CPU 内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")
    # tpu_mem_info = xm.get_memory_info(xm.xla_device())
    # print(tpu_mem_info)
    # tpu_used = tpu_mem_info["kb_total"] - tpu_mem_info["kb_free"]
    # print(f"{msg} TPU 内存占用：{tpu_used/1024**3:.3f}GB/{tpu_mem_info['kb_total']/1024**3:.3f}GB")

    # # 获取当前进程ID
    # pid = psutil.Process().pid
    # # 获取当前进程对象
    # process = psutil.Process(pid)
    # # 获取当前进程占用的内存信息
    # memory_info = process.memory_info()
    # # 将字节大小转换为GB
    # memory_gb = memory_info.rss / (1024 ** 3)
    # # 打印内存大小
    # logger.debug(f"{msg} 占用的内存：{memory_gb:.3f}GB")


    # # 获取当前进程ID
    # current_pid = os.getpid()

    # # 获取当前进程对象
    # current_process = psutil.Process(current_pid)

    # # 获取当前进程占用的内存信息
    # memory_info = current_process.memory_info()

    # # 将字节大小转换为GB
    # memory_gb = memory_info.rss / (1024 ** 3)

    # # 打印当前进程的内存大小
    # # print(f"当前进程ID: {current_pid}, 当前进程占用的内存：{memory_gb:.3f}GB")

    # # 统计所有子进程的内存使用情况
    # total_memory_gb = memory_gb

    # # 获取所有进程ID
    # pids = psutil.pids()

    # for pid in pids:
    #     try:
    #         # 获取进程对象
    #         process = psutil.Process(pid)

    #         # 获取进程的父进程ID
    #         parent_pid = process.ppid()

    #         # 如果父进程ID与当前进程ID相同，则属于当前程序的子进程
    #         if parent_pid == current_pid:
    #             # 获取进程占用的内存信息
    #             memory_info = process.memory_info()

    #             # 将字节大小转换为GB
    #             memory_gb = memory_info.rss / (1024 ** 3)

    #             # 累加子进程的内存大小到总内存大小
    #             total_memory_gb += memory_gb
    #             # 打印子进程的内存大小
    #             # print(f"子进程ID: {pid}, 子进程占用的内存：{memory_gb:.3f}GB")
    #     except (psutil.NoSuchProcess, psutil.AccessDenied):
    #         # 跳过无法访问的进程
    #         continue
    
    # # # 打印合并统计后的内存大小
    # msg = '合并统计后的内存大小' if msg == '' else f'{msg}'
    # print(f"{msg}: {total_memory_gb:.3f}GB")


class AsyncLockWithLog:
    def __init__(self, lock, log_func=print, header=''):
        self.lock = lock
        self.log_func = log_func
        self.header = header

    async def __aenter__(self):
        await self.lock.acquire()  # 异步获取锁
        self.log_func(f"{self.header} Lock acquired!")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()  # 释放锁
        self.log_func(f"{self.header} Lock released!")

class LockWithLog:
    def __init__(self, lock, log_func=print, header=''):
        self.lock = lock
        self.log_func = log_func
        self.header = header

    def __enter__(self):
        self.lock.acquire()
        self.log_func(f"{self.header} Lock acquired!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        self.log_func(f"{self.header} Lock released!")


