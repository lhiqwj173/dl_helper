from enum import Flag
from re import I
import copy, shutil
import psutil, pickle, torch, os, time
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import platform
import time, math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Union, Optional
from scipy.ndimage import label
from itertools import groupby 

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
from dl_helper.rl.rl_env.lob_trade.lob_const import MAX_SEC_BEFORE_CLOSE

UPLOAD_INTERVAL = 300  # 5分钟 = 300秒
MAX_FLAT_RATIO = 0.2  # 平段占比最大值
# NO_MOVE_THRESHOLD = 50  # 无移动阈值
NO_MOVE_THRESHOLD = 10  # 无移动阈值

# 设置价格浮动阈值：中间段的价格与基准价格的最大允许偏差
# 如果中间段的任何价格偏离超过这个值，则不考虑合并
PRICE_DEVIATION_THRESHOLD = 0.0012

# 设置"扰动"段占比阈值：(中间扰动段的总长度 / 两个相同价格段的总长度)
# 如果这个比例超过阈值，则认为扰动过大，不进行合并
OTHER_TO_SAME_RATIO_THRESHOLD = 0.15
OTHER_TO_SAME_RATIO_THRESHOLD = 0.25

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


def blank_logout(*args, **kwargs):
    pass

def clear_folder(folder_path: str or Path) -> None:
    """
    清空指定文件夹下的所有文件和子文件夹，但保留该文件夹本身。

    此方法结合了 pathlib 的易用性和 shutil 的高效性，是推荐的首选方案。

    Args:
        folder_path (str or Path): 目标文件夹的路径。
                                           可以是字符串或 Path 对象。

    Raises:
        FileNotFoundError: 如果提供的路径不是一个存在的文件夹。
        PermissionError: 如果没有足够的权限执行删除操作。
    """
    # 1. 将输入路径转换为 Path 对象，以利用其面向对象的特性
    target_dir = Path(folder_path)

    # 2. 安全性检查：确保路径存在并且确实是一个文件夹
    if not target_dir.is_dir():
        print(f"错误：路径 '{target_dir}' 不是一个有效的文件夹或不存在。")
        # 或者可以抛出异常，让调用者处理
        # raise FileNotFoundError(f"路径 '{target_dir}' 不是一个有效的文件夹或不存在。")
        return

    print(f"准备清空文件夹：'{target_dir.resolve()}'")
    print("-" * 30)

    # 3. 遍历文件夹中的所有项目（文件和子文件夹）
    for item in target_dir.iterdir():
        try:
            if item.is_file():
                # 如果是文件，使用 unlink 方法删除
                print(f"删除文件: {item.name}")
                item.unlink()
            elif item.is_dir():
                # 如果是文件夹，使用 shutil.rmtree 删除整个目录树
                print(f"删除子文件夹: {item.name}")
                shutil.rmtree(item)
        except Exception as e:
            print(f"删除 {item.name} 时出错: {e}")

    print("-" * 30)
    print(f"文件夹 '{target_dir.resolve()}' 已清空。")

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

def _project_vertically(a_y: np.ndarray) -> np.ndarray:
    """
    将一维折线数组的内部点垂直投影到其中点折线上。

    此函数假设折线的 x 坐标是顺序整数 (0, 1, 2, ...)。
    输入和输出都是代表折线 y 坐标的一维数组。

    Args:
        a_y (np.ndarray): 输入的原始折线的 y 坐标，形状为 (N,)。

    Returns:
        np.ndarray: 修改后的新折线的 y 坐标，形状为 (N,)。
    """
    # --- 输入验证 ---
    if not isinstance(a_y, np.ndarray) or a_y.ndim != 1:
        raise TypeError("输入 'a_y' 必须是一个一维的 NumPy 数组。")
    
    num_points = a_y.shape[0]
    
    # 如果点的数量少于3，无法形成内部点或中点折线，直接返回副本
    if num_points < 3:
        return a_y.copy()

    # --- 步骤 1: 计算中点折线 b 的 y 坐标 ---
    # b_y[i] 是原始线段 (i, a_y[i]) -> (i+1, a_y[i+1]) 的中点 y 坐标
    b_y = (a_y[:-1] + a_y[1:]) / 2.0

    # --- 步骤 2: 计算内部点的新 y 坐标 (全向量化) ---
    # 原始点 a_y[i] (x=i) 的新 y 坐标是 b 上 x=i 处的插值。
    # 这个位置正好在 b 的两个点 (x=i-0.5) 和 (x=i+0.5) 的中间。
    # 因此，新 y 值是 b_y[i-1] 和 b_y[i] 的平均值。
    # 我们可以对整个 b_y 数组进行此操作来一次性计算所有内部点的新 y 值。
    new_internal_y = (b_y[:-1] + b_y[1:]) / 2.0

    # --- 步骤 3: 构建最终结果 ---
    # 创建一个 a_y 的副本以保留首尾点
    a_y_modified = a_y.copy()
    
    # 将计算出的新 y 值赋给内部点
    a_y_modified[1:-1] = new_internal_y
    
    return a_y_modified

def find_all_col_blocks(df, col_name, col_val):
    # 1. 创建布尔掩码，True 表示 col_name 等于 col_val
    is_action_zero = (df[col_name] == col_val)

    # 2. 识别分块边界：当值发生变化（True<->False）时，视为新块起点
    #    (is_action_zero != is_action_zero.shift()) 为每个新块起点标记 True
    #    .cumsum() 为每个连续分块分配唯一的整数 ID
    block_ids = (is_action_zero != is_action_zero.shift()).cumsum()

    # 3. 只保留 col_name == col_val 的行
    action_zero_rows = df[is_action_zero]

    # 4. 按分块 ID 分组，获取每个分块的起始和结束索引
    #    block.index[0] 是分块起始索引，block.index[-1] 是结束索引
    results = action_zero_rows.groupby(block_ids[is_action_zero]).apply(
        lambda block: (block.index[0], block.index[-1])
    )

    return results

def _smooth_price_series(prices):
    """
    更优化的版本，完全向量化实现
    prices: ndarray 价格序列
    Returns:
        smoothed: ndarray 平滑后的价格序列
    """
    n = len(prices)
    
    if n < 3:
        return prices.copy()
    
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
            original_arr = smoothed[start-1:end + 1]
            fix_arr = _project_vertically(original_arr)
            smoothed[start-1:end + 1] = fix_arr

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
                ((peaks_num_points[peak_idx] > 1) and ((bid[t2] == bid[pre_t2]) and mid[t2] >= mid[pre_t2]))
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
                    ((peaks_num_points[peak_idx] > 1) and ((bid[t2] == bid[pre_t2]) and mid[t2] >= mid[pre_t2]))
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
    
    # 6. 遍历所有 action==0 的分块，过滤不稳定的正负号的点
    for b, e in find_all_col_blocks(df, 'action', 0):
        not_stable_mask_profit = find_not_stable_sign(df.loc[b:e, 'profit'].values)
        rows_to_update_index = df.loc[b:e].index[not_stable_mask_profit]
        if not rows_to_update_index.empty:
            df.loc[rows_to_update_index, 'profit'] = 0

    # # 7. 过滤所有 not_stable_bid_ask 的点
    # df.loc[df['not_stable_bid_ask'], 'profit'] = 0

    return df

def calculate_sell_save(df, fee=5e-5):
    """
    真正的完全向量化实现，不使用任何循环：
    对于所有 action==1 的行:
    1. 在下一行的 BASE买1价 卖出
    2. 在下一个 action==0 后的下一行的 BASE卖1价 买入
    3. 计算对数收益，考虑交易费用 fee=5e-5
    """
    # 最后 2 行添加一个买入点，
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

    # 遍历所有 action==1 的分块，过滤不稳定的正负号的点
    for b, e in find_all_col_blocks(df, 'action', 1):
        not_stable_mask_sell_save = find_not_stable_sign(df.loc[b:e, 'sell_save'].values)
        rows_to_update_index = df.loc[b:e].index[not_stable_mask_sell_save]
        if not rows_to_update_index.empty:
            df.loc[rows_to_update_index, 'sell_save'] = 0

    # # 8. 过滤所有 not_stable_bid_ask 的点
    # df.loc[df['not_stable_bid_ask'], 'sell_save'] = 0

    return df

def _merge_price_segments_logout(logout, merge_start_row, merge_end_row, df):
    # **** df 表格图片日志 ****
    title = f'{merge_start_row}-{merge_end_row}_df'
    logout(title=title, df=df.iloc[merge_start_row:merge_end_row])

    # **** df 绘图日志 ****
    title = f'{merge_start_row}-{merge_end_row}_plot'
    extra_len = 30
    merge_start_row -= extra_len
    merge_end_row += extra_len
    merge_start_row = max(0, merge_start_row)
    merge_end_row = min(merge_end_row, len(df) - 1)
    plot_dict = {
        'mid_price': df.loc[merge_start_row:merge_end_row, 'mid_price'].values,
        'BASE买1价': df.loc[merge_start_row:merge_end_row, 'BASE买1价'].values,
        'BASE卖1价': df.loc[merge_start_row:merge_end_row, 'BASE卖1价'].values,
    }
    plot_file_path = logout(title=title, plot=True)
    if plot_file_path is not None:
        # 输出图片，前后排除 extra_len 个数据的中间区域，使用淡蓝色填充底色
        df = pd.DataFrame(plot_dict)
        ax = df.plot()
        # 计算填充区域
        n = len(df)
        start = extra_len
        end = n - extra_len
        if end > start:
            ax.axvspan(start, end-1, color='#b3e5fc', alpha=0.3, zorder=0)  # 淡蓝色填充
        plt.tight_layout()
        plt.savefig(plot_file_path)
        plt.close()

def merge_price_segments(df: pd.DataFrame, price_col: str='mid_price', logout=blank_logout) -> pd.DataFrame:
    """
    根据特定规则合并价格平稳段，并平滑掉中间的微小扰动。

    该函数首先将数据按连续相同的价格（`price_col`）划分为不同的段。
    然后，它遍历这些段，尝试将一个价格段（例如，价格为 P1）与后续出现的
    另一个相同价格（P1）的段进行合并。

    占比类合并的条件如下：
    1.  两个 P1 段之间的所有“扰动段”的价格，相对于 P1 的变动绝对值
        不能超过 `PRICE_DEVIATION_THRESHOLD`。
    2.  所有“扰动段”的总长度与两个 P1 段的总长度之比，必须小于
        `OTHER_TO_SAME_RATIO_THRESHOLD`。

    二价类合并的条件如下：
    1.  区间内的价格变动不能超过 0.001（只能有一次价格变动）
    2.  各个价格的起点终止的位置应该在 全长*0.3 以内（价格起点距离全长起点，价格终点距离全长终点）

    如果满足条件，则从第一个 P1 段的开始到第二个 P1 段的结束，所有这些
    数据点都被视为一个大的合并段。如果不满足，则不进行合并。

    Args:
        df (pd.DataFrame): 包含价格数据的 DataFrame。
        price_col (str): DataFrame 中代表价格的列名，例如 'mid_price'。

    Returns:
        pd.DataFrame: 增加了 ['no_move_len_pct', 'no_move_len_2price', 'no_move_len_raw'] 列的原始 DataFrame。
    """
    # --- 步骤 1: 预处理，识别基础的价格段 ---
    
    # 确保 DataFrame 有一个从 0 开始的连续索引，这对于后续基于索引的操作至关重要
    df = df.reset_index(drop=True)

    # 标记价格变化的位置 (True/False)
    changes = df[price_col].ne(df[price_col].shift(1))
    # 使用 cumsum() 创建分组，相同值的连续段属于同一组
    groups = changes.cumsum()
    df['groups'] = groups

    # 计算每个原始组的长度
    group_lengths = groups.map(groups.value_counts())
    df['no_move_len_raw'] = group_lengths

    # --- 步骤 2: 提取每个段的元信息，为循环做准备 ---

    # 使用 groupby 和 agg 高效地获取每个段的起始/结束索引、价格和长度
    # 这比在循环中反复查询 DataFrame 快得多
    segment_info = df.groupby('groups').agg(
        start_index=('groups', 'idxmin'),
        end_index=('groups', lambda x: x.index[-1]),  # Last index of the group
        mid_price=(price_col, 'first'),  # 同一段内价格相同，取第一个即可
        length=('groups', 'size')
    ).reset_index()
    try:
        segment_info.to_csv(os.path.join(os.getenv('TEMP'), 'segment_info.csv'))
    except:
        pass

    num_segments = len(segment_info)
    # 初始化结果数组，使用 numpy 数组比在循环中修改 Pandas Series 更高效
    merged_lengths = group_lengths.copy()
    # merged_lengths_pct = group_lengths.copy()
    # merged_lengths_2price = group_lengths.copy()
    
    # --- 步骤 3: 主循环，实现合并逻辑 ---

    # 使用 while 循环遍历所有段，因为合并操作会使我们一次性跳过多个段
    i = 0
    while i < num_segments:
        current_segment = segment_info.iloc[i]
        base_price = current_segment['mid_price']
        begin_idx = int(current_segment['start_index'])

        # 初始化向前查找的变量
        potential_merge_end_segment_idx = -1

        # 从当前段的下一个段开始，向后寻找可以合并的目标
        for j in range(i + 1, num_segments):
            next_segment = segment_info.iloc[j]

            # 检查条件 1: 中间价格变动是否大于阈值
            if abs(next_segment['mid_price'] - base_price) > PRICE_DEVIATION_THRESHOLD:
                break  # 价格偏差过大，停止当前查找，该块无法合并

            # 如果找到价格相同的段，则将其作为潜在的合并终点
            if next_segment['mid_price'] == base_price:
                potential_merge_end_segment_idx = j

        # --- 步骤 4: 根据查找结果决定是否合并 ---

        # 如果找到了有效的合并终点 (potential_merge_end_segment_idx != -1)
        if potential_merge_end_segment_idx != -1:
            # 提取从当前段 i 到合并终点 j 的所有段
            block_segments = segment_info.iloc[i : potential_merge_end_segment_idx + 1]
            
            # 计算 no_move (价格与 base_price 相同的段的总长度)
            # 和 diff (价格不同的段的总长度)
            same_price_mask = block_segments['mid_price'] == base_price
            diff_total_len = block_segments.loc[~same_price_mask, 'length'].sum()
            total_len = block_segments['length'].sum()

            # 获取合并块在原始 DataFrame 中的起始和结束行索引
            merge_start_row = int(current_segment['start_index'])
            merge_end_row = int(segment_info.iloc[potential_merge_end_segment_idx]['end_index'])

            # 获取合并块的中间价格
            unique_mid_prices = np.sort(df.loc[merge_start_row:merge_end_row, 'mid_price'].unique())

            # 检查条件 2: "扰动"段占比是否小于阈值
            # 增加 diff_total_len > 0 的判断以避免除零错误
            if diff_total_len > 0 and (diff_total_len / total_len) < OTHER_TO_SAME_RATIO_THRESHOLD:
                # **执行合并**
                
                # 将合并后的长度填充到结果数组的相应位置
                # merged_lengths_pct[merge_start_row : merge_end_row + 1] = total_len
                merged_lengths[merge_start_row : merge_end_row + 1] = total_len

                # 更新循环变量 i，跳过所有已寻找的段
                i = potential_merge_end_segment_idx + 1

                # 日志
                if total_len > NO_MOVE_THRESHOLD:
                    logout(f'[diff_pct_merge] merge_start_row={merge_start_row}, merge_end_row={merge_end_row}, total_len={total_len}, diff_pct={(diff_total_len / total_len):.2f}')
                    _merge_price_segments_logout(logout, merge_start_row, merge_end_row, df)

                continue # 继续下一次 while 循环

            # 另一种允许合并的条件
            # 1. 不同的中间价格只能有两个
            # 2. 设总长度 a
            #    各个中间价格 第一个索引 <= int(a * 0.2)
            #    各个中间价格 最后一个索引 >= len(df) - int(a * 0.2)
            elif unique_mid_prices.max() - unique_mid_prices.min() < 0.0012:
                # 检查其他的中间价格是否可以向后扩展
                _begin_idx = potential_merge_end_segment_idx + 1
                for j in range(_begin_idx, num_segments):
                    next_segment = segment_info.iloc[j]
                    # 检查条件 1: 中间价格变动是否仍然在一致的范围内
                    if unique_mid_prices.min() > next_segment['mid_price'] or unique_mid_prices.max() < next_segment['mid_price']:
                        break
                    # 如果找到价格相同的段，则将其作为潜在的合并终点
                    potential_merge_end_segment_idx = j

                # 需要检查其他的中间加个是否可以向前扩展
                _start_begin_idx = i-1
                potential_merge_start_segment_idx = i
                for j in range(_start_begin_idx, -1, -1):
                    pre_segment = segment_info.iloc[j]
                    # 检查条件 1: 中间价格变动是否仍然在一致的范围内
                    if unique_mid_prices.min() > pre_segment['mid_price'] or unique_mid_prices.max() < pre_segment['mid_price']:
                        break
                    # 如果找到价格相同的段，则将其作为潜在的合并起点
                    potential_merge_start_segment_idx = j

                # 若有向前扩展
                if _start_begin_idx + 1 != potential_merge_start_segment_idx:
                    ext_merge_start_row = int(segment_info.iloc[potential_merge_start_segment_idx]['start_index'])
                else:
                    ext_merge_start_row = merge_start_row

                # 若有扩展, 需要更新计算指标
                if _begin_idx - 1 != potential_merge_end_segment_idx:
                    # 获取合并块在原始 DataFrame 中的结束行索引
                    ext_merge_end_row = int(segment_info.iloc[potential_merge_end_segment_idx]['end_index'])
                else:
                    ext_merge_end_row = merge_end_row
                ext_total_len = ext_merge_end_row - ext_merge_start_row + 1# 需要重新计算一遍

                unique_mid_price_idxs = []
                for _p in unique_mid_prices:
                    all_idxs = df.loc[ext_merge_start_row:ext_merge_end_row, 'mid_price_raw'] == _p
                    if all_idxs.any():
                        unique_mid_price_idxs.append((all_idxs[all_idxs].index[0]-ext_merge_start_row, all_idxs[all_idxs].index[-1]-ext_merge_start_row))

                if len(unique_mid_price_idxs) >= 2:
                    assert len(unique_mid_price_idxs) in [2, 3], f"Unexpected number of unique mid price segments found: {len(unique_mid_price_idxs)}"

                    # 最大最小值在首尾都存在
                    # if unique_mid_price_idxs[0][0] <= int(ext_total_len * 0.3)\
                    #     and unique_mid_price_idxs[0][1] >= ext_total_len - int(ext_total_len * 0.3)\
                    #     and unique_mid_price_idxs[-1][0] <= int(ext_total_len * 0.3)\
                    #     and unique_mid_price_idxs[-1][1] >= ext_total_len - int(ext_total_len * 0.3):

                    # 不同值在首尾都存在
                    limit_length = int(ext_total_len * 0.3)
                    begin_satisfys = [i[0]<=limit_length for i in unique_mid_price_idxs]
                    end_satisfys = [i[1]>=(ext_total_len - limit_length) for i in unique_mid_price_idxs]
                    if sum(begin_satisfys) >= 2 and sum(end_satisfys) >= 2:
                        # 满足条件，执行合并
                        # 将合并后的长度填充到结果数组的相应位置
                        # merged_lengths_2price[ext_merge_start_row : ext_merge_end_row + 1] = ext_total_len
                        merged_lengths[ext_merge_start_row : ext_merge_end_row + 1] = ext_total_len

                        # 更新循环变量 i，跳过所有已寻找的段
                        i = potential_merge_end_segment_idx + 1

                        # 日志
                        if ext_total_len > NO_MOVE_THRESHOLD:
                            logout(f'[2price_merge] merge_start_row={ext_merge_start_row}, merge_end_row={ext_merge_end_row}, total_len={ext_total_len}, max_border={int(ext_total_len * 0.3)}, [0]={[int(i) for i in unique_mid_price_idxs[0]]}, [-1]={[int(i) for i in unique_mid_price_idxs[-1]]}')
                            _merge_price_segments_logout(logout, ext_merge_start_row, ext_merge_end_row, df)
                        
                        continue # 继续下一次 while 循环

            # elif diff_total_len > 0 and (diff_total_len / total_len) < 0.4:
            #     # same_price 占大多数, 但不满足合并条件
            #     # 本次的寻找有效
            #     # 更新循环变量 i，跳过所有已寻找的段
            #     i = potential_merge_end_segment_idx + 1

            #     continue # 继续下一次 while 循环

            else:
                pass

        # **不合并** (所有不满足合并条件的情况)
        # 无论是没找到相同价格的段、中间价格偏差过大，还是扰动比例过高
        i += 1 # 移动到下一个段

    # 删除 groups 列
    df = df.drop(columns=['groups'])

    # 将最终计算出的合并长度赋给 DataFrame 的新列
    # df['no_move_len_pct'] = merged_lengths_pct
    # df['no_move_len_2price'] = merged_lengths_2price
    df['no_move_len'] = merged_lengths

    return df

def set_last_two_to_zero(df, col_name):
    # 识别连续的 col_name 组
    # 创建一个辅助列，标记连续组
    df['group'] = (df[col_name] != df[col_name].shift()).cumsum()
    
    # 对每个连续组处理
    for _, group in df.groupby('group'):
        if len(group) >= 2:
            # 修改最后两行的 col_name 值为 0
            df.loc[group.index[-2:], col_name] = 0
        elif len(group) == 1:
            # 如果只有一行，修改该行
            df.loc[group.index, col_name] = 0
    
    # 删除辅助列
    df = df.drop('group', axis=1)
    return df

def _cal_need_keep(segs, df, need_keep, no_move_threshold):
    for seg_start, seg_end in segs:
        # 1. 切片出当前盈利段的DataFrame视图
        df_profit_seg = df.loc[seg_start:seg_end]
        # 2. 在该段内，创建'no_move_len_pct/no_move_len_2price'超阈值的条件掩码
        # is_no_move = (df_profit_seg['no_move_len_pct'] > no_move_threshold) | (df_profit_seg['no_move_len_2price'] > no_move_threshold)
        is_no_move = df_profit_seg['no_move_len'] > no_move_threshold
        # 如果该段内没有任何满足'no_move'条件的行，则跳到下一个盈利段
        if not is_no_move.any():
            continue
        # 3. 获取所有满足'no_move'条件的行的索引
        # .index对于性能至关重要，避免了对值的操作
        no_move_indices = df_profit_seg.index[is_no_move]
        # 4. 识别连续的'no_move'子段 (核心技巧)
        #    - no_move_indices.to_series().diff() 计算索引之间的差值
        #    - .ne(1) 标记非连续索引的位置 (即新子段的开始)
        #    - .cumsum() 为每个连续的子段分配一个唯一的ID
        block_ids = no_move_indices.to_series().diff().ne(1).cumsum()
        # 5. 按子段ID分组，并获取每个子段的最后两个索引
        #    groupby().tail(2) 是一个非常高效的操作
        indices_to_keep = no_move_indices.to_series().groupby(block_ids).tail(2).index
        # 6. 将这些需要保留的行的标志位置为True
        need_keep.loc[indices_to_keep] = True

def filte_no_move(df, no_move_threshold=NO_MOVE_THRESHOLD, logout=blank_logout):
    """
    去除超过阈值价格没有波动的连续块，处理 profit 和 sell_save 两列
    功能一（profit）：连续 no move 超过阈值个，空仓情况下不进行买入（profit=0）
                     会保留最后一个 no move 的买入动作，因为之后价格开始变化（信号变动有效）

    功能一（sell_save）：连续 no move 超过阈值个，持仓情况下不进行卖出（sell_save=0）
                        会保留最后一个 no move 的卖出动作，因为之后价格开始变化（信号变动有效）
    """
    # 检测 mid_price 是否变化（与前一行比较）
    df['mid_price_raw'] = ((df['BASE卖1价'] + df['BASE买1价']) / 2).round(4)
    df['mid_price'] = _smooth_price_series(df['mid_price_raw'].values).round(4)

    df = merge_price_segments(df, logout=logout)

    # 若连续的 profit/sell_save > 0 的段，完全在 no_move_len_pct/no_move_len_2price > NO_MOVE_THRESHOLD 的段中，则需要保留
    need_keep = df['mid_price'] < 0
    # 1. profit 连续段
    profit_segs = find_segments(df['profit'] > 0)
    _cal_need_keep(profit_segs, df, need_keep, no_move_threshold)
    # 2. sell_save 连续段
    # sell_save_segs = find_segments(df['raw_sell_save'] > 0)
    sell_save_segs = find_segments(df['sell_save'] > 0)
    _cal_need_keep(sell_save_segs, df, need_keep, no_move_threshold)

    # 标记每组的最后两行
    # 将每组最后两个值的 no_move_len_pct/no_move_len_2price 设为 0
    # df = set_last_two_to_zero(df, 'no_move_len_pct')
    # df = set_last_two_to_zero(df, 'no_move_len_2price')
    df = set_last_two_to_zero(df, 'no_move_len')

    # 功能一：超过阈值的组的 profit 和 sell_save 置为 0
    # df.loc[((df['no_move_len_pct'] > no_move_threshold) & (~need_keep)), ['profit', 'sell_save']] = 0
    # df.loc[((df['no_move_len_2price'] > no_move_threshold) & (~need_keep)), ['profit', 'sell_save']] = 0
    df.loc[((df['no_move_len'] > no_move_threshold) & (~need_keep)), ['profit', 'sell_save']] = 0
    
    return df

def _find_last_max_b1_value(range_data, _type='strict'):
    """
    _type: strict / normal
    """
    range_b1 = range_data['BASE买1价']
    # range_b1[:] = _smooth_price_series(range_b1.values)
    b1_set = range_b1.drop_duplicates().sort_values(ascending=False).to_list()
    # 遍历 mid_set，连续的最大值的最后一个
    max_idx = len(range_b1) - 1
    max_b1_value = 0
    range_b1 = range_b1.values
    for i_v, max_v in enumerate(b1_set):
        _idx = np.where(range_b1 == max_v)[0]

        # 优先处理连续的相同的值
        for _i in _idx[::-1]:
            if _i == 0:
                # 第一个值, 不再继续
                break

            # 连续的相同的值
            if range_b1[_i] == range_b1[_i-1]:
                max_idx = _i
                max_b1_value = range_b1[_i]
                break

        # 若第一次处理没有找到，且是 normal 类型，则继续处理 相邻递增 的情况
        if max_b1_value == 0 and _type == 'normal':
            for _i in _idx[::-1]:
                if _i == 0:
                    # 第一个值, 不再继续
                    break

                # 2.相邻递增（且前一个值为除了当前值外的最大值）,注意 i_v+1 是否越界
                # 3.相邻递增（且前一个值为除了当前值外的最小值）,注意 i_v-1 是否越界
                if ((i_v<(len(b1_set)-1)) and (range_b1[_i-1] == b1_set[i_v+1])) or\
                        ((i_v>0) and (range_b1[_i-1] == b1_set[i_v-1])):
                    max_idx = _i
                    max_b1_value = min(range_b1[_i-1], range_b1[_i])
                    break
        
        if max_b1_value != 0:
            break

    return max_idx, max_b1_value

def find_last_max_b1_value(range_data):
    return _find_last_max_b1_value(range_data, 'normal')

def find_last_max_b1_value_strict(range_data):
    return _find_last_max_b1_value(range_data, 'strict')

def _find_last_min_a1_value(range_data, _type='strict'):
    """
    _type: strict / normal
    """
    range_a1 = range_data['BASE卖1价']
    # range_a1[:] = _smooth_price_series(range_a1.values)
    a1_set = range_a1.drop_duplicates().sort_values().to_list()
    # 遍历 mid_set，连续的最小值的最后一个
    min_idx = len(range_a1) - 1
    min_a1_value = 0
    range_a1 = range_a1.values
    for i_v, min_v in enumerate(a1_set):
        _idx = np.where(range_a1 == min_v)[0]
        for _i in _idx[::-1]:
            if _i == 0:
                # 第一个值了
                break

            # 优先处理连续的相同的值
            if range_a1[_i] == range_a1[_i-1]:
                min_idx = _i
                min_a1_value = range_a1[_i]
                break

        # 若第一次处理没有找到，且是 normal 类型，则继续处理 相邻递减 的情况
        if min_a1_value == 0 and _type == 'normal':
            for _i in _idx[::-1]:
                if _i == 0:
                    # 第一个值, 不再继续
                    break

                # 2.相邻递减（且前一个值次大于当前值）,注意 i_v+1 是否越界
                # 3.相邻递减（且前一个值次小于当前值）,注意 i_v-1 是否越界
                if ((i_v<(len(a1_set)-1)) and (range_a1[_i-1] == a1_set[i_v+1])) or\
                        ((i_v>0) and (range_a1[_i-1] == a1_set[i_v-1])):
                    min_idx = _i
                    min_a1_value = max(range_a1[_i-1], range_a1[_i])
                    break

        if min_a1_value != 0:
            break
    return min_idx, min_a1_value

def find_last_min_a1_value(range_data):
    return _find_last_min_a1_value(range_data, 'normal')

def find_last_min_a1_value_strict(range_data):
    return _find_last_min_a1_value(range_data, 'strict')

def find_equal_min_last_mid_idx(range_data, must2=False):
    """
    寻找与最小值相等的最后一个索引
    用于上涨段中找到上涨的起点
    """
    range_mid = range_data['mid_price']
    begin_value = range_mid.min()
    last_idx = 0
    min_mid_value = begin_value
    range_mid = range_mid.values
    _idx = np.where(range_mid == min_mid_value)[0]
    for _i in _idx[::-1]:
        if _i == 0:
            # 第一个值
            break

        if must2:
            # 至少连续的2个
            if range_mid[_i] == range_mid[_i-1]:
                last_idx = _i
                break

        else:
            # 1个即可满足
            last_idx = _i
            break

    return last_idx

def find_equal_max_first_mid_idx(range_data):
    """
    寻找与最大值相等的第一个索引
    用于上涨段中找到上涨的终点
    """
    range_mid = range_data['mid_price']
    end_value = range_mid.max()
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

def _extend_profit_start(df: pd.DataFrame, start_idx: int, end_idx: int, profits: np.ndarray, mid_prices: np.ndarray, no_move_lens: np.ndarray) -> pd.DataFrame:
    """
    a. 记录该点的 profit 值和 mid_price (p0)。
    b. 从该点的前一个位置开始，向索引减小的方向（向前）回溯搜索。
    c. 在回溯过程中，如果遇到某点的 mid_price < p0，则认为找到了一个
        潜在的更早的起点，更新 p0 为这个更低的 mid_price，并继续向前搜索。
    d. 回溯在以下任一情况发生时停止：
        - 遇到某点的 mid_price >= p0。
        - 遇到某点的 no_move_lens > 50。
    """
    # 更新 end_idx 为最高点(close点)
    max_idx, max_mid_value = find_last_max_b1_value(df.loc[start_idx:end_idx])
    end_idx = start_idx + max_idx

    # 记录原始起点的重要信息
    original_profit = profits[start_idx]
    p0 = mid_prices[start_idx]  # p0 是搜索过程中的动态最低价

    # 用于记录在回溯中找到的、符合条件的、最靠前的索引
    new_start_idx = None
    _pre_new_start_idx = None

    # 从 start_idx 的前一个位置开始，向索引 0 的方向回溯
    # range(start, stop, step) -> `step`为-1表示反向迭代
    for i in range(start_idx - 1, -1, -1):
        # 检查停止条件 2: 中间价升高 0.001
        if mid_prices[i] > p0 + 0.0008:
            break

        if mid_prices[i] < p0:
            p0 = mid_prices[i]
            _pre_new_start_idx = new_start_idx
            if i > 0 and mid_prices[i] == mid_prices[i-1]:
                # 更新 p0 为当前更低的价格，并记录这个新的索引
                new_start_idx = i-1
            else:
                new_start_idx = i

        if new_start_idx is not None:
            # 需要检查 new_start_idx - start_idx，是否引入的 no_move 的段
            # no_move 段的定义: no_move_lens连续相同值的数量 > NO_MOVE_THRESHOLD
            # 若发现，则退回到 _pre_new_start_idx 并break
            # 1. 提取需要检查的 no_move_lens 切片
            segment_to_check = no_move_lens[new_start_idx : start_idx]
            # 2. 计算该切片中，连续相同值的最大长度
            if len(segment_to_check) > 0:
                # 使用 itertools.groupby 高效计算最长连续次数
                max_run = max(len(list(g)) for _, g in groupby(segment_to_check))
            else:
                max_run = 0 # 如果切片为空，则不存在连续段
            # 3. 判断是否超过阈值
            if max_run > NO_MOVE_THRESHOLD:
                # 如果引入了 "no_move" 段，则停止搜索。
                # final_start_idx 保持为上一个有效的值 (pre_final_start_idx)
                new_start_idx = _pre_new_start_idx
                break

    # --- 步骤 3: 如果找到了新的起点，则更新 profit 值 ---
    # 只有当 new_start_idx 被成功赋值后，才执行更新操作
    if new_start_idx is not None:
        # df.loc[a:b] 的切片是包含 a 和 b 两端的。
        # 我们要更新的区间是 [new_start_idx, start_idx - 1]
        # 使用 .loc 可以确保精确和安全地对DataFrame进行赋值
        df.loc[new_start_idx : start_idx - 1, 'profit'] = original_profit
        # fix act
        df.loc[new_start_idx : start_idx - 1, 'action'] = 0

    return new_start_idx if new_start_idx is not None else start_idx

def _extend_sell_save_start(df: pd.DataFrame, start_idx: int, end_idx: int, sell_saves: np.ndarray, mid_prices: np.ndarray, no_move_lens: np.ndarray) -> pd.DataFrame:
    """
    a. 记录该点的 sell_save 值和 mid_price (p0)。
    b. 从该点的前一个位置开始，向索引减小的方向（向前）回溯搜索。
    c. 在回溯过程中，如果遇到某点的 mid_price > p0，则认为找到了一个
        潜在的更早的起点，更新 p0 为这个更高的 mid_price，并继续向前搜索。
    d. 回溯在以下任一情况发生时停止：
        - 遇到某点的 mid_price < p0。
        - 遇到某点的 no_move_lens > 50。
    """
    # 更新 end_idx 为最高点(close点)
    min_idx, min_mid_value = find_last_min_a1_value(df.loc[start_idx:end_idx])
    end_idx = start_idx + min_idx

    # 记录原始起点的信息
    original_sell_save = sell_saves[start_idx]
    p0 = mid_prices[start_idx]  # p0 是搜索过程中的动态最高价

    # 记录回溯中找到的、符合条件的、最靠前的索引
    new_start_idx = None
    _pre_new_start_idx = None

    # 从起点的前一个位置开始，向索引 0 的方向回溯
    for i in range(start_idx - 1, -1, -1):
        # 停止条件 2: 中间价降低 0.001
        if mid_prices[i] < p0 - 0.0008:
            break

        if mid_prices[i] > p0:
            # 更新 p0 为当前更高的价格，并记录这个新索引
            p0 = mid_prices[i]
            if i > 0 and mid_prices[i] == mid_prices[i-1]:
                new_start_idx = i-1
            else:
                new_start_idx = i

        if new_start_idx is not None:
            # 需要检查 new_start_idx - start_idx，是否引入的 no_move 的段
            # no_move 段的定义: no_move_lens连续相同值的数量 > NO_MOVE_THRESHOLD
            # 若发现，则退回到 _pre_new_start_idx 并break
            # 1. 提取需要检查的 no_move_lens 切片
            segment_to_check = no_move_lens[new_start_idx : start_idx]
            # 2. 计算该切片中，连续相同值的最大长度
            if len(segment_to_check) > 0:
                # 使用 itertools.groupby 高效计算最长连续次数
                max_run = max(len(list(g)) for _, g in groupby(segment_to_check))
            else:
                max_run = 0 # 如果切片为空，则不存在连续段
            # 3. 判断是否超过阈值
            if max_run > NO_MOVE_THRESHOLD:
                # 如果引入了 "no_move" 段，则停止搜索。
                # final_start_idx 保持为上一个有效的值 (pre_final_start_idx)
                new_start_idx = _pre_new_start_idx
                break

    # --- 步骤 3: 如果找到了新的起点，则更新 sell_save 值 ---
    if new_start_idx is not None:
        # 根据题意，将 new_idx 到原起点之间的 sell_save 进行赋值
        # 赋值区间为 [new_start_idx, start_idx - 1]
        df.loc[new_start_idx : start_idx - 1, 'sell_save'] = original_sell_save
        # fix act
        df.loc[new_start_idx : start_idx - 1, 'action'] = 1

    return new_start_idx if new_start_idx is not None else start_idx

def update_non_positive_blocks(
    a: pd.Series,
    b: pd.Series,
    valid_mask: pd.Series
) -> pd.Series:
    """
    检查 a 中的连续非正值块，若对应位置的 b 值全为正 且 valid_mask 全为 True，
    则将 a 的这些值替换为 b 的值。

    Args:
        a (pd.Series): 原始的 a series。
        b (pd.Series): 用于条件检查和提供替换值的 series。
                       必须与 a 具有相同的索引。
        valid_mask (pd.Series): 用于条件检查的布尔掩码。
                                必须与 a 具有相同的索引。

    Returns:
        pd.Series: 更新后的 a series。
    """
    # 确保输入是 Series 且索引一致
    if not all(isinstance(s, pd.Series) for s in [a, b, valid_mask]):
        raise TypeError("输入 a, b, valid_mask 都必须是 pandas Series 类型。")
    if not a.index.equals(b.index) or not a.index.equals(valid_mask.index):
        raise ValueError("三个 Series (a, b, valid_mask) 的索引必须完全相同。")
    if not valid_mask.dtype == 'bool':
        raise TypeError("valid_mask 必须是布尔类型的 Series。")

    # 创建 a 的副本以避免修改原始数据
    a_updated = a.copy()

    # 步骤 1: 标记所有非正值的位置
    is_non_positive = a <= 0
    if not is_non_positive.any():
        return a_updated # 如果没有非正值，直接返回副本

    # 步骤 2: 识别连续的非正值块
    block_ids = is_non_positive.ne(is_non_positive.shift()).cumsum()
    non_positive_block_ids = block_ids[is_non_positive]

    # --- **主要修改点** ---
    # 步骤 3: 检查每个非正值块是否满足组合条件
    block_condition = (b > 0) & valid_mask

    # --- 修复开始 ---
    # 仅对非正值块内的条件进行分组，以避免索引不对齐导致 NaN 分组键
    # 1. 筛选出与 non_positive_block_ids 索引相同的 relevant_conditions
    relevant_conditions = block_condition[non_positive_block_ids.index]

    # 2. 对这个索引对齐的子集进行 groupby 和 transform。
    #    结果 block_met_partial 的索引将是 non_positive_block_ids.index。
    # block_met_partial = relevant_conditions.groupby(non_positive_block_ids).transform('all')
    block_met_partial = relevant_conditions.groupby(non_positive_block_ids.values).transform('all')

    # 3. 将部分结果 reindex 回完整的索引，并将所有其他位置（即 a > 0 的位置）填充为 False。
    #    这样 is_block_condition_met 就是一个完整的、纯布尔的 Series。
    is_block_condition_met = block_met_partial.reindex(a.index, fill_value=False)
    # --- 修复结束 ---

    # 步骤 4: 创建最终的替换掩码
    # 现在的 is_block_condition_met 是一个纯布尔 Series，
    # final_update_mask 的计算更安全、更清晰。
    final_update_mask = is_non_positive & is_block_condition_met

    # 步骤 5: 执行替换
    a_updated.loc[final_update_mask] = b[final_update_mask]

    return a_updated

def update_non_positive_blocks_fixed(
    a: pd.Series,
    b: pd.Series,
    valid_mask: pd.Series
) -> pd.Series:
    """
    检查 a 中的连续非正值块，若对应位置的 b 值全为正 且 valid_mask 全为 True，
    则将 a 的这些值替换为 b 的值。(修复版)

    Args:
        a (pd.Series): 原始的 a series。
        b (pd.Series): 用于条件检查和提供替换值的 series。
                       必须与 a 具有相同的索引。
        valid_mask (pd.Series): 用于条件检查的布尔掩码。
                                必须与 a 具有相同的索引。

    Returns:
        pd.Series: 更新后的 a series。
    """
    # 确保输入是 Series 且索引一致 (代码与原版一致，此处省略)
    if not all(isinstance(s, pd.Series) for s in [a, b, valid_mask]):
        raise TypeError("输入 a, b, valid_mask 都必须是 pandas Series 类型。")
    if not a.index.equals(b.index) or not a.index.equals(valid_mask.index):
        raise ValueError("三个 Series (a, b, valid_mask) 的索引必须完全相同。")
    if not valid_mask.dtype == 'bool':
        raise TypeError("valid_mask 必须是布尔类型的 Series。")

    a_updated = a.copy()
    is_non_positive = a <= 0
    if not is_non_positive.any():
        return a_updated

    # 步骤 1: 识别所有连续块（包括正值和非正值块）
    # 这里的 block_ids 是一个完整的 Series，与 a 索引相同
    block_ids = is_non_positive.ne(is_non_positive.shift()).cumsum()

    # 步骤 2: 计算每一行的替换条件
    # block_condition 也是一个完整的 Series
    block_condition = (b > 0) & valid_mask

    # --- **核心修复点** ---
    # 步骤 3: 在完整的 Series 上执行 groupby.transform
    # 对完整的 block_condition 按完整的 block_ids 分组
    # 检查每个块内的所有条件是否都为 True。
    # .transform('all') 会返回一个与 a 索引相同的布尔 Series，
    # 其中每个元素的值是其所在块的 `all()` 聚合结果。
    # 这种方式避免了对稀疏/过滤后的数据进行 groupby，从而规避了潜在的 bug。
    is_block_condition_met = block_condition.groupby(block_ids).transform('all')

    # 步骤 4: 创建最终的替换掩码
    # 只有当 a 是非正数 且 其所在的整个块都满足条件时，才进行替换
    final_update_mask = is_non_positive & is_block_condition_met

    # 步骤 5: 执行替换
    a_updated.loc[final_update_mask] = b[final_update_mask]

    return a_updated

def fix_profit(df, begin, end):
    end+=2# 避免有效的卖出点被 act=1 占用

    max_mid_value = None

    # 备份 end+1, end+2 的 profit
    backup_profit = df.loc[end-1:end, 'profit'].copy().values

    # 修改成使用 no_move_len_pct
    # new_begin = _extend_profit_start(df, begin, df['profit'].values, df['mid_price'].values, df['no_move_len_raw'].values)
    new_begin = _extend_profit_start(df, begin, end, df['profit'].values, df['mid_price'].values, df['no_move_len'].values)

    # 检查 BASE卖1价 的 min/max 之差
    range_data = df.loc[new_begin:end, :]
    price_range = range_data['BASE卖1价']
    if price_range.max() - price_range.min() < 0.0012:
        # 将范围内 profit 置为 0
        df.loc[new_begin:end, 'profit'] = 0
    else:
        # 检查 这个区间是否也满足可盈利

        # 限制平段占比
        # 上涨段中找到上涨的起点
        up_begin_idx = find_equal_min_last_mid_idx(range_data)
        up_end_idx = find_equal_max_first_mid_idx(range_data)
        
        if up_end_idx > up_begin_idx:
            # 防止出现 up_end_idx < up_begin_idx 的情况
            if up_begin_idx != 0:
                # 检查平段在全部范围的占比（限制平段占比）
                total = up_end_idx + 1
                flat = up_begin_idx
                # 小于5的话 flat 可能会被限制为 0，所以不限制
                # total小于10进行限制不太合理
                if total >= 20:
                    if flat / total > MAX_FLAT_RATIO:
                        max_flat_length = MAX_FLAT_RATIO * (up_end_idx + 1 - up_begin_idx) / (1 - MAX_FLAT_RATIO)
                        max_flat_length = math.floor(max_flat_length)
                        max_flat_idx = up_begin_idx - max_flat_length
                        # 找到 max_flat_idx 开始的范围内的最小值的起始点，不一定就是 max_flat_idx
                        new_range_data = range_data.iloc[max_flat_idx:]
                        new_begin_idx = find_equal_min_last_mid_idx(new_range_data, must2=True) - 1
                        new_real_begin_idx = new_begin_idx + new_range_data.iloc[0].name
                        # new_begin_idx 之前的 profit 需要重新计算
                        # 而不是直接置为 0
                        to_0_indexs = range_data.iloc[:max_flat_idx + new_begin_idx].index
                        _begin, _end = to_0_indexs[0], to_0_indexs[-1]
                        _begin, _end, _max_mid_value = fix_profit(df, _begin, _end)
                        # 更新 range_data
                        range_data = df.loc[new_real_begin_idx:end, :]
            
        # 按照区间最后一个最高价格作为卖出价（必须是连续的，数量>1）
        max_idx, max_mid_value = find_last_max_b1_value(range_data)

        # 时间大于等于卖出价时刻-1的 profit 置为 0
        df.loc[range_data.iloc[max_idx-1:].index, 'profit'] = 0
        if max_idx > 1:
            # 时间早于卖出时刻-1的 profit 重新计算: 下一个时刻卖1价买入成交， 卖出价时刻的买1价卖出成交，计算 profit
            buy_cost = range_data.iloc[1:max_idx]['BASE卖1价'] * (1 + 5e-5)
            sell_gain = max_mid_value * (1 - 5e-5)
            profit = np.log(sell_gain / buy_cost)
            # 过滤不稳定的 +- 点
            _range_idx = range_data.iloc[:max_idx-1].index
            not_stable_mask_profit = find_not_stable_sign(profit.values)
            buy_cost_b1_diff = (range_data.iloc[1:max_idx]['BASE买1价'] + 0.001) * (1 + 5e-5)
            profit_b1_diff = np.log(sell_gain / buy_cost_b1_diff)
            diff_valid_mask = (range_data.iloc[1:max_idx]['BASE卖1价'] - range_data.iloc[1:max_idx]['BASE买1价']) < 0.0022
            profit = update_non_positive_blocks(profit, profit_b1_diff, diff_valid_mask)
            profit.loc[not_stable_mask_profit] = 0
            df.loc[_range_idx, 'profit'] = profit.values
            # # 过滤所有 not_stable_bid_ask 的点
            # df.loc[df['not_stable_bid_ask'], 'profit'] = 0

    # 第一个 profit > 0/ sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
    data_0 = df.loc[new_begin-1, 'profit'] if new_begin > 0 else None
    _begin = new_begin - 1 if new_begin > 0 else new_begin
    data_m1 = df.loc[end+1, 'profit'] if end < len(df) - 1 else None
    _end = end + 1 if end < len(df) - 1 else end
    range_data = reset_profit(df.loc[_begin:_end, :].copy(), data_0, data_m1)
    df.loc[_begin:_end, 'profit'] = range_data['profit']

    # 恢复额外的 profit
    df.loc[end-1:end, 'profit'] = backup_profit

    # return new_begin, end
    return new_begin, end - 2, max_mid_value

def fix_profit_segs(begin_idx, df, profit_segs, close_price):
    """
    根据 close_price 调整 profit_segs 中的段的 profit
    """
    if len(profit_segs) == 0:
        return profit_segs
    
    # 1. 先转为 real_idx
    real_segs = [(begin_idx + seg[0], begin_idx + seg[1]) for seg in profit_segs]
    # 2. 遍历 real_segs, 更新 profit
    for seg in real_segs:
        sell_gain = close_price * (1 - 5e-5)
        buy_cost = df.loc[seg[0]+1:seg[1]+1, 'BASE卖1价'] * (1 + 5e-5)
        profit = np.log(sell_gain / buy_cost)
        # 过滤不稳定的 +- 点
        # 构造临时计算用的 df, 只需要包含 BASE卖1价, BASE买1价, profit
        not_stable_mask_profit = find_not_stable_sign(profit.values)
        buy_cost_b1_diff = (df.loc[seg[0]+1:seg[1]+1, 'BASE买1价']+0.001) * (1 + 5e-5)
        profit_b1_diff = np.log(sell_gain / buy_cost_b1_diff)
        diff_valid_mask = (df.loc[seg[0]+1:seg[1]+1, 'BASE卖1价'] - df.loc[seg[0]+1:seg[1]+1, 'BASE买1价']) < 0.0022
        profit = update_non_positive_blocks(profit, profit_b1_diff, diff_valid_mask)
        profit.loc[not_stable_mask_profit] = 0
        df.loc[seg[0]:seg[1], 'profit'] = profit.values

    # # 过滤 not_stable_bid_ask 的点
    # df.loc[df['not_stable_bid_ask'], 'profit'] = 0

    # 第一个 profit > 0/ sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
    data_0 = df.loc[begin_idx-1, 'profit'] if begin_idx > 0 else None   
    _begin = begin_idx - 1 if begin_idx > 0 else begin_idx
    data_m1 = df.loc[real_segs[-1][1]+1, 'profit'] if real_segs[-1][1] < len(df) - 1 else None
    _end = real_segs[-1][1] + 1 if real_segs[-1][1] < len(df) - 1 else real_segs[-1][1]
    range_data = reset_profit(df.loc[_begin:_end, :].copy(), data_0, data_m1)
    df.loc[_begin:_end, 'profit'] = range_data['profit']

    # 3. 获取新的 profit_segs
    new_profit_segs = find_segments(df.loc[begin_idx:real_segs[-1][1], 'profit'] > 0)

    # 4. 删除之间只有一个间隔的分组（合并成一个大的分组）
    skip_1_point(begin_idx, df, new_profit_segs, 'profit')
    
    return new_profit_segs

def process_non_positive_blocks(series: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    处理数值序列中的非正数（<=0）连续块。

    规则如下:
    1. 如果序列中只有一个非正数连续块（按规则必然在末尾），则原样返回。
    2. 如果有多个非正数连续块，则将除了最后一个块之外的所有非正数块，
       用其前方最近的一个正数（>0）值进行替换。

    Args:
        series (Union[pd.Series, np.ndarray]): 
            输入的数值序列。函数内部会确保其为 pd.Series 类型。
            该序列被假定以一个或多个 <=0 的值结束。

    Returns:
        pd.Series: 处理后的新序列。

    Raises:
        ValueError: 如果一个非正数块之前找不到任何正数，则会引发此异常。
                    例如，序列以一个需要被替换的非正数块开始。
    """
    # 确保输入是 pandas Series 以便使用其强大的索引和分组功能
    if not isinstance(series, pd.Series):
        s = pd.Series(series)
    else:
        # 创建一个副本以避免修改原始输入
        s = series.copy()

    # --- 步骤 1: 识别所有连续块 ---
    # 创建一个布尔掩码，True 代表值 > 0
    is_positive_mask = s > 0
    
    # 使用 diff() 和 cumsum() 的经典技巧来为每个连续块分配唯一ID
    # 当 is_positive_mask 的值发生变化时（从True到False或反之），diff()不为0
    # cumsum() 会在每个变化点上加一，从而为每个块生成了ID
    block_ids = (is_positive_mask.diff() != 0).cumsum()

    # --- 步骤 2: 定位所有非正数块 ---
    # 找出所有非正数块的ID
    # s[~is_positive_mask] 是所有非正数值
    # block_ids[~is_positive_mask] 是这些值对应的块ID
    non_positive_block_ids = block_ids[~is_positive_mask].unique()

    # --- 步骤 3: 根据非正数块的数量决定操作 ---
    # 如果非正数块少于2个，则无需处理，直接返回副本
    if len(non_positive_block_ids) <= 1:
        return s

    # --- 步骤 4: 执行替换逻辑 ---
    # 既然有多个非正数块，我们需要进行替换操作
    # 获取最后一个非正数块的ID，这个块将保持不变
    last_non_positive_block_id = non_positive_block_ids[-1]
    
    # 用于存储每个块前方最近的正数值
    last_seen_positive_value = np.nan

    # 遍历所有块 (按ID分组)
    for block_id, group in s.groupby(block_ids):
        # 检查当前块是否为正数块
        # 我们只需要检查组里的第一个值即可判断整个块的性质
        if group.iloc[0] > 0:
            # 如果是正数块，更新“最近的正数值”为该块的最后一个元素
            last_seen_positive_value = group.iloc[-1]
        else:
            # 如果是非正数块，检查它是否是需要被替换的“中间块”
            if block_id != last_non_positive_block_id:
                # 安全检查：确保我们已经找到了一个可用于替换的正数值
                if np.isnan(last_seen_positive_value):
                    # raise ValueError(
                    #     f"无法处理ID为 {block_id} 的非正数块（起始索引 "
                    #     f"{group.index[0]}）: 在它之前没有找到任何正数。"
                    # )
                    pass
                else:
                    # 执行替换：将这个块的所有值更新为最近的正数值
                    s.loc[group.index] = last_seen_positive_value
    
    return s

def fix_sell_save(df, begin, end):
    end += 2# 避免有效的买入点被 act=0 占用

    min_mid_value = None

    # 备份 end+1, end+2 的 sell_save
    backup_sell_save = df.loc[end-1:end, 'sell_save'].copy().values

    # 修改成使用 no_move_len_pct
    # new_begin = _extend_sell_save_start(df, begin, df['sell_save'].values, df['mid_price'].values, df['no_move_len_raw'].values)
    new_begin = _extend_sell_save_start(df, begin, end, df['sell_save'].values, df['mid_price'].values, df['no_move_len'].values)

    range_data = df.loc[new_begin:end, :]
    price_range = range_data['BASE买1价']
    if price_range.max() - price_range.min() < 0.0012:
        # 将范围内 sell_save 置为 0
        df.loc[new_begin:end, 'sell_save'] = 0
    else:
        # 检查 这个区间是否也满足可节省 TODO
        # 按照区间最后一个最低价格作为买入价（必须是连续的，数量>1）
        min_idx, min_mid_value = find_last_min_a1_value(range_data)
        # 时间大于等于买入价时刻-1的 sell_save 置为 0
        df.loc[range_data.iloc[min_idx-1:].index, 'sell_save'] = 0
        if min_idx > 1:
            # 时间早于买入时刻-1的 sell_save 重新计算: 下一个时刻买1价卖出成交， 买入价时刻的卖1价买入成交，计算 sell_save
            buy_cost = min_mid_value * (1 + 5e-5)
            sell_gain = range_data.iloc[1:min_idx]['BASE买1价'] * (1 - 5e-5)
            sell_save = np.log(sell_gain / buy_cost)
            # 过滤不稳定的 +- 点
            # 构造临时计算用的 df, 只需要包含 BASE卖1价, BASE买1价, sell_save
            _range_idx = range_data.iloc[:min_idx-1].index
            not_stable_mask_sell_save = find_not_stable_sign(sell_save.values)
            sell_gain_a1_diff = (range_data.iloc[1:min_idx]['BASE卖1价'] - 0.001) * (1 - 5e-5)
            sell_save_a1_diff = np.log(sell_gain_a1_diff / buy_cost)
            diff_valid_mask = (range_data.iloc[1:min_idx]['BASE卖1价'] - range_data.iloc[1:min_idx]['BASE买1价']) < 0.0022 # 价差 < 0.0022
            sell_save = update_non_positive_blocks(sell_save, sell_save_a1_diff, diff_valid_mask)
            # sell_save = process_non_positive_blocks(remove_spikes_vectorized(sell_save.values))
            sell_save.loc[not_stable_mask_sell_save] = 0
            df.loc[_range_idx, 'sell_save'] = sell_save.values
            # # 过滤 not_stable_bid_ask 的点
            # df.loc[df['not_stable_bid_ask'], 'sell_save'] = 0

    # 第一个 sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
    data_0 = df.loc[new_begin-1, 'sell_save'] if new_begin > 0 else None
    _begin = new_begin - 1 if new_begin > 0 else new_begin
    data_m1 = df.loc[end+1, 'sell_save'] if end < len(df) - 1 else None
    _end = end + 1 if end < len(df) - 1 else end
    range_data = reset_sell_save(df.loc[_begin:_end, :].copy(), data_0, data_m1)
    df.loc[_begin:_end, 'sell_save'] = range_data['sell_save']

    # 恢复额外的 sell_save
    df.loc[end-1:end, 'sell_save'] = backup_sell_save

    return new_begin, end-2, min_mid_value

def fix_sell_save_segs(begin_idx, df, sell_save_segs, close_price):
    """
    根据 close_price 调整 sell_save_segs 中的段的 sell_save
    """
    if len(sell_save_segs) == 0:
        return sell_save_segs
    
    # 1. 先转为 real_idx
    real_segs = [(begin_idx + seg[0], begin_idx + seg[1]) for seg in sell_save_segs]
    # 2. 遍历 real_segs, 更新 sell_save
    for seg in real_segs:
        buy_cost = close_price * (1 + 5e-5)
        sell_gain = df.loc[seg[0]+1:seg[1]+1, 'BASE买1价'] * (1 - 5e-5)
        sell_save = np.log(sell_gain / buy_cost)
        # 过滤不稳定的 +- 点
        not_stable_mask_sell_save = find_not_stable_sign(sell_save.values)
        sell_gain_a1_diff = (df.loc[seg[0]+1:seg[1]+1, 'BASE卖1价'] - 0.001) * (1 - 5e-5)
        sell_save_a1_diff = np.log(sell_gain_a1_diff / buy_cost)
        diff_valid_mask = (df.loc[seg[0]+1:seg[1]+1, 'BASE卖1价'] - df.loc[seg[0]+1:seg[1]+1, 'BASE买1价']) < 0.0022
        sell_save = update_non_positive_blocks(sell_save, sell_save_a1_diff, diff_valid_mask)
        # sell_save = process_non_positive_blocks(remove_spikes_vectorized(sell_save.values))
        sell_save.loc[not_stable_mask_sell_save] = 0
        df.loc[seg[0]:seg[1], 'sell_save'] = sell_save.values

    # # 过滤所有 not_stable_bid_ask 的点
    # df.loc[df['not_stable_bid_ask'], 'sell_save'] = 0

    # 第一个 profit > 0/ sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
    data_0 = df.loc[begin_idx-1, 'sell_save'] if begin_idx > 0 else None
    _begin = begin_idx - 1 if begin_idx > 0 else begin_idx
    data_m1 = df.loc[real_segs[-1][1]+1, 'sell_save'] if real_segs[-1][1] < len(df) - 1 else None
    _end = real_segs[-1][1] + 1 if real_segs[-1][1] < len(df) - 1 else real_segs[-1][1]
    range_data = reset_sell_save(df.loc[_begin:_end, :].copy(), data_0, data_m1)
    df.loc[_begin:_end, 'sell_save'] = range_data['sell_save']

    # 3. 获取新的 sell_save_segs
    new_sell_save_segs = find_segments(df.loc[begin_idx:real_segs[-1][1], 'sell_save'] > 0)
    # 4. 删除之间只有一个间隔的分组（合并成一个大的分组）
    skip_1_point(begin_idx, df, new_sell_save_segs, 'sell_save')
    return new_sell_save_segs

def find_segments(condition_series):
    cond = condition_series.values
    edges = np.diff(np.concatenate([[0], cond.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts, ends))

def clear_dup(segments):
    seen = set()
    i = 0
    while i < len(segments):
        if segments[i] in seen:
            segments.pop(i)
        else:
            seen.add(segments[i])
            i += 1

def skip_1_point(begin_idx, df, profit_segs, col_name):
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
            df.loc[begin_idx+_e+1, col_name] = df.loc[begin_idx+_e, col_name]

    # 去重
    clear_dup(profit_segs)
    return profit_segs

def _merge_segs(segs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    检查并合并列表中重叠的线段。

    该函数接收一个由 (begin, end) 元组组成的列表，代表多个线段。
    它会合并所有重叠或相邻的线段，并返回一个无重叠的新线段列表。

    例如: [(1, 5), (2, 6), (8, 10)] -> [(1, 6), (8, 10)]

    Args:
        segs (List[Tuple[int, int]]): 一个线段列表，每个线段是一个 (begin, end) 元组。
                                     假设在每个元组中，begin <= end。

    Returns:
        List[Tuple[int, int]]: 合并后且无重叠的线段列表。
    """
    # 1. 处理边界情况：如果输入列表为空或只有一个元素，则无需合并。
    if not segs or len(segs) < 2:
        return segs

    # 2. 核心步骤：按线段的起始点 (begin) 对列表进行升序排序。
    #    这是能够高效合并的关键。排序后，我们只需比较相邻的线段即可。
    #    使用 sorted() 函数会返回一个新的排序列表，不会修改原始输入列表。
    sorted_segs = sorted(segs, key=lambda x: x[0])

    # 3. 初始化结果列表，并将第一个线段作为合并的起点。
    merged_segs = [sorted_segs[0]]

    # 4. 遍历排序后的线段（从第二个开始）。
    for current_begin, current_end in sorted_segs[1:]:
        # 获取结果列表中最后一个已合并的线段。
        last_begin, last_end = merged_segs[-1]

        # 5. 检查当前线段是否与最后一个合并线段重叠。
        #    因为列表已按 begin 排序，所以 current_begin >= last_begin 总是成立。
        #    因此，我们只需检查当前线段的起点是否小于或等于上一个线段的终点。
        if current_begin <= last_end + 1:
            # 如果重叠，则合并它们。
            # 新的合并线段的终点是两者终点中的较大值。
            # 更新结果列表中的最后一个线段。
            merged_segs[-1] = (last_begin, max(last_end, current_end))
        else:
            # 如果不重叠，说明遇到了一个新的、不连续的线段。
            # 将这个新线段直接添加到结果列表中。
            merged_segs.append((current_begin, current_end))

    return merged_segs

def check_last_profit_segment(begin_idx, df, real_b, real_e, profit_segs):
    """
    返回 real_b（可能会被向前扩展，profit_segs会自动处理）
    """
    # 测试用
    print(f'check_last_profit_segment begin_idx: {begin_idx}, real_b: {real_b}, real_e: {real_e}, profit_segs: {profit_segs}')

    # 检查最后一段内是否存在 profit<=0 and no_move_len_raw>NO_MOVE_THRESHOLD
    # 若存在的话，将 real_e 调整到 no_move_len_raw>NO_MOVE_THRESHOLD 的第一个点
    # 替换成 no_move_len
    # _cond = (df.loc[real_b:real_e, 'no_move_len_raw'] > NO_MOVE_THRESHOLD) & (df.loc[real_b:real_e, 'profit'] <= 0)
    _cond = (df.loc[real_b:real_e, 'no_move_len'] > NO_MOVE_THRESHOLD) & (df.loc[real_b:real_e, 'profit'] <= 0)
    # Get indices where _cond is True
    satisfying_indices = df.loc[real_b:real_e][(_cond)].index
    if _cond.any() and len(satisfying_indices) > NO_MOVE_THRESHOLD - 2:
        # 使用第10个 no move 的点，避免一些突刺的极端情况，保证至少有2个相同中间价格的点，可以用于计算 profit/sell_save
        # _idx = satisfying_indices[9]
        _idx = satisfying_indices[NO_MOVE_THRESHOLD - 2]# 最大使用前50个数据（no_move_len > 50 的段中，避免中间仍然间隔超过 50）
        _idx, _ = find_last_max_b1_value(df.loc[real_b:_idx, :])
        _idx += real_b

        # 先置为0
        df.loc[_idx:real_e, 'profit'] = 0
        real_e = _idx

    real_b, real_e, close_price = fix_profit(df, real_b, real_e)
    if real_b < begin_idx:
        # real_b 可能小于 begin_idx
        diff = begin_idx - real_b
        begin_idx = real_b
        # 更新 profit_segs
        profit_segs[:] = [(i[0]+diff, i[1]+diff) for i in profit_segs]

    # 检查新的 profit 是否有夹杂 profit <= 0 的段
    _new_profit_segs = find_segments(df.loc[real_b:real_e, 'profit'] > 0)
    _new_profit_segs = [(i[0]+real_b, i[1]+real_b) for i in _new_profit_segs]
    _new_profit_segs = [(i[0]-begin_idx, i[1]-begin_idx) for i in _new_profit_segs]
    # 需要向前推并合并
    _extend_segs = []
    for _b, _e in _new_profit_segs:
        new_begin = _extend_profit_start(df, _b+begin_idx, _e+begin_idx, df['profit'].values, df['mid_price'].values, df['no_move_len'].values)
        _extend_segs.append((new_begin-begin_idx, _e))
    _new_profit_segs = _merge_segs(_extend_segs)
    # 删除之间只有一个间隔的分组（合并成一个大的分组）
    skip_1_point(begin_idx, df, _new_profit_segs, 'profit')

    # 根据 close_price 调整 profit_segs[:-1] 中的段
    if close_price:
        _new_profit_segs_before = fix_profit_segs(begin_idx, df, profit_segs[:-1], close_price)
        # 合并到 profit_segs
        profit_segs[:-1] = _new_profit_segs_before

    if len(_new_profit_segs) == 0:
        # 最后一段不成立, 需要更新 real_e 
        # 删除最后一段
        profit_segs.pop(-1)
        if len(profit_segs):
            real_e = real_b - 1
            real_b, _ = profit_segs[-1]
            real_b += begin_idx
            return check_last_profit_segment(begin_idx, df, real_b, real_e, profit_segs)

    else:
        # 更新最后一段
        old_last_profit_seg = profit_segs.pop(-1)
        new_last_profit_seg = copy.deepcopy(_new_profit_segs[-1])
        profit_segs.extend([i for i in _new_profit_segs])
        # 检查最后一段是否与之前有重叠
        profit_segs[:] = _merge_segs(profit_segs)[:]

        if len(_new_profit_segs) == 1:
            # 只有一个分段, 需要判断 _check_segs 后是否有变化，若有变化，则需要重新检查最后一段
            if new_last_profit_seg == profit_segs[-1]:
                # 若相等，直接返回, 否则会死循环
                return begin_idx
        else:
            # 多个分段
            if old_last_profit_seg == profit_segs[-1]:
                # 若相等，直接返回, 否则会死循环
                return begin_idx
        # 若不相等
        # 需要重新检查最后一段
        real_b, _ = [i+begin_idx for i in profit_segs[-1]]
        return check_last_profit_segment(begin_idx, df, real_b, real_e, profit_segs)

def check_last_sell_save_segment(begin_idx, df, real_b, real_e, sell_save_segs):
    """
    返回 real_b（可能会被向前扩展，sell_save_segs会自动处理）
    """

    # 检查最后一段内是否存在 sell_save<=0 and no_move_len_raw>NO_MOVE_THRESHOLD
    # 若存在的话，将 real_e 调整到 no_move_len_raw>NO_MOVE_THRESHOLD 的第一个点
    # 替换成 no_move_len
    # _cond = (df.loc[real_b:real_e, 'no_move_len_raw'] > NO_MOVE_THRESHOLD) & (df.loc[real_b:real_e, 'sell_save'] <= 0)
    _cond = (df.loc[real_b:real_e, 'no_move_len'] > NO_MOVE_THRESHOLD) & (df.loc[real_b:real_e, 'sell_save'] <= 0)
    # Get indices where _cond is True
    satisfying_indices = df.loc[real_b:real_e][(_cond)].index
    if _cond.any() and len(satisfying_indices) > NO_MOVE_THRESHOLD - 2:
        # 使用第10个 no move 的点，避免一些突刺的极端情况，保证至少有2个相同中间价格的点，可以用于计算 profit/sell_save
        # _idx = satisfying_indices[min(9, len(satisfying_indices)-1)]
        _idx = satisfying_indices[NO_MOVE_THRESHOLD - 2]# 最大使用前50个数据（no_move_len > 50 的段中，避免中间仍然间隔超过 50）
        _idx, _ = find_last_min_a1_value(df.loc[real_b:_idx, :])
        _idx += real_b

        # 先置为0
        df.loc[_idx:real_e, 'sell_save'] = 0
        real_e = _idx

    real_b, real_e, close_price = fix_sell_save(df, real_b, real_e)
    if real_b < begin_idx:
        # real_b 可能小于 begin_idx
        diff = begin_idx - real_b
        begin_idx = real_b
        # 更新 sell_save_segs
        sell_save_segs[:] = [(i[0]+diff, i[1]+diff) for i in sell_save_segs]

    # 检查新的 sell_save 是否有夹杂 sell_save <= 0 的段
    _new_sell_save_segs = find_segments(df.loc[real_b:real_e, 'sell_save'] > 0)
    _new_sell_save_segs = [(i[0]+real_b, i[1]+real_b) for i in _new_sell_save_segs]
    _new_sell_save_segs = [(i[0]-begin_idx, i[1]-begin_idx) for i in _new_sell_save_segs]
    # 需要向前推并合并
    _extend_segs = []
    for _b, _e in _new_sell_save_segs:
        new_begin = _extend_sell_save_start(df, _b+begin_idx, _e+begin_idx, df['sell_save'].values, df['mid_price'].values, df['no_move_len'].values)
        _extend_segs.append((new_begin-begin_idx, _e))
    _new_sell_save_segs = _merge_segs(_extend_segs)
    # 删除之间只有一个间隔的分组（合并成一个大的分组）
    skip_1_point(begin_idx, df, _new_sell_save_segs, 'sell_save')
    
    if close_price:
        # 根据 close_price 调整 sell_save_segs[:-1] 中的段
        _new_sell_save_segs_before = fix_sell_save_segs(begin_idx, df, sell_save_segs[:-1], close_price)
        # 合并到 sell_save_segs
        sell_save_segs[:-1] = _new_sell_save_segs_before

    if len(_new_sell_save_segs) == 0:
        # 最后一段不成立，需要更新 real_e
        # 删除最后一段
        sell_save_segs.pop(-1)
        if len(sell_save_segs):
            real_e = real_b - 1
            real_b, _ = sell_save_segs[-1]
            real_b += begin_idx
            return check_last_sell_save_segment(begin_idx, df, real_b, real_e, sell_save_segs)

    else:
        new_last_sell_save_seg = copy.deepcopy(_new_sell_save_segs[-1]) 
        # 更新最后一段
        old_last_sell_save_seg = sell_save_segs.pop(-1)
        sell_save_segs.extend([i for i in _new_sell_save_segs])
        # 检查最后一段是否与之前有重叠
        sell_save_segs[:] = _merge_segs(sell_save_segs)[:]

        # 多个分段
        if len(_new_sell_save_segs) == 1:
            # 只有一个分段, 需要判断 _check_segs 后是否有变化，若有变化，则需要重新检查最后一段
            if new_last_sell_save_seg == sell_save_segs[-1]:
                # 若相等，直接返回, 否则会死循环
                return begin_idx
        else:
            # 多个分段
            if old_last_sell_save_seg == sell_save_segs[-1]:
                # 若相等，直接返回, 否则会死循环
                return begin_idx

        # 需要重新检查最后一段
        real_b, _ = [i+begin_idx for i in sell_save_segs[-1]]
        return check_last_sell_save_segment(begin_idx, df, real_b, real_e, sell_save_segs)

def find_start_of_continuous_block(
    df: pd.DataFrame, 
    real_b: int, 
    find_type: int
) -> Optional[int]:
    """
    从指定索引的前一个位置开始，在 'action' 列中向前回溯，
    寻找一个连续值为 `find_type` 的数据块，并返回该块的起始索引。

    例如，如果 action 列为 [1, 1, 0, 0, 1, 1, 1, 0]，real_b=7, find_type=1,
    函数会从索引 6 (real_b - 1) 开始向前查找连续的 1。
    它会检查索引 6, 5, 4 (值都是 1)，然后在索引 3 (值为 0) 停止。
    因此，这个连续块是 [1, 1, 1]，其起始索引为 4，函数将返回 4。

    Args:
        df (pd.DataFrame): 
            输入的 DataFrame，必须包含一个名为 'action' 的列。
            我们假设 DataFrame 的索引是标准的 0-based 整数索引。
        real_b (int): 
            回溯的起始参考点。函数将从 `real_b - 1` 的位置开始向前搜索。
            `real_b` 必须是一个有效的索引位置，且大于 0。
        find_type (int): 
            要寻找的连续值，必须是 0 或 1。

    Returns:
        Optional[int]: 
            如果找到了符合条件的连续块，则返回该块的第一个值的索引（即最小的索引）。
            如果在 `real_b - 1` 的位置就不是 `find_type`，或者没有找到任何匹配项，
            则返回 None。
    """
    # --- 1. 输入参数校验，确保代码健壮性 ---
    if 'action' not in df.columns:
        raise ValueError("输入 DataFrame 中必须包含 'action' 列。")

    # real_b 必须在 (0, len(df)] 范围内，这样 real_b - 1 才是一个有效的起始索引
    if not (0 < real_b <= len(df)):
        return None

    if find_type not in [0, 1]:
        raise ValueError(f"参数 'find_type' ({find_type}) 必须是 0 或 1。")

    # --- 2. 核心回溯逻辑 ---
    
    # 用于存储找到的连续块的起始索引
    block_start_index: Optional[int] = None

    # 使用 range 从 real_b - 1 开始，一直回溯到索引 0
    # range(start, stop, step) -> stop 参数是不包含的，所以用 -1
    for i in range(real_b - 1, -1, -1):
        # 使用 .loc 来安全地访问指定索引和列的值
        # .loc 对于基于标签的索引是首选，这里我们的整数索引也是标签
        current_action = df.loc[i, 'action']

        if current_action == find_type:
            # 如果当前值匹配，说明我们仍在连续块内或刚刚找到了块的末尾。
            # 我们将找到的这个索引记为可能是块的起始点。
            # 随着循环继续向前，这个值会被更小的索引覆盖，
            # 直到循环结束时，它就是真正的起始点。
            block_start_index = i
        elif block_start_index is not None:
            # 如果当前值不匹配，说明连续块在这里中断了。
            # 且 block_start_index 有值，说明曾经存在 current_action == find_type
            # 我们立即停止搜索。
            break
            
    # --- 3. 返回结果 ---
    return block_start_index

def check_first_profit_segment(begin_idx, df, real_b, real_e, profit_segs):
    # 检查 real_b - 50 与 real_b 之间是否存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 的段
    pre_act1_begin_idx = find_start_of_continuous_block(df, real_b, 1)# 限制范围在前 act==1 的段内
    check_begin_idx = max(0, real_b-50, 0 if pre_act1_begin_idx is None else pre_act1_begin_idx)
    check_end_idx = real_b+1
    # if (df.loc[check_begin_idx:check_end_idx, 'no_move_len_pct'].max() > NO_MOVE_THRESHOLD or
    #     df.loc[check_begin_idx:check_end_idx, 'no_move_len_2price'].max() > NO_MOVE_THRESHOLD):
    if df.loc[check_begin_idx:check_end_idx, 'no_move_len'].max() > NO_MOVE_THRESHOLD:
        # 若存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 段，需要改用check_begin_idx起的最低点（至少连续2个点）
        df_range = df.loc[check_begin_idx:check_end_idx, :]
        __min_idx, _ = find_last_min_a1_value_strict(df_range)
        __min_idx -= 1 # find_first_min_a1_value 返回的是连续2个点中的后一个，此处修正成前一个
        _real_b = __min_idx + check_begin_idx
        if _real_b < 0:
            raise ValueError(f"check_first_profit_segment: _real_b < 0, _real_b: {_real_b}, check_begin_idx: {check_begin_idx}, check_end_idx: {check_end_idx}")
        if _real_b < real_b:
            # 修正 df
            df.loc[_real_b:real_b-1, 'profit'] = 0.1
            # 修正 profit_segs
            if _real_b < begin_idx:
                begin_diff = begin_idx - _real_b
                begin_idx = _real_b
                profit_segs[:] = [(i[0]+begin_diff, i[1]+begin_diff) for i in profit_segs]
            old = profit_segs.pop(0)
            profit_segs.insert(0, (_real_b-begin_idx, old[1]))
        
    return begin_idx

def check_first_sell_save_segment(begin_idx, df, real_b, real_e, sell_save_segs):
    # 检查 real_b - 50 与 real_b 之间是否存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 的段
    pre_act0_begin_idx = find_start_of_continuous_block(df, real_b, 0)# 限制范围在前 act==0 的段内
    check_begin_idx = max(0, real_b-50, 0 if pre_act0_begin_idx is None else pre_act0_begin_idx)
    check_end_idx = real_b+1
    # if (df.loc[check_begin_idx:check_end_idx, 'no_move_len_pct'].max() > NO_MOVE_THRESHOLD or
    #     df.loc[check_begin_idx:check_end_idx, 'no_move_len_2price'].max() > NO_MOVE_THRESHOLD):
    if df.loc[check_begin_idx:check_end_idx, 'no_move_len'].max() > NO_MOVE_THRESHOLD:
        # 若存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 段，需要改用check_begin_idx起的最低点（至少连续2个点）
        df_range = df.loc[check_begin_idx:check_end_idx, :]
        __max_idx, _ = find_last_max_b1_value_strict(df_range)
        __max_idx -= 1 # find_first_max_b1_value 返回的是连续2个点中的后一个，此处修正成前一个
        _real_b = __max_idx + check_begin_idx
        if _real_b < 0:
            raise ValueError(f"check_first_sell_save_segment: _real_b < 0, _real_b: {_real_b}, check_begin_idx: {check_begin_idx}, check_end_idx: {check_end_idx}")
        if _real_b < real_b:
            # 修正 df
            df.loc[_real_b:real_b-1, 'sell_save'] = 0.1
            # 修正 sell_save_segs
            if _real_b < begin_idx:
                begin_diff = begin_idx - _real_b
                begin_idx = _real_b
                sell_save_segs[:] = [(i[0]+begin_diff, i[1]+begin_diff) for i in sell_save_segs]
            old = sell_save_segs.pop(0)
            sell_save_segs.insert(0, (_real_b-begin_idx, old[1]))

    return begin_idx

def split_segments_at_midday(
    df: pd.DataFrame, 
    act_segs: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    遍历活动分段列表，如果某个分段在 'before_market_close_sec' 列上跨越了 0.5（中午），
    则将其拆分为两个新的分段。

    Args:
        df (pd.DataFrame): 包含 'before_market_close_sec' 列的 DataFrame。
                           该列是时间到收盘秒数的归一化浮点值。
        act_segs (List[Tuple[int, int]]): 一个包含 (起始索引, 结束索引) 元组的列表。

    Returns:
        List[Tuple[int, int]]: 处理后的分段列表。跨越中午的分段被拆分，
                               其余分段保持不变。
    """
    # 用于存储最终结果的列表
    processed_segs = []
    
    # 检查输入列是否存在
    if "before_market_close_sec" not in df.columns:
        raise ValueError("DataFrame 中必须包含 'before_market_close_sec' 列")

    # 遍历每一个给定的分段
    for i, (start_idx, end_idx) in enumerate(act_segs):
        # 使用 .loc 获取指定索引范围内的子序列，确保包含起始和结束索引
        segment_series = df.loc[start_idx:end_idx, "before_market_close_sec"]

        # 若第一个 before_market_close_sec 小于 0.5，意味着全部数据都是下午的
        # 拼接剩下未处理的 act_segs，然后返回
        if segment_series.iloc[0] < 0.5:
            processed_segs.extend(act_segs[i:])
            return processed_segs

        # 若最后一个 before_market_close_sec 大于 0.5，意味着全部数据都是上午的
        # 直接加入 processed_segs 中
        if segment_series.iloc[-1] > 0.5:
            processed_segs.append((start_idx, end_idx))
            continue

        # 如果分段为空或只有一个元素，不可能跨越0.5，直接跳过
        if segment_series.shape[0] < 2:
            processed_segs.append((start_idx, end_idx))
            continue

        # 检查分段内是否同时存在小于0.5和大于0.5的值
        # .any() 是一个高效的Pandas/Numpy操作
        has_before_midday = (segment_series < 0.5).any()
        has_after_midday = (segment_series > 0.5).any()

        # 如果一个分段同时包含中午前和中午后的时间点，则需要分割
        if has_before_midday and has_after_midday:
            # --- 执行分割逻辑 ---
            
            # 1. 找到所有大于 0.5 的值
            # 属于 早市 的部分
            before_midday_values = segment_series[segment_series > 0.5]
            # 获取 最后一个数据（中午）的索引
            split_point_before = int(before_midday_values.index[-1])

            # 2. 找到所有小于 0.5 的值
            after_midday_values = segment_series[segment_series < 0.5]
            # 获取 第一个数据（中午）的索引
            split_point_after = int(after_midday_values.index[0])
            
            # 3. 根据要求创建两个新的分段并添加到结果列表中
            #    第一个分段：从原始起点到“中午”的索引
            processed_segs.append((start_idx, split_point_before))
            #    第二个分段：从“中午”的索引到原始终点
            processed_segs.append((split_point_after, end_idx))
            
            # print(f"分段 ({start_idx}, {end_idx}) 已被拆分 -> "
            #       f"({start_idx}, {split_point_before}) 和 ({split_point_after}, {end_idx})")

        else:
            # 如果分段没有跨越0.5，则保持原样
            processed_segs.append((start_idx, end_idx))
            # print(f"分段 ({start_idx}, {end_idx}) 无需拆分，保持原样。")

    return processed_segs

def _plot_df_with_segs(extra_len, b, e, df, col, segs=None, _type_name='profit', extra_name='', logout=blank_logout):
    """
    绘制带有高亮分段区域和边界垂直线的 DataFrame 图表。（健壮版）

    Args:
        extra_len (int): 在主要范围 b-e 前后额外扩展的数据点数。
        b (int): 主要绘图范围的起始索引。
        e (int): 主要绘图范围的结束索引。
        df (pd.DataFrame): 包含价格数据的 DataFrame。
        col (str): 用于计算segs的列名
        _type_name (str, optional): 日志文件名的类型前缀。默认为 'profit'。
        logout (function, optional): 用于记录日志和获取文件路径的函数。
    """
    if None is segs:
        segs = find_segments(df.loc[b:e, col] > 0)
    
    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(15, 6))
    
    try:
        # 准备绘图所需的数据
        plot_df = df.loc[b - extra_len: e + extra_len, ['mid_price', 'BASE买1价', 'BASE卖1价']].copy()

        # 绘制价格曲线
        ax.plot(plot_df.index, plot_df['mid_price'], color='C0', label='mid_price')
        ax.plot(plot_df.index, plot_df['BASE买1价'], color='C1', alpha=0.3, label='BASE买1价')
        ax.plot(plot_df.index, plot_df['BASE卖1价'], color='C2', alpha=0.3, label='BASE卖1价')

        # 绘制垂直线
        ax.axvline(x=b, color='red', linestyle='--', linewidth=1.5, label=f'范围起始 (b={b})')
        ax.axvline(x=e, color='red', linestyle='--', linewidth=1.5, label=f'范围结束 (e={e})')
        ax.axvline(x=e + 2, color='red', linestyle='--', linewidth=1.5, label=f'延申部分 (e={e+2})')

        # 初始化一个集合，用于存储所有自定义的X轴刻度
        custom_x_ticks = {b, e, e + 2}

        # 填充 segs 区域，并收集刻度
        for _b, _e in segs:
            ax.axvspan(_b + b, _e + b, color='#b3e5fc', alpha=0.4, zorder=0)
            custom_x_ticks.add(_b + b)
            custom_x_ticks.add(_e + b)

        # 设置X轴的刻度
        ax.set_xticks(sorted(list(custom_x_ticks)))
        ax.tick_params(axis='x', rotation=45, labelsize=10)

        # 添加图表标题和坐标轴标签
        ax.set_title(f'价格走势与分析区间 ({_type_name})', fontsize=16)
        ax.set_xlabel('数据索引', fontsize=12)
        ax.set_ylabel('价格', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # 保存图表
        plot_file_path = logout(
            title=f'{_type_name}_{extra_name}_plot' if extra_name else f'{_type_name}_plot',
            plot=True)
        if plot_file_path is not None:
            plt.savefig(plot_file_path, dpi=150)
            
    finally:
        # 无论 try 块中是否发生异常，都确保关闭 figure，防止内存泄漏
        plt.close(fig)

def fix_profit_sell_save(df, logout=blank_logout):
    """ 
    df 列字段: action, profit, sell_save, BASE卖1价, BASE买1价
    action==0 中间若有多段 profit>0, 需要对每段重新计算 profit
    action==1 中间若有多段 sell_save>0, 需要对每段重新计算 sell_save
    """
    extra_len = 30

    # 向量化处理 profit 段
    act_segs = find_segments(df['action'] == 0)
    # 对中午的数据进行拆分
    act_segs = split_segments_at_midday(df, act_segs)
    print(f'fix_profit begin')
    for b, e in act_segs:
        pic_type_name = f'profit_{b}-{e}'

        _act_0_data = df.loc[b: e]
        # 查找 profit>0 的 b, e
        profit_segs = find_segments(_act_0_data['profit'] > 0)
        _act_0_data_begin_idx = _act_0_data.iloc[0].name
        # 删除之间只有一个间隔的分组（合并成一个大的分组）
        skip_1_point(_act_0_data_begin_idx, df, profit_segs, 'profit')

        original_plot = False

        print(f'fix_profit {pic_type_name} 0')

        # 对第一段 profit 进行修正
        _original_profit_segs = [i for i in profit_segs]
        _original_act_0_data_begin_idx = _act_0_data_begin_idx
        if len(profit_segs) > 0:
            _b, _e = profit_segs[0]
            real_b, real_e = _act_0_data_begin_idx + _b, _act_0_data_begin_idx + _e
            _act_0_data_begin_idx = check_first_profit_segment(_act_0_data_begin_idx, df, real_b, real_e, profit_segs)

        # 若出现变化 输出 df 表格图片
        if _original_profit_segs != profit_segs:
            # 原始图片
            _plot_df_with_segs(extra_len, _original_act_0_data_begin_idx, e, df, 'profit', segs=_original_profit_segs, _type_name=pic_type_name, extra_name='0', logout=logout)
            original_plot = True
            # 检查第一段后的调整图片
            _plot_df_with_segs(extra_len, _act_0_data_begin_idx, e, df, 'profit', _type_name=pic_type_name, extra_name='1_check_first', logout=logout)

        print(f'fix_profit {pic_type_name} 1')

        # 对最后一段 profit 进行修正
        _old_profit_segs = [i for i in profit_segs]
        if len(profit_segs) > 0:
            _b, _ = profit_segs[-1]
            real_b, real_e = _act_0_data_begin_idx + _b, e
            # 递归检查最后一段是否包含 profit<-0 的 no move 段
            # 若有的话，需要调整结束点，并重新计算 profit
            # 直到只包含一段 profit>0
            # 新产生的非最后一段的 profit>0段，会合并到 profit_segs 中，待之后处理
            _act_0_data_begin_idx = check_last_profit_segment(_act_0_data_begin_idx, df, real_b, real_e, profit_segs)
            # print(f'check_last_profit_segment done')

        # 若出现变化 输出 df 表格图片
        if _old_profit_segs != profit_segs and len(profit_segs):
            if not original_plot:
                _plot_df_with_segs(extra_len, _original_act_0_data_begin_idx, e, df, 'profit', segs=_original_profit_segs, _type_name=pic_type_name, extra_name='0', logout=logout)
                original_plot = True
            _plot_df_with_segs(extra_len, _act_0_data_begin_idx, e, df, 'profit', _type_name=pic_type_name, extra_name='2_check_last', logout=logout)

        print(f'fix_profit {pic_type_name} 2')
        print(profit_segs)

        # 需要对除了最后一段的 profit 进行修正
        idx = 0
        dones = []
        while len(profit_segs) > 1:
            assert idx < 30, '异常的 profit_segs 长度，可能是死循环'
            # print(f'profit_segs length: {len(profit_segs)}')
            # 从最后一段开始，向前处理
            _b, _e = profit_segs[-2]
            # profit_segs[-2] 与 profit_segs[-1] 之间存在多个 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 连续的段，需要处理
            # 若存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 段，需要改用第一个 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 段begin+5
            # 否则都使用 profit_segs[-1][0]-1
            _ee = profit_segs[-1][0]-1
            real_b, real_e, real_ee = _act_0_data_begin_idx + _b, _act_0_data_begin_idx + _e, _act_0_data_begin_idx + _ee
            # 检查 real_e 与 real_ee 之间是否存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 的段
            # if (df.loc[real_e:real_ee, 'no_move_len_pct'].max() > NO_MOVE_THRESHOLD or
            #     df.loc[real_e:real_ee, 'no_move_len_2price'].max() > NO_MOVE_THRESHOLD):
            if df.loc[real_e:real_ee, 'no_move_len'].max() > NO_MOVE_THRESHOLD:
                # 若存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 段，需要改用第一个 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 段的最高点
                df_range = df.loc[real_e:real_ee, :]
                # if df_range['no_move_len_pct'].max() > NO_MOVE_THRESHOLD:
                #     __b = df_range[df_range['no_move_len_pct'] > NO_MOVE_THRESHOLD].index[0]
                # else:
                #     __b = df_range[df_range['no_move_len_2price'] > NO_MOVE_THRESHOLD].index[0]
                __b = df_range[df_range['no_move_len'] > NO_MOVE_THRESHOLD].index[0]
                __e = __b + NO_MOVE_THRESHOLD - 2 - 1
                __max_idx, _ = find_last_max_b1_value(df.loc[__b:__e, :])
                real_e = min(__max_idx + __b, real_ee)
            else:
                real_e = real_ee
            real_b, real_e, close_price = fix_profit(df, real_b, real_e)
            if real_b < _act_0_data_begin_idx:
                # real_b 可能小于 _act_0_data_begin_idx
                diff = _act_0_data_begin_idx - real_b
                _act_0_data_begin_idx = real_b
                # 更新 profit_segs
                profit_segs[:] = [(i[0]+diff, i[1]+diff) for i in profit_segs]

            # 检查新的 profit 是否有夹杂 profit <= 0 的段
            _new_profit_segs = find_segments(df.loc[real_b:real_e, 'profit'] > 0)
            # 需要向前推并合并
            _extend_segs = []
            for _b, _e in _new_profit_segs:
                new_begin = _extend_profit_start(df, _b+real_b, _e+real_b, df['profit'].values, df['mid_price'].values, df['no_move_len'].values)
                _extend_segs.append((new_begin-real_b, _e))
            _new_profit_segs = _merge_segs(_extend_segs)
            skip_1_point(real_b, df, _new_profit_segs, 'profit')

            change = False
            old_profit_segs = [i for i in profit_segs[:-2]]
            if close_price:
                # 根据 close_price 调整 profit_segs[:-2] 中的段
                _new_profit_segs_before = fix_profit_segs(b, df, profit_segs[:-2], close_price)
                # 合并到 profit_segs
                profit_segs[:-2] = _new_profit_segs_before
                if old_profit_segs != profit_segs[:-2]:
                    change = True

            # 删除最后一段
            _last = profit_segs.pop(-1)
            old_segs = [i for i in profit_segs]
            old = profit_segs.pop(-1)# 删除当前段
            if len(_new_profit_segs) > 0:
                profit_segs.extend([(i[0]+real_b-_act_0_data_begin_idx, i[1]+real_b-_act_0_data_begin_idx) for i in _new_profit_segs])# 合并最新段
                profit_segs = _merge_segs(profit_segs)# 去重检查

                # 判断是否有改变
                if old_segs != profit_segs:
                    change = True

                    # 判断最后一段是否一致
                    if profit_segs[-1] == old:
                        # 不需要加回 _last
                        dones.insert(0, _last)
                    else:
                        # 需要加回最后一段，重新检查
                        profit_segs.append(_last)
                else:
                    # 没有变化，就不需要加回 _last
                    dones.insert(0, _last)

            else:
                # _new_profit_segs 为空
                # 需要把 _last 加回来
                profit_segs.append(_last)
                # 手动调整 _last 的起点（至old[0]），避免重新计算
                profit_segs[-1] = (old[0], profit_segs[-1][1])
                change = True

            if change:
                all = profit_segs + dones
                if len(all):
                    if not original_plot:   
                        _plot_df_with_segs(extra_len, _original_act_0_data_begin_idx, e, df, 'profit', segs=_original_profit_segs, _type_name=pic_type_name, extra_name='0', logout=logout)
                        original_plot = True
                    _plot_df_with_segs(extra_len, _act_0_data_begin_idx, e, df, 'profit', _type_name=pic_type_name, extra_name=f'3_fix_{idx}', logout=logout)

            print(f'fix_profit {pic_type_name} 3 {idx}')

            idx += 1
    # report_memory_usage(f'fix_profit end')

    # 向量化处理 sell_save 段
    act_segs = find_segments(df['action'] == 1)
    # 对中午的数据进行拆分
    act_segs = split_segments_at_midday(df, act_segs)
    print(f'fix_sell_save begin')
    for b, e in act_segs:
        pic_type_name = f'sell_save_{b}-{e}'

        _act_1_data = df.loc[b: e]
        # 查找 sell_save>0 的 b, e
        sell_save_segs = find_segments(_act_1_data['sell_save'] > 0)
        _act_1_data_begin_idx = _act_1_data.iloc[0].name
        # 删除之间只有一个间隔的分组（合并成一个大的分组）
        skip_1_point(_act_1_data_begin_idx, df, sell_save_segs, 'sell_save')

        original_plot = False

        # 对第一段 sell_save 进行修正
        _original_sell_save_segs = [i for i in sell_save_segs]
        _original_act_1_data_begin_idx = _act_1_data_begin_idx
        if len(sell_save_segs) > 0:
            _b, _e = sell_save_segs[0]
            real_b, real_e = _act_1_data_begin_idx + _b, _act_1_data_begin_idx + _e
            _act_1_data_begin_idx = check_first_sell_save_segment(_act_1_data_begin_idx, df, real_b, real_e, sell_save_segs)

        # 若出现变化 输出 df 表格图片
        if _original_sell_save_segs != sell_save_segs:
            _plot_df_with_segs(extra_len, _original_act_1_data_begin_idx, e, df, 'sell_save', segs=_original_sell_save_segs, _type_name=pic_type_name, extra_name='0', logout=logout)
            original_plot = True
            _plot_df_with_segs(extra_len, _act_1_data_begin_idx, e, df, 'sell_save', _type_name=pic_type_name, extra_name='1_check_first', logout=logout)

        # 对最后一段 sell_save 进行修正
        _old_sell_save_segs = [i for i in sell_save_segs]
        if len(sell_save_segs) > 0:
            _b, _ = sell_save_segs[-1]
            real_b, real_e = _act_1_data_begin_idx + _b, e
            # 递归检查最后一段是否包含 sell_save<-0 的 no move 段
            # 若有的话，需要调整结束点，并重新计算 sell_save
            # 直到只包含一段 sell_save>0
            # 新产生的非最后一段的 sell_save>0段，会合并到 sell_save_segs 中，待之后处理
            _act_1_data_begin_idx = check_last_sell_save_segment(_act_1_data_begin_idx, df, real_b, real_e, sell_save_segs)

            # 重新计算一遍 process_lob_data_extended_sell_save
            _df = process_lob_data_extended_sell_save(df.loc[real_b:real_e, :].reset_index(drop=True).copy())
            df.loc[real_b:real_e, 'sell_save'] = _df['sell_save'].values
            # print(f'check_last_sell_save_segment done')

        # 若出现变化 输出 df 表格图片
        if _old_sell_save_segs != sell_save_segs and len(sell_save_segs):
            if not original_plot:
                _plot_df_with_segs(extra_len, _original_act_1_data_begin_idx, e, df, 'sell_save', segs=_original_sell_save_segs, _type_name=pic_type_name, extra_name='0', logout=logout)
                original_plot = True
            _plot_df_with_segs(extra_len, _act_1_data_begin_idx, e, df, 'sell_save', _type_name=pic_type_name, extra_name='2_check_last', logout=logout)

        # 需要对除了最后一段的 sell_save 进行修正
        idx = 0
        dones = []
        while len(sell_save_segs) > 1:
            assert idx < 30, '异常的 sell_save_segs 长度，可能是死循环'
            # print(f'sell_save_segs length: {len(sell_save_segs)}')
            # 从最后一段开始，向前处理
            _b, _e = sell_save_segs[-2]
            _ee= sell_save_segs[-1][0]-1
            real_b, real_e, real_ee = _act_1_data_begin_idx + _b, _act_1_data_begin_idx + _e, _act_1_data_begin_idx + _ee
            # 检查 real_e 与 real_ee 之间是否存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 的段
            # if (df.loc[real_e:real_ee, 'no_move_len_pct'].max() > NO_MOVE_THRESHOLD or
            #     df.loc[real_e:real_ee, 'no_move_len_2price'].max() > NO_MOVE_THRESHOLD):
            if df.loc[real_e:real_ee, 'no_move_len'].max() > NO_MOVE_THRESHOLD:
                # 若存在 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 段，需要改用第一个 no_move_len_pct/no_move_len_2price>NO_MOVE_THRESHOLD 段的最低点
                df_range = df.loc[real_e:real_ee, :]
                # if df_range['no_move_len_pct'].max() > NO_MOVE_THRESHOLD:
                #     __b = df_range[df_range['no_move_len_pct'] > NO_MOVE_THRESHOLD].index[0]
                # else:
                #     __b = df_range[df_range['no_move_len_2price'] > NO_MOVE_THRESHOLD].index[0]
                __b = df_range[df_range['no_move_len'] > NO_MOVE_THRESHOLD].index[0]
                __e = __b + NO_MOVE_THRESHOLD - 2 - 1
                __min_idx, _ = find_last_min_a1_value(df.loc[__b:__e, :])
                real_e = min(__min_idx + __b, real_ee)
            else:
                real_e = real_ee
            real_b, real_e, close_price = fix_sell_save(df, real_b, real_e)
            if real_b < _act_1_data_begin_idx:
                # real_b 可能小于 _act_1_data_begin_idx
                diff = _act_1_data_begin_idx - real_b
                _act_1_data_begin_idx = real_b
                # 更新 sell_save_segs
                sell_save_segs[:] = [(i[0]+diff, i[1]+diff) for i in sell_save_segs]

            # 检查新的 sell_save 是否有夹杂 sell_save <= 0 的段
            _new_sell_save_segs = find_segments(df.loc[real_b:real_e, 'sell_save'] > 0)
            # 需要向前推并合并
            _extend_segs = []
            for _b, _e in _new_sell_save_segs:
                new_begin = _extend_sell_save_start(df, _b+real_b, _e+real_b, df['sell_save'].values, df['mid_price'].values, df['no_move_len'].values)
                _extend_segs.append((new_begin-real_b, _e))
            _new_sell_save_segs = _merge_segs(_extend_segs)
            skip_1_point(real_b, df, _new_sell_save_segs, 'sell_save')

            change = False
            old_sell_save_segs = [i for i in sell_save_segs[:-2]]
            if close_price:
                # 根据 close_price 调整 sell_save_segs[:-2] 中的段
                _new_sell_save_segs_before = fix_sell_save_segs(b, df, sell_save_segs[:-2], close_price)
                # 合并到 sell_save_segs
                sell_save_segs[:-2] = _new_sell_save_segs_before
                if old_sell_save_segs != sell_save_segs[:-2]:
                    change = True

            # 删除最后一段
            _last = sell_save_segs.pop(-1)
            old_segs = [i for i in sell_save_segs]
            old = sell_save_segs.pop(-1)# 删除当前段
            if len(_new_sell_save_segs) > 0:
                sell_save_segs.extend([(i[0]+real_b-_act_1_data_begin_idx, i[1]+real_b-_act_1_data_begin_idx) for i in _new_sell_save_segs])# 合并最新段
                sell_save_segs = _merge_segs(sell_save_segs)# 去重检查

                # 判断是否有改变
                if old_sell_save_segs != sell_save_segs:
                    change = True

                    # 判断最后一段是否一致
                    if sell_save_segs[-1] == old:
                        # 不需要加回 _last
                        dones.insert(0, _last)
                    else:
                        # 需要加回最后一段，重新检查
                        sell_save_segs.append(_last)
                else:
                    # 没有变化，就不需要加回 _last
                    dones.insert(0, _last)

            else:
                # _new_sell_save_segs 为空
                # 需要把 _last 加回来
                sell_save_segs.append(_last)
                # 手动调整 _last 的起点（至old[0]），避免重新计算
                sell_save_segs[-1] = (old[0], sell_save_segs[-1][1])
                change = True

            if change:
                all = sell_save_segs + dones
                if len(all):
                    if not original_plot:
                        _plot_df_with_segs(extra_len, _original_act_1_data_begin_idx, e, df, 'sell_save', segs=_original_sell_save_segs, _type_name=pic_type_name, extra_name='0', logout=logout)
                        original_plot = True
                    _plot_df_with_segs(extra_len, _act_1_data_begin_idx, e, df, 'sell_save', _type_name=pic_type_name, extra_name=f'3_fix_{idx}', logout=logout)

            idx += 1
    # report_memory_usage(f'fix_sell_save end')

    return df

def replace_isolated_value_inplace(df, column='profit'):
    """
    直接在原始 DataFrame 上修改，将指定列的孤立正值（profit > 0 且前后无正值）
    和孤立非正值（profit <= 0 且前后无非正值）替换为前一个值。
    
    参数：
        df (pd.DataFrame): 输入的 DataFrame，将被直接修改
        column (str): 要处理的列名，默认为 'profit'
    
    返回：
        None: 直接修改原始 DataFrame，无返回值
    """
    # 确保列存在
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # 1. 标记正值行 (profit > 0)
    df['is_positive'] = df[column] > 0
    
    # 2. 标记孤立正值行（前后都不是正值）
    df['is_isolated_positive'] = (
        df['is_positive'] & 
        ~df['is_positive'].shift(1, fill_value=False) & 
        ~df['is_positive'].shift(-1, fill_value=False)
    )
    
    # 3. 标记非正值行 (profit <= 0)
    df['is_non_positive'] = df[column] <= 0
    
    # 4. 标记孤立非正值行（前后都不是非正值）
    df['is_isolated_non_positive'] = (
        df['is_non_positive'] & 
        ~df['is_non_positive'].shift(1, fill_value=False) & 
        ~df['is_non_positive'].shift(-1, fill_value=False)
    )
    
    # 5. 替换孤立正值行和孤立非正值行为前一个值
    df.loc[df['is_isolated_positive'] | df['is_isolated_non_positive'], column] = \
        df[column].shift(1)[df['is_isolated_positive'] | df['is_isolated_non_positive']]
    
    # 6. 删除辅助列
    df.drop(['is_positive', 'is_isolated_positive', 'is_non_positive', 'is_isolated_non_positive'], 
            axis=1, inplace=True)
    
def reset_profit(lob_data, data_0=None, data_m1=None, logout=blank_logout):
    """
    剔除掉动作之后下一个点的价格变化带来的优势 (成交价格带来的优势不允许利用)
    第一个profit>0信号点，下一个时点价格不能下跌
    最后一个profit>0信号点，下一个时点价格不能下跌

    第一个profit<=0信号点, 若与前一个时点的中间价一致，下一个时点价格不能上涨

    参数：
        lob_data (pd.DataFrame): 输入的 DataFrame，将被直接修改
        data_0 (float): 备份 lob_data 前一个数据，不在操作范围内，只用于 shift 避免nan
        data_m1 (float): 备份 lob_data 后一个数据，不在操作范围内，只用于 shift 避免nan
        logout (function): 输出函数，用于输出日志

    返回：
        pd.DataFrame: 修改后的 DataFrame
    """
    next_sell_price = lob_data['BASE卖1价'].shift(-1)

    # # 先替换掉孤立值
    # replace_isolated_value_inplace(lob_data, 'profit')

    count = 0
    while True:
        stop = False
        while not stop:
            # 持续到没有满足条件的行
            # 剔除 profit
            # 第一个profit>0信号点，下一个时点价格不能下跌
            prev_profit = lob_data['profit'].shift(1)
            pprev_profit = lob_data['profit'].shift(2)
            # 条件1：profit > 0, 前一行 profit <= 0, 且前二行 profit <= 0(容忍一个突刺价格), 且下一个 BASE卖1价 < 当前 BASE卖1价
            profit_cond1 = (lob_data['profit'] > 0) & (prev_profit <= 0) & (pprev_profit <= 0) & (next_sell_price < lob_data['BASE卖1价'])
            # 将满足条件1的行的 profit 置为 0
            lob_data.loc[profit_cond1, 'profit'] = 0

            # 条件1：profit > 0, 前一行 profit > 0, 下一行 profit <= 0，且下二行 profit <= 0（容忍一个突刺价格）, 且下一个 BASE卖1价 < 当前 BASE卖1价
            # 最后一个profit>0信号点，下一个时点价格不能下跌
            prev_profit = lob_data['profit'].shift(1)
            next_profit = lob_data['profit'].shift(-1)
            nnext_profit = lob_data['profit'].shift(-2)
            profit_cond2 = (lob_data['profit'] > 0) & (prev_profit > 0) & (next_profit <= 0) & (nnext_profit <= 0) & (next_sell_price < lob_data['BASE卖1价'])
            # 将满足条件1的行的 profit 置为 0
            lob_data.loc[profit_cond2, 'profit'] = 0

            if not profit_cond1.any() and not profit_cond2.any():
                stop = True
                if count > 0:
                    # 恢复备份
                    if data_0 is not None:
                        lob_data.loc[lob_data.index[0], 'profit'] = data_0
                    if data_m1 is not None:
                        lob_data.loc[lob_data.index[-1], 'profit'] = data_m1

                    return lob_data

        stop = False
        while not stop:
            # 条件2：profit <= 0, 前一行 profit > 0, 且当前 BASE卖1价 == 前一个 BASE卖1价, 且下一个 BASE卖1价 > 当前 BASE卖1价
            # 第一个profit<=0信号点, 若与前一个时点的中间价一致，下一个时点价格不能上涨
            prev_sell_price = lob_data['BASE卖1价'].shift(1)
            prev_profit = lob_data['profit'].shift(1) #需要重新更新, 可能被 cond1 修改
            profit_cond2 = (lob_data['profit'] <= 0) & (prev_profit > 0) & (prev_sell_price == lob_data['BASE卖1价']) & (next_sell_price > lob_data['BASE卖1价'])
            # 将前一行的profit值赋给当前行
            lob_data.loc[profit_cond2, 'profit'] = prev_profit.loc[profit_cond2]

            if not profit_cond2.any():
                stop = True

        count += 1

        if count > 10:
            print('reset_profit 循环次数超过10次')

# find_binary_extrema 函数保持不变
def find_binary_extrema(arr: np.ndarray) -> np.ndarray:
    """
    在一个一维 ndarray 中找到所有二元分类的极值点。
    """
    # (代码与上一版本完全相同)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim != 1:
        raise ValueError("输入数组必须是一维的 (Input array must be 1-dimensional)")
    if arr.size < 3:
        return np.zeros_like(arr, dtype=bool)
    is_positive = arr > 0
    is_positive_center = is_positive[1:-1]
    is_positive_prev = is_positive[:-2]
    is_positive_next = is_positive[2:]
    is_extremum_center = (is_positive_center != is_positive_prev) & \
                         (is_positive_center != is_positive_next)
    result = np.zeros_like(arr, dtype=bool)
    result[1:-1] = is_extremum_center
    return result

# --- 新增功能：扩展布尔掩码中的 True 区域 ---
def expand_true_regions_0(mask: np.ndarray, n: int) -> np.ndarray:
    """
    扩展布尔掩码中的 True 区域，将其前后 n 个元素也设置为 True。

    该函数使用一维卷积实现高效的向量化操作。

    Args:
        mask (np.ndarray): 输入的布尔掩码数组。
        n (int): 向前后扩展的邻域大小。如果 n=0，则不进行扩展。

    Returns:
        np.ndarray: 扩展后的布尔掩码数组。
        
    Raises:
        ValueError: 如果 n 为负数。
    """
    # --- 输入验证 ---
    if n < 0:
        raise ValueError("扩展邻域大小 n 不能为负数 (n cannot be negative)")
    if n == 0:
        return mask.copy() # 如果 n=0，无需操作，返回副本
    if mask.size == 0 or not np.any(mask):
        return mask.copy() # 如果掩码为空或全为False，无需操作

    # --- 核心逻辑：卷积 ---
    # 1. 将布尔掩码转换为整数 (True->1, False->0)
    mask_int = mask.astype(int)
    
    # 2. 创建卷积核，长度为 2*n+1，所有元素为1
    #    例如 n=2, kernel = [1, 1, 1, 1, 1]
    kernel = np.ones(2 * n + 1, dtype=int)
    
    # 3. 执行卷积。mode='same' 确保输出数组与输入数组等长。
    #    卷积操作会将每个 True (1) 的影响扩散到其邻域。
    expanded_int = np.convolve(mask_int, kernel, mode='same')
    
    # 4. 将结果转回布尔值。任何非零值都表示该位置在扩展区域内。
    expanded_mask = expanded_int > 0
    
    return expanded_mask

def expand_true_regions(mask: np.ndarray, n: int) -> np.ndarray:
    """
    扩展布尔掩码中的 True 区域，将其前后 n 个元素也设置为 True。

    该函数通过先对数组进行填充（padding），再使用一维卷积（convolution）
    的方式实现高效的向量化操作。这种方法逻辑清晰，结果可预测，
    避免了不同库或版本中 `mode='same'` 参数可能带来的歧义。

    Args:
        mask (np.ndarray): 输入的布尔掩码数组，必须是一维的。
        n (int): 向前后扩展的邻域大小。如果 n=0，则不进行扩展。

    Returns:
        np.ndarray: 扩展后的布尔掩码数组，其长度与输入数组相同。
        
    Raises:
        ValueError: 如果 n 为负数或 mask 不是一维数组。
    """
    # --- 输入验证 ---
    if mask.ndim != 1:
        raise ValueError("输入掩码必须是一维数组 (Input mask must be a 1D array)")
    if n < 0:
        raise ValueError("扩展邻域大小 n 不能为负数 (n cannot be negative)")
    if n == 0:
        return mask.copy()
    if mask.size == 0 or not np.any(mask):
        return mask.copy()

    # --- 核心逻辑：两个方向的扩展 ---
    # 为了精确控制扩展，我们分别处理向前和向后的情况，然后合并结果。
    # 这比使用一个中心化的卷积核(2*n+1)在边界处理上更精确、更直观。

    # 1. 将布尔掩码转换为整数 (True->1, False->0)
    mask_int = mask.astype(np.int8)

    # 2. 向右（向前）扩展：
    #    创建一个长度为 n+1 的卷积核 [1, 1, ..., 1]
    #    通过卷积，任何一个 True (1) 都会使其后的 n 个元素变为非零。
    kernel = np.ones(n + 1, dtype=np.int8)
    #    我们在右侧填充n个0，确保卷积输出长度与原数组一致。
    padded_forward = np.pad(mask_int, (0, n), 'constant')
    expanded_forward = np.convolve(padded_forward, kernel, mode='valid') > 0

    # 3. 向左（向后）扩展：
    #    同样的逻辑，只是填充方向相反。
    padded_backward = np.pad(mask_int, (n, 0), 'constant')
    expanded_backward = np.convolve(padded_backward, kernel, mode='valid') > 0
    
    # 4. 合并两个方向的扩展结果
    #    使用逻辑或操作，任何一个方向扩展到的区域都为 True。
    #    注意：这里有一个小技巧，expanded_backward 需要向右移动n个位置
    #    才能与原始数组对齐。一个更简单的方法是反转原数组和结果。
    mask_int_reversed = mask_int[::-1]
    padded_reversed = np.pad(mask_int_reversed, (0, n), 'constant')
    expanded_reversed_temp = np.convolve(padded_reversed, kernel, mode='valid') > 0
    expanded_backward = expanded_reversed_temp[::-1] # 再次反转回来

    # 将原始为True的位置、向前扩展的位置、向后扩展的位置合并
    final_mask = mask | expanded_forward | expanded_backward
    
    return final_mask

def keep_long_true_blocks(arr: np.ndarray, min_length: int = 5) -> np.ndarray:
    """
    在一个布尔型NumPy数组中，只保留长度大于等于min_length的连续True块。

    该函数利用scipy.ndimage.label高效地识别和处理连续块。

    参数:
        arr (np.ndarray): 输入的布尔型一维数组。
        min_length (int): 连续True块的最小长度阈值。

    返回:
        np.ndarray: 处理后的布尔型数组，不满足长度要求的True块被置为False。
        
    示例:
        >>> test_arr = np.array([F, T, T, F, T, T, T, T, T, F, T, T, T, F, T, T, T, T, T, T, T])
        >>> # 期望保留长度为5和7的块，丢弃长度为2和3的块
        >>> keep_long_true_blocks(test_arr, 5)
        array([F, F, F, F, T, T, T, T, T, F, F, F, F, F, T, T, T, T, T, T, T])
        
    """
    # 检查输入类型
    if not isinstance(arr, np.ndarray) or arr.dtype != bool:
        raise TypeError("输入必须是一个布尔型的NumPy数组。")
    if arr.ndim != 1:
        raise ValueError("该函数目前只支持一维数组。")

    # 1. 使用label函数标记所有连续的True块
    # labeled_array中，每个连续的True块会被赋予一个从1开始的唯一整数标签
    # background (False) 的标签为0
    # num_features 是找到的True块的总数
    labeled_array, num_features = label(arr)

    # 2. 如果没有找到任何True块，直接返回原数组的副本（或一个全False数组）
    if num_features == 0:
        return arr.copy()

    # 3. 计算每个标签（即每个True块）的大小
    # np.bincount会返回一个数组，其索引i处的值是labeled_array中i出现的次数
    # 我们需要ravel()将数组展平，以防未来扩展到多维
    # bincount的结果长度为 num_features + 1，第一个元素(索引0)是背景False的计数
    component_sizes = np.bincount(labeled_array.ravel())

    # 4. 找出大小不满足min_length要求的True块的标签
    # 我们只关心标签>0的块，并且其大小小于min_length
    too_small_labels = np.where(component_sizes < min_length)[0]

    # 5. 创建最终的输出数组
    # 首先创建一个与输入形状相同的全False数组
    output_arr = np.zeros_like(arr, dtype=bool)
    
    # 将labeled_array中不属于too_small_labels的元素位置，在output_arr中置为True
    # np.isin(labeled_array, too_small_labels)会返回一个布尔掩码，
    # 标记了所有属于"太小"块的元素。我们用~将其反转，即选中所有"合格"的块。
    # 同时，我们还要确保不选中背景（标签0），所以`labeled_array != 0`
    valid_mask = ~np.isin(labeled_array, too_small_labels) & (labeled_array != 0)
    
    return valid_mask

def calculate_er_metrics(np_arr: np.ndarray, window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """
    计算给定价格序列的效率比率（ER）和相关的波动指标。

    计算基于一个指定大小（window）的中心移动窗口。

    1. ER (效率比率) = Signal / Noise
       - Signal (信号) = |窗口最后一个点的价格 - 窗口第一个点的价格|
       - Noise (噪声) = sum(|Price[t] - Price[t-1]|) for t over the window

    2. Wasted Movement (无效波动) = Noise - Signal
       - 这个值代表了窗口内所有未对净方向变化做出贡献的价格波动总和。

    Args:
        np_arr (np.ndarray): 输入的价格序列，一个一维的 NumPy 数组。
        window (int, optional): 用于计算的移动窗口大小。必须是一个大于1的奇数。
                                默认为 5。

    Returns:
        tuple[np.ndarray, np.ndarray]: 一个包含两个 NumPy 数组的元组：
            - er_series (np.ndarray): 与输入等长的 ER 序列。
            - wasted_movement_series (np.ndarray): 与输入等长的无效波动序列。
            对于无法形成完整窗口的边界点，两个序列中的对应值为 np.nan。

    Raises:
        ValueError: 如果 window 不是一个大于1的奇数。
    """
    # --- 输入验证 ---
    if not isinstance(window, int) or window <= 1 or window % 2 == 0:
        raise ValueError("窗口大小（window）必须是一个大于1的奇数。")

    n = len(np_arr)

    # 如果序列长度小于窗口大小，无法进行任何计算
    if n < window:
        return (np.full(n, np.nan), np.full(n, np.nan))

    # --- 初始化结果数组 ---
    er = np.full(n, np.nan)
    wasted_movement = np.full(n, np.nan)
    
    # --- 向量化计算 ---

    # 1. 计算信号 (Signal)
    # 窗口内最后一个价格与第一个价格之差的绝对值
    signal = np.abs(np_arr[window - 1:] - np_arr[:-(window - 1)])

    # 2. 计算噪声 (Noise)
    # 窗口内所有相邻价格差的绝对值之和
    price_diffs = np.abs(np.diff(np_arr))
    noise = np.convolve(price_diffs, np.ones(window - 1, dtype=float), mode='valid')

    # 3. 计算 ER 核心部分
    er_core = np.divide(signal, noise, out=np.ones_like(signal, dtype=float), where=noise != 0)

    # 4. 计算无效波动 (Wasted Movement) 的核心部分
    wasted_movement_core = noise - signal

    # --- 填充结果 ---
    
    # 5. 计算结果的起始填充位置
    offset = (window - 1) // 2
    
    # 6. 将计算出的核心部分填充到结果数组中
    er[offset:-offset] = er_core
    wasted_movement[offset:-offset] = wasted_movement_core

    return er, wasted_movement


def find_not_stable_bid_ask(df, N_EXPANSION = 1) -> np.ndarray:
    """
    df: 所有的数据

    找到不稳定的 bid ask 点
    返回需要 屏蔽 的bool索引

    会保留 bid ask 起始的第一个数据
    """
    bid = df['BASE买1价'].values
    ask = df['BASE卖1价'].values

    # 计算 er / wasted_movement 的窗口大小
    window = 5
    extra_length = (window - 1) // 2

    # 计算 er / wasted_movement
    er_bid, wasted_movement_bid = calculate_er_metrics(bid, window=window)
    er_ask, wasted_movement_ask = calculate_er_metrics(ask, window=window)

    # 找到极端的值 趋势越小 and 无效波动越大
    _er_mask = ((er_bid <= 0.15) | (er_ask <= 0.15)) & ((wasted_movement_bid >= 0.003) | (wasted_movement_ask >= 0.003))

    # 扩展极值点区域
    _er_mask = expand_true_regions(_er_mask, n=5)

    # # 只有连续 5 个 True 才保留
    # _er_mask = keep_long_true_blocks(_er_mask, min_length=12)

    # # 扩展极值点区域
    # _er_mask = expand_true_regions(_er_mask, n=5)

    return _er_mask

def find_not_stable_sign(arr, N_EXPANSION = 1) -> np.ndarray:
    """
    找到不稳定的信号点
    返回需要 屏蔽 的bool索引

    会保留 arr 起始的第一个数据
    """
    # 步骤 1.0: 找到初始的极值点 profit / sell_save
    initial_extrema_mask = find_binary_extrema(arr)

    # 步骤 2: 扩展极值点区域
    expanded_extrema_mask = expand_true_regions(initial_extrema_mask, n=N_EXPANSION)

    # 保证 arr 起始的第一个数据不参与计算
    expanded_extrema_mask[0] = False

    # 最终只保留 连续 True 数量>5 的区域
    expanded_extrema_mask = keep_long_true_blocks(expanded_extrema_mask, min_length=5)

    return expanded_extrema_mask

def reset_sell_save(lob_data, data_0=None, data_m1=None, logout=blank_logout):
    """
    剔除掉动作之后下一个点的价格变化带来的优势 (成交价格带来的优势不允许利用)  
    第一个sell_save>0信号点，下一个时点价格不能上涨  
    最后一个sell_save>0信号点，下一个时点价格不能上涨  

    第一个sell_save<=0信号点, 若与前一个时点的中间价一致，下一个时点价格不能下跌  

    参数：
        lob_data (pd.DataFrame): 输入的 DataFrame，将被直接修改
        data_0 (float): 备份 lob_data 前一个数据，不在操作范围内，只用于 shift 避免nan
        data_m1 (float): 备份 lob_data 后一个数据，不在操作范围内，只用于 shift 避免nan
        logout (function): 输出函数，用于输出日志

    返回：
        pd.DataFrame: 修改后的 DataFrame    
    """ 
    next_buy_price = lob_data['BASE买1价'].shift(-1)   

    # # 先替换掉孤立值
    # replace_isolated_value_inplace(lob_data, 'sell_save')

    count = 0
    while True:
        stop = False
        while not stop:
            # 计算前一行和下一行的值
            prev_sell_save = lob_data['sell_save'].shift(1)
            pprev_sell_save = lob_data['sell_save'].shift(2)

            # 剔除 sell_save
            # 条件1：sell_save > 0, 前一行 sell_save <= 0, 且前二行 sell_save <= 0(容忍一个突刺价格), 且下一个 BASE买1价 > 当前 BASE买1价
            # 第一个sell_save>0信号点，下一个时点价格不能上涨
            sell_save_cond1 = (lob_data['sell_save'] > 0) & (prev_sell_save <= 0) & (pprev_sell_save <= 0) & (next_buy_price > lob_data['BASE买1价'])
            # 将满足条件1的行的 sell_save 置为 0
            lob_data.loc[sell_save_cond1, 'sell_save'] = 0

            # 条件1：sell_save > 0, 前一行 sell_save > 0, 下一行 sell_save <= 0, 下二行 sell_save <= 0（容忍一个突刺价格）, 且下一个 BASE买1价 > 当前 BASE买1价
            # 最后一个sell_save>0信号点，下一个时点价格不能上涨
            prev_sell_save = lob_data['sell_save'].shift(1)
            next_sell_save = lob_data['sell_save'].shift(-1)
            nnext_sell_save = lob_data['sell_save'].shift(-2)
            sell_save_cond2 = (lob_data['sell_save'] > 0) & (prev_sell_save > 0) & (next_sell_save <= 0) & (nnext_sell_save <= 0) & (next_buy_price > lob_data['BASE买1价'])
            # 将满足条件1的行的 sell_save 置为 0
            lob_data.loc[sell_save_cond2, 'sell_save'] = 0

            if not sell_save_cond1.any() and not sell_save_cond2.any():
                stop = True
                if count > 0:
                    # 恢复备份
                    if data_0 is not None:
                        lob_data.loc[lob_data.index[0], 'sell_save'] = data_0
                    if data_m1 is not None:
                        lob_data.loc[lob_data.index[-1], 'sell_save'] = data_m1

                    return lob_data

        stop = False
        while not stop:
            # 条件2：sell_save <= 0, 前一行 sell_save > 0, 且当前 BASE买1价 == 前一个 BASE买1价, 且下一个 BASE买1价 < 当前 BASE买1价
            # 第一个sell_save<=0信号点, 若与前一个时点的中间价一致，下一个时点价格不能下跌
            prev_buy_price = lob_data['BASE买1价'].shift(1)
            prev_sell_save = lob_data['sell_save'].shift(1) #需要重新更新, 可能被 cond1 修改
            sell_save_cond2 = (lob_data['sell_save'] <= 0) & (prev_sell_save > 0) & (prev_buy_price == lob_data['BASE买1价']) & (next_buy_price < lob_data['BASE买1价'])    
            # 将前一行的sell_save值赋给当前行
            lob_data.loc[sell_save_cond2, 'sell_save'] = prev_sell_save.loc[sell_save_cond2]
            if not sell_save_cond2.any():
                stop = True

        count += 1
  
        if count > 10:
            print('reset_sell_save 循环次数超过10次')

def reset_profit_sell_save(lob_data, logout=blank_logout):
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
    处理lob_data DataFrame，根据规则调整'sell_save'列。
    
    规则:
    1. 对于sell_save<=0连续块，若之间无action=0，只保留最后一个与action=0相连的块/或早盘午盘最后一个块（若无则全不保留），其余替换为之前最近的sell_save>0值；若无sell_save>0值，替换为0。
    
    参数:
    df (pd.DataFrame): 包含'sell_save'和'action'列的DataFrame
    
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

    # 找到每个组的最后一个 sell_save<=0 块，并检查是否与 action=0 相连
    last_blocks = df[df['is_neg_sell_save']].groupby('group_sell_save')['block_sell_save'].max()
    for group, last_block in last_blocks.items():
        last_block_rows = df[(df['group_sell_save'] == group) & (df['block_sell_save'] == last_block) & (df['is_neg_sell_save'])]
        if not last_block_rows.empty:
            last_row_idx = last_block_rows.index[-1]
            if df.loc[last_row_idx, 'next_action'] == 0:
                df.loc[last_block_rows.index, 'is_last_block_with_action0'] = True
    
    def neg_sell_save_ending(df):
        # 检查 DataFrame 是否以 sell_save<=0 块结束，并标记为保留
        last_block_id = df[df['is_neg_sell_save']]['block_sell_save'].max()
        last_block_rows = df[(df['block_sell_save'] == last_block_id) & (df['is_neg_sell_save'])]
        if not last_block_rows.empty and last_block_rows.index[-1] == df.index[-1]:
            df.loc[last_block_rows.index, 'is_last_block_with_action0'] = True
        return df

    if df['is_neg_sell_save'].any():
        df = neg_sell_save_ending(df)

        am_cond = df['before_market_close_sec'] >= (np.float64(12600 / MAX_SEC_BEFORE_CLOSE))
        am_part = df.loc[am_cond]
        am_part = neg_sell_save_ending(am_part)
        df.loc[am_cond, 'is_last_block_with_action0'] = am_part['is_last_block_with_action0'].values

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

    # for file in tqdm(train_files):
    for file in train_files:
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

    # for file in tqdm(train_files):
    for file in train_files:
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
        log("Gradient norm histogram saved as 'gradient_norm_histogram.png'")
    plt.close()

    return total_grad_norm

def check_dependencies(model, input_tensor, test_idx):
    """
    测试一个模型的输出是否正确地只依赖于对应的输入。

    Args:
        model (nn.Module): 要测试的模型。
        input_tensor (torch.Tensor): 输入张量。
        test_idx (int): 我们要关注的批次中的样本索引。
    """
    print(f"--- Testing {model.__class__.__name__} ---")
    print(f"目标：仅让第 {test_idx} 个样本的输出产生损失。")

    # 1. 确保输入张量可以计算梯度
    input_tensor.requires_grad = True

    # 2. 清除旧的梯度
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()

    # 3. 前向传播
    output = model(input_tensor)

    # 4. 定义 "探针" 损失：只取第 test_idx 个样本的输出总和
    loss = output[test_idx].sum()
    print(f"损失函数: output[{test_idx}].sum()")

    # 5. 反向传播
    loss.backward()

    # 6. 检查输入的梯度
    input_grad = input_tensor.grad
    print("输入的梯度（按样本求和后的绝对值）:")
    # 为了方便观察，我们计算每个样本梯度的总和
    grad_sum_per_sample = input_grad.abs().sum(dim=1)
    print(grad_sum_per_sample)

    # 7. 分析结果
    is_bug_found = False
    for i in range(input_tensor.shape[0]):
        # 检查梯度是否为零
        is_zero = torch.all(input_grad[i] == 0)
        if i == test_idx:
            if is_zero:
                print(f"❌ 错误! 样本 {i} (目标样本) 的梯度为零，这不应该发生。")
                is_bug_found = True
            else:
                print(f"✅ 正确! 样本 {i} (目标样本) 的梯度非零。")
        else:
            if not is_zero:
                print(f"🚨🚨🚨 Bug 发现! 样本 {i} 的梯度非零，说明发生了信息泄露！")
                is_bug_found = True
            else:
                print(f"✅ 正确! 样本 {i} 的梯度为零。")
    
    if not is_bug_found:
        print("结论：依赖关系正确！\n")
    else:
        print("结论：依赖关系错误！\n")

def run_dependency_check_without_bn(model, test_input, test_idx):
    """
    一个封装好的函数，用于测试一个模型在禁用BatchNorm后的依赖关系。

    Args:
        model (nn.Module): 要测试的模型。
        test_input (torch.Tensor): 测试用的输入张量。
        test_idx (int): 目标样本的索引。
    """
    print("="*50)
    print("开始验证（已临时禁用 BatchNorm）...")
    print("="*50)

    # 2. 定义一个替换函数
    def replace_bn_with_identity(model):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                # 替换为恒等映射
                setattr(model, name, nn.Identity())
            else:
                # 递归处理子模块
                replace_bn_with_identity(module)
        return model
    
    # 3. 使用 .apply() 方法，将替换函数递归地应用到模型的所有子模块
    #    这会创建一个没有BatchNorm效应的新模型实例
    #    注意：这会原地修改模型，所以我们在原始模型的副本上操作
    import copy
    model_without_bn = copy.deepcopy(model)
    replace_bn_with_identity(model_without_bn)
    
    # 将模型设置为训练模式，以确保其他层（如Dropout）正常工作
    model_without_bn.train()
    
    print("模型中的 BatchNorm 层已被 nn.Identity 临时替换。")
    
    # 4. 使用你原来的 check_dependencies 函数来测试这个“净化”过的模型
    check_dependencies(model_without_bn, test_input, test_idx)

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


if __name__ == '__main__':
    print(find_not_stable_bid_ask(
        pd.DataFrame({
            'BASE卖1价': [
                7.132,
                7.132,
                7.132,
                7.131,
                7.13,
                7.129,
                7.13,
                7.129,
                7.131,
                7.133,
                7.134,
            ],
            'BASE买1价': [
                7.131,
                7.131,
                7.131,
                7.13,
                7.129,
                7.128,
                7.129,
                7.128,
                7.13,
                7.131,
                7.133,
            ]
        })

    ))