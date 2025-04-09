import psutil, pickle, torch, os, time
import numpy as np
from tqdm import tqdm
import platform
import time
import pandas as pd
import numpy as np
from pathlib import Path

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

def _identify_peaks_valleys(plateaus, rep_select):
    """
    根据平台期识别波峰和波谷。
    
    参数:
    plateaus (list): 平台期列表
    rep_select: 波峰波谷的选取方式, 'mid' 表示中间点, 'random' 表示随机点
    
    返回:
    tuple: (peaks, valleys)，分别为波峰和波谷的代表点列表
    """
    peaks = []
    valleys = []
    n = len(plateaus)
    
    for i in range(n):
        start, end, value = plateaus[i]

        if rep_select == 'mid':
            rep = (start + end) // 2  # 选择中间点作为代表
        elif rep_select == 'random':
            if start == end:
                rep = start
            else:
                rep = np.random.randint(start, end)

        if i == 0:  # 第一个平台期
            if n > 1 and value < plateaus[1][2]:
                valleys.append(rep)
            elif n > 1 and value > plateaus[1][2]:
                peaks.append(rep)
        elif i == n - 1:  # 最后一个平台期
            if value < plateaus[i - 1][2]:
                valleys.append(rep)
            elif value > plateaus[i - 1][2]:
                peaks.append(rep)
        else:  # 中间平台期
            prev_value = plateaus[i - 1][2]
            next_value = plateaus[i + 1][2]
            if value > prev_value and value > next_value:
                peaks.append(rep)
            elif value < prev_value and value < next_value:
                valleys.append(rep)
    return peaks, valleys 

def _find_max_profitable_trades(bid, ask, mid, valleys, peaks, fee=5e-5):
    """
    寻找所有可盈利的交易对，最大化对数收益率总和，考虑更低波谷的潜在更大利润。

    参数:
    bid (np.array): bid 价格序列
    ask (np.array): ask 价格序列
    valleys (list): 波谷位置列表
    peaks (list): 波峰位置列表
    fee (float): 交易费率，默认为 5e-5

    返回:
    trades (list): 所有可盈利的交易对列表，每个元素为 (valley, peak)
    total_log_return (float): 所有可盈利交易的对数收益率总和
    """
    trades = []
    total_log_return = 0.0
    valley_idx = 0
    peak_idx = 0

    pre_t2 = 0# 上一个波峰t
    while valley_idx < len(valleys) and peak_idx < len(peaks):
        t1 = valleys[valley_idx]
        t2 = peaks[peak_idx]

        if t1 >= t2:
            # 波谷在波峰之后，跳到下一个波峰
            peak_idx += 1
            continue

        if t1 < pre_t2:
            # 新的波谷肯定药在前一个波峰之后
            valley_idx += 1
            continue

        if mid[t2] < mid[t1]:
            # 波峰比波谷还低，测试下一个波谷
            valley_idx += 1
            continue

        # 当前波峰波谷中的其他波谷,
        # 选择一个最小值
        _valley_idx = valley_idx + 1
        while _valley_idx < len(valleys) and valleys[_valley_idx] < t2:
            # 时间药满足小于 t2
            _valley_t = valleys[_valley_idx]
            if ask[_valley_t] <= ask[t1]:
                t1 = _valley_t
                valley_idx = _valley_idx
            _valley_idx += 1

        # 若当前波谷ask >= 上一个波峰bid, 且 当前波峰bid >= 上一个波峰bid, 则修改上一个波峰为当前波峰
        # 上一个波峰卖出，当前波谷买入 平添交易费/损失持仓量 
        if pre_t2 > 0 and ask[t1] >= bid[pre_t2] and bid[t2] >= bid[pre_t2]:
            # print(f'前一个波峰卖出损失')
            trades[-1] = (trades[-1][0], t2)
            pre_t2 = t2
            peak_idx += 1
            valley_idx += 1
            continue

        # 计算当前交易的成本和收入
        buy_cost = ask[t1] * (1 + fee)
        sell_income = bid[t2] * (1 - fee)

        if sell_income > buy_cost:
            # 当前交易盈利，计算收益率
            current_return = np.log(sell_income / buy_cost)

            # 确认当前交易
            trades.append((t1, t2))
            total_log_return += current_return
            pre_t2 = t2
            valley_idx += 1
            peak_idx += 1
        else:
            # 不盈利
            # 1. 对比前一个交易对的波峰，若当前的波峰更高，则修改上一个交易对的波峰为当前波峰
            # 2. 若当前的波峰更低，则尝试下一个波峰
            if pre_t2 > 0 and bid[t2] > bid[pre_t2]:
                # print(f'当前波峰波谷不盈利，合并到上一个交易对')
                # 修改上一个交易对的波峰为当前波峰
                trades[-1] = (trades[-1][0], t2)
                pre_t2 = t2
                valley_idx += 1
                peak_idx += 1
            else:
                # 尝试下一个波峰
                peak_idx += 1

    return trades, total_log_return

def max_profit_reachable(bid, ask, rep_select='mid'):
    """
    计算bid/ask序列中 潜在的最大利润
    rep_select: 波峰波谷的选取方式, 'mid' 表示中间点, 'random' 表示随机点

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

    # 识别平台期
    plateaus = _find_plateaus(mid)

    # 识别波峰和波谷
    peaks, valleys = _identify_peaks_valleys(plateaus, rep_select)

    # 匹配交易对
    # 计算可盈利交易的对数收益率总和
    trades,total_log_return = _find_max_profitable_trades(bid, ask, mid, valleys, peaks)
    return trades,total_log_return, valleys, peaks

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


def plot_trades_plt(mid, trades, valleys, peaks):
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
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制 mid-price 曲线
    plt.plot(time, mid, label='Mid Price', color='blue')
    
    # 添加波谷点
    plt.scatter(valleys, [mid[v] for v in valleys], color='red', label='Valleys', s=50)
    
    # 添加波峰点
    plt.scatter(peaks, [mid[p] for p in peaks], color='green', label='Peaks', s=50)
    
    # 添加交易对的连接线
    for t1, t2 in trades:
        plt.plot([t1, t2], [mid[t1], mid[t2]], color='green', linestyle='--', label=f'Trade {t1}-{t2}')
    
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

def _check_nan(output):
    has_nan = torch.isnan(output).any(dim=-1)  # 检查是否有 NaN
    has_inf = torch.isinf(output).any(dim=-1)  # 检查是否有 inf

    # 找出包含 NaN 或 inf 值的批次索引
    batch_indices = torch.nonzero(has_nan | has_inf, as_tuple=True)[0]
    return batch_indices

def check_nan(output, ids):
    # 找出包含 NaN 或 inf 值的批次索引
    batch_indices = _check_nan(output)
    if batch_indices.numel() > 0:
        bad_ids = []
        for i in list(batch_indices):
            bad_ids.append(ids[i])
        msg = f'训练数据异常 nan/inf ids:{bad_ids}'
        wx.send_message(msg)
        raise Exception(msg)

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


