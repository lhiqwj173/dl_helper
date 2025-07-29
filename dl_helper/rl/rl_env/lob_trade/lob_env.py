import cProfile
import os, time
import random
import datetime
import numpy as np
import pandas as pd
import gymnasium as gym
import gymnasium.spaces as spaces
import pickle
from collections import deque
from multiprocessing import Process, Queue
import queue
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pytz
from matplotlib.widgets import Button

from dl_helper.tool import in_windows
if in_windows():
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

from py_ext.tool import log, init_logger,get_exception_msg, get_log_file, share_tensor
from py_ext.datetime import beijing_time
from py_ext.wechat import send_wx

from dl_helper.tool import calc_sharpe_ratio, calc_sortino_ratio, calc_drawdown, calc_return, calc_drawup_ticks, max_profit_reachable
from dl_helper.train_param import in_kaggle

from dl_helper.rl.rl_utils import date2days, days2date

from dl_helper.rl.rl_env.lob_trade.lob_env_data_augmentation import random_his_window, gaussian_noise_vol
from dl_helper.rl.rl_env.lob_trade.lob_env_reward import ClosePositionRewardStrategy, HoldPositionRewardStrategy, NoPositionRewardStrategy, BlankRewardStrategy, ForceStopRewardStrategy, RewardCalculator

from dl_helper.rl.rl_env.lob_trade.lob_const import USE_CODES, STD_REWARD, FINAL_REWARD
from dl_helper.rl.rl_env.lob_trade.lob_const import MEAN_SEC_BEFORE_CLOSE, STD_SEC_BEFORE_CLOSE, MAX_SEC_BEFORE_CLOSE
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL
from dl_helper.rl.rl_env.lob_trade.lob_const import RESULT_OPEN, RESULT_CLOSE, RESULT_HOLD
from dl_helper.rl.rl_env.lob_trade.lob_const import LOCAL_DATA_FOLDER, KAGGLE_DATA_FOLDER, DATA_FOLDER

from dl_helper.rl.rl_env.lob_trade.lob_data_helper import fix_raw_data

debug_bid_ask_accnet_code = r"""from dl_helper.tool import max_profit_reachable, plot_trades_plt
import pandas as pd
import numpy as np
import pickle 

bid, ask, accnet = pickle.load(open(r"C:\Users\lh\Desktop\temp\bid_ask_accnet.pkl", 'rb'))
mid_price = (pd.Series(ask)+ pd.Series(bid)) / 2

# 计算潜在收益
trades, total_log_return, valleys, peaks = max_profit_reachable(bid, ask)
print(f'total_log_return:', total_log_return)
print(f'acc_log_ret:', np.log(accnet[-1]) - np.log(accnet[0]))
plot_trades_plt(mid_price, trades, valleys, peaks)"""

class data_producer:
    """
    遍历日期文件，每天随机选择一个标的
    当天的数据读取完毕后，需要强制平仓

    特征列:
        [
            'OF买1量', 'OF卖1量', 'OF买2量', 'OF卖2量', 'OF买3量', 'OF卖3量', 'OF买4量', 'OF卖4量', 'OF买5量', 'OF卖5量', 'OF买6量', 'OF卖6量', 'OF买7量', 'OF卖7量', 'OF买8量', 'OF卖8量', 'OF买9量', 'OF卖9量', 'OF买10量', 'OF卖10量', 
            'BASE卖1价', 'BASE卖1量', 'BASE买1价', 'BASE买1量', 'BASE卖2价', 'BASE卖2量', 'BASE买2价', 'BASE买2量', 'BASE卖3价', 'BASE卖3量', 'BASE买3价', 'BASE买3量', 'BASE卖4价', 'BASE卖4量', 'BASE买4价', 'BASE买4量', 'BASE卖5价', 'BASE卖5量', 'BASE买5价', 'BASE买5量', 'BASE卖6价', 'BASE卖6量', 'BASE买6价', 'BASE买6量', 'BASE卖7价', 'BASE卖7量', 'BASE买7价', 'BASE买7量', 'BASE卖8价', 'BASE卖8量', 'BASE买8价', 'BASE买8量', 'BASE卖9价', 'BASE卖9量', 'BASE买9价', 'BASE买9量', 'BASE卖10价', 'BASE卖10量', 'BASE买10价', 'BASE买10量', 
            'OB卖5量', 'OB卖4量', 'OB卖3量', 'OB卖2量', 'OB卖1量', 'OB买1量', 'OB买2量', 'OB买3量', 'OB买4量', 'OB买5量', 'OB买6量', 'OB买7量', 'OB买8量', 'OB买9量', 'OB买10量', 
            'OS卖10量', 'OS卖9量', 'OS卖8量', 'OS卖7量', 'OS卖6量', 'OS卖5量', 'OS卖4量', 'OS卖3量', 'OS卖2量', 'OS卖1量', 'OS买1量', 'OS买2量', 'OS买3量', 'OS买4量', 'OS买5量', 
            'OBC买1量', 'OBC买2量', 'OBC买3量', 'OBC买4量', 'OBC买5量', 'OBC买6量', 'OBC买7量', 'OBC买8量', 'OBC买9量', 'OBC买10量', 
            'OSC卖10量', 'OSC卖9量', 'OSC卖8量', 'OSC卖7量', 'OSC卖6量', 'OSC卖5量', 'OSC卖4量', 'OSC卖3量', 'OSC卖2量', 'OSC卖1量', 
            'DB卖5量', 'DB卖4量', 'DB卖3量', 'DB卖2量', 'DB卖1量', 'DB买1量', 'DB买2量', 'DB买3量', 'DB买4量', 'DB买5量', 
            'DS卖5量', 'DS卖4量', 'DS卖3量', 'DS卖2量', 'DS卖1量', 'DS买1量', 'DS买2量', 'DS买3量', 'DS买4量', 'DS买5量'
        ]

    """
    def __init__(
            self,
            data_type='train', 
            his_len=100, 
            simple_test=False, 
            need_cols=[], 
            use_symbols=[], 
            data_std=True, 
            save_folder="", 
            debug_date=[],
            debug_time=None,
            random_begin_in_day=True,
            latest_dates=-1,
            data_folder=DATA_FOLDER,

            # 数据增强
            use_random_his_window=True,
            use_gaussian_noise_vol=True,
            use_spread_add_small_limit_order=True,
        ):
        """
        'data_type': 'train',# 训练/测试
        'his_len': 100,# 每个样本的 历史数据长度
        'simple_test': False,# 是否为简单测试
        'need_cols': [],# 需要的特征列名
        'use_symbols': []# 只使用指定的标的
        'random_begin_in_day': True,# 是否在日内随机开始
        'latest_dates': -1,# 使用最近日期的数据，-1 表示使用所有数据

        'use_random_his_window': True,# 随机窗口开关
        'use_gaussian_noise_vol': True,# 高斯噪声开关
        'use_spread_add_small_limit_order': True,# 价差中添加小单
        """
        # 数据增强
        self.use_random_his_window = use_random_his_window  # 随机窗口开关
        self.use_gaussian_noise_vol = use_gaussian_noise_vol  # 高斯噪声开关
        self.use_spread_add_small_limit_order = use_spread_add_small_limit_order  # 价差中添加小单

        # 随机数
        self.np_random = np.random.default_rng()

        # 快速测试
        self.simple_test = simple_test

        self.his_len = his_len
        self.data_std = data_std
        self.save_folder = save_folder
        self.debug_date = [i.replace('-', '').replace(' ', '') for i in debug_date]
        if self.debug_date:
            log(f'[{data_type}] debug_date: {self.debug_date}')
        self.debug_time = debug_time
        if self.debug_time:
            # 使用指定的时间开始
            log(f'[{data_type}] debug_time: {self.debug_time}')

        self.use_symbols = use_symbols
        self.random_begin_in_day = random_begin_in_day
        self.latest_dates = latest_dates

        # 需要的特征列名
        self.need_cols = need_cols
        self.need_cols_idx = []

        self.cols_num = 130 if not self.need_cols else len(self.need_cols)

        # 训练数据
        self.data_folder = data_folder

        self.data_type = data_type
        self.cur_data_type = data_type
        self.files = []
        self.cur_data_file = ''

        # 当前数据日期/code
        self.date = ''
        self.code = ''
        
        # 数据内容
        # ids, mean_std, x, all_self.all_raw_data_data
        self.ids = []
        self.mean_std = []
        self.x = []
        self.all_raw_data = None
        # 距离市场关闭的秒数
        self.before_market_close_sec = []

        # 数据索引
        self.idxs = []

        # 最近一个数据的索引
        self.last_data_idx = None

        # 记录价格用于成交/计算潜在收益
        self.bid_price = []
        self.ask_price = []

        # 数据列字段对于的索引
        self.col_idx = {}

        # id
        self.id = ''

        log(f'[{self.data_type}] data_producer init done')

    def pre_plot_data(self):
        """
        预先读取绘图数据
        col_idx: BASE买1价, BASE卖1价, BASE中间价
        """
        # 直接选择需要的列并创建DataFrame
        cols = ['BASE买1价', 'BASE卖1价']
        self.plot_data = self.all_raw_data.iloc[:, [self.col_idx[col] for col in cols]].copy()
        self.plot_data.columns = ['bid', 'ask']
        # 高效计算中间价格
        self.plot_data['mid_price'] = self.plot_data.mean(axis=1)

    def _get_data_type_files(self):
        """
        获取数据类型对应的文件列表(路径)
        """
        assert self.debug_date or (self.data_type in ['train', 'val']), f'非指定日期数据，暂时不支持 {self.data_type} 数据类型'
        if self.data_type == 'test':
            return [os.path.join(self.data_folder, self.data_type, i) for i in os.listdir(os.path.join(self.data_folder, self.data_type))]
        else:
            if not hasattr(self, 'train_files'):
                # train/val 数据
                files = []
                for root, dirs, _files in os.walk(self.data_folder):
                    for _file in _files:
                        if _file.endswith('.pkl'):
                            files.append(os.path.join(root, _file))

                # 按文件名排序
                files.sort(key=lambda x: os.path.basename(x))

                # 只使用最近 latest_dates 个数据
                if self.latest_dates != -1:
                    files = files[-self.latest_dates:]
            
                # # 随机抽取 30 个文件作为val
                # # 使用固定的随机种子， 确保一致
                # rng = np.random.default_rng(0)
                # self.val_files = rng.choice(files, 30, replace=False)
                # self.train_files = [i for i in files if i not in self.val_files]

                # 取最后20个文件作为val
                self.val_files = files[-20:]

                # 其余文件作为train
                self.train_files = files[:-20]

            if self.debug_date:
                # 返回全部的 files
                return self.train_files + self.val_files

            elif self.data_type == 'train':
                return [i for i in self.train_files]
            elif self.data_type == 'val':
                return [i for i in self.val_files]

    def _pre_files(self):
        """
        准备文件列表
        若是训练数据，随机读取
        若是验证/测试数据，按顺序读取
        """
        # 当天的数据用完，且没有其他日期数据可以load，
        # 或 数据类型发生变化，需要重新准备数据
        if (not self.files) or (self.cur_data_type != self.data_type):
            # 若 文件列表为空，重新准备
            self.files = self._get_data_type_files()
            # log(f'[{self.data_type}] all files: {[os.path.basename(i) for i in self.files]}')

            if self.data_type == 'train':
                self.np_random.shuffle(self.files)

            if self.debug_date:
                # 按照 debug_date 的顺序重新排列文件
                ordered_files = []
                for debug_date in self.debug_date:
                    for file in self.files:
                        if file.split('.')[0][-8:] == debug_date:
                            ordered_files.append(file)
                self.files = ordered_files
            log(f'[{self.data_type}] prepare files: {[os.path.basename(i) for i in self.files]}')
        
        assert self.files, f'[{self.data_type}] no datas'
            
    def _load_data(self):
        """
        按照文件列表读取数据
        每次完成后从文件列表中剔除
        """
        if (not self.idxs) or (self.cur_data_type != self.data_type):
            while self.files:
                # 更新在用数据类型
                self.cur_data_type = self.data_type

                _cur_data_file = self.files.pop(0)

                # 路径中只取文件名
                self.cur_data_file = os.path.basename(_cur_data_file)

                log(f'[{self.data_type}] load date file: {self.cur_data_file}')
                self.ids, self.mean_std, self.x, self.all_raw_data = pickle.load(open(_cur_data_file, 'rb'))
                # 转换数据类型为float32
                for col in self.all_raw_data.iloc[:, :-3].columns:
                    self.all_raw_data[col] = self.all_raw_data[col].astype('float32')
                self.all_raw_data = self.all_raw_data.set_index('时间')

                # 列过滤
                if self.need_cols:
                    self.need_cols_idx = [self.all_raw_data.columns.get_loc(col) for col in self.need_cols]
                    # 只保留需要的列
                    self.all_raw_data = self.all_raw_data.loc[:, self.need_cols]

                if self.simple_test:
                    self.ids = self.ids[:8000]
                    self.mean_std = self.mean_std[:8000]
                    self.x = self.x[:8000]

                # 距离市场关闭的秒数
                self.date = self.cur_data_file[:8]
                dt = datetime.datetime.strptime(f'{self.date} 15:00:00', '%Y%m%d %H:%M:%S')
                dt = pytz.timezone('Asia/Shanghai').localize(dt)
                close_ts = int(dt.timestamp())
                self.before_market_close_sec = np.array([int(i.split('_')[1]) for i in self.ids])
                self.before_market_close_sec = close_ts - self.before_market_close_sec

                # 解析标的 随机挑选一个标的数据
                symbols = np.array([i.split('_')[0] for i in self.ids])
                unique_symbols = np.unique(symbols)

                if self.use_symbols:
                    unique_symbols = [i for i in unique_symbols if i in self.use_symbols]
                
                if not unique_symbols:
                    log(f'[{self.data_type}] no data for date: {self.date}' + '' if not self.use_symbols else f', symbols: {self.use_symbols}')
                    continue

                # 获取所有标的的起止索引
                self.full_idxs = []# begin 没有被截断的索引
                self.idxs = []
                for symbol in unique_symbols:
                    symbol_mask = symbols == symbol
                    symbol_indices = np.where(symbol_mask)[0]
                    begin = symbol_indices[0]
                    end = symbol_indices[-1]
                    self.full_idxs.append([begin, end, USE_CODES.index(symbol)])
                    if not self.debug_time and (self.random_begin_in_day and self.data_type == 'train'):
                        if hasattr(self, 'begin_before_market_close_sec'):
                            _idx = begin
                            while _idx <= end:
                                if self.before_market_close_sec[_idx] == int(self.begin_before_market_close_sec):
                                    break
                                _idx += 1
                            assert _idx != end + 1, f'{int(self.begin_before_market_close_sec)} not found'
                            begin = _idx
                        else:
                            # begin = random.randint(begin, end-50)# 至少50个数据
                            begin = self.np_random.integers(begin, end-50+1)# 至少50个数据
                    self.idxs.append([begin, end, USE_CODES.index(symbol)])

                if not self.idxs:
                    log(f'[{self.data_type}] no data for date: {self.date}' + '' if not self.use_symbols else f', symbols: {self.use_symbols}')
                    continue

                # 训练数据随机选择一个标的
                # 一个日期文件只使用其中的一个标的的数据，避免同一天各个标的之间存在的相关性 对 训练产生影响
                if self.data_type == 'train' and not self.use_symbols:
                    # self.idxs = [random.choice(self.idxs)]
                    choose_idx = self.np_random.choice(len(self.idxs))
                    self.idxs = [self.idxs[choose_idx]]
                    self.full_idxs = [self.full_idxs[choose_idx]]

                # 截取测试的时间点
                if self.debug_time:
                    for idx_obj in self.idxs:
                        begin, end, _ = idx_obj
                        _idx = begin
                        while _idx <= end:
                            if self.before_market_close_sec[_idx] <= int(self.debug_time):
                                break
                            _idx += 1
                        assert _idx != end + 1, f'{int(self.debug_time)} not found'
                        begin = _idx
                        idx_obj[0] = begin

                # 当前的标的
                self.cur_symbol = USE_CODES[self.idxs[0][2]]

                log(f'[{self.data_type}] init idxs: {self.idxs}')

                # 调整数据
                self.all_raw_data = fix_raw_data(self.all_raw_data) 

                # 记录需要的索引，供后续转为numpy时使用
                # BASE买1价 / BASE卖1价
                for col in ['BASE买1价', 'BASE卖1价']:
                    self.col_idx[col] = self.all_raw_data.columns.get_loc(col)

                # 准备绘图数据
                self.pre_plot_data()

                # # 测试用
                # pickle.dump((self.all_raw_data, self.mean_std, self.x), open(f'{self.data_type}_raw_data.pkl', 'wb'))
                break

    def set_data_type(self, data_type):
        self.data_type = data_type

    def data_size(self):
        # 运行只获取部分列， 简化数据
        return self.cols_num*self.his_len

    def use_data_split(self, raw, ms):
        """
        使用数据分割
        raw 是完整的 pickle 切片
        ms 是标准化数据df
        都是 numpy 数组
        TODO 运行只获取部分列， 简化数据
        """
        if self.need_cols:
            return raw, ms.iloc[self.need_cols_idx, :].values
        else:
            return raw[:, :130], ms.iloc[:130, :].values

    def store_bid_ask_1st_data(self, raw):
        """
        存储买卖1档数据 用于撮合交易
        raw 是完整的 pickle 切片
        """
        last_row = raw[-1]  # 最后一个数据
        self.bid_price.append(last_row[self.col_idx['BASE买1价']])
        self.ask_price.append(last_row[self.col_idx['BASE卖1价']])

    def get_ask_bid(self):
        """
        获取 ask/bid
        """
        a, b = self.x[self.last_data_idx]
        ask = self.all_raw_data.iloc[b-1, self.col_idx['BASE卖1价']]
        bid = self.all_raw_data.iloc[b-1, self.col_idx['BASE买1价']]

        assert ask == self.ask_price[-1], f'ask: {ask} != {self.ask_price[-1]}'
        assert bid == self.bid_price[-1], f'bid: {bid} != {self.bid_price[-1]}'
        return ask, bid

    def get(self):
        """
        输出观察值
            返回 symbol_id, before_market_close_sec, x, need_close, self.id

            若data_std=False
            返回 symbol_id, before_market_close_sec, x, need_close, _id, x_std, sec_std
        """
        # # 测试用
        # print(self.idxs[0])

        # 记录数据索引
        self.last_data_idx = self.idxs[0][0]

        # 准备观察值
        a, b = self.x[self.idxs[0][0]]
        self.step_use_data = self.all_raw_data.iloc[max(b - int(self.his_len * 1.5), a): b, :]# 多截取 50% 数据，用于增强
        raw_0 = self.step_use_data.values# 多截取 50% 数据

        # 原始数据增强
        # 1. use_random_his_window 从更长的历史窗口（如15个）随机截取 所需的时间窗口的数据
        if_modified = False
        if self.use_random_his_window and self.data_type == 'train':
            if_modified, raw_0 = random_his_window(raw_0, self.his_len, rng = self.np_random)

        # 修正历史数据长度
        if if_modified:
            # 无需复制，返回的是新对象
            raw = raw_0[-self.his_len:]
        else:
            # 如果未修改，需要复制一份
            raw = raw_0[-self.his_len:].copy()

        # 记录 买卖1档 的价格
        self.store_bid_ask_1st_data(raw)

        # 截断数据后的增强
        # 1. use_gaussian_noise_vol 添加高斯噪声，在成交量上添加符合正态分布的小幅随机噪声，模拟市场中的微小波动(50以内)。
        # 判断是否有 _gaussian_noise_vol_col_idxs 属性
        if self.use_gaussian_noise_vol and self.data_type == 'train':
            if not hasattr(self, '_gaussian_noise_vol_col_idxs'):
                vol_cols = [i for i in list(self.all_raw_data) if i.startswith('BASE') and '价' not in i]
                self._gaussian_noise_vol_col_idxs = [self.all_raw_data.columns.get_loc(col) for col in vol_cols]
            # 噪音控制在 -50 到 50 之间
            raw[:, self._gaussian_noise_vol_col_idxs] += gaussian_noise_vol(raw[:, self._gaussian_noise_vol_col_idxs].shape, rng = self.np_random)
            # 控制最小值为 1    
            raw[:, self._gaussian_noise_vol_col_idxs] = np.clip(raw[:, self._gaussian_noise_vol_col_idxs], 1, None)

        # 2. use_spread_add_small_limit_order 在价差（若可以）中添加小单（5以内）
        # TODO

        # 数据标准化
        std_data = self.mean_std[self.idxs[0][0]]
        # 未实现收益率 使用 zscore
        if 'unrealized_log_return' not in std_data:
            unrealized_log_return_std_data = None
        else:   
            unrealized_log_return_std_data = std_data['unrealized_log_return']['zscore']

        ###################################
        # 价格量 使用 robust
        ms = pd.DataFrame(std_data['all_std']['all'], dtype=np.float32)
        x, ms = self.use_data_split(raw, ms)

        if self.data_std:
            x -= ms[:, 0]
            x /= ms[:, 1]
        else:
            x_std = ms.copy()
        ###################################

        # ####################################
        # # (价格 - mid_price) / spread
        # # 数量 / total_vol/total_bid_vol/total_ask_vol(基于最近tick计算)
        # mid_price = (self.bid_price[-1] + self.ask_price[-1]) / 2
        # spread = self.ask_price[-1] - self.bid_price[-1]
        # if not hasattr(self, 'p_cols_idxs'):
        #     self.p_cols_idxs = [i for i in range(len(self.all_raw_data.columns)) if '价' in self.all_raw_data.columns[i]]
        #     self.v_cols_idxs = [i for i in range(len(self.all_raw_data.columns)) if '量' in self.all_raw_data.columns[i]]
        # x = raw
        # total_vol = x[-1, self.v_cols_idxs].sum()
        # if self.data_std:
        #     x[:, self.p_cols_idxs] = (x[:, self.p_cols_idxs] - mid_price) / spread
        #     x[:, self.v_cols_idxs] = x[:, self.v_cols_idxs] / (total_vol + 1e-6)
        # else:
        #     x_std = (mid_price, spread, total_vol)
        # ####################################

        # 标的id
        symbol_id = self.idxs[0][2]

        # 当前标的
        self.code = USE_CODES[int(symbol_id)]

        # 距离市场关闭的秒数
        before_market_close_sec = self.before_market_close_sec[self.idxs[0][0]]

        # 记录数据id
        self.id = self.ids[self.idxs[0][0]]

        # 检查本次数据是否是最后一个数据
        self.need_close = False
        if self.idxs[0][0] == self.idxs[0][1]:
            # 当组 begin/end 完成，需要平仓
            self.need_close = True
            log(f'[{self.data_type}] need_close {self.idxs[0][0]} {self.idxs[0][1]}')
            # 更新剩余的 begin/end 组
            self.idxs = self.idxs[1:]
            self.full_idxs = self.full_idxs[1:]
            log(f'[{self.data_type}] idxs: {self.idxs}')
            if not self.idxs:
                # 当天的数据没有下一个可读取的 begin/end 组
                log(f'[{self.data_type}] date file done')
            else:
                # 更新当前的标的
                self.cur_symbol = USE_CODES[self.idxs[0][2]]

                # 准备绘图数据
                self.pre_plot_data()
        else:
            self.idxs[0][0] += 1

        if self.data_std:
            # 额外数据的标准化
            # 距离收盘秒数
            # ZSCORE
            # before_market_close_sec -= MEAN_SEC_BEFORE_CLOSE
            # before_market_close_sec /= STD_SEC_BEFORE_CLOSE
            # 归一化
            before_market_close_sec /= MAX_SEC_BEFORE_CLOSE

            # id 不需要标准化, 模型中嵌入
            # 0 - 29 共30个
            return symbol_id, before_market_close_sec, x, self.need_close, self.id, unrealized_log_return_std_data
        else:
            sec_std = (MEAN_SEC_BEFORE_CLOSE, STD_SEC_BEFORE_CLOSE, MAX_SEC_BEFORE_CLOSE)
            return symbol_id, before_market_close_sec, x, self.need_close, self.id, unrealized_log_return_std_data, x_std, sec_std

    def get_plot_data(self):
        """
        获取绘图数据,
        返回 绘图范围 a, b, 当前时点的时间索引
        """
        a, b = self.x[self.last_data_idx]
        a = max(b - self.his_len, a)

        extra_data_length = 600
        extra_a, extra_b = a - extra_data_length, b + extra_data_length

        return max(0, extra_a), min(len(self.plot_data), extra_b), self.plot_data.iloc[b-1].name

    def reset(self, rng=None):
        if rng is not None:
            self.np_random = rng

        self._pre_files()
        self._load_data()
        if not self.idxs:
            # 没有成功加载到可用数据，需要重新准备文件
            # 当天数据中没有指定标的
            log(f'[{self.data_type}] no data for date: {self.date}' + ('' if not self.use_symbols else f', symbols: {self.use_symbols}, will repre files'))
            return False

        self.last_data_idx = None
        self.bid_price = []
        self.ask_price = []
        return True

class Account:
    """
    账户类，用于记录交易状态和计算收益
    """
    fee_rate = 5e-5

    def __init__(self):
        # 持仓状态 
        self.status = 0
        self.init_status = 0
        # 持仓量
        self.pos = 0
        self.cash = 0
        # 持仓时间
        self.hold_length = 0
        # 净值序列
        self.net_raw = []
        self.net_raw_bm = []# 基准，一直持有

    def step(self, bid_price, ask_price, action, last_close_idx):
        """
        执行交易
        :param bid_price: 最优买价, 用于结算
        :param ask_price: 最优卖价, 用于结算 
        :param action: 0-空仓 1-持仓

        :return: (持仓状态, 日内对数收益率, 未实现对数收益率, 本次操作平仓的净值序列, 动作结果(0-平仓 1-开仓 -1-无))
        """
        # nan 检查
        assert not np.isnan(bid_price), f'bid_price is nan, {bid_price}'
        assert not np.isnan(ask_price), f'ask_price is nan, {ask_price}'

        # 检查
        if self.status:
            # 持仓状态
            assert self.cash == 0 and self.pos>0, f'status:{self.status} cash:{self.cash} pos:{self.pos}'
        else:
            # 空仓状态
            assert self.pos == 0 and self.cash >0, f'status:{self.status} cash:{self.cash} pos:{self.pos}'

        # 先处理动作
        act_result = RESULT_HOLD # 无
        if action == ACTION_BUY:  # 持仓动作
            if self.status == 0:  # 空仓可以买入
                self.status = 1

                # 更新 pos/cash
                self.pos = self.cash / (ask_price * (1 + Account.fee_rate))
                self.cash = 0

                # 动作结果:开仓
                act_result = RESULT_OPEN

        elif action == ACTION_SELL:  # 空仓动作
            if self.status == 1:  # 有多仓可以卖出
                self.status = 0

                # 更新 pos/cash
                self.cash = self.pos * bid_price * (1 - Account.fee_rate)
                self.pos = 0

                # 动作结果:平仓
                act_result = RESULT_CLOSE

        # 持仓时间
        if self.status == 1:
            self.hold_length += 1

        # 基于最新数据 更新净值
        # 转为现金的价值
        net_bm = bid_price * (1 - Account.fee_rate)
        self.net_raw_bm.append(net_bm)
        net = self.cash + self.pos * bid_price * (1 - Account.fee_rate)
        self.net_raw.append(net)

        # 计算持仓对数收益率
        # 上一次平仓后至今的对数收益率
        unrealized_return = 0
        if self.status == 1:
            latest_trade_net = self.net_raw[last_close_idx:]
            unrealized_return = np.log(latest_trade_net[-1] / latest_trade_net[0])

        # 日内收益率
        inday_return = np.log(self.net_raw[-1] / self.net_raw[0])

        # print("acc::step", action, self.status, self.pos, self.cash)
        return self.status, inday_return, unrealized_return, act_result

    def get_plot_data(self):
        """
        获取绘图数据

        net_raw, status
        """
        return self.net_raw[-1], self.status

    @staticmethod
    def cal_res(net_raw, net_raw_bm, last_close_idx, need_cal_drawup=True):
        # 评价指标
        res = {
            'max_drawdown': np.nan,
            'max_drawdown_ticks': np.nan,
            'trade_return': np.nan,
            'step_return': np.nan,
            'max_drawdown_bm': np.nan,
            'max_drawdown_ticks_bm': np.nan,
            'max_drawup_ticks_bm': np.nan,
            'drawup_ticks_bm_count': np.nan,
            'trade_return_bm': np.nan,
            'step_return_bm': np.nan,
        }

        # 数据足够 > 1
        # 需要计算评价指标， 储存在info中
        if (len(net_raw) > 1):
            # 计算策略净值的评价指标
            net = np.array(net_raw)
            # 计算对数收益率序列
            log_returns = np.diff(np.log(net))
            # 计算指标
            res['max_drawdown'], res['max_drawdown_ticks'] = calc_drawdown(net)
            res['trade_return'] = calc_return(log_returns, annualize=False)
            res['step_return'] = log_returns[-1]
            # log(f'step_return: {res["step_return"]}')

            # 计算基准净值的评价指标
            net_bm = np.array(net_raw_bm)
            # 计算对数收益率序列
            log_returns_bm = np.diff(np.log(net_bm))
            res['max_drawdown_bm'], res['max_drawdown_ticks_bm'] = calc_drawdown(net_bm)
            if need_cal_drawup:
                res['max_drawup_ticks_bm'], res['drawup_ticks_bm_count'] = calc_drawup_ticks(net_bm[last_close_idx:])# 针对上一次平仓到当前的数据
            else:
                res['max_drawup_ticks_bm'], res['drawup_ticks_bm_count'] = 0, 0
            res['trade_return_bm'] = calc_return(log_returns_bm, annualize=False)
            res['step_return_bm'] = log_returns_bm[-1]

        return res
        
    def reset(self, bid_price, status=None, rng=None):
        """
        重置账户状态
        """
        if status is None:
            # 随机持仓
            if rng is None:
                self.status = random.randint(0, 1)
            else:
                self.status = rng.integers(0, 2)
        else:
            # 指定持仓
            self.status = status

        self.init_status = self.status

        # self.status = 1# FOR DEBUG
        self.pos = 0
        self.cash = 0
        self.hold_length = 0
        self.net_raw = []
        self.net_raw_bm = []

        # 初始化净值 / cash
        net_bm = bid_price * (1 - Account.fee_rate)
        self.net_raw_bm.append(net_bm)
        self.net_raw.append(net_bm)

        if self.status == 0:
            self.cash = net_bm
        else:
            self.pos = 1

        return self.status

class RewardTracker:
    def __init__(self):
        # 初始化奖励列表和连续负奖励计数器
        self.rewards = []  # 存储所有奖励
        self.consecutive_negative = 0  # 当前连续负奖励次数
        
    def add_reward(self, reward):
        """
        添加新的奖励值并更新统计信息
        :param reward: float or int, 当前的奖励值
        :return: tuple (连续负奖励次数, 平均奖励)
        """
        # 添加奖励到列表
        self.rewards.append(reward)
        
        # 更新连续负奖励计数
        if reward < 0:
            self.consecutive_negative += 1
        else:
            self.consecutive_negative = 0
            
        # 计算平均奖励
        if len(self.rewards) > 0:
            avg_reward = sum(self.rewards) / len(self.rewards)
        else:
            avg_reward = 0.0
            
        return (self.consecutive_negative, avg_reward)

    def reset(self):
        """
        重置统计信息
        """
        self.rewards = []
        self.consecutive_negative = 0
        
class Render:
    def __init__(self, human_play=False):
        self.human_play = human_play
        self.app = None  # QApplication 实例
        self.win = None  # GraphicsLayoutWidget 实例

        # 中间数据
        self.std_data = {}
        self.full_plot_data = None
        self.data_deque = {}
        self.potential_data = None
        self.open_idx = None
        self.pre_n = None
        self.forbiden_begin_idx = None
        self.forbiden_end_idx = None

        # 绘图对象（延迟初始化）
        self.p1 = None
        self.p2 = None
        self.p1_mid_price = None
        self.p1_net_raw = None
        self.p1_mid_price_future = None
        self.p2_rewards = None
        self.p1_vline = None
        self.p2_vline = None
        self.p1_text = None
        self.p1_valley_points = None
        self.p1_peak_points = None
        self.p2_scatter = None
        self.p1_arrows_buy = []
        self.p1_arrows_sell = []

        # 前一个绘图点的时间
        self.pre_latest_tick_time = None
        # 是否需要fix net_raw / rewards
        self.net_data_fixed = False

        # 按钮相关变量
        self.button1 = None
        self.button2 = None
        self.clicked_button = None

        # 是否需要继续运行
        self.keep_play = None

        self._init_plot()
        log('Render 初始化完成')

    def _init_plot(self):
        """初始化绘图环境"""
        # 创建或重用 QApplication
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication([])
            log('创建新的 QApplication')
        else:
            self.app = QtWidgets.QApplication.instance()
            log('重用现有的 QApplication')

        # 创建主窗口 QWidget
        self.main_widget = QtWidgets.QWidget()
        self.main_widget.setWindowTitle('LOB Trade Environment')
        self.main_widget.resize(1000, 660)

        # 创建主布局
        main_layout = QtWidgets.QVBoxLayout(self.main_widget)

        # 创建 GraphicsLayoutWidget
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win)

        # 初始化绘图对象
        log('创建绘图对象')
        self.p1_mid_price = pg.PlotCurveItem(pen=pg.mkPen('b', width=2))
        self.p1_net_raw = pg.PlotCurveItem(pen=pg.mkPen('r', width=2))
        self.p1_mid_price_future = pg.PlotCurveItem(pen=pg.mkPen(color=(0, 0, 255, 128), width=2, style=QtCore.Qt.DashLine))
        self.p2_rewards = pg.PlotCurveItem(pen=pg.mkPen('g', width=2))

        try:
            self.p1_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
        except Exception as e:
            log(f'创建 p1_vline 失败: {e}')
            self.p1_vline = None

        try:
            self.p1_vline_acc_begin = pg.InfiniteLine(angle=90, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
        except Exception as e:
            log(f'创建 p1_vline_acc_begin 失败: {e}')
            self.p1_vline_acc_begin = None

        try:
            self.p2_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
        except Exception as e:
            log(f'创建 p2_vline 失败: {e}')
            self.p2_vline = None

        try:
            self.p2_vline_acc_begin = pg.InfiniteLine(angle=90, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
        except Exception as e:
            log(f'创建 p2_vline_acc_begin 失败: {e}')
            self.p2_vline_acc_begin = None

        try:
            self.p1_text = pg.TextItem(anchor=(0, 1))
        except Exception as e:
            log(f'创建 p1_text 失败: {e}')
            self.p1_text = None

        try:
            self.p1_text_acc_begin = pg.TextItem(anchor=(0, 1))
        except Exception as e:
            log(f'创建 p1_text_acc_begin 失败: {e}')
            self.p1_text_acc_begin = None

        self.p1_valley_points = pg.ScatterPlotItem(symbol='o', pen=pg.mkPen(color=(255, 0, 0, 128), width=2), brush=None, size=10)
        self.p1_peak_points = pg.ScatterPlotItem(symbol='o', pen=pg.mkPen(color=(0, 255, 0, 128), width=2), brush=None, size=10)
        self.p2_scatter = pg.ScatterPlotItem(symbol='o', pen=pg.mkPen(None), brush=pg.mkBrush('g'), size=10)

        self.p1_arrows_buy = [pg.ArrowItem(angle=90, tipAngle=60, baseAngle=0, headLen=15, tailLen=0, tailWidth=2, pen=None, brush='r') for _ in range(100)]
        self.p1_arrows_sell = [pg.ArrowItem(angle=-90, tipAngle=60, baseAngle=0, headLen=15, tailLen=0, tailWidth=2, pen=None, brush='g') for _ in range(100)]

        # 持仓背景
        self.p1_region = pg.LinearRegionItem(
            brush=pg.mkBrush(color=(173, 216, 230, 50)),
            movable=False
        )
        self.p2_region = pg.LinearRegionItem(
            brush=pg.mkBrush(color=(173, 216, 230, 50)),
            movable=False
        )

        # 下午开盘的禁止时间背景
        self.p1_region_forbiden = pg.LinearRegionItem(
            brush=pg.mkBrush(color=(255, 0, 0, 50)),  # 填充：淡红色，透明度 50
            pen=pg.mkPen(color=(255, 0, 0, 255)),     # 边线：纯红色，不透明
            movable=False
        )
        self.p2_region_forbiden = pg.LinearRegionItem(
            brush=pg.mkBrush(color=(255, 0, 0, 50)),  # 填充：淡红色，透明度 50
            pen=pg.mkPen(color=(255, 0, 0, 255)),     # 边线：纯红色，不透明
            movable=False
        )

        # 设置绘图窗口
        self.p1 = self.win.addPlot(row=0, col=0, title="mid_price and net")
        self.p2 = self.win.addPlot(row=1, col=0, title="Cumulative Rewards")
        self.p1.getViewBox().parentItem().setFixedHeight(400)
        self.p2.getViewBox().parentItem().setFixedHeight(150)
        self.p2.setXLink(self.p1)

        # 添加绘图对象
        self.p1.addItem(self.p1_mid_price)
        self.p1.addItem(self.p1_net_raw)
        self.p1.addItem(self.p1_mid_price_future)
        if self.p1_vline:
            self.p1.addItem(self.p1_vline)
        if self.p1_vline_acc_begin:
            self.p1.addItem(self.p1_vline_acc_begin)
        if self.p1_text:
            self.p1.addItem(self.p1_text)
        if self.p1_text_acc_begin:
            self.p1.addItem(self.p1_text_acc_begin)
        self.p1.addItem(self.p1_valley_points)
        self.p1.addItem(self.p1_peak_points)

        self.p2.addItem(self.p2_rewards)
        if self.p2_vline:
            self.p2.addItem(self.p2_vline)
        if self.p2_vline_acc_begin:
            self.p2.addItem(self.p2_vline_acc_begin)
        self.p2.addItem(self.p2_scatter)

        for arrow in self.p1_arrows_buy + self.p1_arrows_sell:
            arrow.setVisible(False)
            self.p1.addItem(arrow)

        self.p1.addItem(self.p1_region, ignoreBounds=True)
        self.p2.addItem(self.p2_region, ignoreBounds=True)
        self.p1.addItem(self.p1_region_forbiden, ignoreBounds=True)
        self.p2.addItem(self.p2_region_forbiden, ignoreBounds=True)

        self.p1_region.setZValue(-10)
        self.p2_region.setZValue(-10)
        self.p1_region_forbiden.setZValue(-10)
        self.p2_region_forbiden.setZValue(-10)

        # 创建按钮
        self.button1 = QtWidgets.QPushButton("BUY")
        self.button2 = QtWidgets.QPushButton("SELL")
        self.button3 = QtWidgets.QPushButton("KEEP_RUN")
        self.button1.clicked.connect(lambda: self.button_clicked(0))
        self.button2.clicked.connect(lambda: self.button_clicked(1))
        self.button3.clicked.connect(lambda: (setattr(self, 'keep_play', True), print('keep_play set to True')))

        # 创建按钮布局
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button1)
        button_layout.addWidget(self.button2)
        button_layout.addWidget(self.button3)

        # 创建按钮容器 widget
        self.button_widget = QtWidgets.QWidget()
        self.button_widget.setLayout(button_layout)

        # 将按钮 widget 添加到主布局
        main_layout.addWidget(self.button_widget)

        # 显示主窗口
        self.main_widget.show()
        self.main_widget.closeEvent = self._on_close  # 绑定关闭事件

        log('绘图和按钮设置完成')

    def button_clicked(self, index):
        """处理按钮点击事件"""
        self.clicked_button = index
        log(f'按钮 {index} 被点击')

    def _on_close(self, event):
        """处理窗口关闭事件"""
        if self.app:
            self.app.quit()
        event.accept()
        log('窗口已关闭')

    def close(self):
        """清理资源"""
        if self.win:
            self.win.close()
            self.win = None
        if self.app:
            self.app.quit()
            self.app = None
        log('渲染资源已清理')

    # def wait_keep_play(self, wait_time=5):
    #     self._wait_counter = 0
    #     self._max_wait = wait_time
    #     self._check_keep_play_timer = QtCore.QTimer()
    #     self._check_keep_play_timer.timeout.connect(self._check_keep_play_status)
    #     self._check_keep_play_timer.start(1000)  # 每秒检查一次

    # def _check_keep_play_status(self):
    #     print(f'wait_keep_play: {self._wait_counter}')
    #     self._wait_counter += 1

    #     if self.keep_play is not None:
    #         self._check_keep_play_timer.stop()

    #     if self._wait_counter >= self._max_wait:
    #         print("Timeout: keep_play still None")
    #         self._check_keep_play_timer.stop()

    def handle_data(self, data):
        """处理输入数据"""
        if not self.win or not self.win.isVisible():
            log('窗口已关闭，跳过数据处理')
            return

        if 'render' in data:
            self._plot_data(*data['render'])
            if self.win.isVisible():
                self.app.processEvents()

            if self.human_play:
                # # 等待按钮点击
                self.clicked_button = None
                while self.clicked_button is None:
                    if not self.main_widget or not self.main_widget.isVisible():
                        log('窗口已关闭，退出等待')
                        return None
                    self.app.processEvents()
                    QtCore.QThread.msleep(10)  # 短暂休眠以减少 CPU 使用率

                # 返回点击的按钮索引
                return self.clicked_button

        elif 'full_plot_data' in data:
            self.full_plot_data = data['full_plot_data']
            if not self.full_plot_data.empty:
                self.open_idx = None
                self.pre_n = None
                self.std_data = {}
                self.std_data['mid_price'] = self.full_plot_data['mid_price'].iloc[0] or 1.0
                self.full_plot_data['mid_price_std'] = self.full_plot_data['mid_price'] / self.std_data['mid_price']
                max_len = len(self.full_plot_data)
                self.data_deque['net_raw'] = deque(maxlen=max_len)
                self.data_deque['rewards'] = deque(maxlen=max_len)
                self.pre_latest_tick_time = None
            self.potential_data = None

        elif 'potential_data' in data:
            self.potential_data = data['potential_data']

    def _plot_data(self, a, b, latest_tick_time, latest_net_raw, status, need_render, recent_reward, done, init_status):
        """更新绘图数据并调整 Y 轴范围"""
        if not self.win or not self.win.isVisible():
            log('窗口已关闭，跳过绘图')
            return

        if not need_render or self.full_plot_data is None or self.full_plot_data.empty:
            log("跳过 _plot_data: full_plot_data 无效")
            return

        plot_data = self.full_plot_data.iloc[a:b]
        if plot_data.empty:
            log("跳过 _plot_data: plot_data 为空")
            return

        if not isinstance(plot_data.index, pd.DatetimeIndex):
            log("plot_data.index 必须是 DatetimeIndex")
            return

        try:
            hist_end = plot_data.index.get_loc(latest_tick_time) + 1
        except KeyError:
            log(f"无效的 latest_tick_time: {latest_tick_time}")
            return

        # 初始化标准化数据
        if 'net_raw' not in self.std_data:
            self.std_data['net_raw'] = latest_net_raw if latest_net_raw != 0 else 1.0

        n = len(plot_data)
        
        # 记录第一个持仓的索引
        if self.open_idx is None:
            if status == 1:
                self.open_idx = hist_end - 1
        else:
            if self.pre_n is not None:
                if n <= self.pre_n:
                    self.open_idx -= 1
                    self.open_idx = max(self.open_idx, 0)

        self.pre_n = n

        # 若 latest_tick_time 大于 12:00:00
        # 且 self.net_data_fixed == False
        # 需要fix net_raw / rewards
        noon_time = pd.Timestamp('12:00:00').time()
        if latest_tick_time.time() > noon_time and self.pre_latest_tick_time is not None and self.pre_latest_tick_time.time() < noon_time and not self.net_data_fixed:
            # 计算 plot_data 中当前时间点 至 前一个绘图点的时间 数据的数量
            need_fix_num = plot_data.index.get_loc(latest_tick_time) - plot_data.index.get_loc(self.pre_latest_tick_time) - 1
            for i in range(need_fix_num):
                self.data_deque['net_raw'].append(self.data_deque['net_raw'][-1])
                self.data_deque['rewards'].append(self.data_deque['rewards'][-1])
                if self.open_idx is not None:
                    self.open_idx -= 1
            self.net_data_fixed = True
            self.forbiden_end_idx = hist_end - 1
            # self.forbiden_begin_idx = hist_end - 1 - need_fix_num
            log(f'fix noon data net_raw / rewards: {need_fix_num} 条')

        else:
            # if self.forbiden_end_idx is not None and self.forbiden_begin_idx is not None:
            if self.forbiden_end_idx is not None:
                if n <= self.pre_n:
                    self.forbiden_end_idx -= 1
                    # self.forbiden_begin_idx -= 1
                    self.forbiden_end_idx = max(self.forbiden_end_idx, 0)
                    # self.forbiden_begin_idx = max(self.forbiden_begin_idx, 0)

        # 查找 plot_data 中大于 12:00 的第一个时间索引
        filtered = plot_data[plot_data.index.time > noon_time]
        if not filtered.empty:
            first_after_noon = filtered.index[0]
            self.forbiden_begin_idx = max(plot_data.index.get_loc(first_after_noon) - 1, 0)
            if (self.forbiden_begin_idx == 0 and self.forbiden_end_idx is None) or (self.forbiden_begin_idx < hist_end - 1):
                self.forbiden_begin_idx = None

        # 更新禁止背景数据
        # 检查 self.open_idx 是否有效
        if self.forbiden_begin_idx is not None and self.forbiden_end_idx!=0:
            # 设置区域范围
            self.p1_region_forbiden.setRegion([self.forbiden_begin_idx, self.forbiden_end_idx if self.forbiden_end_idx is not None else len(plot_data) - 1])
            self.p2_region_forbiden.setRegion([self.forbiden_begin_idx, self.forbiden_end_idx if self.forbiden_end_idx is not None else len(plot_data) - 1])
            self.p1_region_forbiden.show()
            self.p2_region_forbiden.show()
        else:
            # 隐藏区域
            self.p1_region_forbiden.hide()
            self.p2_region_forbiden.hide()

        net_value = latest_net_raw / self.std_data['net_raw'] if self.std_data['net_raw'] != 0 else 1.0
        self.data_deque['net_raw'].append(net_value)
        self.data_deque['rewards'].append(recent_reward if (recent_reward is not None and not done) else 0.0)

        net_raw = np.array(list(self.data_deque['net_raw']))
        rewards = np.array(list(self.data_deque['rewards']))

        acc_begin_pos = 0
        if len(net_raw) < hist_end:
            pad_length = hist_end - len(net_raw)
            acc_begin_pos = pad_length
            net_raw = np.concatenate([np.full(pad_length, net_raw[0] if len(net_raw) > 0 else 1.0), net_raw])
            rewards = np.concatenate([np.zeros(pad_length), rewards])

        net_raw = net_raw[-hist_end:]
        rewards = rewards[-hist_end:]
        cumsum_rewards = np.cumsum(rewards)

        # 获取 mid_price 数据
        mid_price_hist = plot_data['mid_price_std'].iloc[:hist_end].values
        mid_price_future = plot_data['mid_price_std'].iloc[hist_end-1:].values

        # 数据清理
        mid_price_hist = np.asarray(mid_price_hist, dtype=np.float64).flatten()
        mid_price_future = np.asarray(mid_price_future, dtype=np.float64).flatten()
        mid_price_hist = np.nan_to_num(mid_price_hist, nan=0.0, posinf=0.0, neginf=0.0)
        mid_price_future = np.nan_to_num(mid_price_future, nan=0.0, posinf=0.0, neginf=0.0)
        net_raw = np.nan_to_num(net_raw, nan=0.0, posinf=0.0, neginf=0.0)
        cumsum_rewards = np.nan_to_num(cumsum_rewards, nan=0.0, posinf=0.0, neginf=0.0)

        # 创建 X 数据
        x_hist = np.arange(hist_end, dtype=np.float64)
        x_future = np.arange(hist_end-1, n, dtype=np.float64)

        # 更新曲线数据
        try:
            self.p1_mid_price.setData(x_hist, mid_price_hist)
            if len(mid_price_future) > 0:
                self.p1_mid_price_future.setData(x_future, mid_price_future)
            else:
                self.p1_mid_price_future.setData([], [])
            self.p1_net_raw.setData(x_hist, net_raw)
            self.p2_rewards.setData(x_hist, cumsum_rewards)
        except Exception as e:
            log(f'更新曲线数据失败: {e}')
            return
        
        # 更新持仓背景数据
        # 检查 self.open_idx 是否有效
        if self.open_idx is not None and self.open_idx < hist_end - 1:
            # 设置区域范围为 [self.open_idx, hist_end - 1]
            self.p1_region.setRegion([self.open_idx, hist_end - 1])
            self.p2_region.setRegion([self.open_idx, hist_end - 1])
            self.p1_region.show()
            self.p2_region.show()
        else:
            # 隐藏区域
            self.p1_region.hide()
            self.p2_region.hide()

        # 更新垂直线和时间标签
        time_str = plot_data.index[hist_end-1].strftime('%Y-%m-%d %H:%M:%S')
        if self.p1_vline:
            self.p1_vline.setPos(hist_end-1)
        if self.p1_vline_acc_begin:
            self.p1_vline_acc_begin.setPos(acc_begin_pos)
        if self.p2_vline:
            self.p2_vline.setPos(hist_end-1)
        if self.p2_vline_acc_begin:
            self.p2_vline_acc_begin.setPos(acc_begin_pos)
        if self.p1_text:
            self.p1_text.setText(time_str)
            vb_range = self.p1.getViewBox().viewRange()
            if vb_range and len(vb_range[1]) > 0:
                self.p1_text.setPos(hist_end-1, vb_range[1][0])
        if self.p1_text_acc_begin:
            self.p1_text_acc_begin.setText(f'ACC_BEGIN')
            vb_range = self.p1.getViewBox().viewRange()
            if vb_range and len(vb_range[1]) > 0:
                self.p1_text_acc_begin.setPos(acc_begin_pos, vb_range[1][0])

        # 更新谷点、峰点和买卖箭头
        if self.potential_data is not None:
            _potential_data = self.potential_data.iloc[a:b]
            valley_mask = _potential_data['valley_peak'] == 0
            peak_mask = _potential_data['valley_peak'] == 1
            valley_indices = np.where(valley_mask)[0]
            peak_indices = np.where(peak_mask)[0]
            mid_price = np.asarray(plot_data['mid_price_std'].values, dtype=np.float64).flatten()
            self.p1_valley_points.setData(valley_indices, mid_price[valley_indices])
            self.p1_peak_points.setData(peak_indices, mid_price[peak_indices])

            buy_mask = _potential_data['action'] == ACTION_BUY
            sell_mask = _potential_data['action'] == ACTION_SELL
            buy_indices = np.where(buy_mask)[0]
            sell_indices = np.where(sell_mask)[0]

            for i, idx in enumerate(buy_indices):
                if i < len(self.p1_arrows_buy):
                    self.p1_arrows_buy[i].setPos(idx, mid_price[idx])
                    self.p1_arrows_buy[i].setVisible(True)
            for i in range(len(buy_indices), len(self.p1_arrows_buy)):
                self.p1_arrows_buy[i].setVisible(False)

            for i, idx in enumerate(sell_indices):
                if i < len(self.p1_arrows_sell):
                    self.p1_arrows_sell[i].setPos(idx, mid_price[idx])
                    self.p1_arrows_sell[i].setVisible(True)
            for i in range(len(sell_indices), len(self.p1_arrows_sell)):
                self.p1_arrows_sell[i].setVisible(False)

        # 更新奖励散点图
        reward_points = [(i, cumsum_rewards[i]) for i in range(len(rewards)) if rewards[i] != 0 and not np.isnan(cumsum_rewards[i])]
        if reward_points:
            x, y = zip(*reward_points)
            self.p2_scatter.setData(x, y)
        else:
            self.p2_scatter.setData([], [])

        # 更新标题
        status_str = '持仓' if status == 1 else '空仓'
        init_status_str = '持仓' if init_status == 1 else '空仓'
        self.p1.setTitle(f'status: {status_str} init_status: {init_status_str}')
        if done:
            self.p2.setTitle(f'累计奖励: {cumsum_rewards[-1]:.6f} + {recent_reward:.6f}')
        else:
            self.p2.setTitle(f'累计奖励: {cumsum_rewards[-1]:.6f}')

        # 动态调整 Y 轴范围
        self.update_p1_y_range(self.p1)
        self.update_p2_y_range(self.p2)

        # 更新 前一个绘图点的时间
        self.pre_latest_tick_time = latest_tick_time

        # 平仓清理标记
        if status == 0:
            self.open_idx = None

    def update_p1_y_range(self, p1):
        """动态调整 p1 的 Y 轴范围"""
        if not p1:
            return
        try:
            xr = p1.vb.viewRange()[0]
            xmin, xmax = xr
            y_mins = []
            y_maxs = []
            for item in [self.p1_mid_price, self.p1_net_raw, self.p1_mid_price_future]:
                data = item.getData()
                if data[0] is not None and len(data[0]) > 0:
                    mask = (data[0] >= xmin) & (data[0] <= xmax)
                    if np.any(mask):
                        y_data = data[1][mask]
                        y_mins.append(np.min(y_data))
                        y_maxs.append(np.max(y_data))
            if y_mins and y_maxs:
                y_min = min(y_mins)
                y_max = max(y_maxs)
                p1.vb.setYRange(y_min, y_max, padding=0.05)
        except Exception as e:
            log(f'调整 p1 Y 轴范围失败: {e}')

    def update_p2_y_range(self, p2):
        """动态调整 p2 的 Y 轴范围"""
        if not p2:
            return
        try:
            xr = p2.vb.viewRange()[0]
            xmin, xmax = xr
            y_mins = []
            y_maxs = []
            for item in [self.p2_rewards]:
                data = item.getData()
                if data[0] is not None and len(data[0]) > 0:
                    mask = (data[0] >= xmin) & (data[0] <= xmax)
                    if np.any(mask):
                        y_data = data[1][mask]
                        y_mins.append(np.min(y_data))
                        y_maxs.append(np.max(y_data))
            if y_mins and y_maxs:
                y_min = min(y_mins)
                y_max = max(y_maxs)
                p2.vb.setYRange(y_min, y_max, padding=0.05)
        except Exception as e:
            log(f'调整 p2 Y 轴范围失败: {e}')

class LOB_trade_env(gym.Env):
    """
    用于 LOB 的强化学习环境
    返回的 obs 结构:
        lob数据 + 持仓 + 未实现收益率
    """

    REG_NAME = 'lob'
    ITERATION_DONE_FILE = os.path.join(os.path.expanduser('~'), '_lob_env_iteration_done')
    
    def __init__(self, config: dict, data_std=True, debug_obs_date=None, debug_obs_time=None, debug_init_pos=None, dump_bid_ask_accnet=False, data_folder=DATA_FOLDER):
        """
        :param config: 配置
            {
                # 用于实例化 数据生产器
                'data_type': 'train'/'val'/'test',# 训练/测试
                'his_len': 100,# 每个样本的 历史数据长度
                'simple_test': False,# 是否为简单测试
                'need_cols': [],# 需要读取的列
                'use_symbols': [],# 只使用某些标的
                'random_begin_in_day': True,# 是否在日内随机开始
                'close_trade_need_reset': True,# 平仓后需要重置
                'latest_dates': -1,# 使用最近日期的数据，-1 表示使用所有数据

                # 数据增强
                'use_random_his_window': False,# 是否使用随机历史窗口
                'use_gaussian_noise_vol': False,# 是否使用高斯噪声
                'use_spread_add_small_limit_order': False,# 是否使用价差添加小单

                # 终止游戏的回撤阈值
                'max_drawdown_threshold': 0.01,# 最大回测阈值

                # 渲染模式
                'render_mode': 'none',

                # 渲染频率, 每N步渲染一次
                'render_freq': 1,

                # 是否需要人工操作
                'human_play': False,

                # 奖励策略
                'end_position': ClosePositionRewardStrategy,
                'close_position': ClosePositionRewardStrategy,
                'open_position_step': BlankRewardStrategy,
                'hold_position': HoldPositionRewardStrategy,
                'no_position': NoPositionRewardStrategy,
                'force_stop': ForceStopRewardStrategy,
            }
        """
        super().__init__()

        self.data_type = config.get('data_type', 'train')
        self.his_len = config.get('his_len', 100)
        self.simple_test = config.get('simple_test', False)
        self.need_cols = config.get('need_cols', [])
        self.use_symbols = config.get('use_symbols', [])
        self.random_begin_in_day = config.get('random_begin_in_day', True)
        self.close_trade_need_reset = config.get('close_trade_need_reset', True)
        self.latest_dates = config.get('latest_dates', -1)
        self.use_random_his_window = config.get('use_random_his_window', False)
        self.use_gaussian_noise_vol = config.get('use_gaussian_noise_vol', False)
        self.use_spread_add_small_limit_order = config.get('use_spread_add_small_limit_order', False)

        self.max_drawdown_threshold = abs(config.get('max_drawdown_threshold', 0.01))

        self.render_mode = config.get('render_mode', 'none')
        self.render_freq = config.get('render_freq', 1)
        self.human_play = config.get('human_play', False)

        end_position_reward_strategy = config.get('end_position', ClosePositionRewardStrategy)
        close_position_reward_strategy = config.get('close_position', ClosePositionRewardStrategy)
        open_position_step_reward_strategy = config.get('open_position_step', BlankRewardStrategy)
        hold_position_reward_strategy = config.get('hold_position', HoldPositionRewardStrategy)
        no_position_reward_strategy = config.get('no_position', NoPositionRewardStrategy)
        force_stop_reward_strategy = config.get('force_stop', ForceStopRewardStrategy)

        self.dump_bid_ask_accnet = dump_bid_ask_accnet

        # 保存文件夹
        if os.path.exists(config['train_folder']):
            config['train_folder'] = ''
        self.save_folder = os.path.join(config['train_folder'], 'env_output')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # 奖励计算器
        self.reward_calculator = RewardCalculator(
            self.max_drawdown_threshold, 
            {
                'end_position': end_position_reward_strategy,
                'close_position': close_position_reward_strategy,
                'open_position_step': open_position_step_reward_strategy,
                'hold_position': hold_position_reward_strategy,
                'no_position': no_position_reward_strategy,
                'force_stop': force_stop_reward_strategy,
            },
        )
    
        # 是否标准化数据
        self.data_std = data_std

        # 测试日期
        self.debug_date = days2date(int(debug_obs_date)) if debug_obs_date else None
        if self.debug_date and isinstance(self.debug_date, str):
            self.debug_date = [self.debug_date]
        else:
            self.debug_date = []

        # 调试时间
        self.debug_time = int(debug_obs_time * MAX_SEC_BEFORE_CLOSE) if debug_obs_time else None

        # 调试初始持仓
        self.debug_init_pos = debug_init_pos

        # 初始化日志
        log_name = f'{config["train_title"]}_{beijing_time().strftime("%Y%m%d")}'
        init_logger(log_name, home=config['train_folder'], timestamp=False)
        log(f'[{id(self)}] init logger: {get_log_file()}')
        
        # 数据生产器
        self.data_producer = data_producer(
            self.data_type, 
            self.his_len, 
            self.simple_test, 
            self.need_cols, 
            self.use_symbols, 
            data_std=self.data_std, 
            save_folder=self.save_folder, 
            debug_date=self.debug_date,
            debug_time=self.debug_time,
            random_begin_in_day=self.random_begin_in_day,
            latest_dates=self.latest_dates,
            data_folder=data_folder,
            use_random_his_window=self.use_random_his_window,
            use_gaussian_noise_vol=self.use_gaussian_noise_vol,
            use_spread_add_small_limit_order=self.use_spread_add_small_limit_order,
        )

        # 最近一次step的时间, 用于判断 迭代次数
        self.last_step_time = time.time()
        # 迭代次数
        self.iteration = 0

        # 账户数据
        self.acc = Account()

        # 是否需要重置（切换模式后需要）
        self.need_reset = False

        # 动作空间 
        # 持仓/空仓 ACTION_BUY/ACTION_SELL
        self.action_space = spaces.Discrete(2)

        # 观察空间 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data_producer.data_size() + 4,), dtype=np.float32)

        # 样本计数
        self.sample_count = 0

        # 最近一次平仓的idx
        self.last_close_idx = 0

        # 环境输出文件
        self.need_upload_file = ''
        self.update_need_upload_file()

        # 记录每个 episode 的步数
        self.mean_episode_lengths = 0
        self.episode_count = 0

        # 是否终止
        self.done = False

        # 渲染模式
        if self.render_mode == 'human':
            self._render = Render(self.human_play)

        log(f'[{id(self)}][{self.data_producer.data_type}] init env done')

    @staticmethod
    def iteration_done():
        """会使用文件来标记迭代结束的时间"""
        with open(LOB_trade_env.ITERATION_DONE_FILE, 'w') as f:
            f.write(f'1')

    def is_iteration_done(self):
        """是否最近迭代完成"""
        if os.path.exists(LOB_trade_env.ITERATION_DONE_FILE):
            return os.path.getmtime(LOB_trade_env.ITERATION_DONE_FILE) > self.last_step_time
        return False

    def update_need_upload_file(self):
        os.makedirs(os.path.join(self.save_folder, self.data_producer.data_type), exist_ok=True)

        # 只保留文件夹中最近的3个文件
        files = [f for f in os.listdir(os.path.join(self.save_folder, self.data_producer.data_type)) if f.endswith('.csv')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.save_folder, self.data_producer.data_type, x)), reverse=True)
        for f in files[3:]:
            os.remove(os.path.join(self.save_folder, self.data_producer.data_type, f))

        self.need_upload_file = os.path.join(self.save_folder, self.data_producer.data_type, f'{id(self)}_{self.iteration}.csv')
        
    def _set_data_type(self, data_type):
        if self.data_producer.data_type != data_type:
            log(f'[{id(self)}][{self.data_producer.data_type}] set data type: {data_type}')
            self.data_producer.set_data_type(data_type)
            self.need_reset = True
            self.sample_count = 0
            self.update_need_upload_file()
            self.data_type = data_type

    def val(self):
        self._set_data_type('val')

    def train(self):
        self._set_data_type('train')

    def test(self):
        self._set_data_type('test')

    def _get_data(self):
        # 获取数据
        if self.data_std:
            symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data = self.data_producer.get()
            x = x.reshape(-1)
            return symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data
        else:
            symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data, x_std, sec_std = self.data_producer.get()
            x = x.reshape(-1)
            return symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data, x_std, sec_std

    def _cal_reward(self, action, need_close, info):
        """
        计算奖励
        
        act_criteria 动作评价:
            0 交易获利
            1 交易亏损
            2 触发止损

        act_result: 动作结果
            RESULT_OPEN / RESULT_CLOSE / RESULT_HOLD
        """
        # 游戏是否终止
        # 1. 最大回测超过阈值                           -STD_REWARD
        acc_done = False

        pos, inday_return, unrealized_return, act_result = self.acc.step(self.data_producer.bid_price[-1], self.data_producer.ask_price[-1], action, self.last_close_idx)
        # log(f'[{id(self)}][{self.data_producer.data_type}][{self.data_producer.step_use_data.iloc[-1].name}] act: {action}, act_result: {act_result}, last_close_idx: {self.last_close_idx}')
        need_cal_drawup = pos == 0 and act_result == RESULT_HOLD
        res = self.acc.cal_res(self.acc.net_raw, self.acc.net_raw_bm, self.last_close_idx, need_cal_drawup)
        res['hold_length'] = self.acc.hold_length
        
        # 记录net/net_bm
        res['net'] = self.acc.net_raw[-1] if self.acc.net_raw else np.nan
        res['net_bm'] = self.acc.net_raw_bm[-1] if self.acc.net_raw_bm else np.nan

        #数据类型
        res['data_type'] = self.data_producer.data_type

        reward = 0.0

        # 同时记录评价指标
        if need_close or act_result==RESULT_CLOSE:
            # 增加操作交易评价结果
            # 体现 win/loss/止损
            res['act_criteria'] = 0 if res['trade_return'] > 0 else 1

        # 检查最大回测 
        force_stop = False
        if abs(res['max_drawdown']) > self.max_drawdown_threshold:
            # 游戏被终止，计入交易失败
            res['act_criteria'] = 2
            force_stop = True

            # 计算奖励
            reward, acc_done = self.reward_calculator.calculate_reward(id(self), STD_REWARD, force_stop, need_close, act_result, res, self.data_producer, self.acc)
            log(f'[{id(self)}][{self.data_producer.data_type}] max drawdown: {res["max_drawdown"]}, force_stop')

        else:
            # 游戏正常进行

            # 平仓时更新一次记录
            if need_close or act_result==RESULT_CLOSE:
                if act_result==RESULT_CLOSE:
                    # 交易对数
                    self.trades += 1

                # if not need_close:
                #     # 潜在收益率（针对最近的交易序列） 需要剔除掉第一个数据，因为第一个数据无法成交
                #     if self.dump_bid_ask_accnet:
                #         dump_file = r'C:\Users\lh\Desktop\temp\bid_ask_accnet.pkl'
                #         log(f'dump (bid, ask, accnet) > {dump_file}')
                #         log(f'debug code:')
                #         log('\n\n'+ debug_bid_ask_accnet_code + '\n')
                #         pickle.dump((self.data_producer.bid_price[self.last_close_idx+1:], self.data_producer.ask_price[self.last_close_idx+1:], self.acc.net_raw[self.last_close_idx:]), open(dump_file, 'wb'))
                #     _, max_profit_reachable_bm, _, _ = max_profit_reachable(self.data_producer.bid_price[self.last_close_idx+1:], self.data_producer.ask_price[self.last_close_idx+1:])
                #     latest_trade_net = self.acc.net_raw[self.last_close_idx:]# 最近一次交易净值序列, 重上一次平仓的净值开始计算 > 上一次平仓后的净值序列
                #     acc_return = np.log(latest_trade_net[-1]) - np.log(latest_trade_net[0])
                # else:
                #     # 游戏完成，使用所有数据完整计算
                #     _, max_profit_reachable_bm, _, _ = max_profit_reachable(self.data_producer.bid_price[1:], self.data_producer.ask_price[1:])
                #     # 日内完整的策略收益率
                #     acc_return = res['trade_return']

                # train/test/val 都会在平仓时结束游戏，所以使用所有数据完整计算
                _, max_profit_reachable_bm, _, _ = max_profit_reachable(self.data_producer.bid_price[1:], self.data_producer.ask_price[1:],rep_select='last')
                # 日内完整的策略收益率
                acc_return = res['trade_return']

                res['potential_return'] = max_profit_reachable_bm
                res['acc_return'] = acc_return

            # 交易对数
            res['trades'] = self.trades

            # 计算奖励
            reward, acc_done = self.reward_calculator.calculate_reward(id(self), STD_REWARD, force_stop, need_close, act_result, res, self.data_producer, self.acc)

            # 奖励范围
            assert abs(reward) <= FINAL_REWARD, f'reward({reward}) > FINAL_REWARD({FINAL_REWARD})'

        # 拷贝 res > info
        for k, v in res.items():
            info[k] = v

        return reward, acc_done, pos, inday_return, unrealized_return, act_result

    def out_test_predict(self, out_text):
        """
        输出测试预测数据 -> predict.csv

        # 状态相关
        id,before_market_close_sec,pos,inday_return,unrealized_return,predict,data_file,episode,step,net,net_bm,

        # 其他
        terminated,truncated,

        # 奖励评价相关
        reward,max_drawdown,max_drawdown_ticks,trade_return,step_return,hold_length,potential_return,acc_return,

        # 基准相关
        max_drawdown_bm,max_drawdown_ticks_bm,max_drawup_ticks_bm,drawup_ticks_bm_count,trade_return_bm,step_return_bm,
        """
        # 输出列名
        if not os.path.exists(self.need_upload_file):
            with open(self.need_upload_file, 'w') as f:
                f.write('id,before_market_close_sec,pos,inday_return,unrealized_return,predict,data_file,episode,step,net,net_bm,terminated,truncated,reward,max_drawdown,max_drawdown_ticks,trade_return,step_return,hold_length,potential_return,acc_return,max_drawdown_bm,max_drawdown_ticks_bm,max_drawup_ticks_bm,drawup_ticks_bm_count,trade_return_bm,step_return_bm\n')
        with open(self.need_upload_file, 'a') as f:
            f.write(out_text)
            f.write('\n')
    
    def step(self, action):
        """
        动作:
            ACTION_BUY / ACTION_SELL
        """
        try:
            assert not self.need_reset, "LOB env need reset"
            
            self.steps += 1

            # 更新样本计数
            self.sample_count += 1
            if self.sample_count % 500 == 0:
                log(f'[{id(self)}][{self.data_producer.data_type}] total sample: {self.sample_count}')

            # 在每个迭代更新文件
            if self.is_iteration_done():
                self.update_need_upload_file()
                self.iteration += 1
            self.last_step_time = time.time()

            # 准备输出数据
            out_text = f'{self.data_producer.id},{self.static_data["before_market_close_sec"]},{self.static_data["pos"]},{self.static_data["inday_return"]},{self.static_data["unrealized_return"]},{int(action)},{self.data_producer.cur_data_file},{self.episode_count},{self.steps}'

            # 先获取下一个状态的数据, 会储存 bid_price, ask_price, 用于acc.step(), 避免用当前状态种的价格结算
            if self.data_std:
                symbol_id, before_market_close_sec, observation, need_close, _id, unrealized_log_return_std_data = self._get_data()
            else:
                symbol_id, before_market_close_sec, observation, need_close, _id, unrealized_log_return_std_data, x_std, sec_std = self._get_data()

            info = {
                'id': _id,
                'action': action,
            }

            if not self.data_std:
                info['x_std'] = x_std
                info['sec_std'] = sec_std

            # 计算奖励
            reward, acc_done, pos, inday_return, unrealized_return, act_result = self._cal_reward(action, need_close, info)

            # 记录静态数据，用于输出预测数据
            # 属于下个step的数据
            self.static_data = {
                'before_market_close_sec': before_market_close_sec,
                'pos': pos,
                'inday_return': inday_return,
                'unrealized_return': unrealized_return,
            }

            # 记录平仓，需要放在 _cal_reward 之后
            if act_result==RESULT_CLOSE:
                self.last_close_idx = self.steps

                if self.close_trade_need_reset and not self.run_twice:
                    # 平仓后游戏结束
                    acc_done = True

                if self.run_twice:
                    log(f'初始化持仓，第一步平仓，需要二次平仓')

            # 准备输出数据
            # net,net_bm,
            out_text += f",{info['net']},{info['net_bm']}"
            # reward,max_drawdown,max_drawdown_ticks,trade_return,step_return,hold_length,
            out_text2 = f",{reward},{info.get('max_drawdown', '')},{info.get('max_drawdown_ticks', '')},{info.get('trade_return', '')},{info.get('step_return', '')},{info.get('hold_length', '')},{info.get('potential_return', '')},{info.get('acc_return', '')}"
            # max_drawdown_bm,max_drawdown_ticks_bm,max_drawup_ticks_bm,drawup_ticks_bm_count,trade_return_bm,step_return_bm,
            out_text2 += f",{info.get('max_drawdown_bm', '')},{info.get('max_drawdown_ticks_bm', '')},{info.get('max_drawup_ticks_bm', '')},{info.get('drawup_ticks_bm_count', '')},{info.get('trade_return_bm', '')},{info.get('step_return_bm', '')}"

            # 标准化 未实现收益率
            if unrealized_log_return_std_data is not None:
                unrealized_return = (unrealized_return - unrealized_log_return_std_data[0]) / unrealized_log_return_std_data[1]
                inday_return = (inday_return - unrealized_log_return_std_data[0]) / unrealized_log_return_std_data[1]

            # 添加 静态特征
            # 20250406 取消收益率
            observation = np.concatenate([observation, [np.float32(symbol_id), np.float32(before_market_close_sec), np.float32(pos), np.float32(date2days(self.data_producer.date))]])

            # 检查是否结束
            terminated = acc_done or need_close# 游戏终止
            # 环境中没有截断结束的情况
            truncated = False

            out_text += f",{terminated},{truncated}"
            out_text += out_text2
            # 记录数据
            self.out_test_predict(out_text)

            self.done = terminated or truncated
            if self.done:
                # 计算平均步数
                self.mean_episode_lengths = (self.mean_episode_lengths * self.episode_count + self.steps) / (self.episode_count + 1)
                self.episode_count += 1
                log(f'[{id(self)}][{self.data_producer.data_type}] episode {self.episode_count} done, mean episode length: {self.mean_episode_lengths}, latest episode length: {self.steps}, reward: {reward}')

            # 记录最近的reward
            self.recent_reward = reward
            # log(f'[{id(self)}][{self.data_producer.data_type}] step {self.steps} status: {self.acc.status} reward: {self.recent_reward}')

            # 一步之后不需要再运行
            if self.run_twice:
                self.run_twice = False

            return observation, reward, terminated, truncated, info

        except Exception as e:
            log(f'[{id(self)}][{self.data_producer.data_type}] step error: {get_exception_msg()}')
            raise e

    def reset(self, seed=None, options=None, acc_status=None):
        try:
            super().reset(seed=seed, options=options)

            # 步数统计
            self.steps = 0

            # 是否终止
            self.done = False

            # 交易对数
            self.trades = 0

            # 重置累计收益率
            self.acc_return = 0

            self.recent_reward = 0
            
            # 重置
            self.need_reset = False

            # 最近一次平仓的idx
            self.last_close_idx = 0

            # 数据
            log(f'[{id(self)}][{self.data_producer.data_type}] reset')

            while True:
                if not self.data_producer.reset(rng = self.np_random):
                    # 指定日期标的，而没有找到数据
                    # 正常不指定的情况下不应该发生
                    raise ValueError(f'指定的数据不存在')

                if self.data_std:
                    symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data = self._get_data()
                else:
                    symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data, x_std, sec_std = self._get_data()
                if need_close:
                    # 若是 val 数据，有可能 need_close 为True
                    # 需要过滤
                    log(f'[{id(self)}][{self.data_producer.data_type}] need_close: True, reset data_producer again')
                else:
                    break
            
            if self.render_mode == 'human':
                full_plot_data = self.data_producer.plot_data
                self._render.handle_data({'full_plot_data': full_plot_data})

            # 账户
            pos = self.acc.reset(self.data_producer.bid_price[-1], rng = self.np_random, status=acc_status or self.debug_init_pos)
            log(f'acc reset: {pos}')

            # 初始化持仓需要记录
            self.run_twice = pos == 1

            # 添加 静态特征
            x = np.concatenate([x, [np.float32(symbol_id), np.float32(before_market_close_sec), np.float32(pos), np.float32(date2days(self.data_producer.date))]])

            # 记录静态数据，用于输出预测数据
            self.static_data = {
                'before_market_close_sec': before_market_close_sec,
                'pos': 0.0,
                'inday_return': 0.0,
                'unrealized_return': 0.0,
            }

            if self.data_std:
                return x, {'id': _id}
            else:
                return x, {'x_std': x_std, 'sec_std': sec_std, 'id': _id}
            
        except Exception as e:
            log(f'[{id(self)}][{self.data_producer.data_type}] reset error: {get_exception_msg()}')
            raise e

    def close(self):
        if self.render_mode == 'human':
            self._render.close()

    def add_potential_data(self, potential_data):
        if self.render_mode == 'human':
            self._render.handle_data({'potential_data': potential_data})

    def set_state(self, t, expert=None):
        """
        expert: 用于可视化指导
        使用轨迹设置 env
        返回是否终止
        """
        # 获取样本数据
        obs = t['obs']
        obs_date = obs[-1]
        obs_symbol = obs[-3]
        before_market_close_sec = obs[-4]
        pos = obs[-2]

        # 获取动作
        action = t['acts']

        if not hasattr(self, 'obs_date') or self.obs_date != obs_date or self.obs_symbol != obs_symbol:
            # 重新设置
            self.obs_date = obs_date
            self.obs_symbol = obs_symbol

            # 步数统计
            self.steps = 0
            self.recent_reward = 0

            # 设置 data_producer
            date = days2date(int(obs_date))
            symbol = USE_CODES[int(obs_symbol)]
            # 设置 data_producer files
            _date_file = f'{date}.pkl'
            if os.path.exists(os.path.join(self.data_producer.data_folder, 'train', _date_file)):
                _data_file_path = os.path.join(self.data_producer.data_folder, 'train', _date_file)
            elif os.path.exists(os.path.join(self.data_producer.data_folder, 'val', _date_file)):
                _data_file_path = os.path.join(self.data_producer.data_folder, 'val', _date_file)
            elif os.path.exists(os.path.join(self.data_producer.data_folder, 'test', _date_file)):
                _data_file_path = os.path.join(self.data_producer.data_folder, 'test', _date_file)
            else:
                raise ValueError(f'{_date_file} not in {self.data_producer.data_folder}')
            self.data_producer.files = [_data_file_path]
            # 设置 data_producer use_symbols
            self.data_producer.use_symbols = [symbol]
            # 设置 data_producer begin_before_market_close_sec
            self.data_producer.begin_before_market_close_sec = before_market_close_sec * MAX_SEC_BEFORE_CLOSE

            _obs, _info = self.reset(acc_status=pos)
            assert np.array_equal(obs, _obs), f'obs数据不一致，请检查!!!'

            if expert is not None:
                _act = expert.get_action(_obs)
                expert.add_potential_data_to_env(self)

        # 走一步
        _obs, _reward, _terminated, _truncated, _info = self.step(action)

        assert np.array_equal(t['next_obs'], _obs), f'obs数据不一致，请检查!!!'
        return _terminated or _truncated

    def render(self):
        if self.render_mode != 'human':
            return
        
        need_render = (self.steps+1) % self.render_freq == 0
        
        # 获取绘图数据
        if need_render:
            # 包含列名: index(ts), bid, ask, mid_price
            a, b, latest_tick_time = self.data_producer.get_plot_data()
        else:
            a, b, latest_tick_time = 0, 0, 0
        latest_net_raw, status = self.acc.get_plot_data()

        # 将数据放入队列，交给更新线程处理
        action = self._render.handle_data({"render": (a, b, latest_tick_time, latest_net_raw, status, need_render, self.recent_reward, self.done, self.acc.init_status)})
        if self.human_play:
            return action

def test_quick_produce_train_sdpk(date, code):
    """
    快速生成训练数据，用于检查对比
    """
    from feature.features.time_point_data import read_sdpk

    begin_t = '09:30'
    end_t = '14:59:50'
    def get_sdpk(date, code, level=5):
        file = rf"D:\L2_DATA_T0_ETF\his_data\{date}\{code}\十档盘口.csv"

        # 使用标准时间api
        # sdpk = read_sdpk(file).iloc[1:]

        # 简单读取原始数据
        sdpk = pd.read_csv(file, encoding='gbk')
        # 删除完全重复的行
        sdpk = sdpk.drop_duplicates(keep='first')
        # 删除列 '卖1价' 和 '买1价' 中存在 NaN 值的行
        sdpk = sdpk.dropna(subset=['卖1价', '买1价'])
        sdpk['时间'] = pd.to_datetime(sdpk['时间'])
        # 格式化
        for col in ['总卖', '总买']:
            try:
                sdpk[col] = sdpk[col].astype(float)
            except:
                sdpk[col] = sdpk[col].apply(
                    lambda x: 10000 * (float(x.replace("万", "")))
                    if "万" in str(x)
                    else 1e8 * (float(x.replace("亿", "")))
                    if "亿" in str(x)
                    else float(x)
                )
        sdpk = sdpk.set_index('时间')
        # 截取时间
        sdpk = sdpk[(sdpk.index.time >= pd.to_datetime(begin_t).time()) & (
            sdpk.index.time < pd.to_datetime(end_t).time())]
        sdpk = sdpk[(sdpk.index.time <= pd.to_datetime('11:30:00').time()) | (
            sdpk.index.time > pd.to_datetime('13:00:00').time())]
        sdpk = sdpk.iloc[1:]

        sdpk['id'] = [f"{code}_{int((x + pd.Timedelta(hours=-8)).timestamp())}" for x in sdpk.index]
        cols = [item for i in range(level) for item in [f'卖{i+1}价', f'卖{i+1}量', f'买{i+1}价', f'买{i+1}量']]
        cols.append('id')
        sdpk = sdpk.loc[:, cols]
        return sdpk
    
    # 获取标准化数据
    all_dates = [i for i in os.listdir(rf"D:\L2_DATA_T0_ETF\his_data") if len(i) == 8]
    all_dates.sort()
    cur_idx = all_dates.index(date)
    std_dates = all_dates[cur_idx-5:cur_idx]
    std_sdpks = pd.DataFrame()
    for std_date in std_dates:
        sdpk = get_sdpk(std_date, code, 5)
        std_sdpks = pd.concat([std_sdpks, sdpk])
    # 计数标准化数据
    std_sdpks = std_sdpks.iloc[:, :-1]
    base_price_col_nums = [i for i in range(len(std_sdpks.columns)) if '价' in std_sdpks.columns[i]]
    base_vol_col_nums = [i for i in range(len(std_sdpks.columns)) if '量' in std_sdpks.columns[i]]
    base_price_data = std_sdpks.iloc[:, base_price_col_nums]
    base_vol_data = std_sdpks.iloc[:, base_vol_col_nums]
    base_price_mean = np.mean(base_price_data.values)
    base_price_std = np.std(base_price_data.values)
    base_vol_mean = np.mean(base_vol_data.values)
    base_vol_std = np.std(base_vol_data.values)

    # 获取当日数据
    sdpk = get_sdpk(date, code).reset_index(drop=True)

    # 合并均值和标准差
    all_col_mean_list = []
    all_col_std_list = []
    for i in list(sdpk)[:-1]:
        if '价' in i:
            all_col_mean_list.append(base_price_mean)
            all_col_std_list.append(base_price_std)
        else:
            all_col_mean_list.append(base_vol_mean)
            all_col_std_list.append(base_vol_std)
    index = sdpk.columns[:-1]
    mean = pd.Series(all_col_mean_list, index=index)
    std = pd.Series(all_col_std_list, index=index)

    return sdpk, mean, std

def check(obs, info, data, mean, std):
    # 切片 检查用data
    _id = info['id']
    code, ts = _id.split('_')
    dt = pd.to_datetime(int(ts), unit='s') + pd.Timedelta(hours=8)
    _idx = data[data['id'] >= _id].index.to_list()[0]
    # 获取该id前10个数据
    start_idx = max(0, _idx - 9)
    data_slice = data.iloc[start_idx:_idx+1, :-1].values
    
    # 对比 检查用data 和 obs
    obs_data = obs[:200].reshape((10, 20))  
    if not np.array_equal(obs_data, data_slice):
        print('obs_data 和 data_slice 不一致')
        diff_mask = data_slice != obs_data
        diff_indices = np.where(diff_mask)
        idxs = []
        for i in range(len(diff_indices[0])):
            idxs.append((diff_indices[0][i], diff_indices[1][i]))
        print(idxs)
        raise Exception(f'obs_data 和 data_slice 不一致, id: {_id}')

    # 对比 mean, std 和 info
    info_mean = info['x_std'][:, 0]
    info_std = info['x_std'][:, 1]
    data_mean = mean.values
    data_std = std.values
    if not np.array_equal(info_mean, data_mean):
        print('info_mean 和 data_mean 不一致')
        raise Exception(f'info_mean 和 data_mean 不一致, id: {_id}')
    if not np.array_equal(info_std, data_std):
        print('info_std 和 data_std 不一致')
        raise Exception(f'info_std 和 data_std 不一致, id: {_id}')

    return True

def test_lob_data(check_data = True, check_reward = True):
    from tqdm import tqdm
    from dl_helper.rl.rl_env.lob_trade.lob_env_reward import BlankRewardStrategy

    code = '513050'
    max_drawdown_threshold = 0.01
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
    data_std=False,
    # debug_date=['20240521'],
    )

    if check_reward:
        from dl_helper.rl.rl_env.lob_trade.lob_env_checker import LobEnvChecker
        checker = LobEnvChecker(env, max_drawdown_threshold=max_drawdown_threshold)

    acts = ([ACTION_BUY]* 30 + [ACTION_SELL] * 30) * 3
    act_idx = 0

    # for i in tqdm(range(5000)):
    for i in range(1):
        print(f'iter: {i}, begin')
        obs, info = env.reset(seed=5)
        if check_reward:
            checker.reset()
        if check_data:
            cur_date = env.data_producer.date
            data, mean, std = test_quick_produce_train_sdpk(cur_date, code)
            check(obs, info, data, mean, std)

        while True:
            act = acts[act_idx] if act_idx < len(acts) else ACTION_SELL
            act_idx += 1
            obs, reward, terminated, truncated, info = env.step(act)
            if check_reward:
                checker.step(act, reward, info)
            if check_data:
                check(obs, info, data, mean, std)
            if terminated or truncated:
                break
            
        print(f'iter: {i}, end')

    print('all done')

def play_lob_data(render=True):
    from tqdm import tqdm

    code = '513050'
    env = LOB_trade_env({
        # 'data_type': 'val',# 训练/测试
        'data_type': 'train',# 训练/测试
        'his_len': 10,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        'use_symbols': [code],

        'train_folder': r'C:\Users\lh\Desktop\temp\lob_env',
        'train_title': 'test',

        'render_mode': 'human',
    },
    data_std=False,
    debug_date=['20240521'],
    )

    act_dict = {}
    act_dict = {
        '2024/5/21 09:37:05':1,
        '2024/5/21 09:41:11':0,
        '2024/5/21 10:19:50':1,
        '2024/5/21 10:27:53':0,
        '2024/5/21 10:51:29':1,
        '2024/5/21 11:07:14':0,
    }

    act_dict = {
        datetime.datetime.strptime(key, '%Y/%m/%d %H:%M:%S'): value
        for key, value in act_dict.items()
    }

    print('reset')
    obs, info = env.reset()

    if not act_dict:
        # 保存数据文件 
        data = env.data_producer.all_raw_data
        data.to_csv(r'C:\Users\lh\Desktop\temp\lob_data.csv', encoding='gbk')

        # # notebook 打开快速查找买卖点

        # import pandas as pd
        # import plotly.express as px

        # d = pd.read_csv(r"C:\Users\lh\Desktop\temp\lob_data.csv", encoding='gbk').loc[:, ['时间', 'BASE卖1价','BASE买1价']]
        # d['mid'] = (d['BASE卖1价'] + d['BASE买1价']) / 2
        # d = d.set_index('时间')

        # # 使用 Plotly 绘制交互式折线图
        # fig = px.line(d, x=d.index, y='mid', title='MID 值随时间变化',
        #             labels={'mid': 'MID 值', '时间': '时间'})

        # # 显示图表
        # fig.show()

        return

    dt= env.data_producer.step_use_data.iloc[-1].name
    if render:
        env.render()

    act = 1
    need_close = False
    while not need_close:
        if dt in act_dict:
            act = act_dict[dt]
        obs, reward, terminated, truncated, info = env.step(act)
        dt= env.data_producer.step_use_data.iloc[-1].name
        if render:
            env.render()
        need_close = terminated or truncated
        if render:
            time.sleep(0.1)
        
    env.close()
    print('all done')


if __name__ == '__main__':
    # test_quick_produce_train_sdpk('20250303', '513050')
    # test_lob_data(check_data=True, check_reward=False)
    # play_lob_data(render = True)
    pass