import os, time
import random
import datetime
import numpy as np
import pandas as pd
import gymnasium as gym
import gymnasium.spaces as spaces
import pickle
from multiprocessing import Process, Queue
import queue
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pytz
from matplotlib.widgets import Button
from py_ext.tool import log, init_logger,get_exception_msg, get_log_file
from py_ext.datetime import beijing_time
from py_ext.wechat import send_wx

from dl_helper.tool import calc_sharpe_ratio, calc_sortino_ratio, calc_drawdown, calc_return, calc_drawup_ticks, max_profit_reachable
from dl_helper.train_param import in_kaggle

from dl_helper.rl.rl_env.lob_trade.lob_env_data_augmentation import random_his_window
from dl_helper.rl.rl_env.lob_trade.lob_env_reward import ClosePositionRewardStrategy, HoldPositionRewardStrategy, NoPositionRewardStrategy, BlankRewardStrategy, RewardCalculator

USE_CODES = [
    '513050',
    '513330',
    '518880',
    '159941',
    '513180',
    '159920',
    '513500',
    '513130',
    '159792',
    '513100',
    '159937',
    '510900',
    '513060',
    '159934',
    '159509',
    '159632',
    '159605',
    '513010',
    '159513',
    '513120',
    '159501',
    '518800',
    '513300',
    '513660',
    '513090',
    '513980',
    '159892',
    '159740',
    '159636',
    '159659',
]
MEAN_CODE_ID = np.mean(np.arange(len(USE_CODES)))
STD_CODE_ID = np.std(np.arange(len(USE_CODES)))
MAX_CODE_ID = len(USE_CODES) - 1

STD_REWARD = 100
FINAL_REWARD = STD_REWARD * 1000

# 时间标准化
MEAN_SEC_BEFORE_CLOSE = 10024.17
STD_SEC_BEFORE_CLOSE = 6582.91
MAX_SEC_BEFORE_CLOSE = 5.5*60*60

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

            # 数据增强
            use_random_his_window=False,
            use_time_scaling=False,
            use_gaussian_noise=False,
            use_impact_noise=False,
            use_progressive_mask=False,
        ):
        """
        'data_type': 'train',# 训练/测试
        'his_len': 100,# 每个样本的 历史数据长度
        'simple_test': False,# 是否为简单测试
        'need_cols': [],# 需要的特征列名
        'use_symbols': []# 只使用指定的标的
        'use_random_his_window': True,# 随机窗口开关
        'use_time_scaling': True,# 时间缩放开关
        'use_gaussian_noise': True,# 高斯噪声开关
        'use_impact_noise': True,# 冲击噪声开关
        'use_progressive_mask': True,# 渐进式遮挡开关
        """
        # 数据增强
        self.use_random_his_window = use_random_his_window  # 随机窗口开关
        self.use_time_scaling = use_time_scaling   # 时间缩放开关
        self.use_gaussian_noise = use_gaussian_noise  # 高斯噪声开关
        self.use_impact_noise = use_impact_noise   # 冲击噪声开关
        self.use_progressive_mask = use_progressive_mask  # 渐进式遮挡开关

        # 快速测试
        self.simple_test = simple_test

        self.his_len = his_len
        self.data_std = data_std
        self.save_folder = save_folder
        self.debug_date = [i.replace('-', '').replace(' ', '') for i in debug_date]
        if self.debug_date:
            log(f'[{data_type}] debug_date: {self.debug_date}')

        self.use_symbols = use_symbols
        
        # 需要的特征列名
        self.need_cols = need_cols
        self.need_cols_idx = []
        # 添加必须的列
        # if self.need_cols:
        #     for must_col in ['mid_pct', 'mid_price', 'mid_vol']:
        #         if must_col not in self.need_cols:
        #             self.need_cols.append(must_col)

        self.cols_num = 130 if not self.need_cols else len(self.need_cols)

        # 训练数据
        if in_kaggle:
            input_folder = r'/kaggle/input'
            try:
                # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
                data_folder_name = os.listdir(input_folder)[0]
                self.data_folder = os.path.join(input_folder, data_folder_name)
            except:
                self.data_folder = r''
        else:
            self.data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\RL_10level_20250316'

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
            self.files = os.listdir(os.path.join(self.data_folder, self.data_type))
            self.files.sort()
            if self.data_type == 'train':
                random.shuffle(self.files)
            if self.debug_date:
                # 按照 debug_date 的顺序重新排列文件
                ordered_files = []
                for debug_date in self.debug_date:
                    for file in self.files:
                        if file.split('.')[0] == debug_date:
                            ordered_files.append(file)
                self.files = ordered_files
            log(f'[{self.data_type}] prepare files: {self.files}')
            
    def _load_data(self):
        """
        按照文件列表读取数据
        每次完成后从文件列表中剔除
        """
        while (self.files) or (self.cur_data_type != self.data_type):
            # 更新在用数据类型
            self.cur_data_type = self.data_type

            self.cur_data_file = self.files.pop(0)
            log(f'[{self.data_type}] load date file: {self.cur_data_file}')
            self.ids, self.mean_std, self.x, self.all_raw_data = pickle.load(open(os.path.join(self.data_folder, self.data_type, self.cur_data_file), 'rb'))
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
            unique_symbols = [i for i in np.unique(symbols) if i != '159941']

            if self.use_symbols:
                unique_symbols = [i for i in unique_symbols if i in self.use_symbols]

            # 获取所有标的的起止索引
            self.idxs = []
            for symbol in unique_symbols:
                symbol_mask = symbols == symbol
                symbol_indices = np.where(symbol_mask)[0]
                self.idxs.append([symbol_indices[0], symbol_indices[-1], USE_CODES.index(symbol)])

            if not self.idxs:
                log(f'[{self.data_type}] no data for date: {self.date}' + '' if not self.use_symbols else f', symbols: {self.use_symbols}')
                continue

            # 训练数据随机选择一个标的
            # 一个日期文件只使用其中的一个标的的数据，避免同一天各个标的之间存在的相关性 对 训练产生影响
            if self.data_type == 'train':
                self.idxs = [random.choice(self.idxs)]

            log(f'[{self.data_type}] init idxs: {self.idxs}')

            # 调整数据
            # fix 在某个时点上所有数据都为0的情况，导致模型出现nan的bug
            all_cols = list(self.all_raw_data)
            if 'OBC买10量' in all_cols and 'OSC卖10量' in all_cols:
                # 订单数据
                order_cols = [i for i in all_cols if i.startswith('OS') or i.startswith('OB')]
                order_raw = self.all_raw_data.loc[:, order_cols]
                self.all_raw_data.loc[(order_raw == 0).all(axis=1), ['OBC买10量', 'OSC卖10量']] = 1
            if 'OF买10量' in all_cols and 'OF卖10量' in all_cols:
                # OF数据
                OF_cols = [i for i in all_cols if i.startswith('OF')]
                OF_raw = self.all_raw_data.loc[:, OF_cols]
                self.all_raw_data.loc[(OF_raw == 0).all(axis=1), ['OF买10量', 'OF卖10量']] = 1
            if '卖10量' in all_cols and '卖10价' not in all_cols:
                # 深度数据
                depth_cols = ['卖10量',
                    '卖9量',
                    '卖8量',
                    '卖7量',
                    '卖6量',
                    '卖5量',
                    '卖4量',
                    '卖3量',
                    '卖2量',
                    '卖1量',
                    '买1量',
                    '买2量',
                    '买3量',
                    '买4量',
                    '买5量',
                    '买6量',
                    '买7量',
                    '买8量',
                    '买9量',
                    '买10量']
                depth_raw = self.all_raw_data.loc[:, depth_cols]
                wait_fix_index = depth_raw[(depth_raw == 0).all(axis=1)].index.to_list()
                if wait_fix_index and wait_fix_index[0] == 0:
                    # 若第一个数据就为0，填充 卖10量/买10量 为1，最小化影响
                    self.all_raw_data.loc[0, '卖10量'] = 1
                    self.all_raw_data.loc[0, '买10量'] = 1
                    # 去掉第一个记录
                    wait_fix_index = wait_fix_index[1:]

                self.all_raw_data.loc[wait_fix_index, depth_cols] = np.nan# 先用nan填充，方便后续处理
                for col in depth_cols:
                    self.all_raw_data[col] = self.all_raw_data[col].ffill()
            if 'DB卖1量' in all_cols and 'DS买1量' in all_cols: 
                # 成交数据
                deal_cols = [i for i in all_cols if i.startswith('D')]
                deal_raw = self.all_raw_data.loc[:, deal_cols]
                self.all_raw_data.loc[(deal_raw == 0).all(axis=1), ['DB卖1量', 'DS买1量']] = 1
            # 40档位价量数据nan处理
            if 'BASE卖1量' in all_cols and 'BASE买1量' in all_cols:
                # 价格nan填充, 使用上一个档位数据 +-0.001 进行填充
                for i in range(2, 11):
                    if f'BASE买{i}价' not in all_cols or f'BASE买{i-1}价' not in all_cols:
                        continue

                    # 买价
                    self.all_raw_data.loc[:, f'BASE买{i}价'] = self.all_raw_data[f'BASE买{i}价'].fillna(self.all_raw_data[f'BASE买{i-1}价'] - 0.001)

                    # 卖价
                    self.all_raw_data.loc[:, f'BASE卖{i}价'] = self.all_raw_data[f'BASE卖{i}价'].fillna(self.all_raw_data[f'BASE卖{i-1}价'] + 0.001)

                # 量nan用0填充
                vol_cols = [i for i in list(self.all_raw_data) if i.startswith('BASE') and '价' not in i]
                self.all_raw_data[vol_cols] = self.all_raw_data[vol_cols].fillna(0)

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

    def get(self):
        """
        输出观察值
            返回 symbol_id, before_market_close_sec, x, need_close, self.id

            若data_std=False
            返回 symbol_id, before_market_close_sec, x, need_close, _id, x_std, sec_std, id_std
        """
        # # 测试用
        # print(self.idxs[0])

        # 记录数据索引
        self.last_data_idx = self.idxs[0][0]

        # 准备观察值
        a, b = self.x[self.idxs[0][0]]
        self.step_use_data = self.all_raw_data.iloc[max(b - int(self.his_len * 1.5), a): b, :].copy()# 多截取 50% 数据，用于增强 TODO copy 是否必要
        raw_0 = self.step_use_data.copy().values# 多截取 50% 数据，用于增强 TODO copy 是否必要

        # 原始数据增强
        # 1. use_random_his_window 从更长的历史窗口（如1ls5个）随机截取 所需的时间窗口的数据
        if self.use_random_his_window and self.data_type == 'train':
            raw_0 = random_his_window(raw_0, self.his_len)
        # 2. use_time_scaling 时间缩放：对时间轴进行非线性缩放，例如加速或减速某些时段的价格变化，以模拟市场的波动性变化。

        # 修正历史数据长度
        raw = raw_0[-self.his_len:]
        # 记录 买卖1档 的价格
        self.store_bid_ask_1st_data(raw)

        # 截断数据后的增强
        # 1. use_gaussian_noise 添加高斯噪声，在买卖五档价格和成交量上添加符合正态分布的小幅随机噪声，模拟市场中的微小波动
        # 2. use_impact_noise 添加冲击噪声，随机增减某些档位数量 ±5%
        # 3. use_progressive_mask 按概率逐步遮挡历史数据（如越早的时间点遮挡概率越高），模拟渐进式信息衰减。

        # 数据标准化
        std_data = self.mean_std[self.idxs[0][0]]
        # 未实现收益率 使用 zscore
        unrealized_log_return_std_data = std_data['unrealized_log_return']['zscore']
        # 价格量 使用 robust
        ms = pd.DataFrame(std_data['price_vol_each']['robust'])
        x, ms = self.use_data_split(raw, ms)

        if self.data_std:
            x -= ms[:, 0]
            x /= ms[:, 1]
        else:
            x_std = ms.copy()

        # 标的id
        symbol_id = self.idxs[0][2]

        # 当前标的
        self.code = USE_CODES[int(symbol_id)]

        # 距离市场关闭的秒数
        before_market_close_sec = self.before_market_close_sec[self.idxs[0][0]]

        # 记录数据id
        self.id = self.ids[self.idxs[0][0]]

        # 检查本次数据是否是最后一个数据
        need_close = False
        if self.idxs[0][0] == self.idxs[0][1]:
            # 当组 begin/end 完成，需要平仓
            need_close = True
            log(f'[{self.data_type}] need_close {self.idxs[0][0]} {self.idxs[0][1]}')
            # 更新剩余的 begin/end 组
            self.idxs = self.idxs[1:]
            log(f'[{self.data_type}] idxs: {self.idxs}')
            if not self.idxs:
                # 当天的数据没有下一个可读取的 begin/end 组
                log(f'[{self.data_type}] date file done')
            else:
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

            # id
            # ZSCORE
            # symbol_id -= MEAN_CODE_ID
            # symbol_id /= STD_CODE_ID
            # 归一化
            symbol_id /= MAX_CODE_ID
            return symbol_id, before_market_close_sec, x, need_close, self.id, unrealized_log_return_std_data
        else:
            sec_std = (MEAN_SEC_BEFORE_CLOSE, STD_SEC_BEFORE_CLOSE, MAX_SEC_BEFORE_CLOSE)
            id_std = (MEAN_CODE_ID, STD_CODE_ID, MAX_CODE_ID)
            return symbol_id, before_market_close_sec, x, need_close, self.id, unrealized_log_return_std_data, x_std, sec_std, id_std

    def get_plot_data(self):
        """
        获取绘图数据,
        返回 绘图范围 a, b, 当前时点的时间索引
        """
        a, b = self.x[self.last_data_idx]
        a = max(b - self.his_len, a)

        extra_data_length = 50
        extra_a, extra_b = a - extra_data_length, b + extra_data_length

        return max(0, extra_a), min(len(self.plot_data), extra_b), self.plot_data.iloc[b-1].name

    def reset(self):
        while True:
            self._pre_files()
            self._load_data()
            if not self.idxs:
                # 没有成功加载到可用数据，需要重新准备文件
                # 当天数据中没有指定标的
                log(f'[{self.data_type}] no data for date: {self.date}' + '' if not self.use_symbols else f', symbols: {self.use_symbols}, will repre files')
                continue
            break

        self.last_data_idx = None
        self.bid_price = []
        self.ask_price = []

class Account:
    """
    账户类，用于记录交易状态和计算收益
    """
    fee_rate = 5e-5

    def __init__(self):
        # 持仓状态 
        self.status = 0
        # 是否初始化了策略净值
        self.init_net_raw = False
        # 持仓量
        self.pos = 0
        self.cash = 0
        # 持仓时间
        self.hold_length = 0
        # 净值序列
        self.net_raw = []
        self.net_raw_bm = []# 基准，一直持有
        self.net_raw_last_change = []# 最近一次开仓策略净值

    def step(self, bid_price, ask_price, action):
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
            assert not self.net_raw_last_change, f'status:{self.status} net_raw_last_change:{self.net_raw_last_change}'
            assert self.pos == 0 and (self.cash >0 or not self.init_net_raw), f'status:{self.status} cash:{self.cash} pos:{self.pos}'

        # 持仓时间
        if self.status == 1:
            self.hold_length += 1

        # 基于最新数据 更新净值
        # 转为现金的价值
        net_bm = bid_price * (1 - Account.fee_rate)
        self.net_raw_bm.append(net_bm)
        # 初始化策略净值（与基准净值一致）
        if len(self.net_raw) == 0:
            net = self.cash = self.net_raw_bm[0]# 现金
            self.init_net_raw = True
        else:
            net = self.cash + self.pos * bid_price * (1 - Account.fee_rate)
        self.net_raw.append(net)
        if self.status == 1:
            self.net_raw_last_change.append(net)

        # 计算持仓对数收益率(最近一次的在持仓位)
        unrealized_return = 0
        if self.status == 1:
            # 未实现收益需考虑卖出时的手续费
            unrealized_return = np.log(self.net_raw_last_change[-1] / self.net_raw_last_change[0])

        # 日内收益率
        inday_return = np.log(self.net_raw[-1] / self.net_raw[0])

        # 本次操作平仓的净值序列
        close_net_raw = []

        act_result = -1 # 无
        if action == 1:  # 持仓动作
            if self.status == 0:  # 空仓可以买入
                self.status = 1
                self.hold_length += 1

                # 更新 pos/cash
                self.pos = self.cash / (ask_price * (1 + Account.fee_rate))
                self.cash = 0

                # 更新净值
                self.net_raw[-1] = self.pos * bid_price * (1 - Account.fee_rate)

                # 新的一次持仓序列
                self.net_raw_last_change.append(self.net_raw[-1])

                # 开仓
                act_result = 1
                
        elif action == 0:  # 空仓动作
            if self.status == 1:  # 有多仓可以卖出
                self.status = 0
                self.hold_length -= 1# 时刻开始就卖出了，需要把多加的剪掉

                # 备份重置 net_raw_last_change
                close_net_raw = self.net_raw_last_change
                self.net_raw_last_change = []

                # 更新 pos/cash
                self.cash = self.net_raw[-1]
                self.pos = 0

                # 平仓
                act_result = 0

        # print("acc::step", action, self.status, self.pos, self.cash)
        return self.status, inday_return, unrealized_return, close_net_raw, act_result

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
        
    def reset(self):
        """
        重置账户状态
        """
        self.status = 0
        self.pos = 0
        self.cash = 0
        self.hold_length = 0
        self.net_raw = []
        self.net_raw_bm = []
        self.net_raw_last_change = []
        self.init_net_raw = False

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
        
class LOB_trade_env(gym.Env):
    """
    用于 LOB 的强化学习环境
    返回的 obs 结构:
        lob数据 + 持仓 + 未实现收益率
    """

    REG_NAME = 'lob'
    ITERATION_DONE_FILE = os.path.join(os.path.expanduser('~'), '_lob_env_iteration_done')
    
    def __init__(self, config: dict, data_std=True, debug_date=''):
        """
        :param config: 配置
            {
                # 用于实例化 数据生产器
                'data_type': 'train'/'val'/'test',# 训练/测试
                'his_len': 100,# 每个样本的 历史数据长度
                'simple_test': False,# 是否为简单测试
                'need_cols': [],# 需要读取的列
                'use_symbols': [],# 只使用某些标的
                'use_random_his_window': False,# 是否使用随机历史窗口
                'use_time_scaling': False,# 是否使用时间缩放
                'use_gaussian_noise': False,# 是否使用高斯噪声
                'use_impact_noise': False,# 是否使用冲击噪声
                'use_progressive_mask': False,# 是否使用渐进式掩码

                # 终止游戏的回撤阈值
                'max_drawdown_threshold': 0.005,# 最大回测阈值

                # 渲染模式
                'render_mode': 'none',

                # 渲染频率, 每N步渲染一次
                'render_freq': 5,

                # 奖励策略
                'end_position': ClosePositionRewardStrategy,
                'close_position': ClosePositionRewardStrategy,
                'open_position_step': BlankRewardStrategy,
                'hold_position': HoldPositionRewardStrategy,
                'no_position': NoPositionRewardStrategy,
            }
        """
        super().__init__()

        self.data_type = config.get('data_type', 'train')
        self.his_len = config.get('his_len', 100)
        self.simple_test = config.get('simple_test', False)
        self.need_cols = config.get('need_cols', [])
        self.use_symbols = config.get('use_symbols', [])
        self.use_random_his_window = config.get('use_random_his_window', False)
        self.use_time_scaling = config.get('use_time_scaling', False)
        self.use_gaussian_noise = config.get('use_gaussian_noise', False)
        self.use_impact_noise = config.get('use_impact_noise', False)
        self.use_progressive_mask = config.get('use_progressive_mask', False)

        self.max_drawdown_threshold = abs(config.get('max_drawdown_threshold', 0.005))

        self.render_mode = config.get('render_mode', 'none')
        self.render_freq = config.get('render_freq', 5)

        end_position_reward_strategy = config.get('end_position', ClosePositionRewardStrategy)
        close_position_reward_strategy = config.get('close_position', ClosePositionRewardStrategy)
        open_position_step_reward_strategy = config.get('open_position_step', BlankRewardStrategy)
        hold_position_reward_strategy = config.get('hold_position', HoldPositionRewardStrategy)
        no_position_reward_strategy = config.get('no_position', NoPositionRewardStrategy)
        force_stop_reward_strategy = config.get('force_stop', BlankRewardStrategy)

        # 保存文件夹
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
        self.debug_date = debug_date
        if self.debug_date and isinstance(self.debug_date, str):
            self.debug_date = [self.debug_date]

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
            use_random_his_window=self.use_random_his_window,
            use_time_scaling=self.use_time_scaling,
            use_gaussian_noise=self.use_gaussian_noise,
            use_impact_noise=self.use_impact_noise,
            use_progressive_mask=self.use_progressive_mask,
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
        # 持仓/空仓
        self.action_space = spaces.Discrete(2)

        # 观察空间 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data_producer.data_size() + 5,), dtype=np.float32)

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

        # 渲染模式
        if self.render_mode == 'human':
            self.input_queue = Queue(maxsize=30)  # 创建多进程队列用于传递数据
            self.update_process = Process(target=self.update_plot, args=(self.input_queue,), daemon=True)
            self.update_process.start()  # 启动更新进程

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
        self.need_upload_file = os.path.join(self.save_folder, self.data_producer.data_type, f'{id(self)}_{self.iteration}.csv')
        
    def _set_data_type(self, data_type):
        if self.data_producer.data_type != data_type:
            log(f'[{id(self)}][{self.data_producer.data_type}] set data type: {data_type}')
            self.data_producer.set_data_type(data_type)
            self.need_reset = True
            self.sample_count = 0
            self.update_need_upload_file()

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
            symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data, x_std, sec_std, id_std = self.data_producer.get()
            x = x.reshape(-1)
            return symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data, x_std, sec_std, id_std

    def _cal_reward(self, action, need_close, info):
        """
        计算奖励
        
        act_criteria 动作评价:
            0 交易获利
            1 交易亏损
            2 触发止损

        act_result: 动作结果
            0 平仓
            1 开仓
            -1 无
        """
        # 游戏是否终止
        # 1. 最大回测超过阈值                           -STD_REWARD
        acc_done = False

        pos, inday_return, unrealized_return, close_net_raw, act_result = self.acc.step(self.data_producer.bid_price[-1], self.data_producer.ask_price[-1], action)
        log(f'[{id(self)}][{self.data_producer.data_type}][{self.data_producer.step_use_data.iloc[-1].name}] act: {action}, act_result: {act_result}, last_close_idx: {self.last_close_idx}')
        need_cal_drawup = pos == 0 and act_result == -1
        res = self.acc.cal_res(self.acc.net_raw, self.acc.net_raw_bm, self.last_close_idx, need_cal_drawup)
        res['hold_length'] = self.acc.hold_length
        
        # 记录net/net_bm
        res['net'] = self.acc.net_raw[-1] if self.acc.net_raw else np.nan
        res['net_bm'] = self.acc.net_raw_bm[-1] if self.acc.net_raw_bm else np.nan

        #数据类型
        res['data_type'] = self.data_producer.data_type

        reward = 0

        # 同时记录评价指标
        if need_close or act_result==0:
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
            if need_close or act_result==0:
                if act_result==0:
                    # 交易对数
                    self.trades += 1

                if not need_close:
                    # 潜在收益率（针对最近的交易序列）
                    _, max_profit_reachable_bm, _, _ = max_profit_reachable(self.data_producer.bid_price[self.last_close_idx:], self.data_producer.ask_price[self.last_close_idx:])
                    acc_return = np.log(close_net_raw[-1]) - np.log(close_net_raw[0])
                else:
                    # 游戏完成，使用所有数据完整计算
                    _, max_profit_reachable_bm, _, _ = max_profit_reachable(self.data_producer.bid_price, self.data_producer.ask_price)
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
        0 空仓
        1 持仓
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
                symbol_id, before_market_close_sec, observation, need_close, _id, unrealized_log_return_std_data, x_std, sec_std, id_std = self._get_data()

            info = {
                'id': _id,
                'action': action,
            }

            if not self.data_std:
                info['x_std'] = x_std
                info['sec_std'] = sec_std
                info['id_std'] = id_std

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

            # 记录平仓
            if act_result==0:
                self.last_close_idx = self.steps

            # 准备输出数据
            # net,net_bm,
            out_text += f",{info['net']},{info['net_bm']}"
            # reward,max_drawdown,max_drawdown_ticks,trade_return,step_return,hold_length,
            out_text2 = f",{reward},{info.get('max_drawdown', '')},{info.get('max_drawdown_ticks', '')},{info.get('trade_return', '')},{info.get('step_return', '')},{info.get('hold_length', '')},{info.get('potential_return', '')},{info.get('acc_return', '')}"
            # max_drawdown_bm,max_drawdown_ticks_bm,max_drawup_ticks_bm,drawup_ticks_bm_count,trade_return_bm,step_return_bm,
            out_text2 += f",{info.get('max_drawdown_bm', '')},{info.get('max_drawdown_ticks_bm', '')},{info.get('max_drawup_ticks_bm', '')},{info.get('drawup_ticks_bm_count', '')},{info.get('trade_return_bm', '')},{info.get('step_return_bm', '')}"

            # 标准化 未实现收益率
            unrealized_return = (unrealized_return - unrealized_log_return_std_data[0]) / unrealized_log_return_std_data[1]
            inday_return = (inday_return - unrealized_log_return_std_data[0]) / unrealized_log_return_std_data[1]

            # 添加 静态特征
            observation = np.concatenate([observation, [before_market_close_sec, symbol_id, pos, inday_return, unrealized_return]])

            # 检查是否结束
            terminated = acc_done or need_close# 游戏终止
            # 环境中没有截断结束的情况
            truncated = False

            out_text += f",{terminated},{truncated}"
            out_text += out_text2
            # 记录数据
            self.out_test_predict(out_text)

            done = terminated or truncated
            if done:
                # 计算平均步数
                self.mean_episode_lengths = (self.mean_episode_lengths * self.episode_count + self.steps) / (self.episode_count + 1)
                self.episode_count += 1
                log(f'[{id(self)}][{self.data_producer.data_type}] episode {self.episode_count} done, mean episode length: {self.mean_episode_lengths}, latest episode length: {self.steps}')

            return observation, reward, terminated, truncated, info

        except Exception as e:
            log(f'[{id(self)}][{self.data_producer.data_type}] step error: {get_exception_msg()}')
            raise e

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed, options=options)

            # 步数统计
            self.steps = 0

            # 交易对数
            self.trades = 0

            # 重置累计收益率
            self.acc_return = 0
            
            # 重置
            self.need_reset = False

            # 最近一次平仓的idx
            self.last_close_idx = 0

            # 数据
            log(f'[{id(self)}][{self.data_producer.data_type}] reset')

            while True:
                self.data_producer.reset()
                if self.data_std:
                    symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data = self._get_data()
                else:
                    symbol_id, before_market_close_sec, x, need_close, _id, unrealized_log_return_std_data, x_std, sec_std, id_std = self._get_data()
                if need_close:
                    # 若是 val 数据，有可能 need_close 为True
                    # 需要过滤
                    log(f'[{id(self)}][{self.data_producer.data_type}] need_close: True, reset data_producer again')
                else:
                    break
            
            full_plot_data = self.data_producer.plot_data
            self.input_queue.put({'full_plot_data': full_plot_data})

            # 账户
            self.acc.reset()
            # 账户需要用动作0(空仓)走一步, 会初始记录act之前的净值
            pos, inday_return, unrealized_return, _, _ = self.acc.step(self.data_producer.bid_price[-1], self.data_producer.ask_price[-1], 0)
            # 标准化 未实现收益率
            unrealized_return = (unrealized_return - unrealized_log_return_std_data[0]) / unrealized_log_return_std_data[1]
            inday_return = (inday_return - unrealized_log_return_std_data[0]) / unrealized_log_return_std_data[1]

            # 添加 静态特征
            x = np.concatenate([x, [before_market_close_sec, symbol_id, pos, inday_return, unrealized_return]])

            # 记录静态数据，用于输出预测数据
            self.static_data = {
                'before_market_close_sec': before_market_close_sec,
                'pos': pos,
                'inday_return': inday_return,
                'unrealized_return': unrealized_return,
            }

            if self.data_std:
                return x, {'id': _id}
            else:
                return x, {'x_std': x_std, 'sec_std': sec_std, 'id_std': id_std, 'id': _id}
            
        except Exception as e:
            log(f'[{id(self)}][{self.data_producer.data_type}] reset error: {get_exception_msg()}')
            raise e

    def close(self):
        if self.render_mode == 'human':
            self.input_queue.put(None)
            self.update_process.join()

    def update_plot(self, input_queue):
        """线程中运行的图形更新函数"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.show(block=False)  # 添加这行，非阻塞方式显示图形

            std_data = {}
            full_plot_data = None
            net_raw = []

            while True:
                try:
                    # 从队列中获取数据，非阻塞方式
                    data = input_queue.get_nowait()
                    if None is data:
                        # 结束
                        return
                    if 'render' in data:
                        self._plot_data(fig, ax, *data['render'], std_data, full_plot_data, net_raw)
                    elif 'full_plot_data' in data:
                        full_plot_data = data['full_plot_data']
                        # 全reset
                        std_data = {}
                        net_raw = []
                except queue.Empty:
                    pass  # 队列为空时跳过
                # plt.pause(0.1)  # 短暂暂停以允许其他线程运行并更新图形
        except Exception as e:
            log(f'[{id(self)}][{self.data_producer.data_type}] update_plot error: {get_exception_msg()}')
            raise e

    @staticmethod
    def _plot_data(fig, ax, a, b, latest_tick_time, latest_net_raw, status, need_render, std_data, full_plot_data, net_raw):
        """绘制图形的具体实现"""
        # 更新净值
        net_raw.append(latest_net_raw)

        if not need_render:
            return

        # 清空当前的axes
        ax.clear()

        # 截取绘图数据
        plot_data = full_plot_data.iloc[a:b]

        # 确定历史数据和未来数据的分界点
        hist_end = plot_data.index.get_loc(latest_tick_time) + 1

        # 截取账户净值并标准化
        net_raw = net_raw[-hist_end:]
        net_raw = np.array(net_raw)

        n = len(plot_data)

        # 记录标准化数据
        if 'net_raw' not in std_data:
            std_data['net_raw'] = net_raw[0]
        # 标准化
        net_raw = net_raw / std_data['net_raw']

        # 对齐长度
        if len(net_raw) < hist_end:
            net_raw = np.concatenate([[net_raw[0]] * (hist_end - len(net_raw)), net_raw])

        # 记录标准化数据
        if 'mid_price' not in std_data:
            std_data['mid_price'] = plot_data['mid_price'].iloc[0]
        # 标准化成净值
        plot_data['mid_price'] = plot_data['mid_price'] / std_data['mid_price']

        # 
        ax.plot(range(hist_end), plot_data['mid_price'].iloc[:hist_end].values, label=f'mid_price({plot_data[f"mid_price"].iloc[hist_end - 1]:.6f})', color='blue', alpha=1)
        ax.plot(range(hist_end), net_raw, label=f'net({net_raw[hist_end - 1]:.6f})', color='red', alpha=1)
        ax.plot(range(hist_end-1, n),plot_data['mid_price'].iloc[hist_end-1:].values, color='blue', alpha=0.3)
        ax.legend(loc='upper left')
        status_str = f'持仓' if status == 1 else '空仓'
        ax.set_title(f'status: {status_str}')

        # 标注最新的历史数据时间
        time_str = plot_data.index[hist_end-1].strftime('%Y-%m-%d %H:%M:%S')
        ax.axvline(x=hist_end-1, color='gray', linestyle='--')
        ax.text(hist_end-1, ax.get_ylim()[0], time_str, rotation=90, verticalalignment='bottom')

        # 设置x轴标签
        ax.set_xlabel('Tick Index')

        # 调整布局并刷新图形
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        

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
        self.input_queue.put({"render": (a, b, latest_tick_time, latest_net_raw, status, need_render)})

def test_quick_produce_train_sdpk(date, code):
    """
    快速生成训练数据，用于检查对比
    """
    from feature.features.time_point_data import read_sdpk
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
        begin_t = '09:30'
        end_t = '14:57'
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
    begin_t = '09:30'
    end_t = '14:57'
    for std_date in std_dates:
        sdpk = get_sdpk(std_date, code, 10)
        # 截取时间
        sdpk = sdpk[(sdpk.index.time >= pd.to_datetime(begin_t).time()) & (
            sdpk.index.time < pd.to_datetime(end_t).time())]
        sdpk = sdpk[(sdpk.index.time <= pd.to_datetime('11:30:00').time()) | (
            sdpk.index.time > pd.to_datetime('13:00:00').time())]
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

def test_lob_data(check_data = True):
    from tqdm import tqdm
    from dl_helper.rl.rl_env.lob_trade.lob_env_reward import BlankRewardStrategy

    code = '513050'
    env = LOB_trade_env({
        # 'data_type': 'val',# 训练/测试
        'data_type': 'train',# 训练/测试
        'his_len': 10,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        'use_symbols': [code],

        'train_folder': r'C:\Users\lh\Desktop\temp\lob_env',
        'train_title': 'test',

        # 奖励策略
        'reward_strategy_class_dict': {
            'end_position': BlankRewardStrategy,
            'close_position': BlankRewardStrategy,
            'open_position_step': BlankRewardStrategy,
            'hold_position': BlankRewardStrategy,
            'no_position': BlankRewardStrategy
        }
    },
    data_std=False,
    debug_date=['20240521'],
    )

    acts = [0, 1, 0, 1, 0, 1]
    acts = [0] * 10# 违法操作, 多次买入
    acts = [0,1,2,0,1,2]# 合法操作
    act_idx = 0

    for i in tqdm(range(5000)):
        print(f'iter: {i}, begin')
        obs, info = env.reset()
        if check_data:
            cur_date = env.data_producer.date
            data, mean, std = test_quick_produce_train_sdpk(cur_date, code)
            check(obs, info, data, mean, std)

        while True:
            act = acts[act_idx] if act_idx < len(acts) else 2
            act_idx += 1
            obs, reward, terminated, truncated, info = env.step(act)
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
    # test_lob_data(check_data=False)
    play_lob_data(render = True)