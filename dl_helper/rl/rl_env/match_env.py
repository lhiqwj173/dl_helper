"""
配对交易的强化学习环境

状态:
    前5个交易日的价差           5
    最近10个tick的价差          10
    距离收盘的秒数              1
    仓位状态                    1
        -1 code1满仓
        0 均衡仓位
        1 code2满仓
    当天的超额对数收益率        1
    最近一次调仓的超额对数收益率 1
动作:
    -1 code1满仓
    0 均衡仓位
    1 code2满仓

每天都是均衡仓位开始交易

目的:
    最大化超额收益
"""

import os, time, json
import random
import datetime
import numpy as np
import pandas as pd
import gymnasium as gym
import gymnasium.spaces as spaces
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pytz
from matplotlib.widgets import Button
from py_ext.tool import log, init_logger,get_exception_msg, get_log_file
from py_ext.datetime import beijing_time
from py_ext.wechat import send_wx

from dl_helper.tool import calc_sharpe_ratio, calc_sortino_ratio, calc_drawdown, calc_return, calc_drawup_ticks, max_profit_reachable
from dl_helper.train_param import in_kaggle

from dl_helper.rl.rl_env.lob_env_data_augmentation import random_his_window
from dl_helper.rl.rl_env.match_env_reward import EndPositionRewardStrategy, ClosePositionRewardStrategy, HoldPositionRewardStrategy, BalanceRewardStrategy, RewardCalculator

STD_REWARD = 100

# 时间标准化
MAX_SEC_BEFORE_CLOSE = 5.5*60*60

class data_producer:
    """
    遍历日期文件，每天随机选择一个标的
    当天的数据读取完毕后，需要强制平仓
    """
    def __init__(
            self,
            data_type='train', 
            his_daily_len=5,
            his_tick_len=10,
            save_folder="", 
            debug_dates='',
        ):
        """
        'data_type': 'train',# 训练/测试
        'his_daily_len': 5,# 前5个交易日的价差
        'his_tick_len': 10,# 最近10个tick的价差
        'save_folder': '',# 保存路径
        """
        self.save_folder = save_folder
        self.his_daily_len = his_daily_len
        self.his_tick_len = his_tick_len
        self.debug_dates = debug_dates

        # 数据路径
        if in_kaggle:
            input_folder = r'/kaggle/input'
            try:
                # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
                data_folder_name = os.listdir(input_folder)[0]
                self.data_folder = os.path.join(input_folder, data_folder_name)
            except:
                self.data_folder = r''
        else:
            self.data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\RL_match'

        # 细分路径
        self.match_folder = os.path.join(self.data_folder, 'match_data')

        self.data_type = data_type
        self.cur_data_type = data_type
        self.files = []
        self.cur_data_file = ''

        # 数据内容
        self.daily_k_data = pd.read_csv(os.path.join(self.data_folder, 'daily_k.csv'))
        self.match_data = None
        self.tick_data = None
        # 最新的tick, 包含 code1_bid  code1_ask  code2_bid  code2_ask
        self.latest_tick = None

        # 配对标的
        self.code1, self.code2 = list(self.daily_k_data)[1:]

        # 当前数据索引
        self.idx = 0

        # id
        self.id = ''

        log(f'[{self.data_type}] data_producer init done')

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
            if self.debug_dates:
                self.files = self.debug_dates
            else:
                self.files = os.listdir(os.path.join(self.data_folder, self.data_type))
                self.files.sort()
                if self.data_type == 'train':
                    random.shuffle(self.files)

            log(f'[{self.data_type}] prepare files(head 5): {self.files[:5]}')
            
    def _load_data(self):
        """
        按照文件列表读取数据
        每次完成后从文件列表中剔除
        """
        while (self.files) or (self.cur_data_type != self.data_type):
            # 更新在用数据类型
            self.cur_data_type = self.data_type

            # 读取数据
            self.cur_data_file = self.files.pop(0)
            log(f'[{self.data_type}] load date file: {self.cur_data_file}')
            self.tick_data = pickle.load(open(os.path.join(self.data_folder, self.data_type, self.cur_data_file), 'rb'))

            # 距离市场关闭的秒数
            self.date = self.cur_data_file[:8]
            dt = datetime.datetime.strptime(f'{self.date} 15:00:00', '%Y%m%d %H:%M:%S')
            dt = pytz.timezone('Asia/Shanghai').localize(dt)
            close_ts = int(dt.timestamp())
            self.tick_data['ts'] = self.tick_data.index.map(lambda x: int(x.timestamp()) - 8*60*60)
            self.tick_data['before_market_close_sec'] = close_ts - self.tick_data['ts']
            # 标准化
            self.tick_data['before_market_close_sec'] /= MAX_SEC_BEFORE_CLOSE

            # 中间价格
            code1_mid = (self.tick_data[self.code1 + '_ask'] + self.tick_data[self.code1 + '_bid']) / 2
            code2_mid = (self.tick_data[self.code2 + '_ask'] + self.tick_data[self.code2 + '_bid']) / 2

            # 读取 match 数据
            cur_trade_date_idx = (self.daily_k_data[self.daily_k_data['时间'] == str(dt)[:10]].index).item()
            pre_trade_date_idx = cur_trade_date_idx - 1
            pre_trade_date = self.daily_k_data.iloc[pre_trade_date_idx]['时间']
            self.match_data = json.load(open(os.path.join(self.match_folder, f'{pre_trade_date.replace("-", "")}.txt'), 'r'))

            # tick 价差
            self.tick_data['price_diff'] = code1_mid * self.match_data['beta'] - code2_mid
            self.tick_data['zscore'] = (self.tick_data['price_diff'] - self.match_data['mean']) / self.match_data['std']

            # 历史日价差
            self.daily_k_data['price_diff'] = self.daily_k_data[self.code1] * self.match_data['beta'] - self.daily_k_data[self.code2]
            self.daily_k_data['zscore'] = (self.daily_k_data['price_diff'] - self.match_data['mean']) / self.match_data['std']
            self.his_daily_zscore = self.daily_k_data['zscore'].iloc[cur_trade_date_idx - self.his_daily_len: cur_trade_date_idx]
            self.his_daily_zscore = self.his_daily_zscore.values

            # 初始化数据索引
            self.idx = self.his_tick_len - 1
        
            break

    def set_data_type(self, data_type):
        self.data_type = data_type

    def data_size(self):
        return self.his_daily_len + self.his_tick_len

    def get_codes_bid_ask(self):
        """
        获取代码的买卖价
        """
        return self.latest_tick[:4].values

    def get(self):
        """
        输出观察值
            返回 before_market_close_sec, x
        """
        # 准备观察值
        a = self.idx - self.his_tick_len + 1
        b = self.idx + 1
        tick = self.tick_data.iloc[a: b]
        tick_x = tick['zscore'].values

        # 最新tick
        self.latest_tick = tick.iloc[-1]

        # 历史日价差
        his_daily_x = self.his_daily_zscore

        # 距离市场关闭的秒数
        before_market_close_sec = self.latest_tick['before_market_close_sec']

        # 检查本次数据是否是最后一个数据
        need_close = False
        if self.idx == len(self.tick_data) - 1:
            # 当日数据完成，需要平仓
            need_close = True
            log(f'[{self.data_type}] need_close {self.idx} {len(self.tick_data)}')
        else:
            self.idx += 1

        return before_market_close_sec, tick_x, his_daily_x, need_close

    def reset(self):
        self._pre_files()
        self._load_data()

        return self.match_data

class Account:
    """
    账户类，用于记录交易状态和计算收益
    """
    num_per_year = int(250*(4*60*60/3))
    fee_rate = 5e-5

    def __init__(self):
        self.match_data = None
        # 持仓量 
        self.default_code1_pos = 0
        self.default_code2_pos = 0
        self.code1_pos = 0
        self.code2_pos = 0
        # 最近一次买入的标的数量
        self.last_pos_increase_diff = 0
        # 持仓状态
        self.pos = 0
        # 净值序列
        self.net_raw = []
        self.net_raw_bm = []# 基准，一直持有均衡仓位
        self.net_raw_last_change = []# 最近一次调仓策略净值
        self.net_raw_last_change_bm = []# 最近一次调仓基准净值

        # 最近一次非均衡调仓前的仓位，用于计算 调仓基准净值
        self.last_change_pre_code1_pos = 0
        self.last_change_pre_code2_pos = 0

    def step(self, codes_bid_ask, action):
        """
        执行交易
        :param codes_bid_ask: 代码的买卖价
        :param action: -1-code1满仓 0-均衡仓位 1-code2满仓

        :return: (持仓量, 超额对数收益率, 最近一次调仓超额对数收益率)
        """
        # nan 检查
        assert not np.isnan(codes_bid_ask).all(), f'codes_bid_ask is nan, {codes_bid_ask}'

        code1_bid_price = codes_bid_ask[0]
        code1_ask_price = codes_bid_ask[1]
        code2_bid_price = codes_bid_ask[2]
        code2_ask_price = codes_bid_ask[3]
        
        # 仓位资产净值(可以换取的现金值)
        net_bm = (self.default_code1_pos * code1_bid_price + self.default_code2_pos * code2_bid_price) * (1-Account.fee_rate)
        self.net_raw_bm.append(net_bm)
        net = (self.code1_pos * code1_bid_price + self.code2_pos * code2_bid_price) * (1-Account.fee_rate)
        self.net_raw.append(net)
        if self.pos != 0:
            net_last_change = (self.last_change_pre_code1_pos * code1_bid_price + self.last_change_pre_code2_pos * code2_bid_price) * (1-Account.fee_rate)
            self.net_raw_last_change_bm.append(net_last_change)
            self.net_raw_last_change.append(self.net_raw[-1])

        if self.pos == 0:
            assert self.last_pos_increase_diff == 0
            assert not self.net_raw_last_change_bm
            assert not self.net_raw_last_change
            assert self.last_change_pre_code1_pos == 0
            assert self.last_change_pre_code2_pos == 0
        else:
            assert self.last_pos_increase_diff > 0
            assert self.net_raw_last_change_bm
            assert self.net_raw_last_change
            assert self.last_change_pre_code1_pos >= 0
            assert self.last_change_pre_code2_pos >= 0   

        # 本次操作平仓的 最近一次非均衡调仓净值序列
        close_net_raw_last_change = []
        close_net_raw_last_change_bm = []

        # 是否有调仓操作
        pos_change_flag = action != self.pos
        if pos_change_flag and action in [-1, 1]:

            # 记录最近一次非均衡调仓前净值
            # 当前时点净值
            if len(self.net_raw_last_change_bm) > 1:
                # 之前已经存在未平仓的 非均衡调仓净值序列
                # 说明本次调仓是 非均衡仓位 之间的调仓
                # 比如 -1 > 1 或 1 > -1
                # 记录本次平仓的序列
                close_net_raw_last_change_bm = [i for i in self.net_raw_last_change_bm]
                close_net_raw_last_change = [i for i in self.net_raw_last_change]

            self.net_raw_last_change = []
            self.net_raw_last_change_bm = [self.net_raw[-1]]
            # 调仓前的仓位
            self.last_change_pre_code1_pos = self.code1_pos
            self.last_change_pre_code2_pos = self.code2_pos

        if action == 0:  # 均衡仓位
            if self.pos == 1:  
                # code2卖出多余仓位， code1买入不足仓位
                # 卖出code2
                self.code2_pos -= self.last_pos_increase_diff
                _free_cash = self.last_pos_increase_diff * code2_bid_price * (1-Account.fee_rate)
                # _free_cash买入code1
                self.code1_pos += _free_cash / (code1_ask_price * (1+Account.fee_rate))

            elif self.pos == -1:
                # code1卖出多余仓位， code2买入不足仓位
                # 卖出code1
                self.code1_pos -= self.last_pos_increase_diff
                _free_cash = self.last_pos_increase_diff * code1_bid_price * (1-Account.fee_rate)
                # _free_cash买入code2
                self.code2_pos += _free_cash / (code2_ask_price * (1+Account.fee_rate))

            # 清空最近一次买入的标的数量
            self.last_pos_increase_diff = 0
            # 记录本次平仓的序列
            close_net_raw_last_change_bm = self.net_raw_last_change_bm
            close_net_raw_last_change = self.net_raw_last_change
            # 清空最近一次非均衡调仓前净值
            self.net_raw_last_change_bm = []
            self.net_raw_last_change = []
            self.last_change_pre_code1_pos = 0
            self.last_change_pre_code2_pos = 0

        elif action == -1:  # code1满仓
            if self.pos == 1:
                # 先恢复均衡仓位
                # 卖出code2
                self.code2_pos -= self.last_pos_increase_diff
                _free_cash = self.last_pos_increase_diff * code2_bid_price * (1-Account.fee_rate)
                # _free_cash买入code1
                self.code1_pos += _free_cash / (code1_ask_price * (1+Account.fee_rate))
                self.pos = 0

            if self.pos == 0:
                # 清仓code2，买入code1
                # 卖出code2
                _free_cash = self.code2_pos * code2_bid_price * (1-Account.fee_rate)
                self.code2_pos -= self.code2_pos
                # _free_cash买入code1
                self.last_pos_increase_diff = _free_cash / (code1_ask_price * (1+Account.fee_rate))
                self.code1_pos += self.last_pos_increase_diff
            
        elif action == 1:   # code2满仓
            if self.pos == -1:
                # 先恢复均衡仓位
                # 卖出code1
                self.code1_pos -= self.last_pos_increase_diff
                _free_cash = self.last_pos_increase_diff * code1_bid_price * (1-Account.fee_rate)
                # _free_cash买入code2
                self.code2_pos += _free_cash / (code2_ask_price * (1+Account.fee_rate))
                self.pos = 0
            
            if self.pos == 0:
                # 清仓code1，买入code2  
                # 卖出code1
                _free_cash = self.code1_pos * code1_bid_price * (1-Account.fee_rate)
                self.code1_pos -= self.code1_pos
                # _free_cash买入code2
                self.last_pos_increase_diff = _free_cash / (code2_ask_price * (1+Account.fee_rate))
                self.code2_pos += self.last_pos_increase_diff

        if pos_change_flag:
            self.pos = action
            # 更新 调仓后的净值
            net = (self.code1_pos * code1_bid_price + self.code2_pos * code2_bid_price) * (1-Account.fee_rate)
            self.net_raw[-1] = net
            if action in [-1, 1]:
                self.net_raw_last_change.append(net)

        if self.net_raw_last_change_bm:
            # 最近一次调仓的对数收益率
            net_change_log_return = np.log(self.net_raw_last_change[-1]) - np.log(self.net_raw_last_change[0])
            # 若未调仓的对数收益率
            net_change_log_return_bm = np.log(self.net_raw_last_change_bm[-1]) - np.log(self.net_raw_last_change_bm[0])
            # 最近一次调仓的超额收益
            last_excess_log_return = net_change_log_return - net_change_log_return_bm
        else:
            last_excess_log_return = 0.0

        return self.pos, last_excess_log_return, close_net_raw_last_change, close_net_raw_last_change_bm

    @staticmethod
    def cal_res(net_raw, net_raw_bm):
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

        if (len(net_raw_bm) > 1):
            # 计算基准净值的评价指标
            net_bm = np.array(net_raw_bm)
            # 计算对数收益率序列
            log_returns_bm = np.diff(np.log(net_bm))
            res['max_drawdown_bm'], res['max_drawdown_ticks_bm'] = calc_drawdown(net_bm)
            res['max_drawup_ticks_bm'], res['drawup_ticks_bm_count'] = calc_drawup_ticks(net_bm)
            res['trade_return_bm'] = calc_return(log_returns_bm, annualize=False)
            res['step_return_bm'] = log_returns_bm[-1]

        return res
        
    def reset(self, match_data):
        """
        重置账户状态
        """
        self.match_data = match_data
        self.default_code1_pos = self.code1_pos = self.match_data['beta']
        self.default_code2_pos = self.code2_pos = 1
        self.pos = 0
        self.net_raw = []
        self.net_raw_bm = []
        self.net_raw_last_change = []# 最近一次调仓策略净值
        self.net_raw_last_change_bm = []# 最近一次调仓基准净值
        return self.pos, 0, 0

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
        
class MATCH_trade_env(gym.Env):
    """
    用于 配对交易 的强化学习环境
    返回的 obs 结构 (19,):
        5 前5个交易日的价差 + 
        10 最近10个tick的价差 +
        1 距离收盘的秒数 +
        1 仓位状态 +
        1 当天的超额对数收益率 + 
        1 最近一次调仓(非均衡仓位)的超额对数收益率
    """

    REG_NAME = 'match'
    ITERATION_DONE_FILE = os.path.join(os.path.expanduser('~'), '_match_env_iteration_done')
    
    def __init__(self, config: dict, debug_dates=[]):
        """
        :param config: 配置
            {
                # 用于实例化 数据生产器
                'data_type': 'train'/'val'/'test',# 训练/测试
                'his_daily_len': 5,# 每个样本的 历史日的价差长度
                'his_tick_len': 10,# 每个样本的 历史tick的价差长度

                # 终止游戏的超额亏损阈值
                'loss_threshold': -0.005,# 最大超额亏损阈值

                # 奖励策略
                'reward_strategy_class_dict': {
                    'end_position': EndPositionRewardStrategy,  
                    'close_position': ClosePositionRewardStrategy,
                    'hold_position': HoldPositionRewardStrategy,
                    'balance': BalanceRewardStrategy
                }
            }
        """
        super().__init__()

        defult_config = {
            # 用于实例化 数据生产器
            'data_type': 'train',# 训练/测试
            'his_daily_len': 5,# 每个样本的 历史日的价差长度
            'his_tick_len': 10,# 每个样本的 历史tick的价差长度

            # 终止游戏的超额亏损阈值
            'loss_threshold': -0.005,# 最大超额亏损阈值

            # 奖励策略
            'reward_strategy_class_dict': {
                'end_position': EndPositionRewardStrategy,
                'close_position': ClosePositionRewardStrategy,
                'hold_position': HoldPositionRewardStrategy,
                'balance': BalanceRewardStrategy
            }
        }

        # 用户配置更新
        for k, v in defult_config.items():
            config[k] = config.get(k, v)
        for k, v in defult_config['reward_strategy_class_dict'].items():
            config['reward_strategy_class_dict'][k] = config['reward_strategy_class_dict'].get(k, v)

        self.loss_threshold = config['loss_threshold']

        # 保存文件夹
        self.save_folder = os.path.join(config['train_folder'], 'env_output')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # 奖励计算器
        self.reward_calculator = RewardCalculator(config['reward_strategy_class_dict'])

        # 测试日期
        self.debug_dates = debug_dates

        # 初始化日志
        log_name = f'{config["train_title"]}_{beijing_time().strftime("%Y%m%d")}'
        init_logger(log_name, home=config['train_folder'], timestamp=False)
        log(f'[{id(self)}] init logger: {get_log_file()}')
        
        # 数据生产器
        self.data_producer = data_producer(
            config['data_type'], 
            config['his_daily_len'], 
            config['his_tick_len'], 
            save_folder=self.save_folder, 
            debug_dates=self.debug_dates,
        )

        # 账户数据
        self.acc = Account()

        # 是否需要重置（切换模式后需要）
        self.need_reset = False

        # 动作空间 
        # -1, 0, 1
        self.action_space = spaces.Discrete(3)

        # 观察空间 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data_producer.data_size() + 4,), dtype=np.float32)

        # 样本计数
        self.sample_count = 0

        # 记录上一个step的时间戳
        self.last_step_time = 0
        self.iteration = 0

        # 环境输出文件
        self.need_upload_file = ''
        self.update_need_upload_file()

        # 记录每个 episode 的步数
        self.mean_episode_lengths = 0
        self.episode_count = 0
        
        log(f'[{id(self)}][{self.data_producer.data_type}] init env done')

    @staticmethod
    def iteration_done():
        """会使用文件来标记迭代结束的时间"""
        with open(MATCH_trade_env.ITERATION_DONE_FILE, 'w') as f:
            f.write(f'1')

    def is_iteration_done(self):
        """是否最近迭代完成"""
        if os.path.exists(MATCH_trade_env.ITERATION_DONE_FILE):
            return os.path.getmtime(MATCH_trade_env.ITERATION_DONE_FILE) > self.last_step_time
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

    def _cal_reward(self, action, need_close, info):
        """
        计算奖励
        """
        # 游戏是否终止
        # 1. 超额亏损超过阈值                           -STD_REWARD
        acc_done = False

        pos, res_last, close_net_raw_last_change, close_net_raw_last_change_bm = self.acc.step(self.data_producer.get_codes_bid_ask(), action)

        # 日内净值的评价指标
        result = self.acc.cal_res(self.acc.net_raw, self.acc.net_raw_bm)

        # 记录close数据
        if len(close_net_raw_last_change) > 0:
            result['close_trade'] = True
            # 计算结束交易的超额收益率
            ret = np.log(close_net_raw_last_change[-1]) - np.log(close_net_raw_last_change[0])
            ret_bm = np.log(close_net_raw_last_change_bm[-1]) - np.log(close_net_raw_last_change_bm[0])
            excess_return = ret - ret_bm
            result['close_excess_return'] = excess_return
        else:
            result['close_trade'] = False

        # 记录net/net_bm
        result['net'] = self.acc.net_raw[-1] if self.acc.net_raw else np.nan
        result['net_bm'] = self.acc.net_raw_bm[-1] if self.acc.net_raw_bm else np.nan

        #数据类型
        result['data_type'] = self.data_producer.data_type

        reward = 0

        # 检查超额收益
        exceed_reward = result['trade_return'] - result['trade_return_bm']
        result['excess_return'] = exceed_reward
        if exceed_reward < self.loss_threshold:
            # 游戏被终止，计入交易失败
            result['force_close'] = 1
            acc_done = True
            reward = -STD_REWARD

            # 标记结束交易
            result['close_trade'] = True
            # 计算结束交易的超额收益率
            ret = np.log(self.acc.net_raw_last_change[-1]) - np.log(self.acc.net_raw_last_change[0])
            ret_bm = np.log(self.acc.net_raw_last_change_bm[-1]) - np.log(self.acc.net_raw_last_change_bm[0])
            excess_return = ret - ret_bm
            result['close_excess_return'] = excess_return

        else:
            result['force_close'] = 0
            # 游戏正常进行
            # 计算奖励
            reward, acc_done = self.reward_calculator.calculate_reward(id(self), STD_REWARD, need_close, action, result, self.data_producer, self.acc, close_net_raw_last_change, close_net_raw_last_change_bm)

            # 奖励范围
            assert abs(reward) <= STD_REWARD, f'reward({reward}) > STD_REWARD({STD_REWARD})'

        # 拷贝 result > info
        for k, v in result.items():
            info[k] = v

        return reward, acc_done, pos, exceed_reward, res_last

    def out_test_predict(self, out_text):
        """
        输出测试预测数据 -> predict.csv

        # 状态相关
        before_market_close_sec,pos,res,res_last,predict,data_file,episode,step,
        net,net_bm,

        # 其他
        terminated,truncated,force_close,close_trade,

        # 奖励评价相关
        reward,max_drawdown,max_drawdown_ticks,trade_return,step_return,hold_length,max_profit_reachable_bm,potential_return,acc_return,

        # 基准相关
        max_drawdown_bm,max_drawdown_ticks_bm,max_drawup_ticks_bm,drawup_ticks_bm_count,trade_return_bm,step_return_bm,excess_return,
        """
        # 输出列名
        if not os.path.exists(self.need_upload_file):
            with open(self.need_upload_file, 'w') as f:
                f.write('before_market_close_sec,pos,res,res_last,predict,data_file,episode,step,net,net_bm,terminated,truncated,force_close,close_trade,reward,max_drawdown,max_drawdown_ticks,trade_return,step_return,hold_length,max_profit_reachable_bm,potential_return,acc_return,max_drawdown_bm,max_drawdown_ticks_bm,max_drawup_ticks_bm,drawup_ticks_bm_count,trade_return_bm,step_return_bm,excess_return\n')
        with open(self.need_upload_file, 'a') as f:
            f.write(out_text)
            f.write('\n')
    
    def step(self, action):
        self.steps += 1
        try:
            assert not self.need_reset, "LOB env need reset"

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
            out_text = f'{self.static_data["before_market_close_sec"]},{self.static_data["pos"]},{self.static_data["res"]},{self.static_data["res_last"]},{int(action)},{self.data_producer.cur_data_file},{self.episode_count},{self.steps}'

            # 先获取下一个状态的数据, 会储存 latest_tick, 用于acc.step(), 避免用当前状态价格结算
            before_market_close_sec, tick_x, his_daily_x, need_close = self.data_producer.get()

            info = {
                'action': action,
            }

            # 计算奖励
            reward, acc_done, pos, res, res_last = self._cal_reward(action, need_close, info)

            # 准备输出数据
            # net,net_bm,
            out_text += f",{info['net']},{info['net_bm']}"
            # reward,max_drawdown,max_drawdown_ticks,trade_return,step_return,hold_length,
            out_text2 = f",{reward},{info.get('max_drawdown', '')},{info.get('max_drawdown_ticks', '')},{info.get('trade_return', '')},{info.get('step_return', '')},{info.get('hold_length', '')},{info.get('max_profit_reachable_bm', '')},{info.get('potential_return', '')},{info.get('acc_return', '')}"
            # max_drawdown_bm,max_drawdown_ticks_bm,max_drawup_ticks_bm,drawup_ticks_bm_count,trade_return_bm,step_return_bm,
            out_text2 += f",{info.get('max_drawdown_bm', '')},{info.get('max_drawdown_ticks_bm', '')},{info.get('max_drawup_ticks_bm', '')},{info.get('drawup_ticks_bm_count', '')},{info.get('trade_return_bm', '')},{info.get('step_return_bm', '')},{info.get('excess_return', '')}"

            # 记录静态数据，用于输出预测数据, 在下一个step中使用
            self.static_data = {
                'before_market_close_sec': before_market_close_sec,
                'pos': pos,
                'res': res,
                'res_last': res_last,
            }

            # 标准化 res / res_last TODO

            # 添加 静态特征
            observation = np.concatenate([his_daily_x, tick_x, [before_market_close_sec, pos, res, res_last]])

            # 检查是否结束
            terminated = acc_done or need_close# 游戏终止，最终回报: 非法操作 / 消极操作，多次错失机会 / 连续3次平仓奖励为负 / 当天连续数据结束时，平均平仓奖励为正
            # 环境中没有截断结束
            truncated = False

            out_text += f",{terminated},{truncated},{info.get('force_close', 0)},{info.get('close_trade', False)}"
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

            # 重置
            self.need_reset = False

            # 数据
            log(f'[{id(self)}][{self.data_producer.data_type}] reset')

            match_data = None
            while True:
                match_data = self.data_producer.reset()
                before_market_close_sec, tick_x, his_daily_x, need_close = self.data_producer.get()
                if need_close:
                    # 若是 val 数据，有可能 need_close 为True
                    # 需要过滤
                    log(f'[{id(self)}][{self.data_producer.data_type}] need_close: True, reset data_producer again')
                else:
                    break

            # 账户
            self.acc.reset(match_data)
            # 账户需要用动作0走一步, 会初始记录act之前的净值
            pos, res, res_last, _ = self.acc.step(self.data_producer.get_codes_bid_ask(), 0)

            # 标准化 res / res_last TODO

            # 添加 静态特征
            observation = np.concatenate([his_daily_x, tick_x, [before_market_close_sec, pos, res, res_last]])

            # 记录静态数据，用于输出预测数据
            self.static_data = {
                'before_market_close_sec': before_market_close_sec,
                'pos': pos,
                'res': res,
                'res_last': res_last,
            }

            return observation, {}
            
        except Exception as e:
            log(f'[{id(self)}][{self.data_producer.data_type}] reset error: {get_exception_msg()}')
            raise e

def test_data_producer():
    dp = data_producer(
        data_type='train',
        his_daily_len=5,
        his_tick_len=10,
        save_folder="", 
        debug_dates=['20241009.pkl'],
    )

    acc = Account()

    act_idx = 0
    acts = [-1, -1, -1, 0, 0, 1, 1, 1, -1, -1, 0, 0]

    for i in range(10):
        if act_idx >= len(acts):
            break

        print('reset')
        match_data = dp.reset()
        before_market_close_sec, tick_x, his_daily_x, need_close = dp.get()
        codes_bid_ask = dp.get_codes_bid_ask()
        acc.reset(match_data)
        # 需要走一遍 act = 0
        pos, res, res_last = acc.step(codes_bid_ask, 0)

        need_close = False
        while not need_close:
            if act_idx >= len(acts):
                break
            act = acts[act_idx]
            act_idx += 1
            before_market_close_sec, tick_x, his_daily_x, need_close = dp.get()
            codes_bid_ask = dp.get_codes_bid_ask()
            pos, res, res_last = acc.step(codes_bid_ask, act)
            print(f'pos: {pos}, res: {res}, res_last: {res_last}')

    """
    pos: -1, res: -0.000408688165584814, res_last: -0.000408688165584814
    pos: -1, res: -0.001102931719229594, res_last: -0.001102931719229594
    pos: -1, res: -0.001102931719229594, res_last: -0.001102931719229594
    pos: 0, res: -0.00168571491457159, res_last: 0.0
    pos: 0, res: -0.0016849070773450325, res_last: 0.0
    pos: 1, res: -0.0022559040012010145, res_last: -0.000570547857931536
    pos: 1, res: -0.0022559040012010145, res_last: -0.000570547857931536
    pos: 1, res: -0.0022559040012010145, res_last: -0.000570547857931536
    pos: -1, res: -0.002915344575315948, res_last: -0.0006594405741149334
    pos: -1, res: -0.0025685324511838248, res_last: -0.00010000000008336674
    pos: 0, res: -0.0026303761830964723, res_last: 0.0
    pos: 0, res: -0.0026303761830964723, res_last: 0.0
    done
    """
    print('done')

def test_env():
    env = MATCH_trade_env(
        config = {
            # 用于实例化 数据生产器
            'data_type': 'train',# 训练/测试
            'his_daily_len': 5,# 每个样本的 历史日的价差长度
            'his_tick_len': 10,# 每个样本的 历史tick的价差长度

            # 终止游戏的超额亏损阈值
            'loss_threshold': -0.005,# 最大超额亏损阈值

            # 奖励策略
            'reward_strategy_class_dict': {
                'end_position': EndPositionRewardStrategy,
                'close_position': ClosePositionRewardStrategy,
                'hold_position': HoldPositionRewardStrategy,
                'balance': BalanceRewardStrategy
            },

            'train_folder': r'C:\Users\lh\Desktop\temp\match_env',
            'train_title': 'test',
        },
        debug_dates=['20241009.pkl'],
    )

    act_idx = 0
    acts = [-1, -1, -1, 0, 0, 1, 1, 1, -1, -1, 0, 0] + [0] * 5000
    acts = [0] * 5000

    for i in range(10):
        if act_idx >= len(acts):
            break

        print('reset')
        obs, info = env.reset()
        need_close = False
        while not need_close:
            if act_idx >= len(acts):
                break
            act = acts[act_idx]
            act_idx += 1
            obs, reward, terminated, truncated, info = env.step(act)
            need_close = terminated or truncated
            print(f'pos: {obs[-3]}, res: {obs[-2]}, res_last: {obs[-1]}, reward: {reward}')

    print('done')

if __name__ == '__main__':
    # test_data_producer()
    test_env()


