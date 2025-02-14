import os
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
from py_ext.tool import log

from dl_helper.tool import calc_sharpe_ratio, calc_sortino_ratio, calc_drawdown, calc_return, calc_drawup_ticks
from dl_helper.train_param import in_kaggle

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

ILLEGAL_REWARD = -10000
# 扩大因子
SCALE_FACTOR = 1e6
MEAN_SEC_BEFORE_CLOSE = 10024.17
STD_SEC_BEFORE_CLOSE = 6582.91

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
    def __init__(self, data_type='train', his_len=100, file_num=0, simple_test=False, need_cols=[]):
        """
        'data_type': 'train',# 训练/测试
        'his_len': 100,# 每个样本的 历史数据长度
        'file_num': 0,# 数据生产器 限制最多使用的文件（日期）数量
        'simple_test': False,# 是否为简单测试
        """
        # 快速测试
        self.simple_test = simple_test
        if self.simple_test:
            file_num = 5

        self.his_len = his_len
        self.file_num = file_num
        
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
            # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
            data_folder_name = os.listdir(input_folder)[0]
            self.data_folder = os.path.join(input_folder, data_folder_name)
        else:
            self.data_folder = r'D:\L2_DATA_T0_ETF\train_data\RL_combine_data'

        self.data_type = data_type
        self.files = []

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
        self.col_idx = {}
        # 用于绘图的索引
        self.plot_begin = 0
        self.plot_end = 0
        self.plot_cur = 0
        self.plot_cur_pre = -1
        self.plot_data = None
        # 买卖1档价格
        self.ask_price = 0
        self.bid_price = 0
        # id
        self.id = ''
        # 当前日期数据停止标志
        self.date_file_done = False

    def pre_plot_data(self):
        """
        预先读取绘图数据
        col_idx: BASE买1价, BASE卖1价, BASE中间价
        """
        # 直接选择需要的列并创建DataFrame
        cols = ['BASE买1价', 'BASE卖1价']
        self.plot_data = pd.DataFrame(self.all_raw_data[self.plot_begin:self.plot_end, [self.col_idx[col] for col in cols]], columns=['bid', 'ask'])
        # 高效计算中间价格
        self.plot_data['mid_price'] = self.plot_data.mean(axis=1)

    def _pre_files(self):
        """
        准备文件列表
        若是训练数据，随机读取
        若是验证/测试数据，按顺序读取
        """
        self.files = os.listdir(os.path.join(self.data_folder, self.data_type))
        if self.data_type == 'train':
            random.shuffle(self.files)

        if self.file_num:
            self.files = self.files[:self.file_num]
            
    def _load_data(self):
        """
        按照文件列表读取数据
        每次完成后从文件列表中剔除
        """
        file = self.files.pop(0)
        log(f'load date file({self.data_type}): {file}')
        self.ids, self.mean_std, self.x, self.all_raw_data = pickle.load(open(os.path.join(self.data_folder, self.data_type, file), 'rb'))

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
        self.date = file[:8]
        dt = datetime.datetime.strptime(f'{self.date} 15:00:00', '%Y%m%d %H:%M:%S')
        dt = pytz.timezone('Asia/Shanghai').localize(dt)
        close_ts = int(dt.timestamp())
        self.before_market_close_sec = np.array([int(i.split('_')[1]) for i in self.ids])
        self.before_market_close_sec = close_ts - self.before_market_close_sec

        # 解析标的 随机挑选一个标的数据
        symbols = np.array([i.split('_')[0] for i in self.ids])
        unique_symbols = [i for i in np.unique(symbols) if i != '159941']
        # 获取所有标的的起止索引
        self.idxs = []
        for symbol in unique_symbols:
            symbol_mask = symbols == symbol
            symbol_indices = np.where(symbol_mask)[0]
            self.idxs.append([symbol_indices[0], symbol_indices[-1], USE_CODES.index(symbol)])
    
        # 训练数据随机选择一个标的
        # 一个日期文件只使用其中的一个标的的数据，避免同一天各个标的之间存在的相关性 对 训练产生影响
        if self.data_type == 'train':
            self.idxs = [random.choice(self.idxs)]

        log(f'init idxs: {self.idxs}')

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
       
        # self.all_raw_data 转为 numpy
        self.all_raw_data = self.all_raw_data.values

        # 初始化绘图索引
        self.plot_begin, self.plot_cur = self.x[self.idxs[0][0]]
        self.plot_cur -= 1
        self.plot_cur_pre = -1
        _, self.plot_end = self.x[self.idxs[0][1]]
        # 准备绘图数据
        self.pre_plot_data()

        # # 测试用
        # pickle.dump((self.all_raw_data, self.mean_std, self.x), open(f'{self.data_type}_raw_data.pkl', 'wb'))

        # 载入了新的日期文件数据
        # 重置日期文件停止标志
        self.date_file_done = False

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
        self.bid_price = last_row[self.col_idx['BASE买1价']]
        self.ask_price = last_row[self.col_idx['BASE卖1价']]

    def get(self):
        """
        输出观察值
        返回 symbol_id, before_market_close_sec, x, done, need_close, period_done
        """
        # # 测试用
        # print(self.idxs[0])

        if self.plot_cur_pre != -1:
            # 更新绘图数据
            self.plot_cur = self.plot_cur_pre

        # 检查日期文件结束
        if self.date_file_done:
            # load 下一个日期文件的数据
            self._load_data()

        # 准备观察值
        a, b = self.x[self.idxs[0][0]]
        if b-a > self.his_len:# 修正历史数据长度
            a = b - self.his_len
        raw = self.all_raw_data[a: b, :].copy()
        # 记录 买卖1档 的价格
        self.store_bid_ask_1st_data(raw)
        # 数据标准化
        ms = pd.DataFrame(self.mean_std[self.idxs[0][0]])
        x, ms = self.use_data_split(raw, ms)
        x -= ms[:, 0]
        x /= ms[:, 1]

        # 标的id
        symbol_id = self.idxs[0][2]

        # 当前标的
        self.code = USE_CODES[int(symbol_id)]

        # 距离市场关闭的秒数
        before_market_close_sec = self.before_market_close_sec[self.idxs[0][0]]

        # 记录数据id
        self.id = self.ids[self.idxs[0][0]]

        # 检查下一个数据是否是最后一个数据
        all_done = False
        need_close = False
        if self.idxs[0][0] == self.idxs[0][1]:
            # 当组 begin/end 完成，需要平仓
            need_close = True
            log(f'need_close {self.idxs[0][0]} {self.idxs[0][1]}')
            # 更新剩余的 begin/end 组
            self.idxs = self.idxs[1:]
            log(f'idxs: {self.idxs}')
            if not self.idxs:
                # 当天的数据没有下一个可读取的 begin/end 组
                log(f'date done')
                self.date_file_done = True
                log(f'len(files): {len(self.files)}')
                if not self.files:
                    # 没有下一个可以读取的日期数据文件
                    log('all date files done')
                    all_done = True
            else:
                # 重置绘图索引
                self.plot_begin, self.plot_cur = self.x[self.idxs[0][0]]
                self.plot_cur -= 1
                self.plot_cur_pre = -1
                _, self.plot_end = self.x[self.idxs[0][1]]
                # 准备绘图数据
                self.pre_plot_data()
        else:
            self.idxs[0][0] += 1
            _, self.plot_cur_pre = self.x[self.idxs[0][0]]
            self.plot_cur_pre -= 1

        return symbol_id, before_market_close_sec, x, all_done, need_close, self.date_file_done

    def get_plot_data(self):
        """
        获取绘图数据, 当前状态(时间点)索引, 当前状态(时间点)id
        """
        return self.plot_data, self.plot_cur - self.plot_begin, self.ids[self.plot_cur]

    def reset(self):
        self._pre_files()
        self._load_data()

class Account:
    """
    账户类，用于记录交易状态和计算收益
    """
    def __init__(self, fee_rate=5e-5, net_file=os.path.join(os.path.expanduser('~'), 'net.txt')):
        # 持仓量 
        self.pos = 0
        # 持仓成本
        self.cost = 0
        # 累计收益率
        self.profit = 0
        # 交易费率
        self.fee_rate = fee_rate
        self.buy_fee = 0
        self.sell_fee = 0   
        # 净值序列
        self.net_raw = []
        self.net_raw_bm = []# 基准，一直持有
        self.net = []
        self.net_bm = []
        self.bm_next_open = True
        # 净值文件
        self.net_file = net_file
        # 若已经存在 则删除
        if os.path.exists(net_file):
            os.remove(net_file)

    def save_net(self, net):
        """
        保存净值序列到文件
        每行保存一个净值序列，用逗号分隔
        """
        with open(self.net_file, 'a') as f:
            f.write(','.join(map(str, net)) + '\n')

    def step(self, bid_price, ask_price, action, need_close):
        """
        执行交易
        :param bid_price: 最优买价
        :param ask_price: 最优卖价 
        :param action: 0-买入 1-卖出 2-不操作
        :param need_close: 是否需要平仓
        :return: (动作是否合法, 持仓量, 对数收益率, 评价指标)
        """
        # 重置fee记录
        if self.pos == 0:
            self.buy_fee = 0
            self.sell_fee = 0

        # 计算持仓对数收益率
        unrealized_profit = 0
        if self.pos == 1:
            # 未实现收益需考虑卖出时的手续费
            self.sell_fee = bid_price * self.fee_rate
            sell_price = bid_price - self.sell_fee
            unrealized_profit = np.log(sell_price / self.cost)

        # 基准净值, 一直持有
        if self.bm_next_open:
            self.bm_next_open = False
            self.net_raw_bm.append(ask_price)
        else:
            self.net_raw_bm.append(bid_price)

        # 无操作任何时间都允许
        legal = True
        if action == 0:  # 买入
            if self.pos == 0:  # 空仓可以买入
                self.pos = 1
                self.net_raw.append(ask_price)
                # 买入成本需要加上手续费
                self.buy_fee = ask_price * self.fee_rate
                self.cost = ask_price + self.buy_fee
            else:
                legal = False
        elif action == 1:  # 卖出
            if self.pos == 1:  # 有多仓可以卖出
                self.pos = 0
                self.profit = unrealized_profit
                self.net_raw.append(bid_price)
            else:
                legal = False
        elif action == 2:   # 不操作
            if self.pos == 0:
                # 无持仓
                if len(self.net_raw):
                    # 有前净值， 保持净值不变
                    self.net_raw.append(self.net_raw[-1])
                # 无前净值 不操作
            else:
                # 有持仓
                self.net_raw.append(bid_price)

        # 起始无持仓的净值矫正
        # 基准净值长度 > 策略净值长度
        # 用策略净值填充至相同的长度
        net_len = len(self.net_raw)
        bm_len = len(self.net_raw_bm)
        if net_len == 1 and bm_len > 1:
            for i in range(bm_len - net_len):
                self.net_raw.append(self.net_raw[-1])

        if not legal:
            self.net_raw.append(bid_price)
        
        # 评价指标
        res = {}
        if legal:
            # 数据足够 > 1
            # 需要平仓 或 卖出， 需要计算评价指标， 储存在info中
            if need_close or action==1:
                if (len(self.net_raw) > 1):
                    # 计算策略净值的评价指标
                    # 平均税费(买入卖出)到每一步
                    # 第一步买入，等价较好，不平均税费
                    step_fee = (self.buy_fee + self.sell_fee) / (len(self.net_raw) - 1)
                    self.net = np.array(self.net_raw)
                    self.net[1:] -= step_fee
                    # 储存净值序列到文件
                    self.save_net(self.net)
                    # 计算对数收益率序列
                    log_returns = np.diff(np.log(self.net))
                    # 计算指标
                    res['sortino_ratio'] = calc_sortino_ratio(log_returns)
                    res['sharpe_ratio'] = calc_sharpe_ratio(log_returns)
                    res['max_drawdown'], res['max_drawdown_ticks'] = calc_drawdown(self.net)
                    res['total_return'] = calc_return(log_returns)
                    res['trade_return'] = res['total_return'] / len(log_returns)
                    res['hold_length'] = len(self.net)# 持仓时间
                else:
                    # 策略净值
                    res['sortino_ratio'] = 0
                    res['sharpe_ratio'] = 0
                    res['max_drawdown'] = 0
                    res['max_drawdown_ticks'] = 0
                    res['total_return'] = 0
                    res['trade_return'] = 0
                    res['hold_length'] = 0

                if (len(self.net_raw_bm) > 1):
                    # 计算基准净值的评价指标
                    buy_fee_bm = self.net_raw_bm[0] * self.fee_rate
                    sell_fee_bm = self.net_raw_bm[-1] * self.fee_rate
                    step_fee_bm = (buy_fee_bm + sell_fee_bm) / (len(self.net_raw_bm) - 1)
                    self.net_bm = np.array(self.net_raw_bm)
                    self.net_bm[1:] -= step_fee_bm
                    # 计算对数收益率序列
                    log_returns_bm = np.diff(np.log(self.net_bm))
                    res['sortino_ratio_bm'] = calc_sortino_ratio(log_returns_bm)
                    res['sharpe_ratio_bm'] = calc_sharpe_ratio(log_returns_bm)
                    res['max_drawdown_bm'], res['max_drawdown_ticks_bm'] = calc_drawdown(self.net_bm)
                    res['max_drawup_ticks_bm'] = calc_drawup_ticks(self.net_bm)
                    res['total_return_bm'] = calc_return(log_returns_bm)
                    res['trade_return_bm'] = res['total_return_bm'] / len(log_returns_bm)
                else:
                    # 基准净值
                    res['sortino_ratio_bm'] = 0
                    res['sharpe_ratio_bm'] = 0
                    res['max_drawdown_bm'] = 0
                    res['max_drawdown_ticks_bm'] = 0
                    res['max_drawup_ticks_bm'] = 0
                    res['total_return_bm'] = 0
                    res['trade_return_bm'] = 0

                # 平仓后，重置净值
                self.reset()
        else:
            # 不合法的操作，交易全部清空
            self.reset()

        # print("acc::step",legal, res)

        return legal, self.pos, unrealized_profit, res
        
    def reset(self):
        """
        重置账户状态
        """
        self.pos = 0
        self.cost = 0 
        self.profit = 0
        self.net_raw = []
        self.net_raw_bm = []
        return self.pos, 0

class LOB_trade_env(gym.Env):
    """
    用于 LOB 的强化学习环境
    返回的 obs 结构:
        lob数据 + 持仓 + 未实现收益率
    """

    REG_NAME = 'lob'
    
    def __init__(self, config: dict):
        """
        :param config: 配置
            {
                # 用于实例化 数据生产器
                'data_type': 'train'/'val'/'test',# 训练/测试
                'his_len': 100,# 每个样本的 历史数据长度
                'file_num': 0,# 数据生产器 限制最多使用的文件（日期）数量
                'simple_test': False,# 是否为简单测试

                'period_done': False,
                'need_close': False,
            }
        """
        super().__init__()

        defult_config = {
            # 用于实例化 数据生产器
            'data_type': 'train',# 训练/测试
            'his_len': 100,# 每个样本的 历史数据长度
            'file_num': 0,# 数据生产器 限制最多使用的文件（日期）数量
            'simple_test': False,# 是否为简单测试
            'need_cols': [],# 需要读取的列

            'period_done': False,
            'need_close': False,
        }
        # 用户配置更新
        for k, v in defult_config.items():
            config[k] = config.get(k, v)
        
        # 数据生产器
        self.data_producer = data_producer(config['data_type'], config['his_len'], config['file_num'], config['simple_test'], config['need_cols'])
        self.period_done = config['period_done']

        # 账户数据
        self.need_close = config['need_close'] # False: 下一个动作无论是什么，都要对做平仓操作(因为没有更多的连续数据)
        self.acc = Account()

        # 是否需要重置（切换模式后需要）
        self.need_reset = False

        # 动作空间 
        # 买 卖 不操作
        self.action_space = spaces.Discrete(3)

        # 观察空间 
        #     标的数据
        #       100 * 130 -> 13000
        #     仓位数据:
        #       #交易数量 固定1单位
        #       0/1 空仓/多头
        # 13001 数组
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data_producer.data_size() + 4,), dtype=np.float32)

        # 测试数据集 预测输出文件
        self.need_upload_file = ''

        # 记录上一个买入的idx
        self.last_buy_idx = -1
        
        # Add notification system
        self.notification_window = None

    def no_need_track_info_item(self):
        return ['close', 'act_criteria', 'period_done']

    def need_wait_close(self):
        return True

    def _set_data_type(self, data_type):
        if data_type in ['val', 'test']:
            # 切换到测试数据集，创建预测输出文件
            self.need_upload_file = f'{data_type}_{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")}.csv'
        self.data_producer.set_data_type(data_type)
        
        self.need_reset = True

    def val(self):
        self._set_data_type('val')

    def train(self):
        self._set_data_type('train')

    def test(self):
        self._set_data_type('test')

    def _get_data(self):
        # 获取数据
        symbol_id, before_market_close_sec, x, all_data_done, need_close, period_done = self.data_producer.get()
        x = x.reshape(-1)

        # 标准化 距离收盘秒数
        before_market_close_sec -= MEAN_SEC_BEFORE_CLOSE
        before_market_close_sec /= STD_SEC_BEFORE_CLOSE

        return symbol_id, before_market_close_sec, x, all_data_done, need_close, period_done

    def _cal_reward(self, action, need_close, info):
        """
        计算奖励
        非法动作 reward=ILLEGAL_REWARD
        只有平仓 reward=收益率 + 最大回撤
        其他 reward=0

        平仓标志: info['close'] = True
        需要在平仓后回溯属于本次交易的所有时间步, 修改 reward=收益率
        """
        legal, pos, profit, res = self.acc.step(self.data_producer.bid_price, self.data_producer.ask_price, action, need_close)

        # 合法性检查
        if not legal:
            # 非法动作
            info['close'] = True
            info['act_criteria'] = -1
            return ILLEGAL_REWARD, False, pos, profit

        # 只有平仓才给与reward
        # 同时记录评价指标
        if need_close or action==1:
            info['close'] = True
            # 增加操作交易评价结果
            # 体现 非法/win/loss
            info['act_criteria'] = 0 if res['total_return'] > 0 else 1

            #########################################################
            # 计算奖励 范围: [ILLEGAL_REWARD, -ILLEGAL_REWARD] 
            
            # 1.0 res['trade_return']平均了所有时间步,通常很小,导致虽有收益但奖励为负 -> 可能会倾向于不发生交易(reward=0)
            # reward = res['trade_return'] + res['max_drawdown']

            # # 2.0 res['total_return']交易对的绝对收益，强调收益加大权重
            # reward = res['total_return']*1e4 + res['max_drawdown']

            # 3.0 sortino_ratio
            # reward = res['sortino_ratio']

            # 4.0 sharpe_ratio 
            # reward = res['sharpe_ratio']

            # 5.0 
            # 交易奖励: 平均收益 * 放大因子 - 做空可盈利的回撤惩罚 
            # 若交易奖励为0，基准净值做多可盈利的收益惩罚
            punish = res['max_drawdown_ticks'] >= 2
            reward = res['trade_return'] * SCALE_FACTOR + punish * res['max_drawdown']
            if reward == 0:
                punish = res['max_drawup_ticks_bm'] >= 2
                reward = -res['trade_return_bm'] * SCALE_FACTOR * punish

            # 限制范围
            reward = max(min(reward, -ILLEGAL_REWARD*0.5), ILLEGAL_REWARD*0.5)
            # reward 校正
            if info['act_criteria'] == 1:
                # loss: reward 一定为负数
                reward = min(reward, -1e-5)
            elif info['act_criteria'] == 0:
                # win: reward 一定为正数
                reward = max(reward, 1e-5)
            #########################################################
            for k, v in res.items():
                info[k] = v
        else:
            reward = 0

        return reward, False, pos, profit

    def out_test_predict(self, action):
        """
        输出测试预测数据 -> predict.csv
        id,predict
        """
        # 输出列名
        if not os.path.exists(self.need_upload_file):
            with open(self.need_upload_file, 'w') as f:
                f.write('id,predict\n')
        with open(self.need_upload_file, 'a') as f:
            f.write(f'{self.data_producer.id},{int(action)}\n')
    
    def step(self, action):
        assert not self.need_reset, "LOB env need reset"

        # 如果是test数据集，需要输出预测数据文件
        if self.data_producer.data_type in ['val', 'test']:
            self.out_test_predict(action)

        # 先获取下一个状态的数据, 会储存 bid_price, ask_price, 用于acc.step(), 避免用当前状态种的价格结算
        # 备份 self.need_close
        _need_close = self.need_close
        symbol_id, before_market_close_sec, observation, data_done, self.need_close, self.period_done = self._get_data()

        info = {
            'close': False,# 若为True, 需要回溯属于本次交易的所有时间步, 修改 reward=收益率
            'period_done': self.period_done,
        }

        # 计算奖励
        reward, acc_done, pos, profit = self._cal_reward(action, _need_close, info)

        if action == 1 or _need_close or info.get('act_criteria', None) == -1:
            # 平仓 / 强制平仓 / 非法平仓， 重置idx
            self.last_buy_idx = -1
        elif action == 0:
            # 记录交易idx
            self.last_buy_idx = self.data_producer.plot_cur - self.data_producer.plot_begin

        # 添加标的持仓数据
        observation = np.concatenate([observation, [before_market_close_sec, symbol_id, pos, profit]])

        # 检查是否结束
        terminated = data_done or acc_done

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # 重置
        self.need_reset = False
        # 清理图形对象
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            del self.fig
        # 数据
        self.data_producer.reset()
        symbol_id, before_market_close_sec, x, _, self.need_close, self.period_done = self._get_data()
        # 账户
        pos, profit = self.acc.reset()
        # 添加标的持仓数据
        x = np.concatenate([x, [before_market_close_sec, symbol_id, pos, profit]])
        # 初始化skip计数器
        self.skip_steps = 0
        return x, {}

    def show_notification(self, title, message, fig=None):
        """
        显示一个模态通知窗口，包含标题和消息内容。
        会阻塞主绘图窗口直到用户关闭通知。
        可选择性地显示一个图形。

        参数:
            title: 通知窗口标题
            message: 通知消息内容
            fig: 可选的matplotlib图形对象
        """
        try:
            from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QPushButton
            from PyQt5.QtCore import Qt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            
            # 关闭已存在的通知窗口
            if self.notification_window is not None:
                self.notification_window.close()
            
            # 创建模态对话框
            self.notification_window = QDialog()
            self.notification_window.setWindowTitle(title)
            self.notification_window.setModal(True)  # 设置为模态窗口，会阻塞其他窗口的交互
            layout = QVBoxLayout()
            
            # 添加消息文本
            msg_label = QLabel(message)
            msg_label.setWordWrap(True)  # 允许文本自动换行
            layout.addWidget(msg_label)
            
            # 如果提供了图形，添加到窗口中
            if fig is not None:
                canvas = FigureCanvasQTAgg(fig)
                layout.addWidget(canvas)
            
            # 添加确定按钮
            ok_button = QPushButton("确定")
            ok_button.clicked.connect(self.notification_window.accept)
            layout.addWidget(ok_button)
            
            self.notification_window.setLayout(layout)
            
            # 设置窗口属性
            self.notification_window.setWindowFlags(Qt.Dialog)
            self.notification_window.resize(1000, 700)
            
            # 显示对话框并等待用户响应
            # exec_() 会阻塞程序执行直到用户关闭窗口
            self.notification_window.exec_()
            
        except ImportError:
            # 如果没有PyQt5，回退到命令行显示
            print(f"\n{title}\n{message}")
            input("按回车键继续...") # 阻塞执行直到用户输入

    def render(self):
        pass

    def close(self):
        pass

    def plot(self, state):
        # 如果在skip模式中，直接返回hold动作
        if self.skip_steps > 0:
            self.skip_steps -= 1
            return 2

        # 重置选择的动作
        self.selected_action = None
        
        # 获取其他数据
        before_market_close_sec, symbol_id, pos, profit = state[-4:].astype(float)
        plot_data, plot_cur, id = self.data_producer.get_plot_data()

        # before_market_close_sec 逆标准化
        before_market_close_sec = before_market_close_sec * STD_SEC_BEFORE_CLOSE + MEAN_SEC_BEFORE_CLOSE
        
        # 创建图形和轴
        if not hasattr(self, 'fig'):
            # 创建一个新的窗口，并设置窗口标题
            self.fig = plt.figure(figsize=(15, 8))
            self.fig.canvas.manager.set_window_title('Trading Visualization')
            plt.subplots_adjust(bottom=0.2)
            self.ax = self.fig.add_subplot(111)
            
            # 创建右侧y轴用于显示收益率
            self.ax_return = self.ax.twinx()
            
            # 创建初始线条
            self.hist_bid_line, = self.ax.plot([], [], color='red', label='Historical Bid Price')
            self.hist_ask_line, = self.ax.plot([], [], color='green', label='Historical Ask Price')
            self.future_bid_line, = self.ax.plot([], [], color='red', alpha=0.3, label='Future Bid Price')
            self.future_ask_line, = self.ax.plot([], [], color='green', alpha=0.3, label='Future Ask Price')
            self.current_line = self.ax.axvline(x=0, color='blue', linestyle='--', label='Current Position')
            self.buy_point = self.ax.plot([], [], marker='^', color='red', markersize=10, label='Buy Point')[0]
            
            # 创建文本显示
            self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                        verticalalignment='top')
            
            # 创建按钮
            self.btn_sell = Button(plt.axes([0.1, 0.05, 0.15, 0.075]), 'SELL')
            self.btn_hold = Button(plt.axes([0.3, 0.05, 0.15, 0.075]), 'HOLD')
            self.btn_buy = Button(plt.axes([0.5, 0.05, 0.15, 0.075]), 'BUY')
            self.btn_skip = Button(plt.axes([0.7, 0.05, 0.15, 0.05]), 'SKIP')
            
            # 创建skip步数滑块
            self.skip_slider_ax = plt.axes([0.7, 0.10, 0.15, 0.03])
            self.skip_slider = Slider(self.skip_slider_ax, 'Skip', 1, 2400, valinit=10, valstep=1)
            
            # 修改回调函数
            def make_callback(action):
                def callback(event):
                    self.selected_action = action
                return callback
            
            def skip_callback(event):
                self.skip_steps = int(self.skip_slider.val)
                self.selected_action = 2  # Hold动作
            
            self.btn_buy.on_clicked(make_callback(0))
            self.btn_sell.on_clicked(make_callback(1))
            self.btn_hold.on_clicked(make_callback(2))
            self.btn_skip.on_clicked(skip_callback)
            
            self.ax.grid(True)
            self.ax.legend()
            plt.ion()  # 打开交互模式

            # 设置窗口属性
            try:
                from PyQt5.QtCore import Qt
                window = self.fig.canvas.manager.window
                # 设置窗口为普通窗口，不置顶
                window.setWindowFlags(Qt.Window)
                window.show()
            except ImportError:
                pass
        
        # 更新数据
        bid_price = plot_data['bid'].values
        ask_price = plot_data['ask'].values
        
        # 修改数据范围计算，固定显示450个点
        display_points = 450
        half_before = 150  # 当前位置前150条数据
        half_after = 300   # 当前位置后300条数据
        
        # 计算实际可用的数据范围
        total_points = len(bid_price)
        
        # 根据当前位置计算起始和结束索引
        start_idx = max(0, min(plot_cur - half_before, total_points - display_points))
        end_idx = min(total_points, start_idx + display_points)
        
        # 如果end_idx到达末尾，向前调整start_idx确保显示450个点
        if end_idx == total_points:
            start_idx = max(0, end_idx - display_points)
        
        # 计算x轴坐标，保持实际数据索引
        x_coords = range(start_idx, end_idx)
        current_x = plot_cur

        # 计算价格显示范围
        visible_bid = bid_price[start_idx:end_idx]
        visible_ask = ask_price[start_idx:end_idx]
        price_min = np.min(visible_bid)
        price_max = np.max(visible_ask)
        price_range = price_max - price_min
        
        # 为价格图留出70%的空间，上下各留5%边距
        price_min = price_min - price_range * 0.05
        price_max = price_max + price_range * 0.05
        
        # 将最小和最大价格调整为0.001的整数倍
        price_min = np.floor(price_min * 1000) / 1000
        price_max = np.ceil(price_max * 1000) / 1000
        
        # 设置y轴范围
        self.ax.set_ylim(price_min, price_max)
        
        # 计算合适的刻度间隔（确保是0.001的整数倍）
        # 根据显示范围自动计算合适的刻度数量（约8-12个刻度）
        desired_ticks = 10
        tick_range = price_max - price_min
        tick_step = tick_range / desired_ticks
        # 将步长调整为0.001的整数倍
        tick_step = np.ceil(tick_step * 1000) / 1000
        
        # 生成刻度位置
        ticks = np.arange(price_min, price_max + tick_step, tick_step)
        self.ax.set_yticks(ticks)
        # 设置刻度标签格式，保证显示3位小数
        self.ax.set_yticklabels([f'{tick:.3f}' for tick in ticks])

        # 更新线条数据，使用实际索引
        self.hist_bid_line.set_data(x_coords[:current_x-start_idx], bid_price[start_idx:current_x])
        self.hist_ask_line.set_data(x_coords[:current_x-start_idx], ask_price[start_idx:current_x])
        self.future_bid_line.set_data(x_coords[current_x-start_idx:], bid_price[current_x:end_idx])
        self.future_ask_line.set_data(x_coords[current_x-start_idx:], ask_price[current_x:end_idx])
        self.current_line.set_xdata([current_x, current_x])
        
        # 设置x轴显示范围
        self.ax.set_xlim(start_idx, end_idx)
        
        # 更新买入点和收益率图
        self.ax_return.clear()  # 清除旧的收益率图
        if self.last_buy_idx != -1:
            # 修改买入点显示逻辑
            buy_point_set_idx = max(start_idx, self.last_buy_idx)
            # 限制买入点价格在可视范围内
            buy_point_set_price = ask_price[self.last_buy_idx] - 0.0003
            buy_point_set_price = min(max(buy_point_set_price, price_min), price_max)
            self.buy_point.set_data([buy_point_set_idx], [buy_point_set_price])
            
            # 计算收益率序列
            buy_price = ask_price[self.last_buy_idx] * (1 + 5e-5)  # 买入价加手续费
            sell_prices = bid_price[self.last_buy_idx:] * (1 - 5e-5)  # 卖出价减手续费
            returns = np.log(sell_prices / buy_price)
            
            # 为收益率图设置独立的显示范围
            # 使用价格范围下方30%的空间显示收益率
            returns_base = price_min  # 基准线位置
            returns_height = price_range * 0.3  # 收益率图高度
            
            # 计算可视范围内的收益率
            valid_start = max(0, start_idx - self.last_buy_idx)
            valid_end = current_x - self.last_buy_idx
            future_start = valid_end
            future_end = end_idx - self.last_buy_idx

            # 使用可视范围内的收益率计算缩放比例
            visible_returns = returns[valid_start:future_end]  # 包含历史和未来部分
            max_abs_return = max(abs(np.min(visible_returns)), abs(np.max(visible_returns)))
            if max_abs_return > 0:
                returns_scaled = returns * (returns_height / (2 * max_abs_return)) + returns_base
            else:
                returns_scaled = returns + returns_base

            # 历史收益率
            if len(returns_scaled) > 0 and valid_end > valid_start:
                hist_returns = returns_scaled[valid_start:valid_end]
                x_coords_returns = range(start_idx + (valid_start-(start_idx-self.last_buy_idx)), current_x)
                
                if len(x_coords_returns) == len(hist_returns):
                    # 使用原始returns来判断颜色，而不是缩放后的值
                    orig_returns = returns[valid_start:valid_end]
                    colors = ['green' if r > 0 else 'red' for r in orig_returns]
                    self.ax_return.bar(x_coords_returns, hist_returns-returns_base, bottom=returns_base,
                                    color=colors, width=1.0, alpha=0.7)
            
                # 未来收益率（淡显）
                if future_start < len(returns_scaled):
                    future_returns = returns_scaled[future_start:future_end]
                    if len(future_returns) > 0:
                        x_coords_future = range(current_x, current_x + len(future_returns))
                        # 使用原始returns来判断颜色，而不是缩放后的值
                        orig_returns = returns[future_start:future_end]
                        colors = ['green' if r > 0 else 'red' for r in orig_returns]
                        self.ax_return.bar(x_coords_future, future_returns-returns_base, bottom=returns_base,
                                        color=colors, alpha=0.3, width=1.0)
                
            # 添加基准线
            self.ax_return.axhline(y=returns_base, color='black', linestyle='-', linewidth=0.5)
            
            # 设置右侧y轴的刻度标签为实际收益率值
            self.ax_return.set_ylim(price_min, price_max)
            # 只在收益率区域显示刻度
            returns_ticks = np.linspace(returns_base, returns_base + returns_height, 5)
            self.ax_return.set_yticks(returns_ticks)
            self.ax_return.set_yticklabels([f'{((y-returns_base)/returns_height * max_abs_return):.2%}' for y in returns_ticks])
        else:
            self.buy_point.set_data([], [])
        
        # 更新信息文本
        text = f'Position: {pos:.0f}\nProfit: {profit:.4f}\n'
        text += f'Time to Close: {before_market_close_sec:.0f}s'
        self.info_text.set_text(text)
        log(text)
        
        # 更新标题
        self.ax.set_title(f'Trading Data {self.data_producer.code} {self.data_producer.date} (ID: {id})')
        
        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # 等待按钮被点击，使用更温和的循环
        while self.selected_action is None:
            plt.pause(0.1)  # 降低检查频率
            
        return self.selected_action

