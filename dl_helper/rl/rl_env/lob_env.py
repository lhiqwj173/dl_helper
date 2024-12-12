import os
import random
import datetime
import numpy as np
import pandas as pd
import gymnasium as gym
import gymnasium.spaces as spaces
import pickle
from py_ext.tool import log

from dl_helper.tool import calc_sharpe_ratio, calc_sortino_ratio, calc_max_drawdown, calc_total_return

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

ILLEGAL_REWARD = -100

class data_producer:
    """
    遍历日期文件，每天随机选择一个标的
    当天的数据读取完毕后，需要强制平仓
    """
    def __init__(self, data_folder, _type='train', his_len=100, file_num=0, simple_test=False):
        # 快速测试
        self.simple_test = simple_test
        if self.simple_test:
            file_num = 5

        self.his_len = his_len
        self.file_num = file_num

        # 数据
        self.data_folder = data_folder
        self.data_type = _type
        self.files = []
        
        # 数据内容
        # ids, mean_std, x, all_self.all_raw_data_data
        self.ids = []
        self.mean_std = []
        self.x = []
        self.all_raw_data = None

        # 数据索引
        self.idxs = []
        self.col_idx = {}
        # 买卖1档价格
        self.ask_price = 0
        self.bid_price = 0
        # id
        self.id = ''
        # 当前日期数据停止标志
        self.date_file_done = False

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

        if self.simple_test:
            self.ids = self.ids[:8000]
            self.mean_std = self.mean_std[:8000]
            self.x = self.x[:8000]

        # 解析标的 随机挑选一个标的数据
        symbols = np.array([i.split('_')[0] for i in self.ids])
        unique_symbols = np.unique(symbols)
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
                raw[col] = self.all_raw_data[col].ffill()
        if 'DB卖1量' in all_cols and 'DS买1量' in all_cols: 
            # 成交数据
            deal_cols = [i for i in all_cols if i.startswith('D')]
            deal_raw = self.all_raw_data.loc[:, deal_cols]
            self.all_raw_data.loc[(deal_raw == 0).all(axis=1), ['DB卖1量', 'DS买1量']] = 1
        # 40档位价量数据nan处理
        if 'BASE卖1量' in all_cols and 'BASE买1量' in all_cols:
            # 价格nan填充, 使用上一个档位数据 +-0.001 进行填充
            for i in range(2, 11):
                # 买价
                self.all_raw_data.loc[:, f'BASE买{i}价'] = self.all_raw_data[f'BASE买{i}价'].fillna(self.all_raw_data[f'BASE买{i-1}价'] - 0.001)

                # 卖价
                self.all_raw_data.loc[:, f'BASE卖{i}价'] = self.all_raw_data[f'BASE卖{i}价'].fillna(self.all_raw_data[f'BASE卖{i-1}价'] + 0.001)

            # 量nan，用0填充
            vol_cols = [i for i in list(self.all_raw_data) if i.startswith('BASE') and '价' not in i]
            self.all_raw_data[vol_cols] = self.all_raw_data[vol_cols].fillna(0)

        # 记录需要的索引，供后续转为numpy时使用
        # BASE买1价 / BASE卖1价
        for col in ['BASE买1价', 'BASE卖1价']:
            self.col_idx[col] = self.all_raw_data.columns.get_loc(col)

        # self.all_raw_data 转为 numpy
        self.all_raw_data = self.all_raw_data.values

        # 测试用
        pickle.dump((self.all_raw_data, self.mean_std, self.x), open(f'{self.data_type}_raw_data.pkl', 'wb'))

        # 载入了新的日期文件数据
        # 重置日期文件停止标志
        self.date_file_done = False

    def set_data_type(self, _type):
        self.data_type = _type

    def data_size(self):
        return 130*100

    def use_data_split(self, raw, ms):
        """
        使用数据分割
        raw 是完整的 pickle 切片
        ms 是标准化数据
        都是 numpy 数组
        """
        return raw[:, :130], ms[:130, :]

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
        返回 symbol_id,x, done, need_close, date_done
        """
        # 测试用
        print(self.idxs[0])

        # 检查日期文件结束
        if self.date_file_done:
            # load 下一个日期文件的数据
            self._load_data()

        # 准备观察值
        a, b = self.x[self.idxs[0][0]]
        if b-a > self.his_len:# 修正历史数据长度
            a = b - self.his_len
        raw = self.all_raw_data[a: b, :]
        # 记录 买卖1档 的价格
        self.store_bid_ask_1st_data(raw)
        # 数据标准化
        ms = pd.DataFrame(self.mean_std[self.idxs[0][0]]).values
        x, ms = self.use_data_split(raw, ms)
        x -= ms[:, 0]
        x /= ms[:, 1]

        # 标的id
        symbol_id = self.idxs[0][2]

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
            self.idxs[0][0] += 1

        return symbol_id, x, all_done, need_close, self.date_file_done

    def reset(self):
        self._pre_files()
        self._load_data()

class Account:
    """
    账户类，用于记录交易状态和计算收益
    """
    def __init__(self, fee_rate=5e-5):
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
        self.bm_next_open = True
        
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
            if self.pos == 0 and len(self.net_raw):
                self.net_raw.append(self.net_raw[-1])
            else:
                self.net_raw.append(bid_price)

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
                    net = np.array(self.net_raw)
                    net[1:] -= step_fee
                    # 计算对数收益率序列
                    log_returns = np.diff(np.log(net))
                    # 计算指标
                    res['sortino_ratio'] = calc_sortino_ratio(log_returns)
                    res['sharpe_ratio'] = calc_sharpe_ratio(log_returns)
                    res['max_drawdown'] = calc_max_drawdown(log_returns)
                    res['total_return'] = calc_total_return(log_returns)

                    # 计算基准净值的评价指标
                    buy_fee_bm = self.net_raw_bm[0] * self.fee_rate
                    sell_fee_bm = self.net_raw_bm[-1] * self.fee_rate
                    step_fee_bm = (buy_fee_bm + sell_fee_bm) / (len(self.net_raw_bm) - 1)
                    net_bm = np.array(self.net_raw_bm)
                    net_bm[1:] -= step_fee_bm
                    # 计算对数收益率序列
                    log_returns_bm = np.diff(np.log(net_bm))
                    res['sortino_ratio_bm'] = calc_sortino_ratio(log_returns_bm)
                    res['sharpe_ratio_bm'] = calc_sharpe_ratio(log_returns_bm)
                    res['max_drawdown_bm'] = calc_max_drawdown(log_returns_bm)
                    res['total_return_bm'] = calc_total_return(log_returns_bm)
                else:
                    res['sortino_ratio'] = 0
                    res['sharpe_ratio'] = 0
                    res['max_drawdown'] = 0
                    res['total_return'] = 0
                    res['sortino_ratio_bm'] = 0
                    res['sharpe_ratio_bm'] = 0
                    res['max_drawdown_bm'] = 0
                    res['total_return_bm'] = 0
        else:
            # 不合法的操作，交易全部清空
            self.reset()

        return legal, self.pos, unrealized_profit, res
        
    def reset(self):
        """
        重置账户状态
        """
        self.pos = 0
        self.cost = 0 
        self.profit = 0
        self.net_raw = []
        return self.pos, 0

class LOB_trade_env(gym.Env):
    """
    用于 LOB 的强化学习环境
    返回的 obs 结构:
        lob数据 + 持仓 + 未实现收益率
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, data_producer: data_producer):
        """
        :param data_producer: 数据生产器
        """
        super().__init__()
        
        # 数据生产器
        self.data_producer = data_producer
        self.date_done = False

        # 账户数据
        self.need_close = False # 下一个动作无论是什么，都要对做平仓操作(因为没有更多的连续数据)
        self.acc = Account()

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data_producer.data_size() + 1,), dtype=np.float32)

        # 测试数据集 预测输出文件
        self.predict_file = ''

    def set_data_type(self, _type):
        if _type in ['val', 'test']:
            # 切换到测试数据集，创建预测输出文件
            self.predict_file = f'{_type}_{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")}.csv'
        self.data_producer.set_data_type(_type)

    def _get_data(self):
        # 获取数据
        symbol_id, x, all_data_done, need_close, date_done = self.data_producer.get()
        x = x.reshape(-1)
        return symbol_id, x, all_data_done, need_close, date_done

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
            return ILLEGAL_REWARD, False, pos, profit

        # 只有平仓才给与reward
        # 同时记录评价指标
        if need_close or action==1:
            info['close'] = True
            reward = res['total_return'] + res['max_drawdown']
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
        if not os.path.exists(self.predict_file):
            with open(self.predict_file, 'w') as f:
                f.write('id,predict\n')
        with open(self.predict_file, 'a') as f:
            f.write(f'{self.data_producer.id},{int(action)}\n')
    
    def step(self, action):
        # 如果是test数据集，需要输出预测数据文件
        if self.data_producer.data_type in ['val', 'test']:
            self.out_test_predict(action)

        info = {
            'close': False,# 若为True, 需要回溯属于本次交易的所有时间步, 修改 reward=收益率
            'date_done': self.date_done,
        }

        # 计算奖励
        reward, acc_done, pos, profit = self._cal_reward(action, self.need_close, info)

        # 获取下一个状态的数据
        symbol_id, observation, data_done, self.need_close, self.date_done = self._get_data()
        # 添加标的持仓数据
        observation = np.concatenate([observation, [symbol_id, pos,profit]])

        # 检查是否结束
        terminated = data_done or acc_done

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 数据
        self.data_producer.reset()
        symbol_id, x, _, self.need_close, self.date_done = self._get_data()
        # 账户
        pos, profit = self.acc.reset()
        # 添加标的持仓数据
        x = np.concatenate([x, [symbol_id, pos,profit]])
        return x, {}

    def render(self):
        pass

    def close(self):
        pass