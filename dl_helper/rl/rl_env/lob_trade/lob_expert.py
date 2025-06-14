"""
专家策略

通过透视未来数据, 给出最佳的交易决策
最大化收益率
"""

import os, pickle
import datetime
import random
import pytz
import pandas as pd
import numpy as np
import psutil
from dl_helper.train_param import in_kaggle
from dl_helper.tool import max_profit_reachable, plot_trades
from dl_helper.rl.rl_env.lob_trade.lob_const import MEAN_SEC_BEFORE_CLOSE, STD_SEC_BEFORE_CLOSE, MAX_SEC_BEFORE_CLOSE
from dl_helper.rl.rl_env.lob_trade.lob_const import USE_CODES, STD_REWARD
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL
from dl_helper.rl.rl_env.lob_trade.lob_const import LOCAL_DATA_FOLDER, KAGGLE_DATA_FOLDER, DATA_FOLDER
from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_utils import date2days, days2date
from dl_helper.tool import calculate_profit, calculate_sell_save, reset_profit_sell_save, process_lob_data_extended, process_lob_data_extended_sell_save, filte_no_move, fix_profit_sell_save

from py_ext.tool import log, share_tensor

class LobExpert_file():
    """
    专家策略
    通过 文件 准备数据
    """
    def __init__(self, env=None, rng=None, pre_cache=False, data_folder=DATA_FOLDER, cache_debug=False):
        """
        pre_cache: 是否缓存数据 

        """
        self._env = env
        self.rng = rng

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
                    self.prepare_train_data_file(date2days(_file.split('.')[0]), _data_file_path=_file_path)

        log(f'cache_all done, cache_data: {len(self.cache_data)} dates')

    def _prepare_data(self, begin_idx, end_idx, x, before_market_close_sec, dtype):
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

        # 对多个 profit<=0 / sell_save<=0 的连续块的处理 
        # 1. 无
        # 2. 第一个 <= 0 后都赋值0
        # 3. 最后一个 <= 0 前都赋值 > 0 
        # lob_data = process_lob_data_extended(lob_data)
        lob_data = process_lob_data_extended_sell_save(lob_data)

        # 第一个 profit > 0/ sell_save > 0 时, 不允许 买入信号后，价格（成交价格）下跌 / 卖出信号后，价格（成交价格）上涨，利用跳价
        lob_data = reset_profit_sell_save(lob_data)

        # no_move filter
        lob_data = filte_no_move(lob_data)

        # fix profit / sell_save
        lob_data = fix_profit_sell_save(lob_data)

        if self.cache_debug:
            try:
                # 写入文件
                _file_path = os.path.join(os.getenv('TEMP'), 'lob_data.csv')
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
            log(f'prepare data for {date_key} {symbol_key}, cache_data: {len(self.cache_data)} dates')
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
                            # 价格与当前一致
                            (data['BASE卖1价'] == data['BASE卖1价'].iloc[0]) & \
                            (data['BASE买1价'] == data['BASE买1价'].iloc[0]) & \
                                # 与 第一行数据之间没有发生 BASE卖1价 的下跌(小于第一行 BASE卖1价 的个数为0)
                                # TODO 需要实验测试
                                # 1. 不允许下跌
                                # 2. 不允许仍和的变化
                                ((data['BASE卖1价'] != data['BASE卖1价'].iloc[0]).cumsum() == 0) & \
                                ((data['BASE买1价'] != data['BASE买1价'].iloc[0]).cumsum() == 0)
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
                            (data['BASE买1价'] == data['BASE买1价'].iloc[0]) & \
                            (data['BASE卖1价'] == data['BASE卖1价'].iloc[0]) & \
                                # 与 第一行数据之间没有发生 BASE买1价 的上涨(小于第一行 BASE买1价 的个数为0)
                                ((data['BASE卖1价'] != data['BASE卖1价'].iloc[0]).cumsum() == 0) & \
                                ((data['BASE买1价'] != data['BASE买1价'].iloc[0]).cumsum() == 0)
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
