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
from dl_helper.rl.rl_env.lob_trade.lob_const import USE_CODES, MEAN_CODE_ID, STD_CODE_ID, MAX_CODE_ID, STD_REWARD
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL
from dl_helper.rl.rl_env.lob_trade.lob_const import LOCAL_DATA_FOLDER, KAGGLE_DATA_FOLDER
from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_utils import date2days, days2date
from dl_helper.tool import calculate_profit, calculate_sell_save

from py_ext.tool import log, share_tensor

class LobExpert_file():
    """
    专家策略
    通过 文件 准备数据
    """
    def __init__(self, env=None, rng=None, pre_cache=False):
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
        if in_kaggle:
            self.data_folder = KAGGLE_DATA_FOLDER
            # input_folder = r'/kaggle/input'
            # try:
            #     # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
            #     data_folder_name = os.listdir(input_folder)[0]
            #     self.data_folder = os.path.join(input_folder, data_folder_name)
            # except:
            #     self.data_folder = r''
        else:
            self.data_folder = LOCAL_DATA_FOLDER

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
        for data_type in os.listdir(self.data_folder):
            type_folder = os.path.join(self.data_folder, data_type)
            if not os.path.isdir(type_folder):
                continue
            for data in os.listdir(type_folder):
                data_file = os.path.join(type_folder, data)
                if not os.path.isfile(data_file):
                    continue
                self.prepare_train_data_file(date2days(data.split('.')[0]), _data_file_path=data_file)

        log(f'cache_all done, cache_data: {len(self.cache_data)} dates')

    def _prepare_data(self, begin_idx, end_idx, x, before_market_close_sec, dtype):
        # 截取范围
        x = x[begin_idx:end_idx]
        lob_data_begin = x[0][0]
        lob_data_end = x[-1][1]
        lob_data = self.full_lob_data.iloc[lob_data_begin: lob_data_end].reset_index(drop=True).copy()

        # 只保留 'BASE买1价', 'BASE卖1价'
        lob_data = lob_data[['BASE买1价', 'BASE卖1价']]

        # 距离市场关闭的秒数
        sample_idxs = [i[1]-1 for i in x]
        lob_data['before_market_close_sec'] = np.nan
        lob_data.loc[sample_idxs,'before_market_close_sec'] = [i for i in before_market_close_sec[begin_idx:end_idx]]
        lob_data['before_market_close_sec'] /= MAX_SEC_BEFORE_CLOSE

        if dtype == np.float32:
            lob_data['before_market_close_sec'] = lob_data['before_market_close_sec'].astype(np.float32)

        # 计算潜在收益
        trades, total_log_return, self.valleys, self.peaks = max_profit_reachable(
            # 去掉第一个, 第一个数据无法成交
            lob_data['BASE买1价'].iloc[1:], 
            lob_data['BASE卖1价'].iloc[1:], 
            rep_select='random',
            rng=self.rng,
        )# 增加随机泛化
        # plot_trades((lob_data['BASE买1价']+lob_data['BASE卖1价'])/2, trades, valleys, peaks)
        # 需要 +1
        self.valleys = [i+1 for i in self.valleys]
        self.peaks = [i+1 for i in self.peaks]

        # 添加到 lob_data 中
        lob_data.loc[self.valleys, 'valley_peak'] = 0
        lob_data.loc[self.peaks, 'valley_peak'] = 1

        # b/s/h
        # 无需提前一个k线，发出信号
        # trades 中的索引0实际是 lob_data 中的索引1
        # 沿用 索引0 就已经提前了一个k线
        buy_idx = [i[0] for i in trades]
        sell_idx = [i[1] for i in trades]
        lob_data.loc[buy_idx, 'action'] = ACTION_BUY
        lob_data.loc[sell_idx, 'action'] = ACTION_SELL

        # 设置 env 的潜在收益数据，用于可视化
        # 恢复到 full_lob_data 中 
        self.full_lob_data['action'] = np.nan
        self.full_lob_data['valley_peak'] = np.nan
        self.full_lob_data.iloc[lob_data_begin: lob_data_end, -2:] = lob_data.loc[:, ['action', 'valley_peak']].values

        lob_data['action'] = lob_data['action'].ffill()
        lob_data['action'] = lob_data['action'].fillna(ACTION_SELL)

        # 计算 action==0 时点买入的收益
        lob_data = calculate_profit(lob_data)

        # 计算 action==1 时点卖出节省的收益
        lob_data = calculate_sell_save(lob_data)

        # 最终数据 action, before_market_close_sec, profit, sell_save
        lob_data = lob_data.loc[:, ['action', 'before_market_close_sec', 'profit', 'sell_save']]
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
            if os.path.exists(os.path.join(self.data_folder, 'train', _date_file)):
                _data_file_path = os.path.join(self.data_folder, 'train', _date_file)
            elif os.path.exists(os.path.join(self.data_folder, 'val', _date_file)):
                _data_file_path = os.path.join(self.data_folder, 'val', _date_file)
            elif os.path.exists(os.path.join(self.data_folder, 'test', _date_file)):
                _data_file_path = os.path.join(self.data_folder, 'test', _date_file)
            else:
                raise ValueError(f'{_date_file} not in {self.data_folder}/train')
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
            obs_symbol = obs[-3]
        elif len(obs.shape) == 2:
            assert obs.shape[0] == 1
            obs_date = obs[0][-1]
            obs_symbol = obs[0][-3]
        else:
            raise ValueError(f'obs.shape: {obs.shape}')
        
        # 如果不在缓存数据中，需要准备数据
        date_key = int(obs_date)
        symbol_key = int(obs_symbol*MAX_CODE_ID)
        if date_key not in self.cache_data or symbol_key not in self.cache_data[date_key]:
            log(f'prepare data for {date_key} {symbol_key}, cache_data: {len(self.cache_data)} dates')
            self.prepare_train_data_file(date_key, symbol_key, dtype=obs.dtype)
            self.need_add_potential_data_to_env = True

        return date_key, symbol_key

    @staticmethod
    def _get_action(obs, lob_data, valleys, peaks):
        """
        获取专家动作
        obs 单个样本
        """
        # 距离市场关闭的秒数 / pos
        before_market_close_sec = obs[-4]
        pos = obs[-2]
        
        # 查找 action
        data = lob_data[lob_data['before_market_close_sec'] == before_market_close_sec]
        assert len(data) == 1, f'len(data): {len(data)}'

        data = data.iloc[0]
        res = data['action']
        if res == ACTION_BUY:
            if pos == 0 and data['profit'] <= 0:
                # 若当前位置无持仓且买入收益为负
                # 维持卖出动作
                res = ACTION_SELL

        elif res == ACTION_SELL:
            if pos == 1 and data['sell_save'] < 0:
                # 若当前位置有持仓且卖出收益为负
                # 维持买入动作
                res = ACTION_BUY

        # if len(obs.shape) == 2:
        #     res = np.array([res])
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
            return self._get_action(obs, self.cache_data[date_key][symbol_key], self.valleys, self.peaks)
        elif len(obs.shape) == 2:
            rets = []
            for i in obs:
                date_key, symbol_key = self.check_need_prepare_data(i)
                rets.append(self._get_action(i, self.cache_data[date_key][symbol_key], self.valleys, self.peaks))
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
    init_pos = 1

    code = '513050'
    env = LOB_trade_env({
        # 'data_type': 'val',# 训练/测试
        'data_type': 'train',# 训练/测试
        'his_len': 30,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        'use_symbols': [code],
        'render_freq': 1,

        'train_folder': r'C:\Users\lh\Desktop\temp\lob_env',
        'train_title': 'test',

        'render_mode': 'human' if render else 'none',
    },
    debug_obs_date=debug_obs_date,
    debug_obs_time=debug_obs_time,
    debug_init_pos = init_pos,
    dump_bid_ask_accnet=True,
    )

    expert = LobExpert_file(pre_cache=False if render else True)

    rounds = 5
    rounds = 1
    for i in range(rounds):
        print('reset')
        seed = random.randint(0, 1000000)
        # seed = 755812
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
    # action = LobExpert._get_action(obs, lob_data, valleys, peaks)
    # print(action)
