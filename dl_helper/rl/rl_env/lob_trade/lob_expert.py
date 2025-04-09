"""
专家策略

通过透视未来数据, 给出最佳的交易决策
最大化收益率
"""

import os, pickle
import datetime
import pytz
import pandas as pd
import numpy as np
from dl_helper.train_param import in_kaggle
from dl_helper.tool import max_profit_reachable, plot_trades
from dl_helper.rl.rl_env.lob_trade.lob_const import MEAN_SEC_BEFORE_CLOSE, STD_SEC_BEFORE_CLOSE, MAX_SEC_BEFORE_CLOSE
from dl_helper.rl.rl_env.lob_trade.lob_const import USE_CODES, MEAN_CODE_ID, STD_CODE_ID, MAX_CODE_ID, STD_REWARD
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL
from dl_helper.rl.rl_env.lob_trade.lob_const import LOCAL_DATA_FOLDER
from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_utils import date2days, days2date

from py_ext.tool import log, share_tensor

class LobExpert_env:
    """
    专家策略
    """
    def __init__(self, lob_env: LOB_trade_env=None):
        self.env = lob_env
        self.cur_data_file = None
        self.cur_symbol = None

        # # 共享内存，用于读取 env 的 日期/code/begin/end
        # self.shared_data = share_tensor('lob_env_data_producer', (4,), np.int64)

        # # 数据文件夹
        # if in_kaggle:
        #     input_folder = r'/kaggle/input'
        #     try:
        #         # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
        #         data_folder_name = os.listdir(input_folder)[0]
        #         self.data_folder = os.path.join(input_folder, data_folder_name)
        #     except:
        #         self.data_folder = r''
        # else:
        #     self.data_folder = LOCAL_DATA_FOLDER

    def _prepare_data(self, begin_idx, end_idx, x, full_lob_data, before_market_close_sec, dtype):
        # 截取范围
        x = x[begin_idx:end_idx]
        lob_data_begin = x[0][0]
        lob_data_end = x[-1][1]
        self.lob_data = full_lob_data.iloc[lob_data_begin: lob_data_end].reset_index(drop=True).copy()

        # 只保留 'BASE买1价', 'BASE卖1价'
        self.lob_data = self.lob_data[['BASE买1价', 'BASE卖1价']]

        # 距离市场关闭的秒数
        sample_idxs = [i[1]-1 for i in x]
        self.lob_data['before_market_close_sec'] = np.nan
        self.lob_data.loc[sample_idxs,'before_market_close_sec'] = [i for i in before_market_close_sec[begin_idx:end_idx]]
        self.lob_data['before_market_close_sec'] /= MAX_SEC_BEFORE_CLOSE

        if dtype == np.float32:
            self.lob_data['before_market_close_sec'] = self.lob_data['before_market_close_sec'].astype(np.float32)

        # 计算潜在收益
        trades, total_log_return, self.valleys, self.peaks = max_profit_reachable(
            # 去掉第一个, 第一个数据无法成交
            self.lob_data['BASE买1价'].iloc[1:], 
            self.lob_data['BASE卖1价'].iloc[1:], 
            rep_select='random'
        )# 增加随机泛化
        # plot_trades((self.lob_data['BASE买1价']+self.lob_data['BASE卖1价'])/2, trades, valleys, peaks)
        # 需要 +1
        self.valleys = [i+1 for i in self.valleys]
        self.peaks = [i+1 for i in self.peaks]

        # 添加到 lob_data 中
        self.lob_data.loc[self.valleys, 'valley_peak'] = 0
        self.lob_data.loc[self.peaks, 'valley_peak'] = 1

        # b/s/h
        # 无需提前一个k线，发出信号
        # trades 中的索引0实际是 lob_data 中的索引1
        # 沿用 索引0 就已经提前了一个k线
        buy_idx = [i[0] for i in trades]
        sell_idx = [i[1] for i in trades]
        self.lob_data.loc[buy_idx, 'action'] = ACTION_BUY
        self.lob_data.loc[sell_idx, 'action'] = ACTION_SELL

        # 设置 env 的潜在收益数据，用于可视化
        # 恢复到 full_lob_data 中 
        full_lob_data['action'] = np.nan
        full_lob_data['valley_peak'] = np.nan
        full_lob_data.iloc[lob_data_begin: lob_data_end, -2:] = self.lob_data.loc[:, ['action', 'valley_peak']].values

        self.lob_data['action'] = self.lob_data['action'].ffill()
        self.lob_data['action'] = self.lob_data['action'].fillna(ACTION_SELL)

    def prepare_train_data(self, dtype):
        """
        通过 env 准备数据
        """
        full_lob_data = self.env.data_producer.all_raw_data.copy()

        # 样本的索引范围
        begin_idx, end_idx, code_idx = self.env.data_producer.full_idxs[0]
        # 最后一个也可以取到
        end_idx += 1

        self._prepare_data(begin_idx, end_idx, self.env.data_producer.x, full_lob_data, self.env.data_producer.before_market_close_sec, dtype)
        self.env.add_potential_data(full_lob_data.loc[:, ['action', 'valley_peak']])

    @staticmethod
    def _get_action(obs, lob_data, valleys, peaks):
        """
        获取专家动作
        """
        # 距离市场关闭的秒数 / pos
        if len(obs.shape) == 1:
            before_market_close_sec = obs[-4]
            pos = obs[-2]
        elif len(obs.shape) == 2:
            assert obs.shape[0] == 1
            before_market_close_sec = obs[0][-4]
            pos = obs[0][-2]
        else:
            raise ValueError(f'obs.shape: {obs.shape}')
        
        # 查找 action
        data = lob_data[lob_data['before_market_close_sec'] == before_market_close_sec]
        assert len(data) == 1, f'len(data): {len(data)}'

        cur_idx = data.index[0]
        try:
            next_valley_idx = [i for i in valleys if i > cur_idx][0]
            next_peak_idx = [i for i in peaks if i > cur_idx][0]
        except:
            next_valley_idx = None
            next_peak_idx = None

        # if pos:
        #     if next_valley_idx is None or next_peak_idx is None:
        #         # 之后没有 底部/顶部
        #         # 维持仓位
        #         res = ACTION_BUY

        #     elif next_valley_idx < next_peak_idx:
        #         # 最近的是底部

        #         if lob_data['action'].isna().all():
        #             # 没有 action 卖出
        #             res = ACTION_SELL
        #         else:
        #             # 未来的数据
        #             future_data = lob_data.iloc[cur_idx:]
        #             # 下一个时刻的bid价格(卖出成交的价格) next_bid
        #             next_bid = lob_data['BASE买1价'].iloc[cur_idx+1]
        #             # 检查下一个买入信号之后的下一时刻的ask价格(之后买入成交的价格) buy_next_ask
        #             next_buy_idx = future_data[future_data['action'] == ACTION_BUY].index[0]
        #             buy_next_ask = lob_data['BASE卖1价'].iloc[next_buy_idx + 1]
        #             if next_bid > buy_next_ask+0.0005:# 最小刻度 0.001，+0.0005 避免误差
        #                 # 立即卖出可以减少亏损
        #                 res = ACTION_SELL
        #             else:
        #                 # 无法减少亏损，平添手续费
        #                 # 继续持有
        #                 res = ACTION_BUY
        #     else:
        #         if data['action'].iloc[0] == ACTION_SELL:
        #             assert next_peak_idx - cur_idx == 1, f'ACTION_SELL, next_peak_idx - cur_idx: {next_peak_idx - cur_idx}(should be 1)'
        #         # 最近的是顶部
        #         if next_peak_idx - cur_idx == 1:
        #             # 下一个时刻即为顶点
        #             # 若有持仓，立即平仓, 下一个时刻成交 > 顶部平仓
        #             res = ACTION_SELL
        #         else:
        #             # 下一个时刻不是顶点
        #             # 继续持有
        #             res = ACTION_BUY
        # else:
        #     # 按照 action 操作
        #     res = data['action'].iloc[0]
        #     if np.isnan(res):
        #         res = ACTION_SELL

        res = data['action'].iloc[0]

        if len(obs.shape) == 2:
            res = np.array([res])

        return res
    
    def check_need_prepare_data(self, obs):
        # 检查数据是否一致 使用 self.env
        if self.cur_data_file != self.env.data_producer.cur_data_file or self.cur_symbol != self.env.data_producer.cur_symbol:
            log(f'prepare train data: {self.env.data_producer.cur_data_file}, {self.env.data_producer.cur_symbol}')
            self.cur_data_file = self.env.data_producer.cur_data_file
            self.cur_symbol = self.env.data_producer.cur_symbol
            self.prepare_train_data(dtype=obs.dtype)

    def get_action(self, obs):
        """
        获取专家动作
        """
        self.check_need_prepare_data(obs)
        return self._get_action(obs, self.lob_data, self.valleys, self.peaks)

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

class LobExpert_file(LobExpert_env):
    """
    专家策略
    通过 文件 准备数据
    """
    def __init__(self):
        self.cur_date = None
        self.cur_symbol = None

        # 数据文件夹
        if in_kaggle:
            input_folder = r'/kaggle/input'
            try:
                # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
                data_folder_name = os.listdir(input_folder)[0]
                self.data_folder = os.path.join(input_folder, data_folder_name)
            except:
                self.data_folder = r''
        else:
            self.data_folder = LOCAL_DATA_FOLDER

    def prepare_train_data_file(self, date, symbol, dtype):
        """
        通过 文件 准备数据
        """
        # 读取数据
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
        symbols = np.array([i.split('_')[0] for i in ids])
        # 获取所有标的的起止索引
        symbol_mask = symbols == symbol
        symbol_indices = np.where(symbol_mask)[0]
        begin_idx = symbol_indices[0]
        end_idx = symbol_indices[-1] + 1

        self._prepare_data(begin_idx, end_idx, x, self.full_lob_data, before_market_close_sec, dtype)

    def add_potential_data_to_env(self, env):
        if self.need_add_potential_data_to_env:
            env.add_potential_data(self.full_lob_data.loc[:, ['action', 'valley_peak']])
            self.need_add_potential_data_to_env = False

    def check_need_prepare_data(self, obs):
        # 检查数据是否一致 使用 file
        if len(obs.shape) == 1:
            obs_date = obs[-1]
            obs_symbol = obs[-3]
        elif len(obs.shape) == 2:
            assert obs.shape[0] == 1
            obs_date = obs[0][-1]
            obs_symbol = obs[0][-3]
        else:
            raise ValueError(f'obs.shape: {obs.shape}')
        
        if self.cur_date != obs_date or self.cur_symbol != obs_symbol:
            self.cur_date = obs_date
            self.cur_symbol = obs_symbol
            _date = days2date(int(obs_date))
            _symbol = USE_CODES[int(obs_symbol * MAX_CODE_ID)]
            self.prepare_train_data_file(_date, _symbol, dtype=obs.dtype)
            self.need_add_potential_data_to_env = True

def test_expert(expert_cls):
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

    if expert_cls == LobExpert_env:
        expert = LobExpert_env(env)
    elif expert_cls == LobExpert_file:
        expert = LobExpert_file()
    else:
        raise ValueError(f'expert_cls: {expert_cls}')

    obs, info = env.reset()
    action = expert.get_action(obs)
    print(action)

def play_lob_data_with_expert(expert_cls, render=True):
    import time

    code = '513050'
    date = '20240521'
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
    # debug_date=[date],
    )

    if expert_cls == LobExpert_env:
        expert = LobExpert_env(env)
    elif expert_cls == LobExpert_file:
        expert = LobExpert_file()
    else:
        raise ValueError(f'expert_cls: {expert_cls}')

    print('reset')
    obs, info = env.reset(4)

    dt= env.data_producer.step_use_data.iloc[-1].name
    if render:
        env.render()

    act = 1
    need_close = False
    while not need_close:
        act = expert.get_action(obs)
        if expert_cls == LobExpert_file:
            expert.add_potential_data_to_env(env)

        obs, reward, terminated, truncated, info = env.step(act)
        dt= env.data_producer.step_use_data.iloc[-1].name
        if render:
            env.render()
        need_close = terminated or truncated
        if render:
            time.sleep(0.1)
        
    input('all done, press enter to close')
    env.close()

def eval_expert(expert_cls):
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

    if expert_cls == LobExpert_env:
        expert = LobExpert_env(env)
    elif expert_cls == LobExpert_file:
        expert = LobExpert_file()
    else:
        raise ValueError(f'expert_cls: {expert_cls}')

    reward, _ = evaluate_policy(
        expert,
        env,
        n_eval_episodes=1,
    )
    print(f"Reward after training: {reward}")

if __name__ == '__main__':
    # test_expert(LobExpert_env)
    # test_expert(LobExpert_file)
    # play_lob_data_with_expert(LobExpert_env, True)
    play_lob_data_with_expert(LobExpert_file, True)
    # eval_expert(LobExpert_env)
    # eval_expert(LobExpert_file)

    # dump_file = r"D:\code\dl_helper\get_action.pkl"
    # data = pickle.load(open(dump_file, 'rb'))
    # obs, lob_data, valleys, peaks = data
    # action = LobExpert._get_action(obs, lob_data, valleys, peaks)
    # print(action)
