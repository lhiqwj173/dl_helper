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
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL
from dl_helper.rl.rl_env.lob_trade.lob_const import LOCAL_DATA_FOLDER
from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env

from py_ext.tool import log, share_tensor

class LobExpert:
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

    def prepare_train_data_file(self, dtype):
        """
        通过 文件 准备数据
        """
        # 读取数据
        _date_file = f'{self.cur_data_file}.pkl'
        if os.path.exists(os.path.join(self.data_folder, 'train', _date_file)):
            _data_file_path = os.path.join(self.data_folder, 'train', _date_file)
        elif os.path.exists(os.path.join(self.data_folder, 'val', _date_file)):
            _data_file_path = os.path.join(self.data_folder, 'val', _date_file)
        elif os.path.exists(os.path.join(self.data_folder, 'test', _date_file)):
            _data_file_path = os.path.join(self.data_folder, 'test', _date_file)
        else:
            raise ValueError(f'{_date_file} not in {self.data_folder}/train')
        ids, mean_std, x, self.lob_data = pickle.load(open(_data_file_path, 'rb'))

        # 距离市场关闭的秒数
        dt = datetime.datetime.strptime(f'{self.cur_data_file} 15:00:00', '%Y%m%d %H:%M:%S')
        dt = pytz.timezone('Asia/Shanghai').localize(dt)
        close_ts = int(dt.timestamp())
        before_market_close_sec = np.array([int(i.split('_')[1]) for i in ids])
        before_market_close_sec = close_ts - before_market_close_sec

        # 样本的索引范围
        begin_idx, end_idx = self.shared_data.data[2], self.shared_data.data[3]
        # 最后一个也可以取到
        end_idx += 1

        # 截取范围
        idx = [i[1]-1 for i in x[begin_idx:end_idx]]
        self.lob_data = self.lob_data.iloc[idx].reset_index(drop=True)

        # 只保留 'BASE买1价', 'BASE卖1价'
        self.lob_data = self.lob_data[['BASE买1价', 'BASE卖1价']]

        # 距离市场关闭的秒数
        self.lob_data['before_market_close_sec'] = [i for i in before_market_close_sec[begin_idx:end_idx]]
        self.lob_data['before_market_close_sec'] /= MAX_SEC_BEFORE_CLOSE

        if dtype == np.float32:
            self.lob_data['before_market_close_sec'] = self.lob_data['before_market_close_sec'].astype(np.float32)

        # 计算潜在收益
        trades, total_log_return, valleys, peaks = max_profit_reachable(
            # 去掉第一个, 第一个数据无法成交
            self.lob_data['BASE买1价'].iloc[1:], 
            self.lob_data['BASE卖1价'].iloc[1:], 
            rep_select='random'
        )# 增加随机泛化
        # plot_trades((self.lob_data['BASE买1价']+self.lob_data['BASE卖1价'])/2, trades, valleys, peaks)

        # b/s/h
        # 无需提前一个k线，发出信号
        # trades 中的索引0实际是 lob_data 中的索引1
        # 沿用 索引0 就已经提前了一个k线
        buy_idx = [i[0] for i in trades]
        sell_idx = [i[1] for i in trades]
        self.lob_data.loc[buy_idx, 'action'] = ACTION_BUY
        self.lob_data.loc[sell_idx, 'action'] = ACTION_SELL
        self.lob_data['action'] = self.lob_data['action'].ffill()
        self.lob_data['action'] = self.lob_data['action'].fillna(ACTION_SELL)

    def prepare_train_data(self, dtype):
        """
        通过 env 准备数据
        """
        self.lob_data = self.env.data_producer.all_raw_data.copy()

        # 样本的索引范围
        begin_idx, end_idx, code_idx = self.env.data_producer.idxs[0]
        # 准备时，env 已经get一个数据，导致 begin_idx+1, 需要还原
        begin_idx -= 1
        # 最后一个也可以取到
        end_idx += 1

        # 截取范围
        idx = [i[1]-1 for i in self.env.data_producer.x[begin_idx:end_idx]]
        self.lob_data = self.lob_data.iloc[idx].reset_index(drop=True)

        # 只保留 'BASE买1价', 'BASE卖1价'
        self.lob_data = self.lob_data[['BASE买1价', 'BASE卖1价']]

        # 距离市场关闭的秒数
        self.lob_data['before_market_close_sec'] = [i for i in self.env.data_producer.before_market_close_sec[begin_idx:end_idx]]
        self.lob_data['before_market_close_sec'] /= MAX_SEC_BEFORE_CLOSE

        if dtype == np.float32:
            self.lob_data['before_market_close_sec'] = self.lob_data['before_market_close_sec'].astype(np.float32)

        # 计算潜在收益
        trades, total_log_return, valleys, peaks = max_profit_reachable(
            # 去掉第一个, 第一个数据无法成交
            self.lob_data['BASE买1价'].iloc[1:], 
            self.lob_data['BASE卖1价'].iloc[1:], 
            rep_select='random'
        )# 增加随机泛化
        # plot_trades((self.lob_data['BASE买1价']+self.lob_data['BASE卖1价'])/2, trades, valleys, peaks)

        # b/s/h
        # 无需提前一个k线，发出信号
        # trades 中的索引0实际是 lob_data 中的索引1
        # 沿用 索引0 就已经提前了一个k线
        buy_idx = [i[0] for i in trades]
        sell_idx = [i[1] for i in trades]
        self.lob_data.loc[buy_idx, 'action'] = ACTION_BUY
        self.lob_data.loc[sell_idx, 'action'] = ACTION_SELL
        self.lob_data['action'] = self.lob_data['action'].ffill()
        self.lob_data['action'] = self.lob_data['action'].fillna(ACTION_SELL)

    def get_action(self, obs):
        """
        获取专家动作
        """
        # 检查数据是否一致 使用 self.env
        if self.cur_data_file != self.env.data_producer.cur_data_file or self.cur_symbol != self.env.data_producer.cur_symbol:
            log(f'prepare train data: {self.env.data_producer.cur_data_file}, {self.env.data_producer.cur_symbol}')
            self.prepare_train_data(dtype=obs.dtype)
            self.cur_data_file = self.env.data_producer.cur_data_file
            self.cur_symbol = self.env.data_producer.cur_symbol

        # # 检查数据是否一致 使用共享内存
        # _env_date = self.shared_data.data[0]
        # _env_code = self.shared_data.data[1]
        # if self.cur_data_file != _env_date or self.cur_symbol != _env_code:
        #     log(f'prepare train data: {_env_date}, {_env_code}')
        #     self.cur_data_file = _env_date
        #     self.cur_symbol = _env_code
        #     self.prepare_train_data_file(dtype=obs.dtype)

        # 距离市场关闭的秒数
        if len(obs.shape) == 1:
            before_market_close_sec = obs[-5]
        elif len(obs.shape) == 2:
            assert obs.shape[0] == 1
            before_market_close_sec = obs[0][-5]
        else:
            raise ValueError(f'obs.shape: {obs.shape}')
        
        # 查找 action
        data = self.lob_data[self.lob_data['before_market_close_sec'] == before_market_close_sec]
        assert len(data) == 1, f'len(data): {len(data)}'

        res = data['action'].iloc[0]
        if len(obs.shape) == 2:
            res = np.array([res])

        return res

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

    expert = LobExpert(env)

    obs, info = env.reset()
    action = expert.get_action(obs)
    print(action)

def play_lob_data_with_expert(render=True):
    import time

    code = '513050'
    date = '20240521'
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
    debug_date=[date],
    )

    expert = LobExpert(env)

    print('reset')
    obs, info = env.reset()

    dt= env.data_producer.step_use_data.iloc[-1].name
    if render:
        env.render()

    act = 1
    need_close = False
    while not need_close:
        act = expert.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(act)
        dt= env.data_producer.step_use_data.iloc[-1].name
        if render:
            env.render()
        need_close = terminated or truncated
        if render:
            time.sleep(0.1)
        
    env.close()
    print('all done')

def eval_expert():
    from stable_baselines3.common.evaluation import evaluate_policy

    code = '513050'
    env = LOB_trade_env({
        # 'data_type': 'val',# 训练/测试
        'data_type': 'train',# 训练/测试
        'his_len': 100,# 每个样本的 历史数据长度
        'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
        'use_symbols': [code],

        'train_folder': r'C:\Users\lh\Desktop\temp\lob_env',
        'train_title': 'test',
    },
    )

    expert = LobExpert(env)

    reward, _ = evaluate_policy(
        expert,
        env,
        n_eval_episodes=1,
    )
    print(f"Reward after training: {reward}")

if __name__ == '__main__':
    # test_expert()
    # play_lob_data_with_expert()
    eval_expert()