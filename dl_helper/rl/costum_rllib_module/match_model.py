
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import pandas as pd
import numpy as np

from dl_helper.rl.rl_env.match_env import MATCH_trade_env
from dl_helper.rl.rl_utils import BaseCustomPlotter

def _get_env(env):
    while not isinstance(env, MATCH_trade_env):
        env = env.unwrapped
    return env

class MatchCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = 50

    def on_evaluate_start(self, *args, **kwargs):
        print('on_evaluate_start')
        # 获取 eval_env_runner
        algo = kwargs['algorithm'] 
        if algo.eval_env_runner_group is None:
            eval_env_runner = algo.env_runner_group.local_env_runner
        else:
            eval_env_runner = algo.eval_env_runner
        # 切换环境到 val模式
        for env in eval_env_runner.env.unwrapped.envs:
            _env = _get_env(env)
            _env.val()

    def on_evaluate_end(self, *args, **kwargs):
        print('on_evaluate_end')
        # 获取 eval_env_runner
        algo = kwargs['algorithm'] 
        if algo.eval_env_runner_group is None:
            eval_env_runner = algo.env_runner_group.local_env_runner
            # 只有本地 eval_env_runner 需要切换回 train模式
            for env in eval_env_runner.env.unwrapped.envs:
                _env = _get_env(env)
                _env.train()

    def on_episode_step(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ) -> None:
        # res = {
        #     'max_drawdown': np.nan,
        #     'max_drawdown_ticks': np.nan,
        #     'trade_return': np.nan,
        #     'step_return': np.nan,
        #     'hold_length': np.nan,
        #     'max_drawdown_bm': np.nan,
        #     'max_drawdown_ticks_bm': np.nan,
        #     'max_drawup_ticks_bm': np.nan,
        #     'drawup_ticks_bm_count': np.nan,
        #     'trade_return_bm': np.nan,
        #     'step_return_bm': np.nan,
        # }
        # 从环境的 info 中提取自定义指标
        info = episode.get_infos(-1)
        # print(f'info: \n{info}')

        key_type = '' if info['data_type'] == 'train' else 'val_'
        if info is not None:
            # 用于统计动作分布
            if info['action'] == 0:
                episode.add_temporary_timestep_data(f"{key_type}act_0_num", 1)
            elif info['action'] == 1:
                episode.add_temporary_timestep_data(f"{key_type}act_1_num", 1)
            else:
                episode.add_temporary_timestep_data(f"{key_type}act_m1_num", 1)

        if info['close_trade']:
            # 平仓后
            episode.add_temporary_timestep_data(f"{key_type}trade_num", 1)# 总交易次数(合法操作次数)
            if info['close_excess_return'] > 0:
                episode.add_temporary_timestep_data(f"{key_type}win_num", 1)# 盈利次数
                episode.add_temporary_timestep_data(f"{key_type}win_ret", info['close_excess_return'])# 盈利金额
            else:
                episode.add_temporary_timestep_data(f"{key_type}loss_ret", abs(info['close_excess_return']))# 亏损金额

            if info['force_close']:
                episode.add_temporary_timestep_data(f"{key_type}force_stop_num", 1)# 触发止损次数

            episode.add_temporary_timestep_data(f"{key_type}max_drawdown", info['max_drawdown'])
            episode.add_temporary_timestep_data(f"{key_type}trade_return", info['trade_return'])
            episode.add_temporary_timestep_data(f"{key_type}excess_return", info['trade_return'] - info['trade_return_bm'])

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        # 从环境的 info 中提取自定义指标
        info = episode.get_infos(-1)
        key_type = '' if info['data_type'] == 'train' else 'val_'

        # 各个动作的次数 每轮取和
        act_0s = episode.get_temporary_timestep_data(f"{key_type}act_0_num")
        act_1s = episode.get_temporary_timestep_data(f"{key_type}act_1_num")
        act_m1s = episode.get_temporary_timestep_data(f"{key_type}act_m1_num")
        metrics_logger.log_value(
            f"{key_type}act_0_num", np.sum(act_0s), reduce='sum', window=self.window
        )
        metrics_logger.log_value(
            f"{key_type}act_1_num", np.sum(act_1s), reduce='sum', window=self.window
        )
        metrics_logger.log_value(
            f"{key_type}act_m1_num", np.sum(act_m1s), reduce='sum', window=self.window
        )

        trade_num = np.sum(episode.get_temporary_timestep_data(f"{key_type}trade_num"))# 可能为 nan
        win_num = np.sum(episode.get_temporary_timestep_data(f"{key_type}win_num"))# 可能为 nan
        force_stop_num = np.sum(episode.get_temporary_timestep_data(f"{key_type}force_stop_num"))# 可能为 nan
        win_ret = np.sum(episode.get_temporary_timestep_data(f"{key_type}win_ret"))# 可能为 nan
        loss_ret = np.sum(episode.get_temporary_timestep_data(f"{key_type}loss_ret"))# 可能为 nan
        max_drawdown = np.min(episode.get_temporary_timestep_data(f"{key_type}max_drawdown")) if len(episode.get_temporary_timestep_data(f"{key_type}max_drawdown")) > 0 else np.nan# 可能为 nan
        trade_return = np.sum(episode.get_temporary_timestep_data(f"{key_type}trade_return"))# 可能为 nan
        excess_return = np.sum(episode.get_temporary_timestep_data(f"{key_type}excess_return"))# 可能为 nan
        # 求和
        metrics_logger.log_value(
            f"{key_type}trade_num", trade_num, reduce='sum', window=self.window  
        )
        metrics_logger.log_value(
            f"{key_type}win_num", win_num, reduce='sum', window=self.window
        )
        metrics_logger.log_value(
            f"{key_type}force_stop_num", force_stop_num, reduce='sum', window=self.window
        )
        metrics_logger.log_value(
            f"{key_type}win_ret", win_ret, reduce='sum', window=self.window
        )
        metrics_logger.log_value(
            f"{key_type}loss_ret", loss_ret, reduce='sum', window=self.window
        )

        # 求均值
        metrics_logger.log_value(
            f"{key_type}max_drawdown", max_drawdown, reduce='mean', window=self.window
        )
        metrics_logger.log_value(
            f"{key_type}trade_return", trade_return, reduce='mean', window=self.window
        )
        metrics_logger.log_value(
            f"{key_type}excess_return", excess_return, reduce='mean', window=self.window
        )
        

    def on_train_result(
        self, *, algorithm, result, metrics_logger, **kwargs
    ):
        
        # 提取自定义指标并添加到训练结果中
        result.setdefault("custom_metrics", {
            "trade_num": float('nan'),
            "win_ratio": float('nan'),
            "force_stop_ratio": float('nan'),
            "profit_loss_ratio": float('nan'),
            "max_drawdown": float('nan'),
            "trade_return": float('nan'),
            "excess_return": float('nan'),
            "act_0_pct": float('nan'),
            "act_1_pct": float('nan'),
            "act_m1_pct": float('nan'),

            "val_trade_num": float('nan'),
            "val_win_ratio": float('nan'),
            "val_force_stop_ratio": float('nan'),
            "val_profit_loss_ratio": float('nan'),
            "val_max_drawdown": float('nan'),
            "val_trade_return": float('nan'),
            "val_excess_return": float('nan'),
            "val_act_0_pct": float('nan'),
            "val_act_1_pct": float('nan'),
            "val_act_m1_pct": float('nan'),
        })

        # print(f'result: \n{result}')
        if 'env_runners' in result:
            total_acts = result["env_runners"]["act_0_num"] + result["env_runners"]["act_1_num"] + result["env_runners"]["act_m1_num"]
            result["custom_metrics"]["act_0_pct"] = result["env_runners"]["act_0_num"] / total_acts
            result["custom_metrics"]["act_1_pct"] = result["env_runners"]["act_1_num"] / total_acts
            result["custom_metrics"]["act_m1_pct"] = result["env_runners"]["act_m1_num"] / total_acts
            for i in ['act_0_pct', 'act_1_pct', 'act_m1_pct']:
                assert not np.isnan(result["custom_metrics"][i]), f'{i} is nan'

            if 'trade_num' in result["env_runners"]:
                result["custom_metrics"]["trade_num"] = result["env_runners"]["trade_num"] / self.window# 每轮平均交易数
                result["custom_metrics"]["win_ratio"] = result["env_runners"]["win_num"] / result["env_runners"]["trade_num"] if result["env_runners"]["trade_num"] > 0 else np.nan
                result["custom_metrics"]["force_stop_ratio"] = result["env_runners"]["force_stop_num"] / result["env_runners"]["trade_num"] if result["env_runners"]["trade_num"] > 0 else np.nan
                result["custom_metrics"]["profit_loss_ratio"] = result["env_runners"]["win_ret"] / result["env_runners"]["loss_ret"] if result["env_runners"]["loss_ret"] > 0 else np.nan
                result["custom_metrics"]["max_drawdown"] = result["env_runners"]["max_drawdown"]
                result["custom_metrics"]["trade_return"] = result["env_runners"]["trade_return"]
                result["custom_metrics"]["excess_return"] = result["env_runners"]["excess_return"]

        if 'evaluation' in result and 'env_runners' in result["evaluation"] and 'val_trade_num' in result["evaluation"]["env_runners"]:
            total_acts = result["evaluation"]["env_runners"]["val_act_0_num"] + result["evaluation"]["env_runners"]["val_act_1_num"] + result["evaluation"]["env_runners"]["val_act_m1_num"]
            result["custom_metrics"]["val_act_0_pct"] = result["evaluation"]["env_runners"]["val_act_0_num"] / total_acts
            result["custom_metrics"]["val_act_1_pct"] = result["evaluation"]["env_runners"]["val_act_1_num"] / total_acts
            result["custom_metrics"]["val_act_m1_pct"] = result["evaluation"]["env_runners"]["val_act_m1_num"] / total_acts
            for i in ['val_act_0_pct', 'val_act_1_pct', 'val_act_m1_pct']:
                assert not np.isnan(result["custom_metrics"][i]), f'{i} is nan'
            
            result["custom_metrics"]["val_trade_num"] = result["evaluation"]["env_runners"]["val_trade_num"] / self.window# 每轮平均交易数
            result["custom_metrics"]["val_win_ratio"] = result["evaluation"]["env_runners"]["val_win_num"] / result["evaluation"]["env_runners"]["val_trade_num"] if result["evaluation"]["env_runners"]["val_trade_num"] > 0 else np.nan
            result["custom_metrics"]["val_force_stop_ratio"] = result["evaluation"]["env_runners"]["val_force_stop_num"] / result["evaluation"]["env_runners"]["val_trade_num"] if result["evaluation"]["env_runners"]["val_trade_num"] > 0 else np.nan
            result["custom_metrics"]["val_profit_loss_ratio"] = result["evaluation"]["env_runners"]["val_win_ret"] / result["evaluation"]["env_runners"]["val_loss_ret"] if result["evaluation"]["env_runners"]["val_loss_ret"] > 0 else np.nan
            result["custom_metrics"]["val_max_drawdown"] = result["evaluation"]["env_runners"]["val_max_drawdown"]
            result["custom_metrics"]["val_trade_return"] = result["evaluation"]["env_runners"]["val_trade_return"]
            result["custom_metrics"]["val_excess_return"] = result["evaluation"]["env_runners"]["val_excess_return"]

class MatchPlotter(BaseCustomPlotter):
    def get_additional_plot_count(self):
        """
        返回需要额外绘制的图表数量
        """
        return 7
    def plot(self, out_data, axes_list):
        """
        子类必须实现
        绘制额外图表
        """
        datetime = pd.to_datetime(out_data['datetime'])

        # 1. win_ratio 和 force_stop_ratio
        ax = axes_list[0]
        win_ratio_last = out_data['custom_metrics_win_ratio'].iloc[-1]
        force_stop_ratio_last = out_data['custom_metrics_force_stop_ratio'].iloc[-1]
        val_win_ratio_last = out_data['custom_metrics_val_win_ratio'].iloc[-1]
        val_force_stop_ratio_last = out_data['custom_metrics_val_force_stop_ratio'].iloc[-1]
        ax.plot(out_data['custom_metrics_win_ratio'], 'g-', label=f'win_ratio({win_ratio_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_force_stop_ratio'], 'y-', label=f'force_stop_ratio({force_stop_ratio_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_val_win_ratio'], 'g-', label=f'val_win_ratio({val_win_ratio_last:.4f})')
        ax.plot(out_data['custom_metrics_val_force_stop_ratio'], 'y-', label=f'val_force_stop_ratio({val_force_stop_ratio_last:.4f})')
        ax.set_title('Win Ratio & Force Stop Ratio')
        ax.legend()

        # 2. act pct
        ax = axes_list[1]
        act_0_pct_last = out_data['custom_metrics_act_0_pct'].iloc[-1]
        act_1_pct_last = out_data['custom_metrics_act_1_pct'].iloc[-1]
        val_act_0_pct_last = out_data['custom_metrics_val_act_0_pct'].iloc[-1]
        val_act_1_pct_last = out_data['custom_metrics_val_act_1_pct'].iloc[-1]  
        ax.plot(out_data['custom_metrics_act_0_pct'], 'b-', label=f'act_0_pct({act_0_pct_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_act_1_pct'], 'g-', label=f'act_1_pct({act_1_pct_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_val_act_0_pct'], 'b-', label=f'val_act_0_pct({val_act_0_pct_last:.4f})')
        ax.plot(out_data['custom_metrics_val_act_1_pct'], 'g-', label=f'val_act_1_pct({val_act_1_pct_last:.4f})')
        ax.set_title('Act Pct')
        ax.legend()
        
        # 3. trade_num
        ax = axes_list[2]
        trade_num_last = out_data['custom_metrics_trade_num'].iloc[-1]
        val_trade_num_last = out_data['custom_metrics_val_trade_num'].iloc[-1]
        ax.plot(out_data['custom_metrics_trade_num'], 'b-', label=f'trade_num({trade_num_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_val_trade_num'], 'b-', label=f'val_trade_num({val_trade_num_last:.4f})')
        ax.set_title('Trade Number')
        ax.legend()

        # 4. profit_loss_ratio
        ax = axes_list[3]
        profit_loss_ratio_last = out_data['custom_metrics_profit_loss_ratio'].iloc[-1]
        val_profit_loss_ratio_last = out_data['custom_metrics_val_profit_loss_ratio'].iloc[-1]
        ax.plot(out_data['custom_metrics_profit_loss_ratio'], 'g-', label=f'profit_loss_ratio({profit_loss_ratio_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_val_profit_loss_ratio'], 'g-', label=f'val_profit_loss_ratio({val_profit_loss_ratio_last:.4f})')
        ax.set_title('Profit Loss Ratio')
        ax.legend()

        # 5. max_drawdown
        ax = axes_list[4]
        max_drawdown_last = out_data['custom_metrics_max_drawdown'].iloc[-1]
        val_max_drawdown_last = out_data['custom_metrics_val_max_drawdown'].iloc[-1]
        ax.plot(out_data['custom_metrics_max_drawdown'], 'r-', label=f'max_drawdown({max_drawdown_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_val_max_drawdown'], 'r-', label=f'val_max_drawdown({val_max_drawdown_last:.4f})')
        ax.set_title('Max Drawdown')
        ax.legend()

        # 6. trade_return
        ax = axes_list[5]
        trade_return_last = out_data['custom_metrics_trade_return'].iloc[-1]
        val_trade_return_last = out_data['custom_metrics_val_trade_return'].iloc[-1]
        ax.plot(out_data['custom_metrics_trade_return'], 'g-', label=f'trade_return({trade_return_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_val_trade_return'], 'g-', label=f'val_trade_return({val_trade_return_last:.4f})')
        ax.set_title('Trade Return')
        ax.legend()

        # 7. excess_return
        ax = axes_list[7]
        excess_return_last = out_data['custom_metrics_excess_return'].iloc[-1]
        val_excess_return_last = out_data['custom_metrics_val_excess_return'].iloc[-1]
        ax.plot(out_data['custom_metrics_excess_return'], 'g-', label=f'excess_return({excess_return_last:.4f})', alpha=0.4)
        ax.plot(out_data['custom_metrics_val_excess_return'], 'g-', label=f'val_excess_return({val_excess_return_last:.4f})')
        ax.set_title('Excess Return')
        ax.legend()






