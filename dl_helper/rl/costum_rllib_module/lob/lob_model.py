
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import pandas as pd
import numpy as np

from dl_helper.rl.rl_env.lob_env import LOB_trade_env
from dl_helper.rl.rl_utils import BaseCustomPlotter

def _get_env(env):
    while not isinstance(env, LOB_trade_env):
        env = env.unwrapped
    return env

class LobCallbacks(DefaultCallbacks):
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
        if info is not None and 'act_criteria' in info:
            if info['data_type'] == 'train':
                metrics_logger.log_value("all_num", 1, reduce="sum")
                metrics_logger.log_value("illegal_num", int(info['act_criteria'] == -1), reduce="sum")
            else:
                metrics_logger.log_value("val_all_num", 1, reduce="sum")
                metrics_logger.log_value("val_illegal_num", int(info['act_criteria'] == -1), reduce="sum")

            if info['act_criteria'] != -1:
                if info['data_type'] == 'train':
                    metrics_logger.log_value("trade_num", 1, reduce="sum")
                    metrics_logger.log_value("win_num", int(info['act_criteria'] == 0), reduce="sum")
                    metrics_logger.log_value("win_ret", info['trade_return'] if info['act_criteria'] == 0 else 0, reduce="sum")
                    metrics_logger.log_value("loss_ret", abs(info['trade_return']) if info['act_criteria'] == 1 else 0, reduce="sum")

                    metrics_logger.log_value("max_drawdown", info['max_drawdown'])
                    metrics_logger.log_value("trade_return", info['trade_return'])
                    metrics_logger.log_value("hold_length", info['hold_length'])
                    metrics_logger.log_value("excess_return", info['trade_return'] - info['trade_return_bm'])
                else:
                    metrics_logger.log_value("val_trade_num", 1, reduce="sum")
                    metrics_logger.log_value("val_win_num", int(info['act_criteria'] == 0), reduce="sum")
                    metrics_logger.log_value("val_win_ret", info['trade_return'] if info['act_criteria'] == 0 else 0, reduce="sum")
                    metrics_logger.log_value("val_loss_ret", abs(info['trade_return']) if info['act_criteria'] == 1 else 0, reduce="sum")

                    metrics_logger.log_value("val_max_drawdown", info['max_drawdown'])
                    metrics_logger.log_value("val_trade_return", info['trade_return'])
                    metrics_logger.log_value("val_hold_length", info['hold_length'])
                    metrics_logger.log_value("val_excess_return", info['trade_return'] - info['trade_return_bm'])
            
    def on_train_result(
        self, *, algorithm, result, metrics_logger, **kwargs
    ):
        # 提取自定义指标并添加到训练结果中
        result.setdefault("custom_metrics", {
            "illegal_ratio": float('nan'),
            "trade_num": float('nan'),
            "win_ratio": float('nan'),
            "profit_loss_ratio": float('nan'),
            "max_drawdown": float('nan'),
            "trade_return": float('nan'),
            "hold_length": float('nan'),
            "excess_return": float('nan'),

            "val_illegal_ratio": float('nan'),
            "val_trade_num": float('nan'),
            "val_win_ratio": float('nan'),
            "val_profit_loss_ratio": float('nan'),
            "val_max_drawdown": float('nan'),
            "val_trade_return": float('nan'),
            "val_hold_length": float('nan'),
            "val_excess_return": float('nan'),
        })

        if 'env_runners' in result:
            result["custom_metrics"]["illegal_ratio"] = result["env_runners"]["illegal_num"] / result["env_runners"]["all_num"]
            
            if 'trade_num' in result["env_runners"]:
                result["custom_metrics"]["trade_num"] = result["env_runners"]["trade_num"]
                result["custom_metrics"]["win_ratio"] = result["env_runners"]["win_num"] / result["env_runners"]["trade_num"]
                result["custom_metrics"]["profit_loss_ratio"] = result["env_runners"]["win_ret"] / result["env_runners"]["loss_ret"]
                result["custom_metrics"]["max_drawdown"] = result["env_runners"]["max_drawdown"]
                result["custom_metrics"]["trade_return"] = result["env_runners"]["trade_return"]
                result["custom_metrics"]["hold_length"] = result["env_runners"]["hold_length"]
                result["custom_metrics"]["excess_return"] = result["env_runners"]["excess_return"]

            if 'val_trade_num' in result["env_runners"]:
                result["custom_metrics"]["val_trade_num"] = result["env_runners"]["val_trade_num"]
                result["custom_metrics"]["val_win_ratio"] = result["env_runners"]["val_win_num"] / result["env_runners"]["val_trade_num"]
                result["custom_metrics"]["val_profit_loss_ratio"] = result["env_runners"]["val_win_ret"] / result["env_runners"]["val_loss_ret"]
                result["custom_metrics"]["val_max_drawdown"] = result["env_runners"]["val_max_drawdown"]
                result["custom_metrics"]["val_trade_return"] = result["env_runners"]["val_trade_return"]
                result["custom_metrics"]["val_hold_length"] = result["env_runners"]["val_hold_length"]
                result["custom_metrics"]["val_excess_return"] = result["env_runners"]["val_excess_return"]

        for i in result["custom_metrics"]:
            assert not np.isnan(result["custom_metrics"][i]), f'custom_metrics:{i} is nan'

class LobPlotter(BaseCustomPlotter):
    def get_additional_plot_count(self):
        """
        返回需要额外绘制的图表数量
        custom_metrics_illegal_ratio,
        custom_metrics_trade_num,
        custom_metrics_win_ratio,
        custom_metrics_profit_loss_ratio,
        custom_metrics_max_drawdown,
        custom_metrics_trade_return,
        custom_metrics_hold_length,
        custom_metrics_excess_return

        custom_metrics_val_illegal_ratio,
        custom_metrics_val_trade_num,
        custom_metrics_val_win_ratio,
        custom_metrics_val_profit_loss_ratio,
        custom_metrics_val_max_drawdown,
        custom_metrics_val_trade_return,
        custom_metrics_val_hold_length,
        custom_metrics_val_excess_return
        """
        return 7
    def plot(self, out_data, axes_list):
        """
        子类必须实现
        绘制额外图表
        """
        datetime = pd.to_datetime(out_data['datetime'])

        # 1. illegal_ratio 和 win_ratio
        ax = axes_list[0]
        illegal_ratio_last = out_data['custom_metrics_illegal_ratio'].iloc[-1]
        win_ratio_last = out_data['custom_metrics_win_ratio'].iloc[-1]
        val_illegal_ratio_last = out_data['custom_metrics_val_illegal_ratio'].iloc[-1]
        val_win_ratio_last = out_data['custom_metrics_val_win_ratio'].iloc[-1]
        ax.plot(datetime, out_data['custom_metrics_illegal_ratio'], 'r-', label=f'illegal_ratio({illegal_ratio_last:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_win_ratio'], 'g-', label=f'win_ratio({win_ratio_last:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_illegal_ratio'], 'r-', label=f'val_illegal_ratio({val_illegal_ratio_last:.2f})')
        ax.plot(datetime, out_data['custom_metrics_val_win_ratio'], 'g-', label=f'val_win_ratio({val_win_ratio_last:.2f})')
        ax.set_title('Illegal Ratio & Win Ratio')
        ax.legend()

        # 2. trade_num
        ax = axes_list[1]
        trade_num_last = out_data['custom_metrics_trade_num'].iloc[-1]
        val_trade_num_last = out_data['custom_metrics_val_trade_num'].iloc[-1]
        ax.plot(datetime, out_data['custom_metrics_trade_num'], 'b-', label=f'trade_num({trade_num_last:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_trade_num'], 'b-', label=f'val_trade_num({val_trade_num_last:.2f})')
        ax.set_title('Trade Number')
        ax.legend()

        # 3. profit_loss_ratio
        ax = axes_list[2]
        profit_loss_ratio_last = out_data['custom_metrics_profit_loss_ratio'].iloc[-1]
        val_profit_loss_ratio_last = out_data['custom_metrics_val_profit_loss_ratio'].iloc[-1]
        ax.plot(datetime, out_data['custom_metrics_profit_loss_ratio'], 'g-', label=f'profit_loss_ratio({profit_loss_ratio_last:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_profit_loss_ratio'], 'g-', label=f'val_profit_loss_ratio({val_profit_loss_ratio_last:.2f})')
        ax.set_title('Profit Loss Ratio')
        ax.legend()

        # 5. max_drawdown
        ax = axes_list[3]
        max_drawdown_last = out_data['custom_metrics_max_drawdown'].iloc[-1]
        val_max_drawdown_last = out_data['custom_metrics_val_max_drawdown'].iloc[-1]
        ax.plot(datetime, out_data['custom_metrics_max_drawdown'], 'r-', label=f'max_drawdown({max_drawdown_last:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_max_drawdown'], 'r-', label=f'val_max_drawdown({val_max_drawdown_last:.2f})')
        ax.set_title('Max Drawdown')
        ax.legend()

        # 6. trade_return
        ax = axes_list[4]
        trade_return_last = out_data['custom_metrics_trade_return'].iloc[-1]
        val_trade_return_last = out_data['custom_metrics_val_trade_return'].iloc[-1]
        ax.plot(datetime, out_data['custom_metrics_trade_return'], 'g-', label=f'trade_return({trade_return_last:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_trade_return'], 'g-', label=f'val_trade_return({val_trade_return_last:.2f})')
        ax.set_title('Trade Return')
        ax.legend()

        # 7. hold_length
        ax = axes_list[5]
        hold_length_last = out_data['custom_metrics_hold_length'].iloc[-1]
        val_hold_length_last = out_data['custom_metrics_val_hold_length'].iloc[-1]
        ax.plot(datetime, out_data['custom_metrics_hold_length'], 'b-', label=f'hold_length({hold_length_last:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_hold_length'], 'b-', label=f'val_hold_length({val_hold_length_last:.2f})')
        ax.set_title('Hold Length')
        ax.legend()

        # 8. excess_return
        ax = axes_list[6]
        excess_return_last = out_data['custom_metrics_excess_return'].iloc[-1]
        val_excess_return_last = out_data['custom_metrics_val_excess_return'].iloc[-1]
        ax.plot(datetime, out_data['custom_metrics_excess_return'], 'g-', label=f'excess_return({excess_return_last:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_excess_return'], 'g-', label=f'val_excess_return({val_excess_return_last:.2f})')
        ax.set_title('Excess Return')
        ax.legend()






