from abc import ABC, abstractmethod
import pickle

from py_ext.tool import log
import numpy as np
"""
奖励系统
游戏终止的本质: 无法继续游戏最终达成目标

最终目标 0 : 日内累计最大的超额收益, 尽可能的控制风险
最终目标 1 : 日内累计超额收益 > 0, 超额亏损 < 0.5%
    游戏终止: 
        成功: 达成目标
        失败: 超额亏损 > 0.5%                   -STD_REWARD
        其他: 未达成目标 and 超额亏损 <= 0.5%
    最终奖励: 超额收益 * STD_REWARD
"""

class RewardStrategy(ABC):
    @abstractmethod
    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        """计算奖励和是否结束游戏"""
        pass

class BlankRewardStrategy(RewardStrategy):
    """
    空白奖励策略
    无奖励 不结束游戏
    """
    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        return 0, False


class ExcessReturnRewardStrategy(RewardStrategy):
    def _excess_return_reward(self, excess_return, STD_REWARD):
        # 假设超额收益上限为 0.01
        excess_return_rate = excess_return / 0.01

        # 限制范围
        excess_return_rate = max(min(excess_return_rate, 1), -1)

        # 计算奖励
        reward = excess_return_rate * STD_REWARD

        return reward

class EndPositionRewardStrategy(ExcessReturnRewardStrategy):
    """
    当天结束奖励策略 
    """
    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        # 超额收益率
        excess_return = res['excess_return']

        # 计算奖励
        reward = self._excess_return_reward(excess_return, STD_REWARD)

        return reward, False

class ClosePositionRewardStrategy(ExcessReturnRewardStrategy):
    """
    平仓奖励策略
    """
    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        # 计算交易的超额收益率 
        excess_return = res['close_excess_return']

        # 计算奖励
        reward = self._excess_return_reward(excess_return, STD_REWARD)

        return reward, False
    
class HoldPositionRewardStrategy(RewardStrategy):
    """
    持仓奖励策略
    处理非均衡仓位的奖励
    """
    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        # # 计算当前步的超额收益率
        # excess_return = res['step_return'] - res['step_return_bm']

        # # 每步的奖励不在放大，直接返回
        # return excess_return, False

        return 0, False

class BalanceRewardStrategy(RewardStrategy):
    """
    均衡仓位奖励策略
    """
    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        return 0, False

class RewardCalculator:
    def __init__(self, strategies_class = {
            'end_position': EndPositionRewardStrategy,
            'close_position': ClosePositionRewardStrategy,
            'hold_position': HoldPositionRewardStrategy,
            'balance': BalanceRewardStrategy
        }):
        """
        策略字典
        end_position: 当天结束奖励策略
        close_position: 平仓奖励策略
        hold_position: 持仓奖励策略
        balance: 均衡仓位奖励策略
        """
        self.strategies = {k: v() for k, v in strategies_class.items()}

    def calculate_reward(self, env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm):
        if need_close:
            # 当天结束
            strategy = self.strategies['end_position']
            print(f'reward end_position')
        elif close_net_raw_last_change_bm:
            # 最近的非均衡仓位平仓了
            strategy = self.strategies['close_position']
            print(f'reward close_position')
        else:
            if acc.pos in [-1, 1]:
                # 非均衡仓位
                strategy = self.strategies['hold_position']
                print(f'reward hold_position')
            else:
                # 均衡仓位
                strategy = self.strategies['balance']
                print(f'reward balance')
        reward, acc_done = strategy.calculate_reward(env_id, STD_REWARD, need_close, action, res, data_producer, acc, close_net_raw_last_change, close_net_raw_last_change_bm)
        return reward, acc_done
