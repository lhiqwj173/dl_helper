from abc import ABC, abstractmethod
import pickle
from dl_helper.tool import max_profit_reachable
import numpy as np

from py_ext.tool import log

from dl_helper.rl.rl_env.lob_trade.lob_const import RESULT_OPEN, RESULT_CLOSE, RESULT_HOLD

"""
奖励系统
游戏终止的本质: 无法继续游戏最终达成目标

最终目标 0 : 日内累计最大的收益(获取所有可能潜在的收益), 尽可能的控制回撤大小
最终目标 1 : 日内累计最大的收益(获取所有可能潜在的收益), 不出现指定最大回撤
最终目标 2 : 日内累计达到 潜在收益*0.3, 最大回撤0.5%
    游戏终止: 
        成功: 达成目标
        失败: 最大回撤 > 0.5%
        其他: 未达成目标 and 最大回撤 <= 0.5%
    最终奖励: min(策略收益率累计 / (日内可盈利收益率累计 * 0.3), 1) * FINAL_REWARD TODO
    终止惩罚: (最大回撤 > 0.5%) * -STD_REWARD
"""

class RewardStrategy(ABC):
    @abstractmethod
    def calculate_reward(
        self, 
        env_id=0, 
        STD_REWARD=100, 
        force_stop=False, 
        need_close=False, 
        action_result=RESULT_HOLD, 
        res={}, 
        data_producer=None,
        acc=None, 
        max_drawdown_threshold=0.005
    ):
        """计算奖励和是否结束游戏"""
        pass

class BlankRewardStrategy(RewardStrategy):
    """
    空白奖励策略
    无奖励 不结束游戏
    """
    def calculate_reward(self, **kwargs):
        return 0, False

class ClosePositionRewardStrategy(RewardStrategy):
    """
    平仓奖励策略
    处理已经开仓且需要平仓（need_close 或 action_result == RESULT_CLOSE）的情况。

    奖励逻辑
        1. 策略收益>0, 奖励恒为正 = 策略收益率 / 潜在收益率 * 标准奖励 (若潜在收益率为0, 则奖励为标准奖励)     >> 策略获得收益占潜在收益的比例
        2. 策略收益<=0, 奖励恒为负(惩罚) = max((策略收益率 - 潜在收益率) / 最大回撤阈值, -1) * 标准奖励        >> 策略亏损占最大回撤阈值的比例

        
    """
    def calculate_reward(
            self, 
            STD_REWARD = 100, 
            res = {}, 
            max_drawdown_threshold = 0.005,
            **kwargs
        ):
        potential_return = res['potential_return']
        acc_return = res['acc_return']

        if acc_return > 0:
            if potential_return == 0:
                raise ValueError(f'Maximum potential log return is 0, but closing log return({acc_return}) is not 0')
            reward = acc_return / potential_return * STD_REWARD if potential_return != 0 else STD_REWARD
        else:
            reward = max((acc_return - potential_return) / max_drawdown_threshold, -1) * STD_REWARD

        # 保留 float32 的精度
        reward = np.float32(reward)

        return reward, False  # acc_done 为 False
    
class HoldPositionRewardStrategy(RewardStrategy):
    """
    持仓奖励策略
    处理已经开仓且既不平仓也不为开仓步的情况。
    持仓下, 奖励与每步收益率正相关
    """
    def calculate_reward(
            self, 
            res={},
            **kwargs
        ):
        reward = res['step_return']  # 持仓每步收益率
        return reward, False

class ForceStopRewardStrategy(RewardStrategy):
    """
    强制止损奖励策略
    """
    def calculate_reward(
            self, 
            STD_REWARD=100,
            **kwargs
        ):
        return -STD_REWARD, True

class NoPositionRewardStrategy_0(RewardStrategy):
    """
    未开仓奖励策略
    处理未开仓的情况，包括错过机会的惩罚逻辑。
    """
    def _punish_reward(self, STD_REWARD, res):
        acc_done = True
        if res['max_drawup_ticks_bm'] >= 10:
            reward = -STD_REWARD * (7/10)  # 错过大机会的惩罚
        else:
            reward = -STD_REWARD * (5/10)  # 错过小机会的惩罚

        return reward, acc_done

    def calculate_reward(self, env_id=0, STD_REWARD=100, res={}, data_producer=None, **kwargs):
        punish = 0
        acc_done = False
        if res['max_drawup_ticks_bm'] >= 10:
            log(f'[{env_id}][{data_producer.data_type if data_producer else "unknown"}] max_drawup_ticks_bm({res["max_drawup_ticks_bm"]}) >= 10, '
                'missed a significant long profit opportunity (continuous 10 tick rise), game over: LOSS')
            punish = 1
        if res['drawup_ticks_bm_count'] >= 3:
            log(f'[{env_id}][{data_producer.data_type if data_producer else "unknown"}] drawup_ticks_bm_count({res["drawup_ticks_bm_count"]}) >= 3, '
                'missed 3 consecutive small long profit opportunities (at least 2 tick rise), game over: LOSS')
            punish = 1

        if punish:
            reward, acc_done = self._punish_reward(STD_REWARD, res)

        elif res['drawup_ticks_bm_count'] == 0:
            reward = -res['step_return_bm']  # 标的一直下跌，不开仓正确
        else:
            reward = 0  # 默认值

        return reward, acc_done

class NoPositionRewardStrategy(RewardStrategy):
    """
    未开仓奖励策略
    无持仓奖励与步收益率负相关
    """
    def calculate_reward(
            self, 
            res={},
            **kwargs
        ):
        reward = -res['step_return_bm']  # 持仓每步收益率的相反数
        return reward, False
    
class RewardCalculator:
    def __init__(self, 
            max_drawdown_threshold,
            strategies_class = {
                # (平仓/最终)收益率奖励
                'end_position': ClosePositionRewardStrategy,
                'close_position': ClosePositionRewardStrategy,
                # 无
                'open_position_step': BlankRewardStrategy,
                # 每步奖励
                'hold_position': HoldPositionRewardStrategy,
                'no_position': NoPositionRewardStrategy,
                # 止损惩罚
                'force_stop': ForceStopRewardStrategy,
            },
        ):
        """
        策略字典

        end_position: 当天结束奖励策略
        close_position: 平仓奖励策略
        open_position_step: 开仓步奖励策略
        hold_position: 持仓奖励策略
        no_position: 未开仓奖励策略
        force_stop: 强制止损奖励策略
        """
        self.max_drawdown_threshold = max_drawdown_threshold    
        self.strategies = {k: v() for k, v in strategies_class.items()}

    def calculate_reward(self, env_id, STD_REWARD, force_stop, need_close, action_result, res, data_producer, acc):

        if force_stop:
            # 止损
            strategy = self.strategies['force_stop']

        elif need_close:
            # 当天结束
            strategy = self.strategies['end_position']

        else:
            # 持仓状态
            if action_result == RESULT_CLOSE:
                # 平仓
                strategy = self.strategies['close_position']
            elif action_result == RESULT_OPEN:
                # 开仓
                strategy = self.strategies['open_position_step']
            elif acc.status == 1:
                # 其他
                strategy = self.strategies['hold_position']
            else:
                # 空仓状态
                strategy = self.strategies['no_position']
    
        reward, acc_done = strategy.calculate_reward(
            env_id = env_id, 
            STD_REWARD = STD_REWARD, 
            force_stop = force_stop, 
            need_close = need_close, 
            action_result = action_result, 
            res = res, 
            data_producer = data_producer, 
            acc = acc, 
            max_drawdown_threshold = self.max_drawdown_threshold
        )
        return reward, acc_done
