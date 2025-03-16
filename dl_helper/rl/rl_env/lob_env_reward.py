from abc import ABC, abstractmethod
import pickle
from dl_helper.tool import max_profit_reachable

from py_ext.tool import log

def base_reward_00(env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc):
    # 交易计算奖励 标准化范围: [-STD_REWARD, STD_REWARD] 
    # 平仓奖励/最终奖励: 平仓对数收益率 / 潜在最大对数收益率 * STD_REWARD
    # 持仓/空仓奖励（步）: 持仓每步收益率 / 每步的标的收益率的相反数​
    # 消极空仓惩罚
    if acc_opened:
        # 已经开仓的奖励计算
        if need_close or action==1:
            _, max_profit_reachable_bm, _, _ = max_profit_reachable(data_producer.bid_price, data_producer.ask_price)

            if res['trade_return'] >= 0:
                # 平仓对数收益率 / 潜在最大对数收益率 * STD_REWARD
                # 奖励 [0, STD_REWARD] 
                # 若 max_profit_reachable_bm 为0，trade_return也必定为0，所以奖励为0
                if max_profit_reachable_bm == 0 and res['trade_return'] != 0:
                    pickle.dump((data_producer.bid_price, data_producer.ask_price, legal, acc.net_raw, acc.net_raw_bm), open(f'bid_ask_{data_producer.data_type}_{data_producer.cur_data_file}.pkl', 'wb'))
                    raise ValueError(f'Maximum potential log return is 0, but closing log return is not 0')
                reward = res['trade_return'] / max_profit_reachable_bm * STD_REWARD if max_profit_reachable_bm != 0 else 0
            else:
                # 平仓对数收益率 - 潜在最大对数收益率，假设收益率==-0.03为最低值(日内的一笔交易，亏损达到-0.03，则认为完全失败，-STD_REWARD)
                # 奖励 [-STD_REWARD, 0]
                reward = max((res['trade_return'] - max_profit_reachable_bm) / 0.03, -1) * STD_REWARD
        
        elif action == 0:
            # 开仓步的奖励需要特别考虑
            # 开仓步的 step_return 因为买入手续费的原因，必定为负
            # 直接给与 step_return 作为奖励，可能会导致模型不愿意开仓

            # 中性
            reward = 0

        else:
            # 持仓奖励（步）: 持仓每步收益率 
            # TODO 如何标准化, step_return 大概率是个很小的值
            reward = res['step_return']
    else:
        # 还未开仓的触发检查
        # 1. 若期间的标的净值 max_drawup_ticks_bm > 10, 代表错过了一个很大的多头盈利机会(连续10个tick刻度的上涨)
        # 2. 若期间的标的净值 drawup_ticks_bm_count > 3, 代表连续错过了3个多头可盈利小机会(至少2个tick刻度的上涨)
        # 3. 若 标的一直在下跌 不开仓是正确的，需要给与奖励 
        # 给一个小惩罚, 且结束本轮游戏
        punish = 0
        if res['max_drawup_ticks_bm'] >= 10:
            log(f'[{env_id}][{data_producer.data_type}] max_drawup_ticks_bm({res["max_drawup_ticks_bm"]}) >= 10, missed a significant long profit opportunity (continuous 10 tick rise), game over: LOSS')
            punish = 1
        if res['drawup_ticks_bm_count'] >= 3:
            log(f'[{env_id}][{data_producer.data_type}] drawup_ticks_bm_count({res["drawup_ticks_bm_count"]}) >= 3, missed 3 consecutive small long profit opportunities (at least 2 tick rise), game over: LOSS')
            punish = 1

        if punish:
            # 游戏终止，任务失败 
            acc_done = True
            # # 奖励: -STD_REWARD/10
            # reward = -STD_REWARD / 10
            # 这个惩罚要比非法操作的惩罚(-STD_REWARD)轻
            # 但要比普通的负收益惩罚重，以体现错过机会的特殊性
            if res['max_drawup_ticks_bm'] >= 10:
                reward = -STD_REWARD * (7/10)  # 加大对错过大机会的惩罚
            else:
                reward = -STD_REWARD * (5/10)  # 适度惩罚错过多次小机会
        
        elif res['drawup_ticks_bm_count'] == 0:
            # 标的一直在下跌 不开仓是正确的，需要给与奖励 
            # TODO 如何标准化, step_return 大概率是个很小的值
            reward = -res['step_return_bm']

    return reward, acc_done

class RewardStrategy(ABC):
    @abstractmethod
    def calculate_reward(self, env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc):
        """计算奖励和是否结束游戏"""
        pass

class ClosePositionRewardStrategy(RewardStrategy):
    """
    平仓奖励策略
    处理已经开仓且需要平仓（need_close 或 action == 1）的情况。
    """
    def calculate_reward(self, env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc):
        _, max_profit_reachable_bm, _, _ = max_profit_reachable(data_producer.bid_price, data_producer.ask_price)
        if res['trade_return'] >= 0:
            if max_profit_reachable_bm == 0 and res['trade_return'] != 0:
                pickle.dump((data_producer.bid_price, data_producer.ask_price, legal, acc.net_raw, acc.net_raw_bm), 
                           open(f'bid_ask_{data_producer.data_type}_{data_producer.cur_data_file}.pkl', 'wb'))
                raise ValueError(f'Maximum potential log return is 0, but closing log return is not 0')
            reward = res['trade_return'] / max_profit_reachable_bm * STD_REWARD if max_profit_reachable_bm != 0 else 0
        else:
            reward = max((res['trade_return'] - max_profit_reachable_bm) / 0.03, -1) * STD_REWARD
        return reward, False  # acc_done 为 False
    
class OpenPositionStepRewardStrategy(RewardStrategy):
    """
    开仓步奖励策略
    处理已经开仓且 action == 0 的情况。
    """
    def calculate_reward(self, env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc):
        reward = 0  # 中性奖励，避免因手续费惩罚导致模型不愿开仓
        return reward, False
    
class HoldPositionRewardStrategy(RewardStrategy):
    """
    持仓奖励策略
    处理已经开仓且既不平仓也不为开仓步的情况。
    """
    def calculate_reward(self, env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc):
        reward = res['step_return']  # 持仓每步收益率
        return reward, False


class NoPositionRewardStrategy(RewardStrategy):
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

    def calculate_reward(self, env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc):
        punish = 0
        acc_done = False
        if res['max_drawup_ticks_bm'] >= 10:
            log(f'[{env_id}][{data_producer.data_type}] max_drawup_ticks_bm({res["max_drawup_ticks_bm"]}) >= 10, '
                'missed a significant long profit opportunity (continuous 10 tick rise), game over: LOSS')
            punish = 1
        if res['drawup_ticks_bm_count'] >= 3:
            log(f'[{env_id}][{data_producer.data_type}] drawup_ticks_bm_count({res["drawup_ticks_bm_count"]}) >= 3, '
                'missed 3 consecutive small long profit opportunities (at least 2 tick rise), game over: LOSS')
            punish = 1

        if punish:
            reward, acc_done = self._punish_reward(STD_REWARD, res)

        elif res['drawup_ticks_bm_count'] == 0:
            reward = -res['step_return_bm']  # 标的一直下跌，不开仓正确
        else:
            reward = 0  # 默认值

        return reward, acc_done

class NoPositionRewardStrategy_00(NoPositionRewardStrategy):
    def _punish_reward(self, STD_REWARD, res):
        acc_done = True
        reward = -STD_REWARD / 10
        return reward, acc_done
    
class RewardCalculator:
    def __init__(self, strategies_class = {
            'close_position': ClosePositionRewardStrategy,
            'open_position_step': OpenPositionStepRewardStrategy,
            'hold_position': HoldPositionRewardStrategy,
            'no_position': NoPositionRewardStrategy
        }):
        """
        策略字典

        close_position: 平仓奖励策略
        open_position_step: 开仓步奖励策略
        hold_position: 持仓奖励策略
        no_position: 未开仓奖励策略
        """
        self.strategies = {k: v() for k, v in strategies_class.items()}

    def calculate_reward(self, env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc):
        if acc_opened:
            if need_close or action == 1:
                strategy = self.strategies['close_position']
            elif action == 0:
                strategy = self.strategies['open_position_step']
            else:
                strategy = self.strategies['hold_position']
        else:
            strategy = self.strategies['no_position']

        reward, acc_done = strategy.calculate_reward(env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc)
        return reward, acc_done
