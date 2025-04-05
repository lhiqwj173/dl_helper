"""
获取环境的 ask/bid 数据
获取策略的 action

内部维护数据
    标的净值序列
    策略净值序列
    根据净值计算奖励
    策略动作
    环境奖励

校对环境计算是否正确
"""
import numpy as np  

from dl_helper.rl.rl_env.lob_trade.lob_const import STD_REWARD
from dl_helper.rl.rl_env.lob_trade.lob_const import ACTION_BUY, ACTION_SELL, RESULT_OPEN, RESULT_CLOSE, RESULT_HOLD
from dl_helper.tool import max_profit_reachable

class LobEnvChecker:
    def __init__(self, env, max_drawdown_threshold=0.005, fee=5e-5):
        self.fee = fee
        self.env = env
        self.max_drawdown_threshold = max_drawdown_threshold

        # 数据
        self._step = 0
        self.last_close_step = 0
        self.pos = -1
        self.cash = -1
        self.net_bm = None
        self.net = None
        self.bid = []
        self.ask = []

    def _get_ask_bid(self):
        """
        获取环境 ask/bid 数据
        """
        ask, bid = self.env.data_producer.get_ask_bid()
        self.bid.append(bid)
        self.ask.append(ask)
        return ask, bid

    def reset(self):
        """
        在 env reset 后调用
        """
        # 数据重置
        self._step = 0
        self.last_close_step = 0
        self.pos = 0
        self.cash = 0
        self.net_bm = []
        self.net = []
        self.bid = []
        self.ask = []

        # 获取 ask/bid, 初始化净值
        ask, bid = self._get_ask_bid()
        # 假设初始标的数量为1
        self.net_bm.append(1 * bid*(1-self.fee))
        # 策略初始净值与 标的净值一致，相对于现金
        self.cash = self.net_bm[-1]
        self.net.append(self.cash)

    def step(self, action, env_reward, env_info):
        """
        在 env step 后调用
        """
        self._step += 1
        print(f'step: {self._step}, action: {action}')

        # 获取 ask/bid
        # 成交价
        ask, bid = self._get_ask_bid()

        # 处理策略交易
        self.act_result = RESULT_HOLD # 无
        if action == ACTION_BUY:
            # 持仓动作
            if self.pos == 0:
                # 当前空仓
                self.pos = self.cash / (ask * (1+self.fee))# 转为标的持仓
                self.cash = 0
                self.act_result = RESULT_OPEN# 开仓
        elif action == ACTION_SELL:
            # 空仓动作
            if self.pos > 0:
                # 当前持仓
                self.cash += self.pos * bid * (1-self.fee)
                self.pos = 0
                self.act_result = RESULT_CLOSE# 平仓
        else:
            raise ValueError(f'action: {action} 非法')

        # 更新净值
        self.net_bm.append(1 * bid * (1-self.fee))
        net = self.cash + self.pos * bid * (1-self.fee)
        self.net.append(net)

        # 计算奖励/指标
        reward, res = self.cal_reward()

        # 对比 env_reward/env_info 是否正确
        assert reward == env_reward, f'reward: {reward} != env_reward: {env_reward}'
        for k, v in res.items():
            assert v == env_info[k], f'{k}: {v} != env_info[{k}]: {env_info[k]}'    

        # 记录平仓step，需要放在最后
        if self.act_result == RESULT_CLOSE:
            self.last_close_step = self._step

    def cal_reward(self):
        """
        计算奖励
        """
        # 计算统计指标
        res = self.env.acc.cal_res(self.net, self.net_bm, self.last_close_step, need_cal_drawup=self.pos == 0 and self.act_result == RESULT_HOLD)

        # 数据是否结束了
        need_close = self.env.data_producer.need_close
        if need_close or self.act_result == RESULT_CLOSE:
            if not need_close:
                # 潜在收益率（针对最近的交易序列）
                _, max_profit_reachable_bm, _, _ = max_profit_reachable(self.bid[self.last_close_step:], self.ask[self.last_close_step:])
                last_open_net = self.net[self.last_close_step:]
                acc_return = np.log(last_open_net[-1]) - np.log(last_open_net[0])
            else:
                # 游戏完成，使用所有数据完整计算
                _, max_profit_reachable_bm, _, _ = max_profit_reachable(self.bid, self.ask)
                # 日内完整的策略收益率
                acc_return = res['trade_return']

            res['potential_return'] = max_profit_reachable_bm
            res['acc_return'] = acc_return
        
        # 计算奖励
        if abs(res['max_drawdown']) > self.max_drawdown_threshold:
            # 最大回撤超过阈值
            reward, _ = self.env.reward_calculator.strategies['force_stop'].calculate_reward(STD_REWARD=STD_REWARD)

        elif need_close:
            # 当天结束
            reward, _ = self.env.reward_calculator.strategies['close_position'].calculate_reward(
                STD_REWARD = STD_REWARD, 
                res = res, 
                max_drawdown_threshold = self.max_drawdown_threshold,
            )

        else:
            if self.act_result == RESULT_CLOSE:
                # 平仓
                reward, _ = self.env.reward_calculator.strategies['close_position'].calculate_reward(
                    STD_REWARD = STD_REWARD, 
                    res = res, 
                    max_drawdown_threshold = self.max_drawdown_threshold,
                )
            elif self.act_result == RESULT_OPEN:
                # 开仓
                reward, _ = self.env.reward_calculator.strategies['open_position_step'].calculate_reward()
            elif self.pos > 0:
                # 持仓
                reward, _ = self.env.reward_calculator.strategies['hold_position'].calculate_reward(
                    res = res,
                )
            else:
                # 空仓
                reward, _ = self.env.reward_calculator.strategies['no_position'].calculate_reward(
                    STD_REWARD= STD_REWARD,
                    res = res,
                )

        return reward, res

