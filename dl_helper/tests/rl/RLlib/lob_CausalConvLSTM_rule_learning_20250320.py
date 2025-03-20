"""
使用新的数据
学习交易规则:
    1. act 0 买入，若当前存在持仓，则不合法
    2. act 1 卖出，若当前没有持仓，则不合法
    3. act 2 无操作，在如何情况下都是合法的
    4. 最大回测超出阈值, 触发强制止损，游戏结束     

目标: 
    1. 止损触发率0
    2. 非法操作率0 > 交易频率最高，且无非法操作

20250319:
    奖励: 
        1. 非法操作     -STD_REWARD * 1000
        2. 止损         -STD_REWARD
        3. 积极交易     +STD_REWARD / 100
    
    结果:
        illegal_ratio: 0.0789
        force_stop_ratio: 0.0000
        act_0_pct: 0.4554
        act_1_pct: 0.4547
        hold_length: 3.3
    
    总结:
        1. ai 积极交易，追求交易奖励，但最终都以 illegal 终止，说明还是没能学到完美的交易规则
        2. force_stop_ratio 为 0, 没有触发止损 / 持仓太短达不到跌幅
        3. 中间阶段出现过峰值 4454, 完美的频繁交易，且不触发 illegal > reward不是全部文件, 而是部分文件, 因此可能是局部文件过拟合

    改进:
        1. 增加止损触发情景 > 使用下跌趋势的行情, 以持仓并临近止损状态开始训练
        2. 增加奖励: 交易次数越高，交易获得的奖励就越高
        3. 正常训练结束, 给与额外奖励
"""
from py_ext.tool import init_logger
from py_ext.datetime import beijing_time
from dl_helper.rl.costum_rllib_module.lob.run import run
from dl_helper.rl.costum_rllib_module.lob.causalconvlstm import CausalConvLSTMPPOCatalog, CausalConvLSTMIntrinsicCuriosityModel
from dl_helper.rl.rl_env.lob_env_reward import RewardStrategy
from dl_helper.rl.rl_env.lob_env import STD_REWARD

class EncourageTradeRewardStrategy(RewardStrategy):
    def calculate_reward(self, env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc, max_drawdown_threshold):
        """鼓励交易"""
        if action in [0, 1]:
            return STD_REWARD / 100, False
        return 0, False

train_title = train_folder = '20250320_rule_learning'
init_logger(f'{train_title}_{beijing_time().strftime("%Y%m%d")}', home=train_folder, timestamp=False)

if __name__ == "__main__":
    run(
        train_folder,
        train_title,
        CausalConvLSTMPPOCatalog,# 自定义自定义编码器
        model_config={
            # 自定义编码器参数  
            'input_dims' : (10, 20),
            'extra_input_dims' : 5,
            'output_dims' : 16,
        },
        env_config ={
            # 终止游戏的回撤阈值
            'max_drawdown_threshold': 1000,# 相当于无止损
            # 全部使用空白的奖励
            'reward_strategy_class_dict': {
                'end_position': EncourageTradeRewardStrategy,
                'close_position': EncourageTradeRewardStrategy,
                'open_position_step': EncourageTradeRewardStrategy,
                'hold_position': EncourageTradeRewardStrategy,
                'no_position': EncourageTradeRewardStrategy
            }
        },
    )