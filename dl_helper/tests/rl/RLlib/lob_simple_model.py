"""
使用新的数据
学习交易规则:
    1. act 0 买入，若当前存在持仓，则不合法
    2. act 1 卖出，若当前没有持仓，则不合法
    3. act 2 无操作，在如何情况下都是合法的
    4. 最大回测超出阈值, 触发强制止损，游戏结束     

目标: 
    1. 止损触发率0
    2. 非法操作率0

"""
from py_ext.tool import init_logger
from py_ext.datetime import beijing_time
from dl_helper.rl.costum_rllib_module.lob.run import run
from dl_helper.rl.costum_rllib_module.lob.simple import SimplePPOCatalog, SimpleIntrinsicCuriosityModel
from dl_helper.rl.rl_env.lob_env_reward import RewardStrategy
from dl_helper.rl.rl_env.lob_trade.lob_env import STD_REWARD

class EncourageTradeRewardStrategy(RewardStrategy):
    def calculate_reward(self, env_id, STD_REWARD, acc_opened, legal, need_close, action, res, data_producer, acc):
        """鼓励交易"""
        if action in [0, 1]:
            return STD_REWARD / 100, False
        return 0, False

train_title = train_folder = '20250319_rule_learning_simple'
init_logger(f'{train_title}_{beijing_time().strftime("%Y%m%d")}', home=train_folder, timestamp=False)

if __name__ == "__main__":
    run(
        train_folder,
        train_title,
        SimplePPOCatalog,# 自定义自定义编码器
        model_config={
            # 自定义编码器参数  
            'input_dims' : (10, 20),
            'extra_input_dims' : 4,
            'output_dims' : 8,
        },
        env_config ={
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