"""
学习交易规则
使用新的数据

1. act 0 买入，若当前存在持仓，则不合法
2. act 1 卖出，若当前没有持仓，则不合法
3. act 2 无操作，在如何情况下都是合法的

"""
from py_ext.tool import init_logger
from py_ext.datetime import beijing_time
from dl_helper.rl.costum_rllib_module.lob.run import run
from dl_helper.rl.costum_rllib_module.lob.causalconvlstm import CausalConvLSTMPPOCatalog, CausalConvLSTMIntrinsicCuriosityModel
from dl_helper.rl.rl_env.lob_env_reward import BlankRewardStrategy

import sys
use_intrinsic_curiosity = True
if len(sys.argv) > 1 and sys.argv[1] == 'no_intrinsic_curiosity':
    use_intrinsic_curiosity = False

train_title = train_folder = '20250317_rule_learning' + '_no_intrinsic_curiosity' if not use_intrinsic_curiosity else ''
init_logger(f'{train_title}_{beijing_time().strftime("%Y%m%d")}', home=train_folder, timestamp=False)

if __name__ == "__main__":
    run(
        train_folder,
        train_title,
        CausalConvLSTMPPOCatalog,# 自定义自定义编码器
        model_config={
            # 自定义编码器参数  
            'input_dims' : (10, 20),
            'extra_input_dims' : 4,
            'output_dims' : 8,
        },
        env_config ={
            # 全部使用空白的奖励
            'reward_strategy_class_dict': {
                'end_position': BlankRewardStrategy,
                'close_position': BlankRewardStrategy,
                'open_position_step': BlankRewardStrategy,
                'hold_position': BlankRewardStrategy,
                'no_position': BlankRewardStrategy
            }
        },
        intrinsic_curiosity_model_class = CausalConvLSTMIntrinsicCuriosityModel if use_intrinsic_curiosity else None,
        intrinsic_curiosity_model_config = {
            # feature(自定义编码器)参数  
            'input_dims' : (10, 20),
            'extra_input_dims' : 4,
            'output_dims' : 8,
            # inverse / forward 网络参数
            "inverse_net_hiddens": (16, 16),
            "inverse_net_activation": "relu",
            "forward_net_hiddens": (16, 16),
            "forward_net_activation": "relu",
        },
    )