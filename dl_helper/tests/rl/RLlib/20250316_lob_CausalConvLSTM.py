from py_ext.tool import init_logger
from py_ext.datetime import beijing_time
# import os
# os.environ["RAY_DEDUP_LOGS"] = "0"

from dl_helper.rl.costum_rllib_module.lob.run import run
from dl_helper.rl.costum_rllib_module.lob.causalconvlstm import CausalConvLSTMPPOCatalog

train_folder = 'lob_CausalConvLSTM'
train_title = f'20250316_lob_CausalConvLSTM'
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
            'output_dims' : 6,
        },
        env_config ={
            'new_data': False,
        }
    )