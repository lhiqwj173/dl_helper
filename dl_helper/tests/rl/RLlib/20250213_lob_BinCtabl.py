from py_ext.tool import init_logger
from py_ext.datetime import beijing_time
# import os
# os.environ["RAY_DEDUP_LOGS"] = "0"

from dl_helper.rl.costum_rllib_module.lob.run import run
from dl_helper.rl.costum_rllib_module.lob.binctabl import BinCtablPPOCatalog

train_folder = 'lob_BinCtabl'
train_title = f'20250213_lob_BinCtabl'
init_logger(f'{train_title}_{beijing_time().strftime("%Y%m%d")}', home=train_folder, timestamp=False)

if __name__ == "__main__":
    run(
        train_folder,
        train_title,
        BinCtablPPOCatalog,# 自定义自定义编码器
        model_config={
            # 自定义编码器参数  
            'input_dims' : (10, 20),
            'extra_input_dims' : 4,
            'ds' : (20, 40, 40, 3),
            'ts' : (10, 6, 3, 1),
        },
    )