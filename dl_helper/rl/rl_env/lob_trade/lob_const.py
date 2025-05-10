import numpy as np
from dl_helper.train_param import in_kaggle

USE_CODES = [
    '513050',
    '513330',
    '518880',
    '159941',
    '513180',
    '159920',
    '513500',
    '513130',
    '159792',
    '513100',
    '159937',
    '510900',
    '513060',
    '159934',
    '159509',
    '159632',
    '159605',
    '513010',
    '159513',
    '513120',
    '159501',
    '518800',
    '513300',
    '513660',
    '513090',
    '513980',
    '159892',
    '159740',
    '159636',
    '159659',
]

STD_REWARD = 100
FINAL_REWARD = STD_REWARD * 1000

# 时间标准化
MEAN_SEC_BEFORE_CLOSE = 10024.17
STD_SEC_BEFORE_CLOSE = 6582.91
MAX_SEC_BEFORE_CLOSE = 5.5*60*60

# 动作
ACTION_BUY, ACTION_SELL = range(2)

# 动作结果
RESULT_OPEN, RESULT_CLOSE, RESULT_HOLD = range(3)

# 本地训练数据文件夹
# TRAIN_DATA_LOCAL_FOLDER_NAME = 'RL_10level_20250404'
# TRAIN_DATA_LOCAL_FOLDER_NAME = 'RL_10level_20250427_train'
TRAIN_DATA_LOCAL_FOLDER_NAME = 'RL_10level_20250508_train'

# 本地 win 数据文件夹
LOCAL_DATA_FOLDER = rf'D:\L2_DATA_T0_ETF\train_data\RAW\{TRAIN_DATA_LOCAL_FOLDER_NAME}'

# kaggle 数据文件夹 r'/kaggle/input'
KAGGLE_DATA_FOLDER = rf"/kaggle/input/{TRAIN_DATA_LOCAL_FOLDER_NAME.replace('_', '-').lower()}"

# 数据文件夹
if in_kaggle:
    DATA_FOLDER = KAGGLE_DATA_FOLDER
else:
    DATA_FOLDER = LOCAL_DATA_FOLDER