import numpy as np
from dl_helper.train_param import in_kaggle

USE_CODES = [
    # 黄金
    '518880',
    '159937',
    '159934',
    '518800',
    '518850',
    '518660',
    '518680',
    '518860',
    '518890',
    '518600',
    '159830',
    '159834',

    # 港股
    '159792',
    '513050',
    '513180',
    '513130',
    '513330',
    '159636',
    '159920',
    '513980',
    '513010',
    '513630',
    '513120',
    '513060',
    '510900',
    '159740',
    '513090',
    '159691',
    '513380',
    '159605',
    '520990',
    '513770',
    '513260',
    '159892',
    '513660',
    '513690',
    '513920',
    '159570',
    '159217',
    '513200',
    '513910',
    '513550',
    '513820',
    '513580',
    '520900',
    '159506',
    '513860',
    '159742',
    '513970',
    '513530',
    '513600',
    '513750',
    '159699',
    '159567',
    '159545',
    '159607',
    '513950',

    # 其他
    '159985',
    '164824',
    '513730',
    '513000',
    '513880',
    '513520',
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
# TRAIN_DATA_LOCAL_FOLDER_NAME = 'RL_10level_20250508_train'
# TRAIN_DATA_LOCAL_FOLDER_NAME = 'BC_train_data_20250518'
TRAIN_DATA_LOCAL_FOLDER_NAME = 'BC_train_data_20250729'

# 本地 win 数据文件夹
LOCAL_DATA_FOLDER = rf'D:\L2_DATA_T0_ETF\train_data\RAW\{TRAIN_DATA_LOCAL_FOLDER_NAME}\train_data'

# kaggle 数据文件夹 r'/kaggle/input'
KAGGLE_DATA_FOLDER = rf"/kaggle/input/{TRAIN_DATA_LOCAL_FOLDER_NAME.replace('_', '-').lower()}/train_data"

# 数据文件夹
if in_kaggle:
    DATA_FOLDER = KAGGLE_DATA_FOLDER
else:
    DATA_FOLDER = LOCAL_DATA_FOLDER