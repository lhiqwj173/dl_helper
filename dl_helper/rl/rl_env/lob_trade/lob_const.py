import numpy as np

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
MEAN_CODE_ID = np.mean(np.arange(len(USE_CODES)))
STD_CODE_ID = np.std(np.arange(len(USE_CODES)))
MAX_CODE_ID = len(USE_CODES) - 1

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

# 本地 win 数据文件夹
LOCAL_DATA_FOLDER = r'D:\L2_DATA_T0_ETF\train_data\RAW\RL_10level_20250404'
