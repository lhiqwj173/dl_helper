import warnings
warnings.filterwarnings("ignore")
import subprocess
import sys
import os, torch
kaggle = any(key.startswith("KAGGLE") for key in os.environ.keys())

if kaggle:
    # # 安装py_ext
    # !pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz > /dev/null 2>&1
    # !pip install akshare > /dev/null 2>&1
    # !git clone https://github.com/lhiqwj173/dl_helper.git > /dev/null 2>&1
    # !cd dl_helper/cpp && python setup.py build_ext --inplace > /dev/null 2>&1

    for cmd in [
            'pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz > /dev/null 2>&1',
            'pip install akshare > /dev/null 2>&1',
            'git clone https://github.com/lhiqwj173/dl_helper.git > /dev/null 2>&1',
            'cd dl_helper/cpp && python setup.py build_ext --inplace > /dev/null 2>&1',
        ]:
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

    sys.path.append('/kaggle/working/dl_helper/cpp')

from py_ext.lzma import compress_folder
from joblib import Parallel, delayed
import cpp_ext
import pickle
import akshare as ak
import datetime
from py_ext.wechat import send_wx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import socket
import shutil
from tqdm import tqdm
from py_ext.lzma import decompress
# from py_ext.tool import full_cpu_run_same_func


CODE = '0QYg9Ky17dWnN4eK'

# ###################
# # 测试用
# os.environ['rank'] = '0'
# ###################

def get_idx(train_title):
    # 定义服务器地址和端口
    HOST = '146.235.33.108'
    PORT = 12345

    # 创建一个 TCP 套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        # 发送消息给服务器
        message = f'{CODE}_{train_title}'
        client_socket.sendall(message.encode())
        # 接收服务器的响应
        response = client_socket.recv(1024).decode()
        return int(response)
    finally:
        # 关闭套接字
        client_socket.close()

    raise Exception('get idx error')

top_n = 5
produce_name = f'once_produce_{top_n}_combine_data'
if 'rank' not in os.environ:
    rank = get_idx(produce_name)
    os.environ['rank'] = str(rank)
print(f"rank {os.environ['rank']}")

FILTER_AM_MAP = {
    '1y': 1e8,
    '5y': 5e8,
    '10y': 1e9,
    '15y': 1.5e9,
    '20y': 2e9,
    '30y': 3e9,
    '50y': 5e9,
    '100y': 1e10,
}

FILTER_VOL_MAP = {
    '100w': 1e6,
    '250w': 2.5e6,
    '1000w': 1e7,
    '2000w': 2e7
}

# FILTER_VOL_NAME = '2000w'
# os.environ['filter_min_vol'] = str(FILTER_VOL_MAP[FILTER_VOL_NAME])

# FILTER_AM_NAME = '10y'
# os.environ['filter_min_am'] = str(FILTER_AM_MAP[FILTER_AM_NAME])

# 是否上传到 alist
os.environ['upload_alist'] = '1'

# 使用截取时间
os.environ['train_data_begin_time'] = '09:30'
os.environ['train_data_end_time'] = '14:57'

# # 用于生成标准化数据的历史天数
# # 5/10/20/60/120/200
# os.environ['train_data_std_pass_day_n'] = str(10)

# 采用的codes列表
_codes = [
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
# os.environ['train_data_codes'] = _codes[int(os.environ['rank'])]
# os.environ['train_data_codes'] = '513050'
# os.environ['train_data_codes'] = '513050,513330'
os.environ['train_data_codes'] = ','.join(_codes[:top_n])

# # 是否分配 train/val/test
# os.environ['train_data_split'] = '0'

# 生成标准化数据的方法
# simple / each_sample / limit_std / stable
os.environ['train_data_std_method'] = 'simple'

# 218 - 4 = 214
# 214 / 6 = 37 / 5 = 8(acc)
max_handle_date_num = 6
# ###########
# # 测试用
# max_handle_date_num = 1
# ###########
os.environ['max_handle_date_num'] = str(max_handle_date_num)
input_raw_folder = r'/kaggle/input/lh-q-t0-raw-data-20240907/train_t0_raw'
os.environ['input_raw_folder'] = input_raw_folder

print('####################################')
print('标准化数据生成')
print('####################################')

##################################
# 检查数据
# C:\\Users\\lh\\Desktop\\temp\\data\\train/20240119.pkl
##################################

kaggle = any(key.startswith("KAGGLE") for key in os.environ.keys())

# 读取环境变量中的变量
max_handle_date_num = os.getenv('max_handle_date_num')
rank = os.getenv('rank')
max_handle_date_num = - \
    1 if None is max_handle_date_num else int(max_handle_date_num)
rank = -1 if None is rank else int(rank)

# 获取历史数据日期列表
l2_data = r'Z:\L2_DATA'
folder = os.path.join(l2_data, 'his_data')
pass_n = 100

# ###########
# # 测试用
# l2_data = r'C:\Users\lh\Desktop\temp\train_data'
# ###########

if kaggle:
    l2_data = r'/kaggle/working/'
    folder = os.environ['input_raw_folder']

dates = []
for file in os.listdir(folder):
    if '.7z' not in file:
        dates.append(file)
dates = sorted(dates)
print(f'len(dates) {len(dates)}')

# t0 标的列表
t0_codes = []
# # ###########
# # # 测试用
# t0_codes = ['159742']
# # ###########
date_folder = os.path.join(folder, dates[-1])
if not kaggle and not t0_codes:
    # 非 kaggle
    # 根据最新的日期目录确定
    # 按照 修改日期 排序获取其中的文件夹列表
    t0_codes = sorted(os.listdir(date_folder),
                      key=lambda x: os.path.getmtime(os.path.join(date_folder, x)))
    # 截断，只选取 t+0 etf
    kzz_idx = 0
    for i, file in enumerate(t0_codes):
        if file.startswith('123'):
            kzz_idx = i
            break
    if kzz_idx == 0:
        send_wx('没有找到 可转债 标的文件夹，请检查')
        sys.exit(0)
    t0_codes = t0_codes[:kzz_idx]
elif not t0_codes:
    # kaggl上直接读取
    t0_codes = sorted(os.listdir(date_folder))
print(f't0_codes {len(t0_codes)}')

# 获取交易日历
trade_date_df = ak.tool_trade_date_hist_sina()
trade_date_df['trade_date'] = pd.to_datetime(trade_date_df['trade_date'])
trade_date_df
trade_dates = [str(i)[:10].replace('-', '') for i in trade_date_df['trade_date'].tolist()]
trade_dates = [i for i in trade_dates if i >= dates[0] and i<= dates[-1]]
trade_dates = sorted(trade_dates)

def full_cpu_run_same_func(func, args, n=-1):
    """
    按照cpu核心数并行执行任务
    args 为参数列表，每个元素为单次任务的参数(可以是列表)
    """
    # 并行执行任务
    if len(args[0]) > 1:
        # 多个参数
        return Parallel(n_jobs=n)(delayed(func)(*_args) for _args in args)

    else:
        # 单个参数
        return Parallel(n_jobs=n)(delayed(func)(_args) for _args in args)

def none_zero_begin_end_data(df, cols):

    if not isinstance(cols, list):
        cols = [cols]

    # print(f'none_zero_begin_end_data: \n{df[cols]}')

    df_nonzero = df
    for col in cols:
        # 如果全为0 直接返回None
        if (df_nonzero[col] == 0).all():
            return None

        # 剔除 列开头一直为 0 的行
        start_nonzero_idx = df_nonzero[col].ne(0).idxmax()
        df_nonzero = df_nonzero.iloc[start_nonzero_idx:].reset_index(drop=True)
        # print("剔除开头一直为0的行:\n", df_nonzero)

        # 剔除 列结尾一直为 0 的行
        end_nonzero_idx = df_nonzero[col][::-1].ne(0).idxmax()
        df_nonzero = df_nonzero.iloc[:end_nonzero_idx +
                                     1].reset_index(drop=True)
        # print("\n剔除结尾一直为0的行:\n", df_nonzero)

        # 仍然存在 0 值, 返回None
        if (df_nonzero[col] == 0).any():
            return None

    return df_nonzero

def produce(date, code, trade_dates, trade_dates_idx, pass_n, save_file, std_pass_n, std_method):
    print(f'produce std data: next{date} {code}')

    # 增加特征 卖均,总卖,买均,总买
    all_cols = [
        '卖1价', '卖1量', '买1价', '买1量', '卖2价', '卖2量', '买2价', '买2量', '卖3价', '卖3量', '买3价', '买3量', '卖4价', '卖4量', '买4价', '买4量', '卖5价', '卖5量', '买5价', '买5量', '卖6价', '卖6量', '买6价', '买6量', '卖7价', '卖7量', '买7价', '买7量', '卖8价', '卖8量', '买8价', '买8量', '卖9价', '卖9量', '买9价', '买9量', '卖10价', '卖10量', '买10价', '买10量', 
        '卖均', '总卖', '买均', '总买', 
    ]
    price_cols = ['卖1价', '买1价', '卖2价', '买2价', '卖3价', '买3价', '卖4价', '买4价', '卖5价', '买5价', '卖6价', '买6价',
                  '卖7价', '买7价', '卖8价', '买8价', '卖9价', '买9价', '卖10价', '买10价', '卖均', '买均']
    other_cols = ['卖1量', '买1量', '卖2量', '买2量', '卖3量', '买3量', '卖4量', '买4量', '卖5量', '买5量', '卖6量', '买6量', '卖7量',
                  '买7量', '卖8量', '买8量', '卖9量', '买9量', '卖10量', '买10量', '总卖', '总买']

    msg = ''
    
    cal_date = trade_dates[trade_dates_idx + 1]

    # 获取历史数据
    # 0, 1, 2, 3, 4, 5
    #            date  
    #             cal_date  
    data = pd.DataFrame()
    for i in range(std_pass_n):
        idx = trade_dates_idx - i

        result_folder = './fix_raw_data'
        result_file_name = os.path.join(result_folder, f'{code}_{trade_dates[idx]}')
        if os.path.exists(result_file_name):
            _data = pickle.load(open(result_file_name, 'rb'))
            data = pd.concat([data, _data], ignore_index=True)

    if len(data) == 0:
        print(msg)
        print(f'缺少数据: {code}, 跳过计算')
        return

    dates = list(set(data['时间'].dt.date.to_list()))
    # print(date, code, 0, 'dates', len(dates))
    # print(date, code, 0, 'data', len(data))
    msg += f'{date} {code} 0 dates {len(dates)}\n'
    msg += f'{date} {code} 0 data {len(data)}\n'

    # 遍历每日的半天的交易时间
    code_data_list = []
    for _date in dates:
        date_data = data.loc[data['时间'].dt.date == _date].sort_values('时间')
        am = date_data.loc[date_data['时间'].dt.time <=
                           pd.to_datetime('11:30:00').time()].reset_index(drop=True)
        pm = date_data.loc[date_data['时间'].dt.time >
                           pd.to_datetime('13:00:00').time()].reset_index(drop=True)

        parts = [am, pm]
        for i in range(len(parts)):
            if len(parts[i]) == 0:
                continue

            code_data_list.append(parts[i])

    # print(date, code, 1, len(code_data_list))
    msg += f'{date} {code} {1} len {len(code_data_list)}\n'
    if len(code_data_list) < 10:
        print(msg)
        print(f'{cal_date} {code} 存在时间间隔全部异常，跳过计算')
        return

    all_col_mean_std_list = []
    all_code_data = pd.concat(code_data_list, ignore_index=True)

    ########################################
    # 记录分段数据的 最小 最大
    # [20] + [40] + [15 + 15 + 10 + 10] + [10 + 10] + [20] + 4 -> 154
    # [20] + [40] + [50] +                [20] +      [20] + 4 -> 154
    # of数据 + 原始价量数据 + 委托数据 + 成交数据 + 深度数据 + 基础数据
    ########################################
    # of数据
    of_data = all_code_data.iloc[:, :20]
    of_min = of_data.values.min()
    of_max = of_data.values.max()

    # 原始价量数据
    base_data = all_code_data.iloc[:, 20:20+40]
    base_min = base_data.values.min()
    base_max = base_data.values.max()

    # 订单数据
    order_data = all_code_data.iloc[:, 60:60+50]
    order_min = order_data.values.min()
    order_max = order_data.values.max()

    # 成交数据
    deal_data = all_code_data.iloc[:, 110:110+20]
    deal_min = deal_data.values.min()
    deal_max = deal_data.values.max()

    # 深度数据
    depth_data = all_code_data.iloc[:, 130:130+20]
    depth_min = depth_data.values.min()
    depth_max = depth_data.values.max()

    # pct 数据
    pct_data = all_code_data.iloc[:, 150:151].dropna()
    pct_min = pct_data.values.min()
    pct_max = pct_data.values.max()

    # 输出均值和标准差
    for i in range(20):
        all_col_mean_std_list.append((of_min, of_max))
    for i in range(40):
        all_col_mean_std_list.append((base_min, base_max))
    for i in range(50):
        all_col_mean_std_list.append((order_min, order_max))
    for i in range(20):
        all_col_mean_std_list.append((deal_min, deal_max))
    for i in range(20):
        all_col_mean_std_list.append((depth_min, depth_max))
    all_col_mean_std_list.append((pct_min, pct_max))

    # for mean_std_data in all_col_mean_std_list:
    #     # 方差 > 0
    #     # assert mean_std_data[1] > 0, f'{msg}\n{cal_date} {code} 方差异常: \n{all_col_mean_std_list} '
    #     if mean_std_data[1] <= 0:
    #         print(f'{msg}\n{cal_date} {code} 方差小于等于0: \n{all_col_mean_std_list}')
    #         return

    mm = torch.tensor(all_col_mean_std_list)
    # 检查是否包含 NaN 或 Inf
    has_nan = torch.isnan(mm).any()
    has_inf = torch.isinf(mm).any()
    if has_nan or has_inf:
        # 发送微信提醒
        msg += f'{cal_date} {code} 包含 NaN 或 Inf'
        send_wx(msg)
        os._exit(0)  # 完全终止程序

    pickle.dump(all_col_mean_std_list, open(save_file, 'wb'))
    print(f'{msg}\n{cal_date} {code} 标准化数据储存完成')

def read_deal_data(date, code):
    # 买 卖 集合竞价
    DEAL_TYPES = ["B", "S", "JHJJ"]
    B, S, JHJJ = range(3)

    file_deal = os.path.join(folder, date, code, '逐笔成交.csv')
    begin_t = os.environ.get('train_data_begin_time', '09:30')
    end_t = os.environ.get('train_data_end_time', '14:45')
    if os.path.exists(file_deal):
        deal = pd.read_csv(file_deal, encoding='gbk', dtype=str)

        # 时间
        date_str = f'{date[:4]}-{date[4:6]}-{date[6:]}'
        deal["时间"] = deal["时间"].apply(lambda x: f'{date_str} {x}')
        deal["时间"] = pd.to_datetime(deal["时间"])
        # 截取时间
        deal = deal[(deal["时间"].dt.time >= pd.to_datetime(begin_t).time()) & (
            deal["时间"].dt.time < pd.to_datetime(end_t).time())].reset_index(drop=True)
        deal = deal[(deal["时间"].dt.time <= pd.to_datetime('11:30:00').time()) | (
            deal["时间"].dt.time > pd.to_datetime('13:00:00').time())].reset_index(drop=True)

        deal["类型"] = deal["类型"].fillna("JHJJ")
        deal["类型"] = deal["类型"].apply(lambda x: DEAL_TYPES.index(x))

        deal["手"] = deal["手"].astype(float).astype(int)

        deal["价格"] = deal["价格"].astype(float)
        deal = deal.iloc[:, [0, 2, 3, 4]]
    
    return deal

def read_order_data(date, code):
    # 成交类型
    ORDER_TYPES = ["B", "S", "BC", "SC", "1B", "1S", "UB", "US"]
    B, S, BC, SC, _1B, _1S, UB, US = range(len(ORDER_TYPES))

    file_order = os.path.join(folder, date, code, '逐笔委托.csv')
    begin_t = os.environ.get('train_data_begin_time', '09:30')
    end_t = os.environ.get('train_data_end_time', '14:45')
    if os.path.exists(file_order):
        order = pd.read_csv(file_order, encoding='gbk', dtype=str)
        order["价格"] = order["价格"].astype(float)

        # 时间
        date_str = f'{date[:4]}-{date[4:6]}-{date[6:]}'
        order["时间"] = order["时间"].apply(lambda x: f'{date_str} {x}')
        order["时间"] = pd.to_datetime(order["时间"])
        # 截取时间
        order = order[(order["时间"].dt.time >= pd.to_datetime(begin_t).time()) & (
            order["时间"].dt.time < pd.to_datetime(end_t).time())].reset_index(drop=True)
        order = order[(order["时间"].dt.time <= pd.to_datetime('11:30:00').time()) | (
            order["时间"].dt.time > pd.to_datetime('13:00:00').time())].reset_index(drop=True)

        # 在委时间
        order.loc[order["在委时间"].isna(), "在委时间"] = "0"
        order["在委时间"] = order["在委时间"].apply(
            lambda x: int(x[:-1])
            if "s" in x
            else int(x[:-1]) * 60
            if "m" in x
            else int(x[:-1]) * 60 * 60
            if "h" in x
            else int(x)
        )

        # 手 卖出会存在小数
        order["手"] = order["手"].astype(float).astype(int)

        # 类型
        order["类型"] = order["类型"].apply(lambda x: ORDER_TYPES.index(x))
        order = order.iloc[:, [0, 3, 4, 5]]

    return order

def slice_deal_vol_data_front(v_data, begin=5):
    v_data.iloc[:, begin] += v_data.iloc[:, :begin].sum(axis=1) 
    return v_data.iloc[:, begin:]

def slice_deal_vol_data_back(v_data, end=-5):
    v_data.iloc[:, end-1] += v_data.iloc[:, end:].sum(axis=1) 
    return v_data.iloc[:, :end]

def slice_deal_vol_data(v_data, begin=5, end=-5):
    v_data = slice_deal_vol_data_front(v_data, begin)
    v_data = slice_deal_vol_data_back(v_data, end)
    return v_data

def fix_raw_data(date, code, result_folder):

    result_file_name = os.path.join(result_folder, f'{code}_{date}')
    if os.path.exists(result_file_name):
        return

    print(f'fix_raw_data: {date} {code}', flush=True)

    file = os.path.join(folder, date, code, '十档盘口.csv')
    begin_t = os.environ.get('train_data_begin_time', '09:30')
    end_t = os.environ.get('train_data_end_time', '14:45')


    if os.path.exists(file):
        _data = pd.read_csv(file, encoding='gbk')

        # 删除完全重复的行
        _data = _data.drop_duplicates(keep='first')

        _data['时间'] = pd.to_datetime(_data['时间'])
        # 截取 10:00 - 14:30
        _data = _data[(_data["时间"].dt.time >= pd.to_datetime(begin_t).time()) & (
            _data["时间"].dt.time < pd.to_datetime(end_t).time())].reset_index(drop=True)
        _data = _data[(_data["时间"].dt.time <= pd.to_datetime('11:30:00').time()) | (
            _data["时间"].dt.time > pd.to_datetime('13:00:00').time())].reset_index(drop=True)
        if len(_data) == 0:
            return

        _data = _data.reset_index(drop=True)

        # 删除列 '卖1价' 和 '买1价' 中存在 NaN 值的行
        _data = _data.dropna(subset=['卖1价', '买1价']).reset_index(drop=True)

        # 格式化
        for col in ['总卖', '总买']:
            try:
                _data[col] = _data[col].astype(float)
            except:
                _data[col] = _data[col].apply(
                    lambda x: 10000 * (float(x.replace("万", "")))
                    if "万" in str(x)
                    else 1e8 * (float(x.replace("亿", "")))
                    if "亿" in str(x)
                    else float(x)
                )

        # 删除连续的重复行
        columns_to_check = list(_data)[1:]
        _data = _data[~_data[columns_to_check].eq(_data[columns_to_check].shift()).all(axis=1)].reset_index(drop=True)
        data = _data

        # 稳健表示
        # 区分成 价格 和 数量
        price_cols = [i for i in list(data)[1:41] if '价' in i]
        vol_cols = [i for i in list(data)[1:41] if '价' not in i]

        # price = (data.loc[:, price_cols].copy() * 1000).astype(int) 
        price = (data.loc[:, price_cols].copy() * 1000 + 0.5).astype(int) 
        vol = data.loc[:, vol_cols].copy()

        # 输出结果
        result = vol.copy()
        result2 = price.copy().astype(float)

        price_array = np.array(price)

        # vol 增加一列 0，表示没有成交量
        vol['0'] = 0

        types = ['买', '卖']

        print_idx = 0
        for _type in range(2):
            # 基准1价
            base_type = (_type + 1) % 2
            base_type = types[base_type]
            base_1_p = price[f'{base_type}1价']
            # print(f'基准{base_type}1价')
            # print(base_1_p[0])
            # print('')

            _type = types[_type]

            # 1价 - 10价
            for i in range(10):
                name = _type + str(i+1) + '价'
                # print(name)
                # msg = ''

                _p = base_1_p + (i+1) * (-1 if _type == '买' else 1)
                # msg += f'{_p[print_idx]} '

                # 成交量
                # 创建一个布尔矩阵，表示每个元素是否等于要查找的值
                col_idx = list(price).index(name)
                mask = price_array == np.array(_p)[:, np.newaxis]
                # msg += f'{mask[print_idx, col_idx]} '
                mask = np.column_stack((mask, ~np.any(mask, axis=1)))# 扩展一列代表对应 vol 新增的 0 列
                # msg += f'{mask[print_idx, col_idx]} '

                # 填充价格
                result2[name] = _p / 1000.0

                # 获取成交量 填充到结果中
                try:
                    result[name.replace('价', '量')] = vol.values[mask]
                except:
                    # 数据异常，不同的档位可能存在相同的价格
                    return

                # msg += f"{result[name.replace('价', '量')][print_idx]}"
                # print(msg)

        mid_price = ((price['买1价'] + price['卖1价']) / 2) / 1000
        result3 = pd.DataFrame({
            'mid_pct' : mid_price.pct_change(),
            'mid_price' : mid_price,
            'mid_vol' : (vol['买1量'] + vol['卖1量']) / 2,
            '时间' : data['时间']
        })
        # # 增加一列中间价格
        # result['mid_price'] = ((price['买1价'] + price['卖1价']) / 2) / 1000
        # result['mid_vol'] = (vol['买1量'] + vol['卖1量']) / 2
        # # 添加时间列
        # result['时间'] = data['时间']

        # 累计深度
        for i in range(10, 0, -1):
            for _type in ['买', '卖']:
                cur_col_name = f'{_type}{i}量'
                wait_sum_col_names = [f'{_type}{i_wait}量' for i_wait in range(1, i)]# 不包含当前档位列名

                # 累加
                for wait_col in wait_sum_col_names:
                    result[cur_col_name] += result[wait_col]

        # 别名
        data_v, data_p, data_other = result, result2, result3

        # 增加成交数据
        data_buy_v = data_v.copy()# 主动买入成交
        data_sell_v = data_v.copy()# 主动卖出成交
        data_buy_v.loc[:, :] = 0.0
        data_sell_v.loc[:, :] = 0.0

        # 读取成交数据
        deal = read_deal_data(date, code)
        # 按秒先进行采样
        d2 = deal.groupby(['时间', '价格', '类型']).sum()
        # 盘口切片时间
        deal_price_vol = data_other[['时间']].copy()
        deal_price_vol['begin'] = deal_price_vol['时间'].shift().fillna(pd.Timestamp('1990-01-01 00:00:00'))

        # 遍历时间点进行采样
        idx = 0
        # for begin, end in tqdm(zip(deal_price_vol['begin'].to_list(), deal_price_vol['时间'].to_list()), total=len(deal_price_vol)):
        for begin, end in zip(deal_price_vol['begin'].to_list(), deal_price_vol['时间'].to_list()):
            _d2 = d2[begin: end].reset_index()
            _d2 = _d2.iloc[:, 1:].groupby(['价格', '类型']).sum().reset_index()

            for p, v, t in zip(_d2['价格'], _d2['手'], _d2['类型']):
                deal_v_data = data_buy_v if t == 0 else data_sell_v
                for i in range(20):
                    if data_p.iloc[idx, i] == p:
                        deal_v_data.iloc[idx, i] = v
                        # break
            idx += 1
        
        # 成交量切片，删除大量0值的范围
        # 暂时定于 5:-5
        data_buy_v = slice_deal_vol_data(data_buy_v)
        data_sell_v = slice_deal_vol_data(data_sell_v)
        # 更改列名
        data_sell_v.columns = ['DS' + i for i in list(data_sell_v)]
        data_buy_v.columns = ['DB' + i for i in list(data_buy_v)]

        # 拼接数据
        # [10 + 10] + [20] + 4 -> 44
        # 成交数据 + 深度数据 + 基础数据
        result = pd.concat([data_buy_v, data_sell_v, data_v, data_other], axis=1)

        # 扩展委托数据
        order_buy_v = data_v.copy()
        order_sell_v = data_v.copy()
        order_buy_cv = data_v.copy()
        order_sell_cv = data_v.copy()
        order_buy_v.loc[:, :] = 0.0
        order_sell_v.loc[:, :] = 0.0
        order_buy_cv.loc[:, :] = 0.0
        order_sell_cv.loc[:, :] = 0.0

        # 读取订单数据
        order = read_order_data(date, code)
        # 按秒先进行采样
        d3 = order.groupby(['时间', '价格', '类型']).sum()

        # 成交类型
        ORDER_TYPES = ["B", "S", "BC", "SC", "1B", "1S", "UB", "US"]
        B, S, BC, SC, _1B, _1S, UB, US = range(len(ORDER_TYPES))

        # 遍历时间进行采样
        idx = 0
        # for begin, end in tqdm(zip(deal_price_vol['begin'].to_list(), deal_price_vol['时间'].to_list()), total=len(deal_price_vol)):
        for begin, end in zip(deal_price_vol['begin'].to_list(), deal_price_vol['时间'].to_list()):
            _d3 = d3[begin: end].reset_index()
            _d3 = _d3.iloc[:, 1:].groupby(['价格', '类型']).sum().reset_index()
            for p, v, t in zip(_d3['价格'], _d3['手'], _d3['类型']):
                # 暂时省略
                if t in [_1B, _1S, UB, US]:
                    continue
                order_v_data = order_buy_v if t == B else order_sell_v if t == S else order_buy_cv if t==BC else order_sell_cv if t==SC else None
                for i in range(20):
                    if data_p.iloc[idx, i] == p:
                        order_v_data.iloc[idx, i] = v
                        # break
            idx += 1
        
        # 成交量切片，删除大量0值的范围
        order_buy_v = slice_deal_vol_data_front(order_buy_v, 5)# 15
        order_sell_v = slice_deal_vol_data_back(order_sell_v, -5)# 15
        order_buy_cv = slice_deal_vol_data_front(order_buy_cv, 10)# 10
        order_sell_cv = slice_deal_vol_data_back(order_sell_cv, -10)# 10
        order_sell_v.columns = ['OS' + i for i in list(order_sell_v)]
        order_buy_v.columns = ['OB' + i for i in list(order_buy_v)]
        order_sell_cv.columns = ['OSC' + i for i in list(order_sell_cv)]
        order_buy_cv.columns = ['OBC' + i for i in list(order_buy_cv)]

        # 拼接数据
        # [15 + 15 + 10 + 10] + [10 + 10] + [20] + 4 -> 94
        # 委托数据 + 成交数据 + 深度数据 + 基础数据
        result = pd.concat([order_buy_v, order_sell_v,order_buy_cv,order_sell_cv, result], axis=1)

        # 拼接原始数据
        # [40] + [15 + 15 + 10 + 10] + [10 + 10] + [20] + 4 -> 134
        # 原始价量数据 + 委托数据 + 成交数据 + 深度数据 + 基础数据
        base_data_all_cols = [
            '卖1价', '卖1量', '买1价', '买1量', '卖2价', '卖2量', '买2价', '买2量', '卖3价', '卖3量', '买3价', '买3量', '卖4价', '卖4量', '买4价', '买4量', '卖5价', '卖5量', '买5价', '买5量', '卖6价', '卖6量', '买6价', '买6量', '卖7价', '卖7量', '买7价', '买7量', '卖8价', '卖8量', '买8价', '买8量', '卖9价', '卖9量', '买9价', '买9量', '卖10价', '卖10量', '买10价', '买10量'
        ]
        base_data = data.loc[:, base_data_all_cols].copy()
        # 更改列名
        base_data.columns = ['BASE' + i for i in list(base_data)]
        result = pd.concat([base_data, result], axis=1)

        # OF 数据
        # 转为特征 deepOF
        of = _data.loc[:, ['买1量']].copy()
        for i in range(10):
            # 买
            col_bid_price = f'买{i+1}价'
            col_bid_vol = f'买{i+1}量'
            of[col_bid_vol] = (_data[col_bid_price] > _data[col_bid_price].shift(1)).apply(lambda x:1 if x else np.nan) * _data[col_bid_vol]
            of[col_bid_vol].fillna((_data[col_bid_price] == _data[col_bid_price].shift(1)).apply(lambda x:1 if x else np.nan) * (_data[col_bid_vol] - _data[col_bid_vol].shift(1)), inplace=True)
            of[col_bid_vol].fillna((_data[col_bid_price] < _data[col_bid_price].shift(1)).apply(lambda x:1 if x else np.nan) * (_data[col_bid_vol].shift(1)) * -1, inplace=True)

            # 卖
            col_ask_price = f'卖{i+1}价'
            col_ask_vol = f'卖{i+1}量'
            of[col_ask_vol] = (_data[col_ask_price] > _data[col_ask_price].shift(1)).apply(lambda x:1 if x else np.nan) * (_data[col_ask_vol].shift(1)) * -1
            of[col_ask_vol].fillna((_data[col_ask_price] == _data[col_ask_price].shift(1)).apply(lambda x:1 if x else np.nan) * (_data[col_ask_vol] - _data[col_ask_vol].shift(1)), inplace=True)
            of[col_ask_vol].fillna((_data[col_ask_price] < _data[col_ask_price].shift(1)).apply(lambda x:1 if x else np.nan) * _data[col_ask_vol], inplace=True)
        # 更改列名
        of.columns = ['OF' + i for i in list(of)]
        # 拼接数据
        # [20] + [40] + [15 + 15 + 10 + 10] + [10 + 10] + [20] + 4 -> 154
        # of数据 + 原始价量数据 + 委托数据 + 成交数据 + 深度数据 + 基础数据
        result = pd.concat([of, result], axis=1)

        # 删除第一行
        result = result.iloc[1:, :].reset_index(drop=True)

        # nan/inf 检查
        has_nan = result.isna().any().any()
        has_inf = (result == float('inf')).any().any()
        if has_nan or has_inf:
            # 发送微信提醒 
            msg += f'{date} {code} fix数据 包含 NaN 或 Inf'
            send_wx(msg)
            os._exit(0)  # 完全终止程序

        # 储存
        pickle.dump(result, open(result_file_name, 'wb'))

etf_code_names = ak.fund_etf_category_sina(symbol="ETF基金").loc[:, ['代码', '名称']]
lof_code_names = ak.fund_etf_category_sina(symbol="LOF基金").loc[:, ['代码', '名称']]
code_names = pd.concat([etf_code_names, lof_code_names], ignore_index=True)
code_names['代码'] = code_names['代码'].apply(lambda x: x[2:])
# 过滤t0
code_names = code_names.loc[code_names['代码'].isin(
    t0_codes), :].reset_index(drop=True)
names = code_names['名称'].to_list()
codes = code_names['代码'].to_list()
code_names

# %%
# 获取所有日线数据
data_file = r'Z:\L2_DATA\his_t0_daily_k.csv' if not kaggle else '/kaggle/input/lh-q-t0-daily-k-20240907/his_t0_daily_k.csv'
# 文件存在且修改时间<24h
# if os.path.exists(data_file) and (os.path.getmtime(data_file) > time.time() - 12 * 3600 or kaggle):
if os.path.exists(data_file):
    ###########
    # code 列应为 str
    data = pd.read_csv(data_file, dtype={'code': str})
else:
    data = pd.DataFrame()
    # for i, code in tqdm(enumerate(codes), total=len(codes)):
    for i, code in enumerate(codes):
        # print(i, code)
        if 'LOF' in names[i]:
            df = ak.fund_lof_hist_em(
                symbol=code, period="daily", start_date="20000101", end_date="20500101", adjust="")
        else:
            df = ak.fund_etf_hist_em(
                symbol=code, period="daily", start_date="20000101", end_date="20500101", adjust="")
        df['code'] = code

        data = pd.concat([data, df], ignore_index=True)
        time.sleep(3)
    data.to_csv(data_file, index=False)

data['日期'] = data['日期'].apply(lambda x: x.replace('-', ''))
data = data.loc[(data['日期'].isin(dates)) & (data['code'].isin(t0_codes)), :].reset_index(drop=True)
# 过滤收盘价大于 10 的标的（大部分货币基金/债券基金）
price_code_data = data.loc[:, ['code', '收盘']].groupby(['code']).last()
gte_10_codes = price_code_data[price_code_data['收盘'] >= 10].index.to_list()
data = data.loc[~data['code'].isin(gte_10_codes), :].reset_index(drop=True)

def func(_data, res):
    date = _data.iloc[0]['日期']

    # 筛选每日成交量 >=
    if 'filter_min_vol' in os.environ:
        min_vol = 1e6 if not kaggle else float(os.environ['filter_min_vol'])
        res[date] = _data.loc[_data['成交量'] >= min_vol]['code'].to_list()

    # 筛选每日成交额 >=
    elif 'filter_min_am' in os.environ:
        min_amount = 1e9 if not kaggle else float(os.environ['filter_min_am'])
        res[date] = _data.loc[_data['成交额'] >= min_amount]['code'].to_list()

res = {}
train_data_codes = os.environ.get('train_data_codes', '')
if not train_data_codes:
    data.groupby('日期', group_keys=False).apply(func, res)
else:
    train_data_codes = train_data_codes.split(',')
    _dates = sorted(list(set(data['日期'].to_list())))
    res = {date: train_data_codes for date in _dates}
print(f'len(res) {len(res)}:\n{list(res.keys())}')

# 用于计算标准化数据的历史天数
std_pass_n = int(os.environ.get('train_data_std_pass_day_n', '5'))

# 生成标准化数据的方法
# simple / each_sample
std_method = os.environ.get('train_data_std_method', 'each_sample')

# 获取以完成的列表
std_folder = os.path.join(l2_data, 'std')
os.makedirs(std_folder, exist_ok=True)
std_done_dates = os.listdir(std_folder)

# ##################
# # 测试用
# std_done_dates = []
# ##################

if len(std_done_dates):
    std_done_dates_idx = [trade_dates.index(i)-1 for i in std_done_dates]
    std_done_dates = [trade_dates[i]
                      for i in std_done_dates_idx]
    std_done_dates.sort()
    wait_dates = [i for i in dates[std_pass_n - 1:] if i >= std_done_dates[-1]]
else:
    wait_dates = dates[std_pass_n - 1:]

if rank > -1 and len([i for i in os.environ.get('train_data_codes', '').split(',') if i])!=1:
    # kaggle上分块运行
    begin = rank*max_handle_date_num
    wait_dates = wait_dates[begin: begin+max_handle_date_num]

result_folder = './fix_raw_data'
os.makedirs(result_folder, exist_ok=True)

print(f'wait_dates: {wait_dates}')
for date in wait_dates:
    # date = '20240719'
    # date = dates[-1]
    trade_dates_idx = trade_dates.index(date)

    if trade_dates_idx + 1 == len(trade_dates):
        print(f'{date} 为最后一天，无需计算下一天的标准化数据')
        continue

    cal_date = trade_dates[trade_dates_idx + 1]

    if cal_date not in res:
        print(f'{cal_date} not in res')
        continue

    begin_dt = trade_dates[trade_dates_idx - std_pass_n + 1]
    end_dt = trade_dates[trade_dates_idx]

    print(f'开始生成 {cal_date} 日的标准化数据')
    print(f'使用的历史数据 {begin_dt} 至 {end_dt}')

    # 按照 修改日期 排序获取其中的文件夹列表
    code_list = res[cal_date]

    # #################################
    # # 测试用
    # code_list = t0_codes
    # #################################

    # T0 标的列表
    code_list = [i for i in code_list if i in t0_codes]
    print('len(code_list):', len(code_list))

    a_std_folder = os.path.join(l2_data, 'std', cal_date)
    os.makedirs(a_std_folder, exist_ok=True)

    # ####################
    # # 测试用
    # produce('20231027', '159605', trade_dates,
    #         trade_dates.index('20231027'), pass_n, os.path.join(r'C:\Users\lh\Desktop\temp\temp', f'159605_mean_std.pkl'), std_pass_n, std_method)
    # args = []
    # ####################

    args = []
    for code in code_list:
        # for code in ['520830']:
        # code = code_list[0]
        # code = '513500'
        # print(code)

        save_file = os.path.join(a_std_folder, f'{code}_mean_std.pkl')
        if os.path.exists(save_file):
            print(f'{cal_date} {code} 标准化数据已存在，跳过计算')
            continue

        # 修复另存原始数据
        # 从计算日期开始 - std_pass_n 天, 共 std_pass_n + 1天
        # std_pass_n = 5
        # 0 - 1 - 2 - 3 - 4 - 5
        #                     ^
        begin_fix_idx = trade_dates_idx + 1
        for i in range(std_pass_n + 1):
            _fix_idx = begin_fix_idx - i
            _fix_date = trade_dates[_fix_idx]
            fix_raw_data(_fix_date, code, result_folder)

        args.append((date, code, trade_dates,
                    trade_dates_idx, pass_n, save_file, std_pass_n, std_method))

        # ####################
        # # 测试用
        # produce(date, code, trade_dates,
        #         trade_dates_idx, pass_n, save_file, std_pass_n, std_method)
        # args = []
        # ####################

    if args:
        full_cpu_run_same_func(produce, args, -1 if kaggle else 4)

    # 检查是否有生成数据 a_std_folder
    if not os.listdir(a_std_folder):
        # 删除文件夹
        shutil.rmtree(a_std_folder)

if kaggle:
    compress_folder(r'/kaggle/working/std', f'std_{rank}.7z', level=9, inplace=False)

print('####################################')
print('生成训练数据')
print('####################################')


code = os.environ.get('produce_train_data_code', default="")
date = os.environ.get('produce_train_data_date', default="")
produce_train_data_folder = os.environ.get('produce_train_data_folder', default="")

kaggle = any(key.startswith("KAGGLE") for key in os.environ.keys())

# %%
std_folder = r'Z:\L2_DATA\std' if not kaggle else r'/kaggle/working/std'
os.makedirs(std_folder, exist_ok=True)
# 读取最新的日期文件夹
if not code:
    # 获取历史数据日期列表
    l2_data = r'Z:\L2_DATA'
    folder = os.path.join(l2_data, 'his_data')
    if kaggle:
        l2_data = r'/kaggle/working/'
        folder = os.environ['input_raw_folder']
    dates = []
    for file in os.listdir(folder):
        if '.7z' not in file:
            dates.append(file)
    dates = sorted(dates)
    # t0 标的列表
    t0_codes = []
    date_folder = os.path.join(folder, dates[-1])
    if not kaggle and not t0_codes:
        # 非 kaggle
        # 根据最新的日期目录确定
        # 按照 修改日期 排序获取其中的文件夹列表
        t0_codes = sorted(os.listdir(date_folder),
                        key=lambda x: os.path.getmtime(os.path.join(date_folder, x)))
        # 截断，只选取 t+0 etf
        kzz_idx = 0
        for i, file in enumerate(t0_codes):
            if file.startswith('123'):
                kzz_idx = i
                break
        if kzz_idx == 0:
            send_wx('没有找到 可转债 标的文件夹，请检查')
            sys.exit(0)
        t0_codes = t0_codes[:kzz_idx]
    elif not t0_codes:
        # kaggl上直接读取
        t0_codes = sorted(os.listdir(date_folder))
    print(f't0_codes {len(t0_codes)}')

    # 按照std数据读取日期
    dates = os.listdir(std_folder)
    dates.sort()
    print(f'len(dates) {len(dates)}:\n{dates}')

    # %%
    etf_code_names = ak.fund_etf_category_sina(symbol="ETF基金").loc[:, ['代码', '名称']]
    lof_code_names = ak.fund_etf_category_sina(symbol="LOF基金").loc[:, ['代码', '名称']]
    code_names = pd.concat([etf_code_names, lof_code_names], ignore_index=True)
    code_names['代码'] = code_names['代码'].apply(lambda x: x[2:])
    # 过滤t0
    code_names = code_names.loc[code_names['代码'].isin(
        t0_codes), :].reset_index(drop=True)
    names = code_names['名称'].to_list()
    codes = code_names['代码'].to_list()
    code_names

    # %%
    # 获取所有日线数据
    data_file = r'Z:\L2_DATA\his_t0_daily_k.csv' if not kaggle else '/kaggle/input/lh-q-t0-daily-k-20240907/his_t0_daily_k.csv'
    # 文件存在且修改时间<24h
    # if os.path.exists(data_file) and (os.path.getmtime(data_file) > time.time() - 12 * 3600 or kaggle):
    if os.path.exists(data_file):
        ###########
        # code 列应为 str
        data = pd.read_csv(data_file, dtype={'code': str})
    else:
        data = pd.DataFrame()
        # for i, code in tqdm(enumerate(codes), total=len(codes)):
        for i, code in enumerate(codes):
            # print(i, code)
            if 'LOF' in names[i]:
                df = ak.fund_lof_hist_em(
                    symbol=code, period="daily", start_date="20000101", end_date="20500101", adjust="")
            else:
                df = ak.fund_etf_hist_em(
                    symbol=code, period="daily", start_date="20000101", end_date="20500101", adjust="")
            df['code'] = code

            data = pd.concat([data, df], ignore_index=True)
            time.sleep(3)
        data.to_csv(data_file, index=False)

    data['日期'] = data['日期'].apply(lambda x: x.replace('-', ''))
    data = data.loc[(data['日期'].isin(dates)) & (data['code'].isin(t0_codes)), :].reset_index(drop=True)
    # 过滤收盘价大于 10 的标的（大部分货币基金/债券基金）
    price_code_data = data.loc[:, ['code', '收盘']].groupby(['code']).last()
    gte_10_codes = price_code_data[price_code_data['收盘'] >= 10].index.to_list()
    data = data.loc[~data['code'].isin(gte_10_codes), :].reset_index(drop=True)

    def func(_data, res):
        date = _data.iloc[0]['日期']

        # 筛选每日成交量 >=
        if 'filter_min_vol' in os.environ:
            min_vol = 1e6 if not kaggle else float(os.environ['filter_min_vol'])
            res[date] = _data.loc[_data['成交量'] >= min_vol]['code'].to_list()

        # 筛选每日成交额 >=
        elif 'filter_min_am' in os.environ:
            min_amount = 1e9 if not kaggle else float(os.environ['filter_min_am'])
            res[date] = _data.loc[_data['成交额'] >= min_amount]['code'].to_list()


    res = {}
    train_data_codes = os.environ.get('train_data_codes', '')
    if not train_data_codes:
        data.groupby('日期', group_keys=False).apply(func, res)
    else:
        train_data_codes = train_data_codes.split(',')
        _dates = sorted(list(set(data['日期'].to_list())))
        res = {date: train_data_codes for date in _dates}

else:
    res = {date: [code]}

print(f'len(res) {len(res)}:\n{list(res.keys())}')

predict_n = [3, 5, 10, 15, 30, 60, 100]
pass_n = 105

def none_zero_begin_end_data(df, cols):

    if not isinstance(cols, list):
        cols = [cols]

    df_nonzero = df
    for col in cols:
        # 如果全为0 直接返回None
        if (df_nonzero[col] == 0).all():
            return None

        # 剔除 列开头一直为 0 的行
        start_nonzero_idx = df_nonzero[col].ne(0).idxmax()
        df_nonzero = df_nonzero.iloc[start_nonzero_idx:].reset_index(drop=True)
        # print("剔除开头一直为0的行:\n", df_nonzero)

        # 剔除 列结尾一直为 0 的行
        end_nonzero_idx = df_nonzero[col][::-1].ne(0).idxmax()
        df_nonzero = df_nonzero.iloc[:end_nonzero_idx +
                                     1].reset_index(drop=True)
        # print("\n剔除结尾一直为0的行:\n", df_nonzero)

        # 仍然存在 0 值, 返回None
        if (df_nonzero[col] == 0).any():
            return None

    return df_nonzero

# 若 produce_train_data_code 不为空 则使用 produce_train_data_code 作为 out_folder
# 若 train_data_codes为空 或 train_data_codes包含,(存在多个标的) 时，使用 train_data 作为 out_folder
# 否则使用 train_data_codes 作为 out_folder
out_folder = std_folder.replace('std', 'train_data') if ((not os.environ.get('train_data_codes', default="")) or (',' in os.environ.get('train_data_codes', default=""))) else os.environ['train_data_codes'] if not os.environ.get('produce_train_data_code', default="") else produce_train_data_folder
os.makedirs(out_folder, exist_ok=True)
done_dates = [f for f in os.listdir(
    out_folder) if os.path.isfile(os.path.join(out_folder, f))] if not os.environ.get('produce_train_data_code', default="") else []
# 排序
done_dates.sort()
print(f'已处理 {len(done_dates)} 个日期的数据')

def produce_train_data(train_data_date, std_folder, out_folder, use_codes):
    std_data_folder = os.path.join(std_folder, train_data_date)
    out_file = os.path.join(out_folder, f'{train_data_date}.pkl')

    # 检查，use_codes 必须与 std_data_folder 中的一致
    std_codes = [i.split('_')[0] for i in os.listdir(std_data_folder)]
    use_codes.sort()
    std_codes.sort()
    for i in std_codes:
        assert i in use_codes, f'use_codes: {use_codes}\nstd_codes: {std_codes}'

    all_raw_data = pd.DataFrame()
    if not use_codes:
        print(f'[{train_data_date}] use_codes为空，跳过')
        assert std_codes == use_codes, f'use_codes: {use_codes}\nstd_codes: {std_codes}'
        return

    print(f'[{train_data_date}] 开始生成训练数据')
    print(f'use_codes: {len(use_codes)}\n{use_codes}')

    begin_t = os.environ.get('train_data_begin_time', '09:30')
    end_t = os.environ.get('train_data_end_time', '14:45')

    result_folder = './fix_raw_data'
    # for code in tqdm(use_codes):
    for code in use_codes:
        std_file_path = os.path.join(std_data_folder, f'{code}_mean_std.pkl')
        raw_file_path = os.path.join(result_folder, f'{code}_{train_data_date}')

        ##########
        # 测试用 
        # 测试时注释
        if not os.path.exists(std_file_path):
            print(f'标准化数据不存在，跳过: {std_file_path}')
            continue

        if not os.path.exists(raw_file_path):
            print(f'原始数据不存在，跳过: {raw_file_path}')
            continue
        ##########

        raw = pickle.load(open(raw_file_path, 'rb'))

        # 检查0值
        am = raw.loc[raw['时间'].dt.time <=
                     pd.to_datetime('11:30:00').time(), :].reset_index(drop=True)
        pm = raw.loc[raw['时间'].dt.time >
                     pd.to_datetime('13:00:00').time(), :].reset_index(drop=True)
        parts = [am, pm]

        raw = pd.concat(parts, ignore_index=True)
        if len(raw) == 0:
            print(f'[{code}] 原始数据长度为0，跳过')
            continue

        # id : code_timestamp btcusdt_1710289478588
        raw['code'] = code
        raw['id'] = code + \
            '_' + \
            raw['时间'].apply(
                lambda x: int(x.timestamp()) - 28800).astype(str)
        # 合并
        all_raw_data = pd.concat([all_raw_data, raw], ignore_index=True)

    if len(all_raw_data) == 0:
        print(f'[{train_data_date}] 缺少原始数据，跳过')
        return

    all_raw_data['idx'] = range(len(all_raw_data))
    all_raw_data

    x = []
    y = []
    mean_std = []
    ids = []

    # loop = tqdm(use_codes, desc=f'')
    for code in use_codes:
        # loop.set_description(f'{code}')

        # 原始数据
        _raw = all_raw_data.loc[all_raw_data['code'] == code]
        if len(_raw) == 0:
            print(f'[{code}] 缺少原始数据，跳过')
            continue

        # 读取标准化数据
        std_data = pickle.load(
            open(os.path.join(std_data_folder, f'{code}_mean_std.pkl'), 'rb'))

        # 区分上下午计算生成
        am = _raw.loc[_raw['时间'].dt.time <=
                      pd.to_datetime('11:30:00').time(), :].copy()
        pm = _raw.loc[_raw['时间'].dt.time >
                      pd.to_datetime('13:00:00').time(), :].copy()
        raws = [am, pm]

        for _data in raws:
            if len(_data) < pass_n + max(predict_n):
                continue

            # _data = raws[0]
            # 计算中间价格
            middle_price = _data['mid_price']

            target_cols = []
            for n in predict_n:
                _target_cols = []

                mean_mid_p0 = middle_price.rolling(n).mean().shift(-n)

                # ###############################################
                # # 标签 0/1/2/3
                # for i in [9, 6, 3, 1]:
                #     # 中间均价变动率
                #     mean_mid_p = middle_price.rolling(min(i, n)).mean().shift(-int(min(i, n)/2))
                #     # # 计算target 0
                #     # 中间价格变化 / 最小价格变动单位
                #     # T0: 0.001
                #     _data[f'{n}_mid_diff_pct_{i}'] = (mean_mid_p0 - mean_mid_p) / 0.001
                #     _target_cols.append(f'{n}_mid_diff_pct_{i}')

                ###############################################
                # 标签 paper
                mean_mid_p = middle_price.rolling(n).mean()
                _data[f'{n}_paper'] = (mean_mid_p0 - mean_mid_p) / 0.001
                _target_cols.append(f'{n}_paper')

                # 标签 paper pct
                _data[f'{n}_paper_pct'] = (mean_mid_p0 / mean_mid_p) - 1
                _target_cols.append(f'{n}_paper_pct')

                ###############################################
                # 标签 1 
                # 目标价格为t + predict_n时点 前后k个中间价格的均值(k=3)
                # 当前价格为t时点中间价
                # 再根据阈值化分类别(目的是使得类别均衡)
                k = 3
                p = middle_price.rolling(2*k + 1).mean()
                p2 = p.shift(-k)
                p_1 = p2.shift(-n)
                _data[f'{n}_label_1'] = (p_1 - middle_price) / 0.001
                _target_cols.append(f'{n}_label_1')
                # pct
                _data[f'{n}_label_1_pct'] = p_1 / middle_price - 1
                _target_cols.append(f'{n}_label_1_pct')

                target_cols.append(_target_cols)

            # print(f"原始数据长度: {len(_data)}")
            for i in range(pass_n, len(_data)):
                x_data = _data.iloc[i - pass_n:i]
                cur_data = _data.iloc[i-1]

                target = []
                idx = 0
                for i in target_cols:
                    for j in i:
                        if j.endswith('_target_5_period_point'):
                            # 5区间统计
                            target.append(cal_period_pct_point(
                                cur_data, predict_n[idx]))
                        else:
                            target.append(cur_data[j])
                    idx += 1

                # 检查是否包含nan
                if np.isnan(np.array(target)).any():
                    continue

                ids.append(cur_data['id'])

                # mean_std.append(price_col_mean_std_list)
                # 加入全部的标准化数据
                mean_std.append(std_data)

                y.append(target)
                # 将切片的起始和结束索引加入到x中，相对于最后张量的索引
                x.append((x_data['idx'].min(), x_data['idx'].max()+1))

    if len(x) == 0:
        return

    # 储存数据
    if os.environ.get('produce_train_data_simple_test', '') == 'True':
        def yfunc(x):
            return 0 if x>0.5 else 1 if x<-0.5 else 2

        begin = min(ids).split('_')[1]
        end = max(ids).split('_')[1]
        code = use_codes[0]
        with open(os.path.join(out_folder,f'{code}_{begin}_{end}.csv'), 'w')as f:
            f.write('timestamp,target,predict\n')
            for _id, _y in zip(ids, y):
                _y = yfunc(_y[3])
                f.write(f"{_id.split('_')[1]},{_y},{_y}\n")

        print(f'样本数量: {len(ids)}')

    else:
        pickle.dump(
            (ids, mean_std, x, y, all_raw_data.iloc[:, :153]), open(out_file, 'wb'))
    print(f'[{train_data_date}] 生成完毕')

# ###########
# # 测试用
# _dates = list(res.keys())[-8:]
# res = {_date: ['513500', '518880', '159941', '513050']
#        for _date in res if _date in _dates}
# res = {'2024-08-05': ['513500', '518880', '159941', '513050']}
# out_folder = r'D:\code\featrue_data\notebook\20240413_滚动标准化\label_test'
# ###########
print(f'待处理 {len(res)} 个日期的数据:\n{list(res.keys())}')
args = []
for train_data_date in res:
    ###########
    # 测试用
    # 测试时注释
    if len(done_dates) > 0 and train_data_date <= done_dates[-1]:
        print(f'[{train_data_date}] 已完成，跳过')
        continue
    ###########

    use_codes = res[train_data_date]
    args.append((train_data_date, std_folder, out_folder, use_codes))

    # ###########
    # # 测试用
    # produce_train_data(train_data_date, std_folder, out_folder, use_codes)
    # args = []
    # ###########

    if code:
        produce_train_data(train_data_date, std_folder, out_folder, use_codes)
        args = []

if args:
    full_cpu_run_same_func(produce_train_data, args, -1 if kaggle else 4)

####################################
# 打包数据
####################################

train_data_folder = out_folder
all_files = os.listdir(train_data_folder)

if os.environ.get('train_data_split', '1') == '1':
    val_files = ['20240124.pkl', '20240219.pkl', '20240628.pkl', '20240611.pkl', '20240627.pkl', '20240822.pkl', '20231123.pkl', '20240521.pkl', '20231214.pkl', '20240621.pkl', '20231229.pkl', '20240819.pkl', '20240314.pkl', '20240110.pkl', '20240109.pkl']
    test_files = ['20240829.pkl', '20240830.pkl', '20240902.pkl', '20240903.pkl', '20240904.pkl', '20240905.pkl', '20240906.pkl']

    train = os.path.join(train_data_folder, 'train')
    val = os.path.join(train_data_folder, 'val')
    test = os.path.join(train_data_folder, 'test')
    os.makedirs(train, exist_ok=True)
    os.makedirs(val, exist_ok=True)
    os.makedirs(test, exist_ok=True)

    for file in all_files:
        src = os.path.join(train_data_folder, file)
        if file in val_files:
            dst = os.path.join(val, file)
        elif file in test_files:
            dst = os.path.join(test, file)
        else:
            dst = os.path.join(train, file)

        shutil.move(src, dst)

# 检查是否全空
# 递归遍历 train_data_folder，统计所有 '.pkl' 后缀文件的数量
num_files = 0
for root, dirs, files in os.walk(train_data_folder):
    for file in files:
        if file.endswith('.pkl'):
            num_files += 1

if (num_files > 0):
    zip_file = f'{produce_name}_{rank}.7z'
    compress_folder(train_data_folder, zip_file, level=9, inplace=False)

    # 上传alist
    if os.environ.get('upload_alist', '0') == '1':
        from py_ext.alist import alist
        client = alist('admin', 'LHss6632673')
        client.upload(zip_file, '/produce_train_data/')

