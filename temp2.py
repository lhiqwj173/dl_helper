import os, pickle
import pandas as pd
import numpy as np

folder = r'D:\L2_DATA_T0_ETF\his_data'

# 使用截取时间
os.environ['train_data_begin_time'] = '09:00:00'
os.environ['train_data_end_time'] = '15:00:00'

def fix_raw_data(date, code, result_folder):
    """
    数据预处理
    """

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

        # 格式化 时间
        _data['时间'] = pd.to_datetime(_data['时间'])
        # 截取 10:00 - 14:30
        _data = _data[(_data["时间"].dt.time >= pd.to_datetime(begin_t).time()) & (
            _data["时间"].dt.time < pd.to_datetime(end_t).time())].reset_index(drop=True)
        _data = _data[(_data["时间"].dt.time <= pd.to_datetime('11:30:00').time()) | (
            _data["时间"].dt.time > pd.to_datetime('13:00:00').time())].reset_index(drop=True)
        if len(_data) == 0:
            print(f'fix_raw_data: {date} {code} 没有数据')
            return

        # 判断是否有 涨跌停
        # 涨跌停 不使用
        zt = ((_data['卖1价'] == 0) & (_data['卖1量'] == 0)).any()
        dt = ((_data['买1价'] == 0) & (_data['买1量'] == 0)).any()
        if zt or dt:
            print(f'fix_raw_data: {date} {code} 存在涨跌停')
            return

        _data = _data.reset_index(drop=True)

        # 删除列 '卖1价' 和 '买1价' 中存在 NaN 值的行
        # _data = _data.dropna(subset=['卖1价', '买1价']).reset_index(drop=True)
        # 暂时不允许 '卖1价', '买1价' 存在 NaN
        msg = ''
        if _data['卖1价'].isna().any():
            msg += f'{date} {code} 卖1价存在 NaN\n'
            _data.to_csv(f'{date}_{code}_卖1价存在NaN.csv', index=False)
        if _data['买1价'].isna().any():
            msg += f'{date} {code} 买1价存在 NaN\n'
            _data.to_csv(f'{date}_{code}_买1价存在NaN.csv', index=False)
        if msg:
            # send_wx(msg)
            raise Exception(msg)

        # 可以容忍的异常值处理
        # 2-10 档位价格nan填充, 使用上一个档位数据 +-0.001 进行填充
        for i in range(2, 11):
            # 买价
            _data.loc[:, f'买{i}价'] = _data[f'买{i}价'].fillna(_data[f'买{i-1}价'] - 0.001)
            # 卖价
            _data.loc[:, f'卖{i}价'] = _data[f'卖{i}价'].fillna(_data[f'卖{i-1}价'] + 0.001)
        # 盘口量nan与0都用1填充
        vol_cols = [i for i in list(_data) if '量' in i]
        _data[vol_cols] = _data[vol_cols].replace(0, np.nan).fillna(1)

        # 格式化 总卖 总买
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

        data = _data

        # 信息数据 1
        info_data = pd.DataFrame({
            '时间' : data['时间']
        })

        # 衍生特征 4
        mid_price = (_data['买1价'] + _data['卖1价']) / 2
        depth = (_data['买1量'] + _data['卖1量']) / 2
        ext_data = pd.DataFrame({
            'EXT_mid_price' : mid_price,
            'EXT_mid_vol' : (_data['买1量'] + _data['卖1量']) / 2,
            'EXT_spread' : _data['卖1价'] - _data['买1价'], # 价差
            'EXT_imbalance' : (_data['买1量'] - _data['卖1量']) / (_data['买1量'] + _data['卖1量']), # 买卖盘不平衡度
            # 深度（买一量+卖一量）/2
            'EXT_depth' : depth,
            # 斜率（价差/深度）
            'EXT_slope' : (_data['卖1价'] - _data['买1价']) / depth,
            # 相邻时间点买一价或卖一价的对数差
            'EXT_log_diff_bid' : np.log(_data['买1价']).diff(),
            'EXT_log_diff_ask' : np.log(_data['卖1价']).diff(),
            # 相邻时间点买一量或卖一量的对数差
            'EXT_log_diff_bid_vol' : np.log(_data['买1量']).diff(),
            'EXT_log_diff_ask_vol' : np.log(_data['卖1量']).diff(),
        })

        # 原始数据 40 
        # 原始价量数据 + 信息数据
        base_data_all_cols = [
            '卖1价', '卖1量', '买1价', '买1量', '卖2价', '卖2量', '买2价', '买2量', '卖3价', '卖3量', '买3价', '买3量', '卖4价', '卖4量', '买4价', '买4量', '卖5价', '卖5量', '买5价', '买5量', '卖6价', '卖6量', '买6价', '买6量', '卖7价', '卖7量', '买7价', '买7量', '卖8价', '卖8量', '买8价', '买8量', '卖9价', '卖9量', '买9价', '买9量', '卖10价', '卖10量', '买10价', '买10量'
        ]
        base_data = data.loc[:, base_data_all_cols].copy()
        # 更改列名
        base_data.columns = ['BASE' + i for i in list(base_data)]

        # 拼接数据
        result = pd.concat([base_data, ext_data, info_data], axis=1)

        # 删除第一个数据 （pct 会产生 nan）
        result = result.iloc[1:].reset_index(drop=True)

        # nan/inf 检查
        has_nan = result.isna().any().any()
        has_inf = (result == float('inf')).any().any()
        if has_nan or has_inf:
            # 发送微信提醒 
            msg = f'{date} {code} fix数据 包含 NaN 或 Inf'
            # send_wx(msg)
            raise Exception(msg)

        # 储存
        pickle.dump(result, open(result_file_name, 'wb'))


if __name__ == '__main__':
    fix_raw_data('20240625', '518880', r'C:\Users\lh\Desktop\temp\fix_data')