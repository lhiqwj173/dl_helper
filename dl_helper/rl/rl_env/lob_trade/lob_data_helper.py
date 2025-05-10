import numpy as np

def fix_raw_data(all_raw_data):
    # fix 在某个时点上所有数据都为0的情况，导致模型出现nan的bug
    all_cols = list(all_raw_data)
    if 'OBC买10量' in all_cols and 'OSC卖10量' in all_cols:
        # 订单数据
        order_cols = [i for i in all_cols if i.startswith('OS') or i.startswith('OB')]
        order_raw = all_raw_data.loc[:, order_cols]
        all_raw_data.loc[(order_raw == 0).all(axis=1), ['OBC买10量', 'OSC卖10量']] = 1
    if 'OF买10量' in all_cols and 'OF卖10量' in all_cols:
        # OF数据
        OF_cols = [i for i in all_cols if i.startswith('OF')]
        OF_raw = all_raw_data.loc[:, OF_cols]
        all_raw_data.loc[(OF_raw == 0).all(axis=1), ['OF买10量', 'OF卖10量']] = 1
    if '卖10量' in all_cols and '卖10价' not in all_cols:
        # 深度数据
        depth_cols = ['卖10量',
            '卖9量',
            '卖8量',
            '卖7量',
            '卖6量',
            '卖5量',
            '卖4量',
            '卖3量',
            '卖2量',
            '卖1量',
            '买1量',
            '买2量',
            '买3量',
            '买4量',
            '买5量',
            '买6量',
            '买7量',
            '买8量',
            '买9量',
            '买10量']
        depth_raw = all_raw_data.loc[:, depth_cols]
        wait_fix_index = depth_raw[(depth_raw == 0).all(axis=1)].index.to_list()
        if wait_fix_index and wait_fix_index[0] == 0:
            # 若第一个数据就为0，填充 卖10量/买10量 为1，最小化影响
            all_raw_data.loc[0, '卖10量'] = 1
            all_raw_data.loc[0, '买10量'] = 1
            # 去掉第一个记录
            wait_fix_index = wait_fix_index[1:]

        all_raw_data.loc[wait_fix_index, depth_cols] = np.nan# 先用nan填充，方便后续处理
        for col in depth_cols:
            all_raw_data[col] = all_raw_data[col].ffill()
    if 'DB卖1量' in all_cols and 'DS买1量' in all_cols: 
        # 成交数据
        deal_cols = [i for i in all_cols if i.startswith('D')]
        deal_raw = all_raw_data.loc[:, deal_cols]
        all_raw_data.loc[(deal_raw == 0).all(axis=1), ['DB卖1量', 'DS买1量']] = 1
    # 40档位价量数据nan处理
    if 'BASE卖1量' in all_cols and 'BASE买1量' in all_cols:
        # 价格nan填充, 使用上一个档位数据 +-0.001 进行填充
        for i in range(2, 11):
            if f'BASE买{i}价' not in all_cols or f'BASE买{i-1}价' not in all_cols:
                continue

            # 买价
            all_raw_data.loc[:, f'BASE买{i}价'] = all_raw_data[f'BASE买{i}价'].fillna(all_raw_data[f'BASE买{i-1}价'] - 0.001)

            # 卖价
            all_raw_data.loc[:, f'BASE卖{i}价'] = all_raw_data[f'BASE卖{i}价'].fillna(all_raw_data[f'BASE卖{i-1}价'] + 0.001)

        # 量nan用0填充
        vol_cols = [i for i in list(all_raw_data) if i.startswith('BASE') and '价' not in i]
        all_raw_data[vol_cols] = all_raw_data[vol_cols].fillna(0)

    return all_raw_data
