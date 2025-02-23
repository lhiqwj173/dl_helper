
import os, random, datetime, pytz, pickle, json
import numpy as np
import pandas as pd
from dl_helper.train_param import in_kaggle
from py_ext.tool import log

class data_producer:
    """
    随机起始日期文件
    当天数据读取完毕，若存在配对仓位，则继续读取下一个交易日的数据

    特征列:
        [
            time,
            code1_p4_ask,code1_p4_bid,code1_p3_ask,code1_p3_bid,code1_p2_ask,code1_p2_bid,code1_p1_ask,code1_p1_bid,code1_p0_ask,code1_p0_bid,
            code2_p4_ask,code2_p4_bid,code2_p3_ask,code2_p3_bid,code2_p2_ask,code2_p2_bid,code2_p1_ask,code2_p1_bid,code2_p0_ask,code2_p0_bid,
            spread_p0,spread_p1,spread_p2,spread_p3,spread_p4,
            code1_p5_close,code1_p4_close,code1_p3_close,code1_p2_close,code1_p1_close,
            code2_p5_close,code2_p4_close,code2_p3_close,code2_p2_close,code2_p1_close,
            spread_p1d,spread_p2d,spread_p3d,spread_p4d,spread_p5d,
            seconds_to_close,id
        ]
    """
    def __init__(self, data_type='train', need_cols=[]):
        """
        'data_type': 'train',# 训练/测试
        'need_cols': []# 需要的特征列名,默认所有列
        """        
        # 需要的特征列名
        self.need_cols = need_cols

        # 训练数据
        if in_kaggle:
            input_folder = r'/kaggle/input'
            # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
            data_folder_name = os.listdir(input_folder)[0]
            self.data_folder = os.path.join(input_folder, data_folder_name)
        else:
            self.data_folder = r'D:\L2_DATA_T0_ETF\train_data\RL_match_data'

        self.data_type = data_type
        self.files = []
        
        # 数据内容 TODO
        # ids, mean_std, x, all_self.all_raw_data_data
        self.ids = []
        self.x = []
        self.all_raw_data = None

        # 数据索引
        self.idxs = []
        self.col_idx = {}
        # 买卖1档价格
        self.code1_ask_price = 0
        self.code1_bid_price = 0
        self.code2_ask_price = 0
        self.code2_bid_price = 0
        # id
        self.id = ''
        # 当前日期数据停止标志
        self.date_file_done = False

    def _pre_files(self):
        """
        准备文件列表
        若是训练数据，随机读取一个日期及之后的文件
        若是验证/测试数据，按顺序读取
        """
        self.files = sorted([i for i in os.listdir(os.path.join(self.data_folder, self.data_type)) if '.csv' in i])
        if self.data_type == 'train':
            # 随机选择一个日期文件
            random_date = random.choice(self.files)[:8]
            # 删除早于这个日期的文件
            self.files = [f for f in self.files if f[:8] >= random_date]

        if self.file_num:
            self.files = self.files[:self.file_num]
            
    def _load_data(self):
        """
        按照文件列表读取数据
        每次完成后从文件列表中剔除
        """
        file = self.files.pop(0)
        log(f'load date file({self.data_type}): {file}')
        # 读取数据
        his_file = os.path.join(self.data_folder, self.data_type, file)
        std_file = os.path.join(self.data_folder, self.data_type, file.replace('.csv', '.json'))
        self.all_raw_data = pd.read_csv(his_file)
        with open(std_file, 'r') as f:
            self.mean_std = json.load(f)

        # 列过滤
        if self.need_cols:
            # 只保留需要的列
            self.all_raw_data = self.all_raw_data.loc[:, self.need_cols]

        # 记录需要的索引，供后续转为numpy时使用
        for col in ['code1_p0_ask', 'code1_p0_bid', 'code2_p0_ask', 'code2_p0_bid']:
            self.col_idx[col] = self.all_raw_data.columns.get_loc(col)
       
        # 标准化数据 TODO


        # self.all_raw_data 转为 numpy
        self.all_raw_data = self.all_raw_data.values

        # 初始化绘图索引
        self.plot_begin, self.plot_cur = self.x[self.idxs[0][0]]
        self.plot_cur -= 1
        self.plot_cur_pre = -1
        _, self.plot_end = self.x[self.idxs[0][1]]
        # 准备绘图数据
        self.pre_plot_data()

        # # 测试用
        # pickle.dump((self.all_raw_data, self.mean_std, self.x), open(f'{self.data_type}_raw_data.pkl', 'wb'))

        # 载入了新的日期文件数据
        # 重置日期文件停止标志
        self.date_file_done = False

    def set_data_type(self, data_type):
        self.data_type = data_type

    def data_size(self):
        # 运行只获取部分列， 简化数据
        return self.cols_num*self.his_len

    def use_data_split(self, raw, ms):
        """
        使用数据分割
        raw 是完整的 pickle 切片
        ms 是标准化数据df
        都是 numpy 数组
        TODO 运行只获取部分列， 简化数据
        """
        if self.need_cols:
            return raw, ms.iloc[self.need_cols_idx, :].values
        else:
            return raw[:, :130], ms.iloc[:130, :].values

    def store_bid_ask_1st_data(self, raw):
        """
        存储买卖1档数据 用于撮合交易
        raw 是完整的 pickle 切片
        """
        last_row = raw[-1]  # 最后一个数据
        self.bid_price = last_row[self.col_idx['BASE买1价']]
        self.ask_price = last_row[self.col_idx['BASE卖1价']]

    def get(self):
        """
        输出观察值
        返回 symbol_id, before_market_close_sec, x, done, need_close, period_done
        """
        # # 测试用
        # print(self.idxs[0])

        if self.plot_cur_pre != -1:
            # 更新绘图数据
            self.plot_cur = self.plot_cur_pre

        # 检查日期文件结束
        if self.date_file_done:
            # load 下一个日期文件的数据
            self._load_data()

        # 准备观察值
        a, b = self.x[self.idxs[0][0]]
        if b-a > self.his_len:# 修正历史数据长度
            a = b - self.his_len
        raw = self.all_raw_data[a: b, :].copy()
        # 记录 买卖1档 的价格
        self.store_bid_ask_1st_data(raw)
        # 数据标准化
        ms = pd.DataFrame(self.mean_std[self.idxs[0][0]])
        x, ms = self.use_data_split(raw, ms)
        x -= ms[:, 0]
        x /= ms[:, 1]

        # 标的id
        symbol_id = self.idxs[0][2]

        # 当前标的
        self.code = USE_CODES[int(symbol_id)]

        # 距离市场关闭的秒数
        before_market_close_sec = self.before_market_close_sec[self.idxs[0][0]]

        # 记录数据id
        self.id = self.ids[self.idxs[0][0]]

        # 检查下一个数据是否是最后一个数据
        all_done = False
        need_close = False
        if self.idxs[0][0] == self.idxs[0][1]:
            # 当组 begin/end 完成，需要平仓
            need_close = True
            log(f'need_close {self.idxs[0][0]} {self.idxs[0][1]}')
            # 更新剩余的 begin/end 组
            self.idxs = self.idxs[1:]
            log(f'idxs: {self.idxs}')
            if not self.idxs:
                # 当天的数据没有下一个可读取的 begin/end 组
                log(f'date done')
                self.date_file_done = True
                log(f'len(files): {len(self.files)}')
                if not self.files:
                    # 没有下一个可以读取的日期数据文件
                    log('all date files done')
                    all_done = True
            else:
                # 重置绘图索引
                self.plot_begin, self.plot_cur = self.x[self.idxs[0][0]]
                self.plot_cur -= 1
                self.plot_cur_pre = -1
                _, self.plot_end = self.x[self.idxs[0][1]]
                # 准备绘图数据
                self.pre_plot_data()
        else:
            self.idxs[0][0] += 1
            _, self.plot_cur_pre = self.x[self.idxs[0][0]]
            self.plot_cur_pre -= 1

        # 额外数据的标准化
        # 距离收盘秒数
        before_market_close_sec -= MEAN_SEC_BEFORE_CLOSE
        before_market_close_sec /= STD_SEC_BEFORE_CLOSE
        # id
        symbol_id -= MEAN_CODE_ID
        symbol_id /= STD_CODE_ID

        return symbol_id, before_market_close_sec, x, all_done, need_close, self.date_file_done

    def get_plot_data(self):
        """
        获取绘图数据, 当前状态(时间点)索引, 当前状态(时间点)id
        """
        return self.plot_data, self.plot_cur - self.plot_begin, self.ids[self.plot_cur]

    def reset(self):
        self._pre_files()
        self._load_data()
