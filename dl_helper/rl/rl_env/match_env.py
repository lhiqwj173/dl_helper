
import os
from dl_helper.train_param import in_kaggle

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
            seconds_to_close,id
        ]
        TODO
    """
    def __init__(self, data_type='train', need_cols=[]):
        """
        'data_type': 'train',# 训练/测试
        'need_cols': []# 需要的特征列名,默认所有列
        """        
        # 需要的特征列名
        self.need_cols = need_cols
        self.need_cols_idx = []

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

        # 当前数据日期/code
        self.date = ''
        self.code1 = ''
        self.code2 = ''
        
        # 数据内容 TODO
        # ids, mean_std, x, all_self.all_raw_data_data
        self.ids = []
        self.mean_std = []
        self.x = []
        self.all_raw_data = None
        # 距离市场关闭的秒数
        self.before_market_close_sec = []

        # 数据索引
        self.idxs = []
        self.col_idx = {}
        # 用于绘图的索引
        self.plot_begin = 0
        self.plot_end = 0
        self.plot_cur = 0
        self.plot_cur_pre = -1
        self.plot_data = None
        # 买卖1档价格
        self.ask_price = 0
        self.bid_price = 0
        # id
        self.id = ''
        # 当前日期数据停止标志
        self.date_file_done = False

    def pre_plot_data(self):
        """
        预先读取绘图数据
        col_idx: BASE买1价, BASE卖1价, BASE中间价
        """
        # 直接选择需要的列并创建DataFrame
        cols = ['BASE买1价', 'BASE卖1价']
        self.plot_data = pd.DataFrame(self.all_raw_data[self.plot_begin:self.plot_end, [self.col_idx[col] for col in cols]], columns=['bid', 'ask'])
        # 高效计算中间价格
        self.plot_data['mid_price'] = self.plot_data.mean(axis=1)

    def _pre_files(self):
        """
        准备文件列表
        若是训练数据，随机读取
        若是验证/测试数据，按顺序读取
        """
        self.files = os.listdir(os.path.join(self.data_folder, self.data_type))
        if self.data_type == 'train':
            random.shuffle(self.files)

        if self.file_num:
            self.files = self.files[:self.file_num]
            
    def _load_data(self):
        """
        按照文件列表读取数据
        每次完成后从文件列表中剔除
        """
        file = self.files.pop(0)
        log(f'load date file({self.data_type}): {file}')
        self.ids, self.mean_std, self.x, self.all_raw_data = pickle.load(open(os.path.join(self.data_folder, self.data_type, file), 'rb'))

        # 列过滤
        if self.need_cols:
            self.need_cols_idx = [self.all_raw_data.columns.get_loc(col) for col in self.need_cols]
            # 只保留需要的列
            self.all_raw_data = self.all_raw_data.loc[:, self.need_cols]

        if self.simple_test:
            self.ids = self.ids[:8000]
            self.mean_std = self.mean_std[:8000]
            self.x = self.x[:8000]

        # 距离市场关闭的秒数
        self.date = file[:8]
        dt = datetime.datetime.strptime(f'{self.date} 15:00:00', '%Y%m%d %H:%M:%S')
        dt = pytz.timezone('Asia/Shanghai').localize(dt)
        close_ts = int(dt.timestamp())
        self.before_market_close_sec = np.array([int(i.split('_')[1]) for i in self.ids])
        self.before_market_close_sec = close_ts - self.before_market_close_sec

        # 解析标的 随机挑选一个标的数据
        symbols = np.array([i.split('_')[0] for i in self.ids])
        unique_symbols = [i for i in np.unique(symbols) if i != '159941']
        # 获取所有标的的起止索引
        self.idxs = []
        for symbol in unique_symbols:
            symbol_mask = symbols == symbol
            symbol_indices = np.where(symbol_mask)[0]
            self.idxs.append([symbol_indices[0], symbol_indices[-1], USE_CODES.index(symbol)])
    
        # 训练数据随机选择一个标的
        # 一个日期文件只使用其中的一个标的的数据，避免同一天各个标的之间存在的相关性 对 训练产生影响
        if self.data_type == 'train':
            self.idxs = [random.choice(self.idxs)]

        log(f'init idxs: {self.idxs}')

        # 调整数据
        # fix 在某个时点上所有数据都为0的情况，导致模型出现nan的bug
        all_cols = list(self.all_raw_data)
        if 'OBC买10量' in all_cols and 'OSC卖10量' in all_cols:
            # 订单数据
            order_cols = [i for i in all_cols if i.startswith('OS') or i.startswith('OB')]
            order_raw = self.all_raw_data.loc[:, order_cols]
            self.all_raw_data.loc[(order_raw == 0).all(axis=1), ['OBC买10量', 'OSC卖10量']] = 1
        if 'OF买10量' in all_cols and 'OF卖10量' in all_cols:
            # OF数据
            OF_cols = [i for i in all_cols if i.startswith('OF')]
            OF_raw = self.all_raw_data.loc[:, OF_cols]
            self.all_raw_data.loc[(OF_raw == 0).all(axis=1), ['OF买10量', 'OF卖10量']] = 1
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
            depth_raw = self.all_raw_data.loc[:, depth_cols]
            wait_fix_index = depth_raw[(depth_raw == 0).all(axis=1)].index.to_list()
            if wait_fix_index and wait_fix_index[0] == 0:
                # 若第一个数据就为0，填充 卖10量/买10量 为1，最小化影响
                self.all_raw_data.loc[0, '卖10量'] = 1
                self.all_raw_data.loc[0, '买10量'] = 1
                # 去掉第一个记录
                wait_fix_index = wait_fix_index[1:]

            self.all_raw_data.loc[wait_fix_index, depth_cols] = np.nan# 先用nan填充，方便后续处理
            for col in depth_cols:
                self.all_raw_data[col] = self.all_raw_data[col].ffill()
        if 'DB卖1量' in all_cols and 'DS买1量' in all_cols: 
            # 成交数据
            deal_cols = [i for i in all_cols if i.startswith('D')]
            deal_raw = self.all_raw_data.loc[:, deal_cols]
            self.all_raw_data.loc[(deal_raw == 0).all(axis=1), ['DB卖1量', 'DS买1量']] = 1
        # 40档位价量数据nan处理
        if 'BASE卖1量' in all_cols and 'BASE买1量' in all_cols:
            # 价格nan填充, 使用上一个档位数据 +-0.001 进行填充
            for i in range(2, 11):
                if f'BASE买{i}价' not in all_cols or f'BASE买{i-1}价' not in all_cols:
                    continue

                # 买价
                self.all_raw_data.loc[:, f'BASE买{i}价'] = self.all_raw_data[f'BASE买{i}价'].fillna(self.all_raw_data[f'BASE买{i-1}价'] - 0.001)

                # 卖价
                self.all_raw_data.loc[:, f'BASE卖{i}价'] = self.all_raw_data[f'BASE卖{i}价'].fillna(self.all_raw_data[f'BASE卖{i-1}价'] + 0.001)

            # 量nan用0填充
            vol_cols = [i for i in list(self.all_raw_data) if i.startswith('BASE') and '价' not in i]
            self.all_raw_data[vol_cols] = self.all_raw_data[vol_cols].fillna(0)

        # 记录需要的索引，供后续转为numpy时使用
        # BASE买1价 / BASE卖1价
        for col in ['BASE买1价', 'BASE卖1价']:
            self.col_idx[col] = self.all_raw_data.columns.get_loc(col)
       
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
