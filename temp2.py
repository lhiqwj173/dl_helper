import asyncio
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号'-'显示为方块的问题

class RealtimePlot:
    def __init__(self, code1, code2, open_grid=[1.8, 2.3, 2.8, -2.8, -2.3, -1.8], close_grid=[0.5, 1.0, 1.5, -1.5, -1.0, -0.5]):
        # 初始化数据列表
        self.times = []
        self.code1_prices = []
        self.code2_prices = []
        # 用于标准化
        self.code1_1st_p = 0.0
        self.code2_1st_p = 0.0
        self.zscore = []

        # 子图横线
        self.red_lines = open_grid
        self.green_lines = close_grid

        self.update_time = None
        self.code1 = code1
        self.code2 = code2

        # 创建图形和子图
        plt.ion()  # 启用交互模式
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, height_ratios=[2, 1], figsize=(12, 8))
        
        # 创建价格线和价差线
        self.line_code1, = self.ax1.plot([], [], label=f'{self.code1} net', color='blue')
        self.line_code2, = self.ax1.plot([], [], label=f'{self.code2} net', color='red')
        self.line_zscore, = self.ax2.plot([], [], label='Zscore', color='green')

        # 添加参考线
        for value in self.red_lines:
            self.ax2.axhline(y=value, color='red', linestyle='-', alpha=0.3)
        for value in self.green_lines:
            self.ax2.axhline(y=value, color='green', linestyle='-', alpha=0.3)

        # 创建最新值标注
        self.latest_value_text = self.ax2.text(0, 0, '', color='black', 
                                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # 设置图表属性
        self.ax1.set_title('实时价格监控')
        self.ax1.set_ylabel('价格')
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.set_title('价差')
        self.ax2.set_xlabel('时间')
        self.ax2.set_ylabel('价差')
        self.ax2.grid(True)
        self.ax2.legend()
        
        # 调整子图间距
        plt.tight_layout()

    async def update_data(self, data_list):
        """
        异步更新数据
        data_list:
            ['date',
            'mid_price_1',
            'mid_price_2',
            'spread',
            'z_score',
            'cur_match_state',
            'dst_match_state',
            'dst_pos_code1_0',
            'dst_pos_code2_0',
            'free_cash',
            'dst_pos_code1_1',
            'dst_pos_code2_1']
        """
        # 更新数据列表
        self.times.append(data_list[0])
        if self.code1_1st_p == 0.0:
            self.code1_1st_p = data_list[1]
            self.code2_1st_p = data_list[2]
        self.code1_prices.append(data_list[1]/self.code1_1st_p)
        self.code2_prices.append(data_list[2]/self.code2_1st_p)
        self.zscore.append(data_list[4])
        
        # 更新图表
        await self.update_plot()
    
    async def update_plot(self):
        """异步更新图表"""
        timestamps = [t.timestamp() for t in self.times]
        
        # 更新价格图
        self.line_code1.set_xdata(timestamps)
        self.line_code1.set_ydata(self.code1_prices)
        self.line_code2.set_xdata(timestamps)
        self.line_code2.set_ydata(self.code2_prices)
        
        # 更新价差图
        self.line_zscore.set_xdata(timestamps)
        self.line_zscore.set_ydata(self.zscore)
        
        # 更新最新值标注
        if len(self.zscore) > 0:
            latest_zscore = self.zscore[-1]
            latest_time = timestamps[-1]
            self.latest_value_text.set_position((latest_time, latest_zscore))
            self.latest_value_text.set_text(f'{latest_zscore:.2f}')

        # 调整坐标轴范围
        if len(self.times) > 0:
            # 价格图坐标轴
            self.ax1.set_xlim(min(timestamps), max(timestamps))
            price_min = min(min(self.code1_prices), min(self.code2_prices))
            price_max = max(max(self.code1_prices), max(self.code2_prices))
            margin = (price_max - price_min) * 0.1
            self.ax1.set_ylim(price_min - margin, price_max + margin)
            
            # 价差图坐标轴
            self.ax2.set_xlim(min(timestamps), max(timestamps))
            zscore_min, zscore_max = min(self.zscore), max(self.zscore)
            margin = (zscore_max - zscore_min) * 0.1
            self.ax2.set_ylim(zscore_min - margin, zscore_max + margin)
        
        # 刷新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

async def main():
    # 创建实时绘图对象
    plotter = RealtimePlot()
    
    for i in range(10):
        await plotter.update_data()

if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())