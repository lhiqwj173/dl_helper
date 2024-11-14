import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import plotly.graph_objects as go

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, HoverTool


class StockPriceSimulator:
    def __init__(self, n_steps, n_sims):
        """
        初始化参数
        S0: 初始股票价格
        mu: 预期年化收益率
        sigma: 波动率(标准差)
        n_steps: 时间步数
        n_sims: 模拟路径数量
        """
        self.S0 = 1
        self.T = n_steps
        self.dt = 1
        self.n_sims = n_sims
        self.n_steps = int(n_steps)
        
    def generate_paths(self):
        """生成蒙特卡罗模拟路径"""
        # 创建时间序列
        self.times = np.linspace(0, self.T, self.n_steps + 1)
        
        # 初始化股票价格数组
        S = np.zeros((self.n_steps + 1, self.n_sims))
        S[0] = self.S0

        # 生成随机正态分布数
        Z = np.random.normal(0, 1, (self.n_steps, self.n_sims))

        # 使用几何布朗运动公式模拟价格路径
        for t in range(1, self.n_steps + 1):
            # 计算漂移和扩散项
            mu = np.random.uniform(-0.01, 0.01)
            sigma = np.random.uniform(0.01, 0.05)

            drift = (mu - 0.5 * sigma**2) * self.dt
            diffusion = sigma * np.sqrt(self.dt) * Z[t-1]

            # 更新股票价格
            S[t] = S[t-1] * np.exp(drift + diffusion)
        
        return S
    
    def plot_paths(self, S, n_paths_to_plot=0, t_idx=0, predict_t=100):
        """绘制模拟路径"""
        if n_paths_to_plot == 0:
            n_paths_to_plot = self.n_sims

        plt.figure(figsize=(12, 6))
        plt.axvline(x=t_idx, color='r', linestyle='--')
        plt.axvline(x=predict_t, color='g', linestyle='--')
        plt.plot(self.times, S[:, :n_paths_to_plot])
        plt.xlabel('time')
        plt.ylabel('price')
        plt.grid(True)

        plt.show()
        
    def plot_paths_interactive(self, S, n_paths_to_plot=0, t_idx=0, predict_t=100):
        """
        绘制交互式模拟路径
        鼠标悬停时会高亮当前路径，其他路径变淡
        """
        if n_paths_to_plot == 0:
            n_paths_to_plot = self.n_sims

        # 创建图形
        fig = go.Figure()

        # 添加每条路径
        for i in range(n_paths_to_plot):
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=S[:, i],
                    mode='lines',
                    name=f'Path {i+1}',
                    line=dict(width=1),
                    hoverinfo='y+name',
                    hoveron='points+fills',  # 路径上任意点都可触发悬停效果
                )
            )

        # 添加垂直参考线
        fig.add_vline(
            x=t_idx,
            line_dash="dash",
            line_color="red",
            annotation_text="Start",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=predict_t,
            line_dash="dash",
            line_color="green",
            annotation_text="Predict",
            annotation_position="top"
        )

        # 更新布局
        fig.update_layout(
            title='Stock Price Simulation Paths',
            xaxis_title='Time',
            yaxis_title='Price',
            hovermode='closest',
            showlegend=True,
            plot_bgcolor='white',  # 白色背景
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
            ),
            # 添加交互效果
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Reset View",
                            method="relayout",
                            args=[{"xaxis.range": [min(self.times), max(self.times)],
                                "yaxis.range": [S.min(), S.max()]}]
                        )
                    ]
                )
            ]
        )

        # 设置悬停时的样式变化
        fig.update_traces(
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
            ),
        )

        # 添加自定义交互效果
        fig.for_each_trace(
            lambda trace: trace.update(
                line=dict(width=1),  # 默认线宽
            )
        )

        # 更新配置
        fig.update_layout(
            hoverdistance=100,  # 增加悬停检测距离
            spikedistance=100,  # 增加标记线检测距离
        )

        # 显示图形
        fig.show(config={
            'scrollZoom': True,  # 启用滚轮缩放
            'modeBarButtonsToAdd': ['drawline', 'eraseshape'],  # 添加画线和擦除工具
            'displayModeBar': True,  # 显示工具栏
        })

    def plot_paths_interactive2(self, S, n_paths_to_plot=0, t_idx=0, predict_t=100):
        """
        绘制交互式模拟路径
        鼠标悬停时会高亮当前路径，其他路径变淡
        """
        if n_paths_to_plot == 0:
            n_paths_to_plot = self.n_sims

        fig = go.Figure()

        # 为每条路径创建两个轨迹：一个用于正常显示，一个用于高亮显示
        for i in range(n_paths_to_plot):
            # 添加正常显示的轨迹（初始可见）
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=S[:, i],
                    mode='lines',
                    name=f'Path {i+1}',
                    line=dict(
                        color='rgba(100, 100, 100, 0.2)',  # 淡灰色
                        width=1
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                    customdata=[i]*len(self.times)  # 存储路径索引
                )
            )
            
            # 添加高亮显示的轨迹（初始隐藏）
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=S[:, i],
                    mode='lines',
                    name=f'Path {i+1}',
                    line=dict(
                        color='rgba(31, 119, 180, 1)',  # 蓝色
                        width=2
                    ),
                    hoverinfo='y+name',
                    visible=False,
                    showlegend=False,
                    customdata=[i]*len(self.times)  # 存储路径索引
                )
            )

        # 添加垂直参考线
        fig.add_vline(
            x=t_idx,
            line_dash="dash",
            line_color="red",
            annotation_text="Start",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=predict_t,
            line_dash="dash",
            line_color="green",
            annotation_text="Predict",
            annotation_position="top"
        )

        # 更新布局
        fig.update_layout(
            title='Stock Price Simulation Paths',
            xaxis_title='Time',
            yaxis_title='Price',
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
            ),
        )

        # 添加交互回调
        fig.update_layout(
            hoverdistance=100,
        )

        # 添加自定义JavaScript回调以处理悬停事件
        fig.update_layout(
            updatemenus=[],  # 移除之前的按钮
            hovermode='closest'
        )

        # 添加交互回调
        fig.update_layout(
            hoverdistance=100,
            spikedistance=100,
        )

        # 添加JavaScript回调来处理hover事件
        fig.update_layout(
            newshape=dict(line_color='blue'),
            # JavaScript回调
            updatemenus=[],
            clickmode='event+select'
        )

        # 添加自定义JavaScript以实现hover效果
        js_code = """
        var pathCount = %d;
        var traces = document.querySelectorAll('.scatter');
        var previousPath = null;

        traces.forEach(function(trace) {
            trace.on('mouseover', function(data) {
                var pathIndex = Math.floor(data.points[0].curveNumber / 2);
                
                // 重置所有路径为淡色
                for(var i = 0; i < pathCount; i++) {
                    var normalTrace = document.querySelector('.scatter:nth-child(' + (2*i + 1) + ')');
                    var highlightTrace = document.querySelector('.scatter:nth-child(' + (2*i + 2) + ')');
                    
                    if(normalTrace) normalTrace.style.opacity = '0.2';
                    if(highlightTrace) highlightTrace.style.display = 'none';
                }
                
                // 高亮当前路径
                var currentNormalTrace = document.querySelector('.scatter:nth-child(' + (2*pathIndex + 1) + ')');
                var currentHighlightTrace = document.querySelector('.scatter:nth-child(' + (2*pathIndex + 2) + ')');
                
                if(currentNormalTrace) currentNormalTrace.style.opacity = '1';
                if(currentHighlightTrace) currentHighlightTrace.style.display = 'block';
                
                previousPath = pathIndex;
            });
        });

        document.addEventListener('mouseout', function(e) {
            // 当鼠标离开图表区域时重置所有路径
            for(var i = 0; i < pathCount; i++) {
                var normalTrace = document.querySelector('.scatter:nth-child(' + (2*i + 1) + ')');
                var highlightTrace = document.querySelector('.scatter:nth-child(' + (2*i + 2) + ')');
                
                if(normalTrace) normalTrace.style.opacity = '1';
                if(highlightTrace) highlightTrace.style.display = 'none';
            }
            previousPath = null;
        });
        """ % n_paths_to_plot

        # 将JavaScript代码添加到图形中
        fig.add_annotation(
            dict(
                text=f'<script>{js_code}</script>',
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=1),
                opacity=0
            )
        )

        # 显示图形
        fig.show(config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
        })

    def plot_paths_interactive3(self, S, n_paths_to_plot=0, t_idx=0, predict_t=100):
        """
        绘制交互式模拟路径
        使用plotly的内置功能实现鼠标悬停高亮效果
        """
        if n_paths_to_plot == 0:
            n_paths_to_plot = self.n_sims

        fig = go.Figure()

        # 添加每条路径
        for i in range(n_paths_to_plot):
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=S[:, i],
                    mode='lines',
                    name=f'Path {i+1}',
                    line=dict(
                        width=1,
                        color='rgba(100, 100, 100, 0.3)'  # 默认淡灰色
                    ),
                    hoverinfo='y+name',
                    hoveron='points+fills',
                    hovertemplate='Path %{customdata}<br>Price: %{y:.2f}<extra></extra>',
                    customdata=[i+1]*len(self.times)
                )
            )

        # 添加垂直参考线
        fig.add_vline(
            x=t_idx,
            line_dash="dash",
            line_color="red",
            annotation_text="Start",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=predict_t,
            line_dash="dash",
            line_color="green",
            annotation_text="Predict",
            annotation_position="top"
        )

        # 更新布局
        fig.update_layout(
            title='Stock Price Simulation Paths',
            xaxis_title='Time',
            yaxis_title='Price',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
            ),
            hovermode='closest'
        )

        # 设置hover时的效果
        for trace in fig.data:
            trace.update(
                # 高亮效果
                line=dict(
                    width=1,
                ),
            )

        # 设置悬停时的样式
        fig.for_each_trace(
            lambda trace: trace.update(
                hovertemplate='<b>Path %{customdata}</b><br>Price: %{y:.2f}<extra></extra>',
                line=dict(width=1),
            )
        )

        # 添加交互效果
        fig.update_layout(
            template='plotly_white',
            modebar={'orientation': 'v'},
            # 增加悬停效果
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Arial'
            ),
            # 添加高亮效果
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Reset',
                    method='relayout',
                    args=[{'shapes': []}]
                )]
            )]
        )

        # 添加交互回调
        fig.update_layout(
            margin=dict(t=100, b=50),
            newshape=dict(line_color='blue'),
            annotations=[
                dict(
                    text="Hover over paths to highlight",
                    xref="paper", yref="paper",
                    x=0, y=1.06,
                    showarrow=False
                )
            ]
        )

        # 使用plotly的新特性来实现hover效果
        fig.update_traces(
            selector=dict(mode='lines'),
            line=dict(width=1),
            # hover时的效果
            hoveron='points+fills',
            hoverinfo='y+name',
        )

        # 显示图形
        fig.show(config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
        })

    def plot_paths_interactive4(self, S, n_paths_to_plot=0, t_idx=0, predict_t=100):
        """
        使用 Bokeh 绘制交互式模拟路径
        """
        if n_paths_to_plot == 0:
            n_paths_to_plot = self.n_sims

        # 创建数据源
        source = ColumnDataSource({
            'xs': [self.times for _ in range(n_paths_to_plot)],
            'ys': [S[:, i] for i in range(n_paths_to_plot)],
            'path_id': [f'Path {i+1}' for i in range(n_paths_to_plot)]
        })

        # 创建图表
        p = figure(
            width=800, 
            height=400,
            title='Stock Price Simulation Paths',
            x_axis_label='Time',
            y_axis_label='Price',
            background_fill_color='white'
        )

        # 添加所有路径
        line_renderer = p.multi_line(
            xs='xs', 
            ys='ys', 
            source=source,
            line_width=1.5,
            line_alpha=0.6,
            line_color='navy',
            hover_line_color='orange',
            hover_line_width=3
        )

        # 添加垂直参考线
        p.line([t_idx, t_idx], [S.min(), S.max()], 
            line_color='red', line_dash='dashed', legend_label='Start')
        p.line([predict_t, predict_t], [S.min(), S.max()], 
            line_color='green', line_dash='dashed', legend_label='Predict')

        # 添加悬停工具
        hover = HoverTool(
            tooltips=[
                ('Path', '@path_id'),
                ('Price', '@ys')
            ],
            renderers=[line_renderer],
            mode='mouse'
        )
        p.add_tools(hover)

        # 添加鼠标悬停效果
        callback = CustomJS(args=dict(source=source), code="""
            const indices = cb_data.index.indices;
            if (indices.length > 0) {
                const colors = Array(source.data.xs.length).fill('rgba(0,0,128,0.2)');
                colors[indices[0]] = 'orange';
                const widths = Array(source.data.xs.length).fill(1.5);
                widths[indices[0]] = 3;
                source.data['line_color'] = colors;
                source.data['line_width'] = widths;
                source.change.emit();
            }
        """)

        # 添加鼠标离开效果
        reset_callback = CustomJS(args=dict(source=source), code="""
            const colors = Array(source.data.xs.length).fill('navy');
            const widths = Array(source.data.xs.length).fill(1.5);
            source.data['line_color'] = colors;
            source.data['line_width'] = widths;
            source.change.emit();
        """)

        # 设置图表样式
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.1
        p.axis.axis_line_color = 'gray'
        p.axis.axis_line_alpha = 0.5
        p.title.text_font_size = '16pt'
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        # 添加交互事件
        p.js_on_event('mousemove', callback)
        p.js_on_event('mouseleave', reset_callback)

        # 显示图表
        show(p)

    def plot_paths_interactive5(self, S, n_paths_to_plot=0, t_idx=0, predict_t=100):
        """
        使用 Bokeh 绘制交互式模拟路径
        """
        if n_paths_to_plot == 0:
            n_paths_to_plot = self.n_sims

        # 创建数据源
        source = ColumnDataSource({
            'xs': [self.times for _ in range(n_paths_to_plot)],
            'ys': [S[:, i] for i in range(n_paths_to_plot)],
            'path_id': [f'Path {i+1}' for i in range(n_paths_to_plot)]
        })

        # 创建图表 - 增加尺寸
        p = figure(
            width=1600,  # 增加宽度
            height=800,  # 增加高度
            title='Stock Price Simulation Paths',
            x_axis_label='Time',
            y_axis_label='Price',
            background_fill_color='white',
            sizing_mode="stretch_width"  # 自适应宽度
        )

        # 添加所有路径 - 修改默认颜色和透明度
        line_renderer = p.multi_line(
            xs='xs', 
            ys='ys', 
            source=source,
            line_width=1.5,
            line_alpha=0.3,  # 降低默认透明度
            line_color='gray',  # 改为灰色
            hover_line_color='#FF6B6B',  # 改为醒目的珊瑚红色
            hover_line_width=3.5
        )

        # 添加垂直参考线
        p.line([t_idx, t_idx], [S.min(), S.max()], 
            line_color='red', line_dash='dashed', legend_label='Start',
            line_width=2)  # 增加线宽
        p.line([predict_t, predict_t], [S.min(), S.max()], 
            line_color='green', line_dash='dashed', legend_label='Predict',
            line_width=2)  # 增加线宽

        # 添加悬停工具
        hover = HoverTool(
            tooltips=[
                ('Path', '@path_id'),
                ('Time', '$x{0.0}'),
                ('Price', '@ys{0,0.00}')  # 增加格式化
            ],
            renderers=[line_renderer],
            mode='mouse'
        )
        p.add_tools(hover)

        # 添加鼠标悬停效果
        callback = CustomJS(args=dict(source=source), code="""
            const indices = cb_data.index.indices;
            if (indices.length > 0) {
                const colors = Array(source.data.xs.length).fill('rgba(128,128,128,0.15)');  // 更淡的灰色
                colors[indices[0]] = '#FF6B6B';  // 珊瑚红色
                const widths = Array(source.data.xs.length).fill(1.5);
                widths[indices[0]] = 3.5;
                source.data['line_color'] = colors;
                source.data['line_width'] = widths;
                source.change.emit();
            }
        """)

        # 添加鼠标离开效果
        reset_callback = CustomJS(args=dict(source=source), code="""
            const colors = Array(source.data.xs.length).fill('gray');
            const widths = Array(source.data.xs.length).fill(1.5);
            source.data['line_color'] = colors;
            source.data['line_width'] = widths;
            source.change.emit();
        """)

        # 设置图表样式
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.1
        p.axis.axis_line_color = 'gray'
        p.axis.axis_line_alpha = 0.5
        p.title.text_font_size = '20pt'  # 增大标题字号
        p.xaxis.axis_label_text_font_size = '14pt'  # 增大轴标签字号
        p.yaxis.axis_label_text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '12pt'  # 增大刻度标签字号
        p.yaxis.major_label_text_font_size = '12pt'
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = '12pt'  # 增大图例字号

        # 添加交互事件
        p.js_on_event('mousemove', callback)
        p.js_on_event('mouseleave', reset_callback)

        # 设置边距
        p.min_border_left = 80
        p.min_border_right = 80
        p.min_border_top = 60
        p.min_border_bottom = 60

        # 显示图表
        show(p)

# 使用示例
if __name__ == "__main__":
    # 设置参数
    n_steps= 220   # 时间步数
    n_sims = 100 # 模拟路径数量
    
    # 创建模拟器实例
    simulator = StockPriceSimulator(n_steps, n_sims)
    
    # 生成路径
    paths = simulator.generate_paths()
    
    # 绘制部分路径
    # simulator.plot_paths(paths)
    # simulator.plot_paths_interactive(paths)
    simulator.plot_paths_interactive5(paths)
