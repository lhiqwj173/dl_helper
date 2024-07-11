import torch

from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.binctabl import transform

"""
batch size: 128
[430][279] val done CPU 内存占用：10.4% (32.560GB/334.562GB)
[440][279] val done CPU 内存占用：10.4% (32.535GB/334.562GB)
[450][279] val done CPU 内存占用：10.4% (32.671GB/334.562GB)
[460][279] val done CPU 内存占用：10.4% (32.611GB/334.562GB)
[470][279] val done CPU 内存占用：10.4% (32.653GB/334.562GB)
[480][279] val done CPU 内存占用：10.4% (32.636GB/334.562GB)
[490][279] val done CPU 内存占用：10.4% (32.649GB/334.562GB)

batch size: 32
19475.3s	431	[130][1116] val done CPU 内存占用：6.4% (19.297GB/334.562GB)
20966.6s	432	[140][1116] val done CPU 内存占用：6.5% (19.397GB/334.562GB)
22468.5s	433	[150][1116] val done CPU 内存占用：6.6% (19.912GB/334.562GB)
23964.8s	434	[160][1116] val done CPU 内存占用：6.6% (19.989GB/334.562GB)
25457.4s	435	[170][1116] val done CPU 内存占用：6.7% (19.993GB/334.562GB)
26956.5s	436	[180][1116] val done CPU 内存占用：6.7% (20.052GB/334.562GB)
28452.3s	437	[190][1116] val done CPU 内存占用：6.7% (20.081GB/334.562GB)
29948.7s	438	[200][1116] val done CPU 内存占用：6.7% (20.088GB/334.562GB)
31445.4s	439	[210][1116] val done CPU 内存占用：6.7% (20.109GB/334.562GB)

batch size: 16
438.8s	426	[0][2233] val done CPU 内存占用：4.6% (13.178GB/334.562GB)
3251.6s	427	[10][2233] val done CPU 内存占用：5.3% (15.572GB/334.562GB)
6132.5s	428	[20][2233] val done CPU 内存占用：5.5% (16.173GB/334.562GB)
9038.4s	429	[30][2233] val done CPU 内存占用：5.6% (16.354GB/334.562GB)
11952.4s	430	[40][2233] val done CPU 内存占用：5.6% (16.451GB/334.562GB)
14858.9s	431	[50][2233] val done CPU 内存占用：5.6% (16.515GB/334.562GB)
17768.0s	432	[60][2233] val done CPU 内存占用：5.6% (16.613GB/334.562GB)
20681.1s	433	[70][2233] val done CPU 内存占用：5.6% (16.618GB/334.562GB)
23593.8s	434	[80][2233] val done CPU 内存占用：5.7% (16.716GB/334.562GB)
26498.7s	435	[90][2233] val done CPU 内存占用：5.7% (16.751GB/334.562GB)
29413.0s	436	[100][2233] val done CPU 内存占用：5.7% (16.847GB/334.562GB)
32313.1s	437	[110][2233] val done CPU 内存占用：5.7% (16.842GB/334.562GB)

batch size: 8
712.8s	428	[0][4466] val done CPU 内存占用：4.5% (12.670GB/334.562GB)
6302.0s	429	[10][4466] val done CPU 内存占用：5.1% (14.796GB/334.562GB)
12021.7s	430	[20][4466] val done CPU 内存占用：5.2% (15.012GB/334.562GB)
17760.0s	431	[30][4466] val done CPU 内存占用：5.2% (15.141GB/334.562GB)
23497.5s	432	[40][4466] val done CPU 内存占用：5.3% (15.317GB/334.562GB)
29236.7s	433	[50][4466] val done CPU 内存占用：5.3% (15.452GB/334.562GB)

batch size: 64
23324.1s	453	[300][558] val done CPU 内存占用：9.3% (28.830GB/334.562GB)
24104.6s	454	[310][558] val done CPU 内存占用：9.3% (28.902GB/334.562GB)
24881.2s	455	[320][558] val done CPU 内存占用：9.3% (28.937GB/334.562GB)
25663.5s	456	[330][558] val done CPU 内存占用：9.3% (28.934GB/334.562GB)
26439.9s	457	[340][558] val done CPU 内存占用：9.3% (28.998GB/334.562GB)
27215.5s	458	[350][558] val done CPU 内存占用：9.3% (28.985GB/334.562GB)
27992.0s	459	[360][558] val done CPU 内存占用：9.3% (29.018GB/334.562GB)
28768.7s	460	[370][558] val done CPU 内存占用：9.4% (29.057GB/334.562GB)
29548.8s	461	[380][558] val done CPU 内存占用：9.4% (29.084GB/334.562GB)
30326.1s	462	[390][558] val done CPU 内存占用：9.4% (29.143GB/334.562GB)
31106.2s	463	[400][558] val done CPU 内存占用：9.4% (29.159GB/334.562GB)
31883.0s	464	[410][558] val done CPU 内存占用：9.4% (29.207GB/334.562GB)

"""

def yfunc_target_long_short(x):
    # long/ short
    x1, x2 = x
    if x1 > 0:# 多头盈利
        return 0
    elif x2 > 0:# 空头盈利
        return 1
    else:
        return 2

def yfunc_target_simple(x):
    if x > 0:
        return 0
    elif x < 0:
        return 1
    else:
        return 2

class test(test_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        vars = []
        for classify_idx, name, yfunc in zip(
            [[0, 1], 2, 3], 
            ['10_target_long_short', '10_target_mid', '10_target_5_period_point'],
            [yfunc_target_long_short, yfunc_target_simple, yfunc_target_simple]
        ):
            vars.append((classify_idx, name, yfunc))

        classify_idx, targrt_name, yfunc = vars[2]
        self.y_n = 3

        batch_n = 16 * 8
        title = f'binctabl_{targrt_name}_v{self.idx}'
        data_parm = {
            'predict_n': [10, 20, 30],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (2, 2, 2),
            'total_hours': int(2*3),
            'symbols': '@'.join(symbols),
            'target': targrt_name,
            'std_mode': '5d'  # 4h/1d/5d
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.00013*batch_n, batch_size=64*batch_n, epochs=10,

            # 数据增强
            random_scale=0.05, random_mask_row=0.7,

            # 每4个样本取一个数据
            down_freq=8,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=self.data_folder,

            describe=f'target={targrt_name}',
            amp=self.amp
        )

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        return m_bin_ctabl(60, 40, 100, 40, 120, 10, self.y_n, 1)

    def get_transform(self, device):
        return transform(device, self.para, 105)