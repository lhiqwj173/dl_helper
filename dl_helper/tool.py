import psutil, pickle, torch, os
import numpy as np
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import imgkit

from py_ext.wechat import wx
from py_ext.tool import debug, log, get_log_file
from dl_helper.train_param import logger, match_num_processes
if match_num_processes() ==8:
    import torch_xla.core.xla_model as xm

import torch
from torchstat import stat
from torchinfo import summary
from torch.nn.utils import parameters_to_vector
import df2img
from py_ext.alist import alist

def upload_log_file():
    """上传日志文件到alist"""
    # 获取日志文件
    log_file = get_log_file()
    if log_file and os.path.exists(log_file):
        # 上传到alist
        client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
        # 上传文件夹
        upload_folder = '/rl_learning_process/logs/'
        client.mkdir(upload_folder)
        client.upload(log_file, upload_folder)
        log(f'Upload log file to alist: {log_file}')


def calc_sharpe_ratio(returns, risk_free_rate=0.0):
    """计算夏普比率
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率,默认0
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(len(returns))  # 根据序列长度标准化

def calc_sortino_ratio(returns, risk_free_rate=0.0):
    """计算索提诺比率
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率,默认0
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    excess_returns = returns - risk_free_rate
    # 只考虑负收益的标准差
    downside_returns = excess_returns[excess_returns < 0]
    
    # 如果没有下行收益率，说明所有收益都是正的
    # 这种情况下可以返回一个较大的正值，因为这是理想情况
    if len(downside_returns) == 0:
        return excess_returns.mean() * np.sqrt(len(returns))  # 使用平均超额收益乘以时间调整因子
        
    # 正常情况下的计算
    return excess_returns.mean() / np.sqrt(downside_returns.var() + 1e-8) * np.sqrt(len(returns))

def calc_max_drawdown(returns):
    """计算最大回撤
    Args:
        returns: 收益率序列
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative/running_max - 1
    return np.min(drawdown)

def calc_total_return(returns):
    """计算总收益率
    Args:
        returns: 收益率序列
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    total_return = (1 + returns).prod() - 1
    return total_return / len(returns)  # 根据序列长度标准化

def calc_volatility(returns):
    """计算波动率
    Args:
        returns: 收益率序列
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.values
    return returns.std() * np.sqrt(len(returns))  # 根据序列长度标准化

def adjust_class_weights_df(predict_df):
    # timestamp,target,0,1,2
    min_class_count = predict_df['target'].value_counts().min()
    
    sampled_indices = []
    for class_label in predict_df['target'].unique():
        indices = predict_df.index[predict_df['target'] == class_label]
        sampled_indices.extend(np.random.choice(indices, min_class_count, replace=False))
    
    adjusted_df = predict_df.loc[sampled_indices].reset_index(drop=True)
    return adjusted_df


def adjust_class_weights_numpy(y_true, y_pred):
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    min_class_count = np.min(class_counts)
    
    sampled_indices = []
    for class_label in unique_classes:
        indices = np.where(y_true == class_label)[0]
        sampled_indices.extend(np.random.choice(indices, min_class_count, replace=False))
    
    adjusted_y_true = y_true[sampled_indices]
    adjusted_y_pred = y_pred[sampled_indices]
    
    return adjusted_y_true, adjusted_y_pred

def adjust_class_weights_torch(y_true, y_pred):
    unique_classes, class_counts = torch.unique(y_true, return_counts=True)
    min_class_count = torch.min(class_counts)
    
    sampled_indices = []
    for class_label in unique_classes:
        indices = (y_true == class_label).nonzero().squeeze()
        sampled_indices.extend(torch.tensor(np.random.choice(indices, min_class_count, replace=False)), device=y_true.device)
    
    adjusted_y_true = y_true[sampled_indices]
    adjusted_y_pred = y_pred[sampled_indices]
    
    return adjusted_y_true, adjusted_y_pred
    

def hex_to_rgb(hex_color):
    # 将十六进制颜色转换为RGB
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def output_leaderboard_html(df, out_file):
    # # 直接从CSV文件读取数据
    # df = pd.read_csv(csv_path)
    
    # 定义颜色
    color_good = hex_to_rgb('#93ff93')
    color_bad = hex_to_rgb('#ff4444') 
    
    # 定义需要着色的数值列
    numeric_cols = ['score_test', 'score_val', 'pred_time_test', 'pred_time_val', 
                   'fit_time', 'pred_time_test_marginal', 'pred_time_val_marginal', 
                   'fit_time_marginal']
    
    html = '''
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #000000; padding: 8px; text-align: left; }
            th { background-color: #e1e1e1; }  /* 淡灰色背景 */
        </style>
    </head>
    <body>
    '''
    
    # 创建表格
    table = '<table>'
    
    # 添加表头
    table += '<tr>'
    for col in df.columns:
        table += f'<th>{col}</th>'
    table += '</tr>'
    
    # 添加数据行
    for _, row in df.iterrows():
        table += '<tr>'
        for col in df.columns:
            value = row[col]
            if col in numeric_cols:
                # 对于数值列，计算相对位置并设置颜色
                min_val = df[col].min()
                max_val = df[col].max()
                if col in ['score_test', 'score_val']:  # 这些指标越大越好
                    norm_val = (value - min_val) / (max_val - min_val)
                    # 在两个颜色之间进行插值
                    red = int(color_good[0] + (color_bad[0] - color_good[0]) * (1-norm_val))
                    green = int(color_good[1] + (color_bad[1] - color_good[1]) * (1-norm_val))
                    blue = int(color_good[2] + (color_bad[2] - color_good[2]) * (1-norm_val))
                else:  # 其他指标越小越好
                    norm_val = (value - min_val) / (max_val - min_val)
                    red = int(color_good[0] + (color_bad[0] - color_good[0]) * norm_val)
                    green = int(color_good[1] + (color_bad[1] - color_good[1]) * norm_val)
                    blue = int(color_good[2] + (color_bad[2] - color_good[2]) * norm_val)
                color = f'rgb({red}, {green}, {blue})'
                table += f'<td style="background-color: {color}">{value:.4f}</td>'
            else:
                table += f'<td>{value}</td>'
        table += '</tr>'
    
    table += '</table>'
    html += table + '</body></html>'
    
    # 保存为HTML
    with open(out_file, 'w') as f:
        f.write(html)
    
    # imgkit.from_file(out_png_file.replace('jpg', 'html'), out_png_file)

def save_df_pic(title, df, fig_size=(500, 140)):
    fig = df2img.plot_dataframe(df, row_fill_color=("#ffffff", "#f2f2f2"), fig_size=fig_size)
    df2img.save_dataframe(fig=fig, filename=title)

def cal_symbol_y_idx_thresholds(train_folder, y_len=3):
    """
    返回
    {
        y_idx:{
            code1: (-0.1666666666666483, 0.11666666666676484),
            code2: (-0.1666666666666483, 0.11666666666676484),
        }
    }
    """

    ys = {}
    thresholds = {}

    train_files = os.listdir(train_folder)
    _,_, _, y, _ = pickle.load(open(os.path.join(train_folder, train_files[0]), 'rb'))
    y_idxs = [i for i in range(len(y[0]))]

    for file in tqdm(train_files):
        file = os.path.join(train_folder, file)
        ids,_, _, y, _ = pickle.load(open(file, 'rb'))

        for y_idx in y_idxs:
            if y_idx not in ys:
                ys[y_idx] = {}
                thresholds[y_idx] = {}

            for i, _id in enumerate(ids):
                code, _ = _id.split('_')
                if code not in ys[y_idx]:
                    ys[y_idx][code] = []
                ys[y_idx][code].append(y[i][y_idx])

    for y_idx in ys:
        for code in ys[y_idx]:
            # 计算 33% 和 66% 分位数
            threshold = (np.percentile(ys[y_idx][code], 33), np.percentile(ys[y_idx][code], 66)) if y_len == 3 else (np.percentile(ys[y_idx][code], 50),) if y_len == 2 else []

            if y_len == 3:
                c_0 = len([i for i in ys[y_idx][code] if i <= threshold[0]])
                c_1 = len([i for i in ys[y_idx][code] if i > threshold[0] and i <= threshold[1]])
                c_2 = len([i for i in ys[y_idx][code] if i > threshold[1]])
                _max = max(c_0, c_1, c_2)
                _min = min(c_0, c_1, c_2)
            elif y_len == 2:
                c_0 = len([i for i in ys[y_idx][code] if i <= threshold[0]])
                c_1 = len([i for i in ys[y_idx][code] if i > threshold[0]])
                _max = max(c_0, c_1)
                _min = min(c_0, c_1)
            else:
                raise Exception('y_len 必须为 2 或 3')

            max_diff = _max - _min
            max_diff_pct = max_diff / len(ys[y_idx][code]) * 100
            if max_diff_pct > 3:
                continue

            thresholds[y_idx][code] = threshold
            print(code, y_idx, thresholds[y_idx][code], round(max_diff_pct, 3))

    # 删除值长度为0的键
    thresholds = {key: value for key, value in thresholds.items() if len(value) != 0}
    return thresholds

def cal_symbol_class_pct(train_folder, y_len=3, thresholds=[-0.5, 0.5], min_class_pct=0.1):
    res = {}
    ys = {}

    train_files = os.listdir(train_folder)
    _,_, _, y, _ = pickle.load(open(os.path.join(train_folder, train_files[0]), 'rb'))
    y_idxs = [i for i in range(len(y[0]))]

    for file in tqdm(train_files):
        file = os.path.join(train_folder, file)
        ids,_, _, y, _ = pickle.load(open(file, 'rb'))

        for y_idx in y_idxs:
            if y_idx not in ys:
                ys[y_idx] = {}
                res[y_idx] = []

            for i, _id in enumerate(ids):
                code, _ = _id.split('_')
                if code not in ys[y_idx]:
                    ys[y_idx][code] = []
                ys[y_idx][code].append(y[i][y_idx])

    for y_idx in ys:
        for code in ys[y_idx]:
            pct = []
            _y = ys[y_idx][code]
            _thresholds = [i for i in thresholds]
            # 添加一个无穷大
            _thresholds.append(float('inf'))
            for i in range(y_len):
                _y2 = [x for x in _y if x >= _thresholds[0]]
                num = len(_y) - len(_y2)
                pct.append(100*num/len(ys[y_idx][code]))
                _thresholds = _thresholds[1:]
                _y = _y2

            if min(pct) > min_class_pct * 100:
                print(f"{y_idx} {code} {[f'{i:.2f}%' for i in pct]}")
                res[y_idx].append(code)

    # 删除值长度为0的键
    res = {key: value for key, value in res.items() if len(value) != 0}
    return res

def model_params_num(model):
    # 将所有参数转换为一个向量
    vector = parameters_to_vector(model.parameters())
    # 获取参数数量
    num_params = vector.numel()
    return num_params

def _check_nan(output):
    has_nan = torch.isnan(output).any(dim=-1)  # 检查是否有 NaN
    has_inf = torch.isinf(output).any(dim=-1)  # 检查是否有 inf

    # 找出包含 NaN 或 inf 值的批次索引
    batch_indices = torch.nonzero(has_nan | has_inf, as_tuple=True)[0]
    return batch_indices

def check_nan(output, ids):
    # 找出包含 NaN 或 inf 值的批次索引
    batch_indices = _check_nan(output)
    if batch_indices.numel() > 0:
        bad_ids = []
        for i in list(batch_indices):
            bad_ids.append(ids[i])
        msg = f'训练数据异常 nan/inf ids:{bad_ids}'
        wx.send_message(msg)
        raise Exception(msg)

def stop_all_python_processes():
    current_pid = os.getpid()

    # 获取当前正在运行的所有进程
    all_processes = psutil.process_iter()

    # 遍历所有进程并停止 Python 进程
    for process in all_processes:
        try:
            process_info = process.as_dict(attrs=['pid', 'name'])
            pid = process_info['pid']
            name = process_info['name']

            # 如果进程是 Python 进程且不是当前进程，则终止该进程
            if name.lower() == 'python' and pid != current_pid:
                process.terminate()
                print(f"Terminated Python process: {pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 忽略无法访问或不存在的进程
            pass

def report_memory_usage(msg=''):
    memory_usage = psutil.virtual_memory()
    print(f"{msg} CPU 内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")
    # tpu_mem_info = xm.get_memory_info(xm.xla_device())
    # print(tpu_mem_info)
    # tpu_used = tpu_mem_info["kb_total"] - tpu_mem_info["kb_free"]
    # print(f"{msg} TPU 内存占用：{tpu_used/1024**3:.3f}GB/{tpu_mem_info['kb_total']/1024**3:.3f}GB")

    # # 获取当前进程ID
    # pid = psutil.Process().pid
    # # 获取当前进程对象
    # process = psutil.Process(pid)
    # # 获取当前进程占用的内存信息
    # memory_info = process.memory_info()
    # # 将字节大小转换为GB
    # memory_gb = memory_info.rss / (1024 ** 3)
    # # 打印内存大小
    # logger.debug(f"{msg} 占用的内存：{memory_gb:.3f}GB")


    # # 获取当前进程ID
    # current_pid = os.getpid()

    # # 获取当前进程对象
    # current_process = psutil.Process(current_pid)

    # # 获取当前进程占用的内存信息
    # memory_info = current_process.memory_info()

    # # 将字节大小转换为GB
    # memory_gb = memory_info.rss / (1024 ** 3)

    # # 打印当前进程的内存大小
    # # print(f"当前进程ID: {current_pid}, 当前进程占用的内存：{memory_gb:.3f}GB")

    # # 统计所有子进程的内存使用情况
    # total_memory_gb = memory_gb

    # # 获取所有进程ID
    # pids = psutil.pids()

    # for pid in pids:
    #     try:
    #         # 获取进程对象
    #         process = psutil.Process(pid)

    #         # 获取进程的父进程ID
    #         parent_pid = process.ppid()

    #         # 如果父进程ID与当前进程ID相同，则属于当前程序的子进程
    #         if parent_pid == current_pid:
    #             # 获取进程占用的内存信息
    #             memory_info = process.memory_info()

    #             # 将字节大小转换为GB
    #             memory_gb = memory_info.rss / (1024 ** 3)

    #             # 累加子进程的内存大小到总内存大小
    #             total_memory_gb += memory_gb
    #             # 打印子进程的内存大小
    #             # print(f"子进程ID: {pid}, 子进程占用的内存：{memory_gb:.3f}GB")
    #     except (psutil.NoSuchProcess, psutil.AccessDenied):
    #         # 跳过无法访问的进程
    #         continue
    
    # # # 打印合并统计后的内存大小
    # msg = '合并统计后的内存大小' if msg == '' else f'{msg}'
    # print(f"{msg}: {total_memory_gb:.3f}GB")