import psutil, pickle, torch
from py_ext.wechat import wx
from .train_param import logger

def check_nan(loss, **kwargs):
    if torch.isnan(loss).any().item():
        pickle.dump((loss, kwargs), open(f'train_data.pkl', 'wb'))
        wx.send_message(f'{params.train_title} 训练异常')
        raise Exception("error train data")

def report_memory_usage():
    # 获取当前进程ID
    pid = psutil.Process().pid

    # 获取当前进程对象
    process = psutil.Process(pid)

    # 获取当前进程占用的内存信息
    memory_info = process.memory_info()

    # 将字节大小转换为GB
    memory_gb = memory_info.rss / (1024 ** 3)

    # 打印内存大小
    logger.debug(f"占用的内存：{memory_gb:.3f}GB")