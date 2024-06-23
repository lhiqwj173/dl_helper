import psutil, pickle, torch, os
from py_ext.wechat import wx
from dl_helper.train_param import logger

def check_nan(data, **kwargs):
    if torch.isnan(data).any().item() or torch.isinf(data).any().item():
        pickle.dump((data, kwargs), open(f'train_data.pkl', 'wb'))
        wx.send_message(f'训练异常')
        raise Exception("error train data")

def report_memory_usage(msg=''):
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

    # 获取当前进程ID
    current_pid = os.getpid()

    # 获取当前进程对象
    current_process = psutil.Process(current_pid)

    # 获取当前进程占用的内存信息
    memory_info = current_process.memory_info()

    # 将字节大小转换为GB
    memory_gb = memory_info.rss / (1024 ** 3)

    # 打印当前进程的内存大小
    # print(f"当前进程ID: {current_pid}, 当前进程占用的内存：{memory_gb:.3f}GB")

    # 统计所有子进程的内存使用情况
    total_memory_gb = memory_gb

    # 获取所有进程ID
    pids = psutil.pids()

    for pid in pids:
        try:
            # 获取进程对象
            process = psutil.Process(pid)

            # 获取进程的父进程ID
            parent_pid = process.ppid()

            # 如果父进程ID与当前进程ID相同，则属于当前程序的子进程
            if parent_pid == current_pid:
                # 获取进程占用的内存信息
                memory_info = process.memory_info()

                # 将字节大小转换为GB
                memory_gb = memory_info.rss / (1024 ** 3)

                # 累加子进程的内存大小到总内存大小
                total_memory_gb += memory_gb
                # 打印子进程的内存大小
                # print(f"子进程ID: {pid}, 子进程占用的内存：{memory_gb:.3f}GB")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # 跳过无法访问的进程
            continue
    
    # # 打印合并统计后的内存大小
    # logger.debug(f"合并统计后的内存大小：{total_memory_gb:.3f}GB")