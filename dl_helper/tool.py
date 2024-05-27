import psutil
from .train_param import logger

def report_memory_usage():
    # # 获取设备的总内存（以GB为单位）
    # total_memory = psutil.virtual_memory().total / (1024 ** 3)

    # pct = psutil.virtual_memory().percent
    # logger.debug(f'memory usage: {pct}% of {total_memory:.2f}GB')

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