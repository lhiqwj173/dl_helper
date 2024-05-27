import psutil
from .train_param import logger

def report_memory_usage():
    # 获取设备的总内存（以GB为单位）
    total_memory = psutil.virtual_memory().total / (1024 ** 3)

    pct = psutil.virtual_memory().percent
    logger.debug(f'memory usage: {pct}% of {total_memory:.2f}GB')