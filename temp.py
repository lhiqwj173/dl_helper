from loguru import logger
import multiprocessing
from datetime import datetime

def worker(num):
    logger.info(f"Worker {num} started")
    for i in range(10):
        logger.info(f"Worker {num}: {i}")

if __name__ == '__main__':
    # 使用时间戳创建日志文件名
    log_filename = fr"C:\Users\lh\Desktop\temp\multiprocess.log"
    # 配置日志文件，enqueue=True 确保多进程写入安全
    logger.add(log_filename, rotation="5 MB", enqueue=True)

    with multiprocessing.Pool() as pool:
        pool.map(worker, range(5))