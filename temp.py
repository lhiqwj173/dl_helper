import multiprocessing
import time
import os
import mmap
import ctypes
import platform
if platform.system() == 'Windows':
    from ctypes import windll, wintypes

class Event:
    """跨平台的Event实现，使用底层系统调用"""
    
    def __init__(self, name):
        self.name = name
        self.system = platform.system()
        
        if self.system == 'Windows':
            # Windows下使用Event对象
            self.handle = windll.kernel32.CreateEventA(
                None,   # 默认安全属性
                True,   # 手动重置
                False,  # 初始状态为未触发
                name.encode()  # 事件名称
            )
            if not self.handle:
                raise RuntimeError("Failed to create event")
                
        else:  # Linux/Unix
            # 使用命名信号量
            self.sem = multiprocessing.get_context('fork').Semaphore(0)
            # 使用共享内存来存储状态
            self.shm = mmap.mmap(-1, 1)
            self.shm.write(b'\x00')
            
    def set(self):
        if self.system == 'Windows':
            windll.kernel32.SetEvent(self.handle)
        else:
            # 设置状态并释放所有等待的进程
            self.shm.seek(0)
            self.shm.write(b'\x01')
            # 释放所有等待的进程
            try:
                while True:
                    self.sem.release()
            except ValueError:  # 没有进程在等待时会抛出异常
                pass
                
    def clear(self):
        if self.system == 'Windows':
            windll.kernel32.ResetEvent(self.handle)
        else:
            self.shm.seek(0)
            self.shm.write(b'\x00')
            
    def wait(self, timeout=None):
        if self.system == 'Windows':
            # Windows下使用WaitForSingleObject
            timeout_ms = int(timeout * 1000) if timeout is not None else 0xFFFFFFFF
            result = windll.kernel32.WaitForSingleObject(self.handle, timeout_ms)
            return result == 0  # 返回是否成功等待
        else:
            # Linux下使用信号量等待
            return self.sem.acquire(timeout=timeout)
            
    def __del__(self):
        if self.system == 'Windows':
            if hasattr(self, 'handle'):
                windll.kernel32.CloseHandle(self.handle)
        else:
            if hasattr(self, 'shm'):
                self.shm.close()

def worker(num, worker_num):
    """进程工作函数"""
    event = Event(name='test_event')
    if num != 0:
        print(f"进程 {num} 等待事件")
        event.wait()
        print(f"进程 {num} 被触发")
    else:
        time.sleep(3)  # 让其他进程有时间启动并等待
        print(f"进程 {num} 触发事件")
        event.set()

def main():
    processes = []
    worker_num = 4
    
    event = Event(name='test_event')
    event.clear()
    
    print(f"创建 {worker_num + 1} 个进程")
    for i in range(worker_num + 1):
        p = multiprocessing.Process(target=worker, args=(i, worker_num))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("所有进程执行完毕")

if __name__ == '__main__':
    main()