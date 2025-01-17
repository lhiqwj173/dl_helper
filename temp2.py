import multiprocessing
import time
import os
import platform

if platform.system() == 'Linux':
    import posix_ipc
    import fcntl
    import struct
else:
    import ctypes
    from ctypes import windll, wintypes

class Event:
    def __init__(self, name):
        self.name = name
        self.system = platform.system()

        if self.system == 'Windows':
            # Windows-specific initialization using Win32 API
            self.handle = windll.kernel32.CreateEventA(
                None,   # default security attributes
                True,   # manual reset
                False,  # initial state is not signaled
                name.encode()  # event name
            )
            if not self.handle:
                raise RuntimeError(f"Failed to create event {name}")
        elif self.system in ('Linux', 'Darwin'):  # Darwin for macOS, although not explicitly handled
            # Initialize for POSIX systems (Linux, macOS)
            try:
                self.sem = posix_ipc.Semaphore(f"/{name}")
            except posix_ipc.ExistentialError:
                self.sem = posix_ipc.Semaphore(f"/{name}", flags=posix_ipc.O_CREAT, initial_value=0)
        else:
            raise RuntimeError(f"Unsupported operating system: {self.system}")

    def set(self):
        if self.system == 'Windows':
            windll.kernel32.SetEvent(self.handle)
        else:
            # Release the semaphore for all waiting processes
            for _ in range(10):  # Assuming no more than 10 processes will wait; adjust if needed
                self.sem.release()

    def clear(self):
        if self.system == 'Windows':
            windll.kernel32.ResetEvent(self.handle)
        else:
            # Reset the event by decrementing the semaphore count back to zero
            while self.sem.value > 0:
                self.sem.acquire()

    def wait(self, timeout=None):
        if self.system == 'Windows':
            timeout_ms = int(timeout * 1000) if timeout is not None else 0xFFFFFFFF
            result = windll.kernel32.WaitForSingleObject(self.handle, timeout_ms)
            return result == 0  # Return whether the wait was successful
        else:
            # For POSIX, timeout functionality isn't natively supported here, 
            # so we'll just block indefinitely for simplicity
            self.sem.acquire()

    def close(self):
        if self.system == 'Windows':
            if hasattr(self, 'handle'):
                windll.kernel32.CloseHandle(self.handle)
        else:
            try:
                self.sem.unlink()
            except posix_ipc.ExistentialError:
                pass  # If the semaphore doesn't exist, just ignore the error

    def __del__(self):
        self.close()

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