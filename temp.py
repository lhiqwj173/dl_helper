import threading
import time, random

from py_ext.tool import Lock
num = 10

# 定义一个线程函数
def test():
    lock = Lock('test_lock')
    global num
    while True:
        r_int = random.randint(0, 100)
        if_change = r_int % 5 == 0

        if if_change:
            with lock:
                print(f"num change to {r_int}")
                num = r_int
        else:
            with lock:
                print(f"num: {num}")

        time.sleep(0.5)

if __name__ == '__main__':
    # 创建两个线程
    ts = []
    for i in range(10):
        t = threading.Thread(target=test)
        ts.append(t)

    for t in ts:    
        t.start()

    # 等待线程完成
    for t in ts:
        t.join()
