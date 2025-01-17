import threading
import time

# 定义一个函数来执行任务
def task1():
    print("Task 1 started")
    time.sleep(2)  # 模拟任务执行时间
    print("Task 1 finished")

def task2():
    print("Task 2 started")
    time.sleep(3)  # 模拟任务执行时间
    print("Task 2 finished")

if __name__ == "__main__":
    # 创建线程对象
    thread1 = threading.Thread(target=task1)
    thread2 = threading.Thread(target=task2)

    # 启动线程
    thread1.start()
    thread2.start()

    print('111')

    # 等待线程执行结束
    thread1.join()
    thread2.join()

    print("All tasks are finished")