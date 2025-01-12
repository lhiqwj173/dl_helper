from multiprocessing import Process, Event
import time

def task_a(event, rounds):
    for round in range(rounds):
        print(f"Process A: Starting round {round + 1} of task A")
        time.sleep(2)  # 模拟 A 完成任务所需的时间
        print(f"Process A: Finished round {round + 1} of task A")
        event.set()  # 通知 B 任务可以开始
        event.clear()  # 重置事件，等待下一轮

def task_b(event, rounds, identifier):
    for round in range(rounds):
        print(f"Process B{identifier}: Waiting for round {round + 1} of task A to complete")
        event.wait()  # 等待 A 发送信号
        print(f"Process B{identifier}: Round {round + 1} of task A completed, starting round {round + 1} of task B{identifier}")
        time.sleep(1)  # 模拟 B 完成任务所需的时间
        print(f"Process B{identifier}: Finished round {round + 1} of task B{identifier}")
        event.clear()  # 虽然 A 已经清除了，但确保 B 不互相干扰

if __name__ == '__main__':
    rounds = 3  # 设定进行三轮任务
    event = Event()

    # 启动进程 A
    process_a = Process(target=task_a, args=(event, rounds))
    process_a.start()

    # 启动多个进程 B
    b_processes = []
    for i in range(3):  # 这里我们创建3个 B 任务
        b_process = Process(target=task_b, args=(event, rounds, i))
        b_processes.append(b_process)
        b_process.start()

    # 等待进程 A 完成
    process_a.join()

    # 等待所有 B 进程完成
    for b_process in b_processes:
        b_process.join()

    print("All tasks completed")