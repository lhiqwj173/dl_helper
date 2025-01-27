import threading
import queue
import time

# 生产者线程
def producer(q, items):
    for item in items:
        print(f"生产者生产了: {item}")
        q.put(item)  # 将数据放入队列
        time.sleep(5)  # 模拟生产时间
    q.put(None)  # 发送结束信号

# 消费者线程
def consumer(q):
    while True:
        item = q.get()  # 从队列中取出数据
        if item is None:
            break  # 收到结束信号，退出循环
        print(f"消费者消费了: {item}")
        time.sleep(2)  # 模拟消费时间
        q.task_done()  # 标记任务完成

# 创建队列
q = queue.Queue()

# 创建生产者线程
producer_thread = threading.Thread(target=producer, args=(q, [1, 2, 3, 4, 5]))

# 创建消费者线程
consumer_thread_list = []
for i in range(3):
    consumer_thread = threading.Thread(target=consumer, args=(q,))
    consumer_thread_list.append(consumer_thread)
        
# 启动线程
producer_thread.start()
for consumer_thread in consumer_thread_list:
    consumer_thread.start()

# 等待生产者线程完成
producer_thread.join()

# 等待队列中的所有任务被处理完毕
q.join()

# 等待消费者线程完成
consumer_thread.join()

print("所有任务完成")