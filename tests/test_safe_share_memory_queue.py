import warnings
import unittest
from py_ext.tool import safe_share_memory_queue

# 禁用警告
warnings.filterwarnings("ignore")

class TestSafeShareMemoryList(unittest.TestCase):
    def setUp(self):
        # 创建一个容量为10，每个槽位1024字节的队列
        self.queue = safe_share_memory_queue("test_queue", size=1024, nums=10)
        self.queue.clear()

    def tearDown(self):
        del self.queue

    def test_basic_operations(self):
        # 测试初始状态
        self.assertTrue(self.queue.is_empty())
        self.assertFalse(self.queue.is_full())
        self.assertEqual(len(self.queue), 0)

        # 测试放入数据
        test_data = b"Hello, World!"
        self.queue.put(test_data)
        self.assertFalse(self.queue.is_empty())
        self.assertEqual(len(self.queue), 1)

        # 测试获取数据
        retrieved_data = self.queue.get()
        self.assertEqual(retrieved_data, test_data)
        self.assertTrue(self.queue.is_empty())

    def test_multiple_operations(self):
        # 测试多次放入和获取
        test_data = [f"data_{i}".encode() for i in range(5)]
        
        # 放入数据
        for data in test_data:
            self.queue.put(data)
        
        self.assertEqual(len(self.queue), 5)

        # 获取数据并验证顺序
        for expected_data in test_data:
            retrieved_data = self.queue.get()
            self.assertEqual(retrieved_data, expected_data)

        self.assertTrue(self.queue.is_empty())

    def test_full_queue(self):
        # 测试队列满的情况
        test_data = b"test data"
        
        # 填满队列
        for _ in range(10):
            self.queue.put(test_data)
        
        self.assertTrue(self.queue.is_full())
        
        # 测试向满队列中继续放入数据
        with self.assertRaises(ValueError):
            self.queue.put(test_data)

    def test_empty_queue(self):
        # 测试从空队列中获取数据
        self.assertIsNone(self.queue.get())

    def test_large_data(self):
        # 测试大小超过限制的数据
        large_data = b"x" * 1025  # 超过size限制
        with self.assertRaises(ValueError):
            self.queue.put(large_data)

def main():
    # 运行所有测试
    unittest.main()

if __name__ == "__main__":
    main()
