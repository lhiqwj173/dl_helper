import numpy as np
import unittest

from dl_helper.rl.rl_env.lob_env_data_augmentation import random_his_window

class TestRandomHisWindow(unittest.TestCase):
    
    def setUp(self):
        # 设置随机种子以确保测试的可重复性
        np.random.seed(42)
    
    ### 基本功能测试 ###
    def test_basic_functionality(self):
        """测试函数在正常情况下是否返回行数 >= his_len 的数据"""
        raw_data = np.arange(20).reshape(10, 2)
        his_len = 5
        result = random_his_window(raw_data, his_len)
        self.assertGreaterEqual(result.shape[0], his_len)
    
    ### 边界条件测试 ###
    def test_exact_his_len(self):
        """测试当 raw_data 行数等于 his_len 时，函数行为是否正确"""
        raw_data = np.arange(10).reshape(5, 2)
        his_len = 5
        result = random_his_window(raw_data, his_len)
        self.assertEqual(result.shape[0], his_len)
    
    ### 异常处理测试 ###
    def test_less_than_his_len(self):
        """测试当 raw_data 行数 < his_len 时是否抛出 ValueError"""
        raw_data = np.arange(8).reshape(4, 2)
        his_len = 5
        with self.assertRaises(ValueError):
            random_his_window(raw_data, his_len)
    
    def test_max_random_num(self):
        """测试 max_random_num 参数是否限制最大删除行数"""
        raw_data = np.arange(20).reshape(10, 2)
        his_len = 5
        max_random_num = 2
        # 多次运行以统计删除行为
        deletions = []
        for _ in range(100):
            result = random_his_window(raw_data, his_len, max_random_num=max_random_num)
            deletions.append(10 - result.shape[0])
        max_deletion = max(deletions)
        self.assertLessEqual(max_deletion, max_random_num)
        self.assertGreaterEqual(result.shape[0], his_len)
    
    def test_random_prob(self):
        """测试 random_prob 参数对删除概率的影响"""
        raw_data = np.arange(20).reshape(10, 2)
        his_len = 5
        
        # random_prob = 0.0，不删除任何行
        result = random_his_window(raw_data, his_len, random_prob=0.0)
        self.assertEqual(result.shape[0], 10)
        
        # random_prob = 1.0，受 max_random_num 限制
        result = random_his_window(raw_data, his_len, random_prob=1.0, max_random_num=3)
        self.assertGreaterEqual(result.shape[0], 10 - 3)
    
    ### 随机性测试 ###
    def test_randomness(self):
        """测试函数的随机性是否导致不同输出"""
        raw_data = np.arange(20).reshape(10, 2)
        his_len = 5
        results = [random_his_window(raw_data, his_len) for _ in range(10)]
        unique_results = set(tuple(r.flatten()) for r in results)
        self.assertGreater(len(unique_results), 1)

if __name__ == '__main__':
    unittest.main()