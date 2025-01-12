import numpy as np
from multiprocessing import Process, shared_memory
import time
import unittest

from dl_helper.rl.costum_rllib_module.client_learner import SharedParam

def worker_process(params_dict, add_value):
    """
    工作进程，用于测试参数读写
    """
    # 创建非创建模式的共享参数对象
    shared_param = SharedParam(params_dict, create=False)
    
    # 读取参数并验证
    params = shared_param.get_param_dict()
    
    # 修改参数
    for k, v in params.items():
        v += add_value
    
    # 设置参数
    shared_param.set_param(params)


class TestSharedParam(unittest.TestCase):
    def setUp(self):
        # 测试用的参数字典
        self.params_dict = {
            'weight': np.random.randn(100, 200).astype(np.float32),
            'bias': np.random.randn(200).astype(np.float32)
        }
        
    def test_basic_operations(self):
        # 测试基本操作
        shared_param = SharedParam(self.params_dict, create=True)
        
        # 测试更新计数
        self.assertEqual(shared_param.update_count(), 0)
        
        # 测试参数设置
        new_params = {
            'weight': np.random.randn(100, 200).astype(np.float32),
            'bias': np.random.randn(200).astype(np.float32)
        }
        shared_param.set_param(new_params)
        self.assertEqual(shared_param.update_count(), 1)

        params = shared_param.get_param_dict()
        for k, v in new_params.items():
            np.testing.assert_array_equal(params[k], v)

        shared_param.clear()

    def test_multiprocess(self):
        # 测试多进程共享
        shared_param = SharedParam(self.params_dict, create=True)

        # 设置参数
        shared_param.set_param(self.params_dict)
        # 验证更新计数被更新
        self.assertEqual(shared_param.update_count(), 1)

        # 创建工作进程
        add_value = 1
        p = Process(target=worker_process, args=(self.params_dict, add_value))
        p.start()
        
        # 等待工作进程完成
        p.join()
        
        # 验证更新计数被更新
        self.assertEqual(shared_param.update_count(), 2)

        # 验证参数被更新
        new_params = {k: v + add_value for k, v in self.params_dict.items()}
        params = shared_param.get_param_dict()
        for k, v in new_params.items():
            np.testing.assert_array_equal(params[k], v)
        
        shared_param.clear()

if __name__ == '__main__':
    unittest.main()