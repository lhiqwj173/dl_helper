import numpy as np
import unittest

from dl_helper.rl.rl_utils import PrioritizedReplayBufferWaitClose

class TestPrioritizedReplayBufferWaitClose(unittest.TestCase):
    def setUp(self):
        self.buffer = PrioritizedReplayBufferWaitClose(capacity=100)
        
    def test_basic_functionality(self):
        # 创建模拟数据
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        next_state = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        action = 1
        reward = 0.5
        done = False
        
        # 测试添加单个经验
        self.buffer.add(state, action, reward, next_state, done)
        
        # 验证临时经验缓冲区
        self.assertEqual(len(self.buffer.temp_experiences), 1)
        self.assertEqual(len(self.buffer.temp_indices), 0)
        
        # 测试更新奖励
        new_reward = 1.0
        self.buffer.update_reward(new_reward)
        
        # 验证经验已转移到主缓冲区
        self.assertEqual(self.buffer.size(), 1)
        self.assertEqual(len(self.buffer.temp_experiences), 0)
        
        # 测试采样
        batch, indices, weights = self.buffer.sample(1)
        states, actions, rewards, next_states, dones = batch
        
        # 验证采样数据
        np.testing.assert_array_equal(states[0], state)
        self.assertEqual(actions[0], action)
        self.assertEqual(rewards[0], new_reward)
        np.testing.assert_array_equal(next_states[0], next_state)
        self.assertEqual(dones[0], done)
        
    def test_multiple_experiences(self):
        # 添加多个经验
        for i in range(5):
            state = np.array([i, i+1, i+2], dtype=np.float32)
            next_state = np.array([i+1, i+2, i+3], dtype=np.float32)
            action = i
            reward = 0.1 * i
            done = False
            self.buffer.add(state, action, reward, next_state, done)
            
        # 验证临时缓冲区大小
        self.assertEqual(len(self.buffer.temp_experiences), 5)
        
        # 更新所有奖励
        final_reward = 1.0
        self.buffer.update_reward(final_reward)
        
        # 验证主缓冲区大小
        self.assertEqual(self.buffer.size(), 5)
        
        # 测试批量采样
        batch_size = 3
        batch, indices, weights = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # 验证批量大小
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)
        self.assertEqual(len(next_states), batch_size)
        self.assertEqual(len(dones), batch_size)
        
        # 验证所有奖励都被更新
        self.assertTrue(all(r == final_reward for r in rewards))
        
    def test_reset(self):
        # 添加一些经验
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        next_state = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        self.buffer.add(state, 1, 0.5, next_state, False)
        
        # 测试重置
        self.buffer.reset()
        
        # 验证缓冲区已清空
        self.assertEqual(self.buffer.size(), 0)
        self.assertEqual(len(self.buffer.temp_experiences), 0)
        self.assertEqual(len(self.buffer.temp_indices), 0)

if __name__ == '__main__':
    unittest.main()