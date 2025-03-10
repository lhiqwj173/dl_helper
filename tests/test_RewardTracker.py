import unittest
from statistics import mean
from dl_helper.rl.rl_env.lob_env import RewardTracker

class TestRewardTracker(unittest.TestCase):
    def setUp(self):
        """在每个测试前初始化一个新的 RewardTracker 实例"""
        self.tracker = RewardTracker()

    def test_initial_state(self):
        """测试初始状态"""
        result = self.tracker.add_reward(0)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)

    def test_single_positive_reward(self):
        """测试单个正奖励"""
        result = self.tracker.add_reward(5.0)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 5.0)

    def test_single_negative_reward(self):
        """测试单个负奖励"""
        result = self.tracker.add_reward(-2.0)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], -2.0)

    def test_consecutive_negatives(self):
        """测试连续负奖励"""
        self.tracker.add_reward(-1.0)
        result = self.tracker.add_reward(-2.0)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], -1.5)

    def test_negative_reset(self):
        """测试正奖励重置负奖励计数"""
        self.tracker.add_reward(-1.0)
        self.tracker.add_reward(-2.0)
        result = self.tracker.add_reward(3.0)
        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(result[1], 0.0)

    def test_mixed_rewards(self):
        """测试混合奖励序列"""
        rewards = [1.0, -2.0, -1.0, 3.0, -1.0]
        expected_neg_counts = [0, 1, 2, 0, 1]
        expected_avgs = [1.0, -0.5, -0.6666666666666666, 0.25, 0.0]
        
        for i, reward in enumerate(rewards):
            result = self.tracker.add_reward(reward)
            self.assertEqual(result[0], expected_neg_counts[i])
            self.assertAlmostEqual(result[1], expected_avgs[i])

    def test_zero_reward(self):
        """测试零奖励"""
        self.tracker.add_reward(-1.0)
        result = self.tracker.add_reward(0.0)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], -0.5)

    def test_float_precision(self):
        """测试浮点数精度"""
        result = self.tracker.add_reward(1.1)
        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(result[1], 1.1)

    def test_reset_empty(self):
        """测试空状态下的重置"""
        self.tracker.reset()
        result = self.tracker.add_reward(1.0)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1.0)

    def test_reset_after_rewards(self):
        """测试有奖励后的重置"""
        self.tracker.add_reward(-1.0)
        self.tracker.add_reward(-2.0)
        self.tracker.reset()
        result = self.tracker.add_reward(2.0)
        self.assertEqual(result[0], 0)  # 负计数被重置
        self.assertEqual(result[1], 2.0)  # 平均值只包含新奖励

        
if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)  # verbosity=2 显示详细测试信息