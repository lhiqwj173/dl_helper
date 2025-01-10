import gymnasium as gym
import unittest
from ray.rllib.algorithms.ppo.ppo import PPOConfig

from dl_helper.rl.param_keeper import AsyncRLParameterServer

class TestAsyncRLParameterServer(unittest.TestCase):
    def setUp(self):
        self.config = PPOConfig()
        self.config.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        self.env = gym.make("CartPole-v1")
        self.ps = AsyncRLParameterServer(self.config, self.env)

    def test_apply_gradients(self):
        # 模拟一个梯度字典
        _params = self.ps.learner._params
        gradients_dict = {pid: p * 0.1 for pid, p in _params.items()}
        
        # 调用 apply_gradients 方法
        new_weights, new_version = self.ps.apply_gradients(gradients_dict)
        
        # 检查版本号是否增加
        self.assertEqual(new_version, 1)

    def test_get_weights(self):
        # 获取权重
        weights, version = self.ps.get_weights()
        
        # 检查返回的版本号
        self.assertEqual(version, 0)  # 因为 apply_gradients 还没有被调用过，版本号还是初始值 0

if __name__ == '__main__':
    unittest.main()