import unittest
import torch
import numpy as np
from dl_helper.rl.dqn.c51 import C51

class DummyFeatureExtractor(torch.nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()
        self.fc = torch.nn.Linear(40, output_dim)  # 40 = 10 * 4 (假设的LOB数据维度)
    
    def forward(self, x):
        # 展平
        # torch.Size([1, 10, 4]) -> (1, 40)
        x = x.view(-1, 40)
        return self.fc(x)

class TestC51(unittest.TestCase):
    def setUp(self):
        # 设置基本参数
        self.obs_shape = (10, 4)  # 假设的LOB数据形状
        self.action_dim = 3
        self.features_dim = 35  # 32 + 3 (feature_extractor输出 + 账户信息)
        
        # 初始化C51智能体
        self.agent = C51(
            obs_shape=self.obs_shape,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=0.1,
            target_update=100,
            buffer_size=1000,
            train_title="test_c51",
            action_dim=self.action_dim,
            features_dim=self.features_dim,
            features_extractor_class=DummyFeatureExtractor,
            features_extractor_kwargs={"output_dim": 32},
            net_arch=[64, self.action_dim],
            n_atoms=51,
            v_min=-10,
            v_max=10
        )

    def test_initialization(self):
        """测试C51类的初始化"""
        self.assertEqual(self.agent.action_dim, 3)
        self.assertEqual(self.agent.n_atoms, 51)
        self.assertEqual(len(self.agent.models), 2)
        self.assertIn('q_net', self.agent.models)
        self.assertIn('target_q_net', self.agent.models)

    def test_take_action(self):
        """测试动作选择"""
        # 创建模拟状态
        state = np.random.random((43,))  # 40 (LOB) + 3 (账户信息)
        
        # 测试训练模式（带epsilon-greedy）
        self.agent.train()
        action = self.agent.take_action(state)
        self.assertTrue(0 <= action < self.action_dim)
        
        # 测试评估模式（无epsilon-greedy）
        self.agent.eval()
        action = self.agent.take_action(state)
        self.assertTrue(0 <= action < self.action_dim)

    def test_update(self):
        """测试更新过程"""
        batch_size = 4
        
        # 创建模拟批次数据
        states = torch.randn(batch_size, 43)
        actions = torch.randint(0, self.action_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 43)
        dones = torch.zeros(batch_size)
        
        # 测试更新过程不会抛出错误
        try:
            self.agent._update(states, actions, rewards, next_states, dones, 'train')
        except Exception as e:
            self.fail(f"Update process raised an exception: {e}")

    def test_sync_update(self):
        """测试目标网络同步更新"""
        # 获取更新前的参数
        q_net_params = self.agent.models['q_net'].state_dict()
        target_params_before = self.agent.models['target_q_net'].state_dict()
        
        # 执行同步更新
        self.agent.sync_update_net_params_in_agent()
        
        # 获取更新后的参数
        target_params_after = self.agent.models['target_q_net'].state_dict()
        
        # 验证参数是否相同
        for key in q_net_params:
            self.assertTrue(torch.equal(q_net_params[key], target_params_after[key]))

if __name__ == '__main__':
    unittest.main()