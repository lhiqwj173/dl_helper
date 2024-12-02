import os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np

from dl_helper.rl.base import BaseAgent
from dl_helper.rl.rl_utils import ReplayBuffer, ReplayBufferWaitClose

VANILLA_DQN = 'VanillaDQN'
DOUBLE_DQN = 'DoubleDQN'
DUELING_DQN = 'DuelingDQN'
DD_DQN = 'DoubleDuelingDQN'
DQN_TYPES = [VANILLA_DQN, DOUBLE_DQN, DUELING_DQN, DD_DQN]

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q

class test_features_extractor_class(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    def forward(self, x):
        return F.relu(self.fc1(x))

class dqn_network(torch.nn.Module):
    def __init__(self, features_extractor_class, features_extractor_kwargs, features_dim, net_arch, dqn_type):
        super().__init__()
        self.features_extractor = features_extractor_class(
            **features_extractor_kwargs
        )

        # 剩余部分
        net_arch = net_arch['pi']

        self.fc_a_length = len(net_arch)
        if self.fc_a_length == 1:
            self.fc_a = torch.nn.Linear(features_dim, net_arch[0])
        else:
            self.fc_a = torch.nn.ModuleList([torch.nn.Linear(features_dim, net_arch[0])])
            for i in range(1, self.fc_a_length):
                self.fc_a.append(torch.nn.Linear(net_arch[i - 1], net_arch[i]))

        self.fc_v = None
        if dqn_type in [DUELING_DQN, DD_DQN]:
            self.fc_v = torch.nn.Linear(features_dim, 1)

    def forward(self, x):
        feature = self.features_extractor(x)

        x = feature
        if self.fc_a_length > 1:
            for i in range(self.fc_a_length - 1):
                x = F.relu(self.fc_a[i](x))
            x = self.fc_a[-1](x)
        else:
            x = self.fc_a(x)

        if self.fc_v is not None:
            v = self.fc_v(feature)
            x = v + x - x.mean(1).view(-1, 1)  # Q值由V值和A值计算得到

        return x

def report_learning_process(watch_data, root):
    """
    强化学习性能指标:
        return_list: 回合累计奖励
        avg_return_list: 回合平均长度奖励
        episode_lens: 回合长度
        max_q_value_list: 最大Q值
    
    交易评价指标:
        sharpe_ratio: 夏普比率
        sortino_ratio: 索提诺比率
        max_drawdown: 最大回撤
        total_return: 总回报
    """
    # 获取数据键,将交易评价指标单独处理
    rl_keys = ['return_list', 'avg_return_list', 'episode_lens', 'max_q_value_list']
    trade_keys = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'total_return']
    
    # 固定颜色
    colors = {
        'return': '#1f77b4',      # 蓝色
        'avg_return': '#2ca02c',  # 绿色
        'episode_lens': '#ff7f0e', # 橙色
        'max_q_value': '#9467bd', # 紫色
        'sharpe_ratio': '#8c564b', # 棕色
        'sortino_ratio': '#e377c2', # 粉色
        'max_drawdown': '#7f7f7f', # 灰色
        'total_return': '#bcbd22'  # 黄绿色
    }
    test_color = '#d62728'  # 红色
    
    # 计算总图数
    n_rl = 3  # return合并为1个,其他2个
    n_trade = 2  # ratio合并为1个,其他1个
    n_total = n_rl + n_trade
    
    # 创建图表,每行一个子图
    fig, axes = plt.subplots(n_total, 1, figsize=(12, 4*n_total), sharex=True)
    if n_total == 1:
        axes = np.array([axes])
    
    # 绘制强化学习指标
    # 1. return_list和avg_return_list
    ax = axes[0]
    ax.plot(watch_data['return_list'], color=colors['return'], alpha=0.3, label='train_return')
    ax.plot(watch_data['avg_return_list'], color=colors['avg_return'], alpha=0.3, label='train_avg_return')
    if 'return_list_val' in watch_data:
        ax.plot(watch_data['return_list_val'], color=colors['return'], label='val_return')
        ax.plot(watch_data['avg_return_list_val'], color=colors['avg_return'], label='val_avg_return')
    if 'return_list_test' in watch_data:
        ax.plot(watch_data['return_list_test'], color=test_color, label='test_return')
        ax.plot(watch_data['avg_return_list_test'], color=test_color, linestyle='--', label='test_avg_return')
    ax.set_ylabel('Return')
    ax.grid(True)
    ax.legend()
    
    # 2. episode_lens
    ax = axes[1]
    ax.plot(watch_data['episode_lens'], color=colors['episode_lens'], alpha=0.3, label='train_episode_lens')
    if 'episode_lens_val' in watch_data:
        ax.plot(watch_data['episode_lens_val'], color=colors['episode_lens'], label='val_episode_lens')
    if 'episode_lens_test' in watch_data:
        ax.plot(watch_data['episode_lens_test'], color=test_color, label='test_episode_lens')
    ax.set_ylabel('Episode Length')
    ax.grid(True)
    ax.legend()
    
    # 3. max_q_value_list
    ax = axes[2]
    ax.plot(watch_data['max_q_value_list'], color=colors['max_q_value'], alpha=0.3, label='train_max_q_value')
    if 'max_q_value_list_val' in watch_data:
        ax.plot(watch_data['max_q_value_list_val'], color=colors['max_q_value'], label='val_max_q_value')
    if 'max_q_value_list_test' in watch_data:
        ax.plot(watch_data['max_q_value_list_test'], color=test_color, label='test_max_q_value')
    ax.set_ylabel('Max Q Value')
    ax.grid(True)
    ax.legend()
    
    # 绘制交易评价指标
    # 1. sharpe_ratio和sortino_ratio
    ax = axes[3]
    ax.plot(watch_data['sharpe_ratio'], color=colors['sharpe_ratio'], alpha=0.3, label='train_sharpe_ratio')
    ax.plot(watch_data['sortino_ratio'], color=colors['sortino_ratio'], alpha=0.3, label='train_sortino_ratio')
    if 'sharpe_ratio_val' in watch_data:
        ax.plot(watch_data['sharpe_ratio_val'], color=colors['sharpe_ratio'], label='val_sharpe_ratio')
        ax.plot(watch_data['sortino_ratio_val'], color=colors['sortino_ratio'], label='val_sortino_ratio')
    if 'sharpe_ratio_test' in watch_data:
        ax.plot(watch_data['sharpe_ratio_test'], color=test_color, label='test_sharpe_ratio')
        ax.plot(watch_data['sortino_ratio_test'], color=test_color, linestyle='--', label='test_sortino_ratio')
    ax.set_ylabel('Ratio')
    ax.grid(True)
    ax.legend()
    
    # 2. max_drawdown和total_return (双y轴)
    ax = axes[4]
    ax2 = ax.twinx()  # 创建共享x轴的第二个y轴
    
    lines = []
    # 在左轴绘制max_drawdown
    l1 = ax.plot(watch_data['max_drawdown'], color=colors['max_drawdown'], alpha=0.3, label='train_max_drawdown')[0]
    lines.append(l1)
    if 'max_drawdown_val' in watch_data:
        l2 = ax.plot(watch_data['max_drawdown_val'], color=colors['max_drawdown'], label='val_max_drawdown')[0]
        lines.append(l2)
    if 'max_drawdown_test' in watch_data:
        l3 = ax.plot(watch_data['max_drawdown_test'], color=test_color, label='test_max_drawdown')[0]
        lines.append(l3)
    ax.set_ylabel('Max Drawdown')
    
    # 在右轴绘制total_return
    l4 = ax2.plot(watch_data['total_return'], color=colors['total_return'], alpha=0.3, label='train_total_return')[0]
    lines.append(l4)
    if 'total_return_val' in watch_data:
        l5 = ax2.plot(watch_data['total_return_val'], color=colors['total_return'], label='val_total_return')[0]
        lines.append(l5)
    if 'total_return_test' in watch_data:
        l6 = ax2.plot(watch_data['total_return_test'], color=test_color, label='test_total_return')[0]
        lines.append(l6)
    ax2.set_ylabel('Total Return')
    
    # 合并两个轴的图例
    ax.legend(handles=lines, loc='upper left')
    ax.grid(True)
    
    # 设置共享的x轴标签
    fig.text(0.5, 0.04, 'Episode', ha='center')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(root, 'learning_process.png'))
    plt.close()

class DQN(BaseAgent):
    def __init__(self,
            action_dim,
            features_dim,
            features_extractor_class,
            learning_rate,
            gamma,
            epsilon,
            target_update,
            device,
            wait_trade_close = True,
            features_extractor_kwargs=None,
            net_arch=None,
            dqn_type=VANILLA_DQN
    ):
        """
        DQN算法

        Args:
            action_dim: 动作空间维度
            features_dim: 特征维度
            features_extractor_class: 特征提取器类,必须提供
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: epsilon-贪婪策略中的epsilon
            target_update: 目标网络更新频率
            device: 设备
            features_extractor_kwargs: 特征提取器参数,可选
            net_arch: 网络架构参数,可选
            dqn_type: DQN类型,可选 VANILLA_DQN/DOUBLE_DQN/DUELING_DQN/DD_DQN
        """
        super().__init__(action_dim, features_dim, features_extractor_class, features_extractor_kwargs, net_arch)   
        assert dqn_type in DQN_TYPES, f'dqn_type 必须是 {DQN_TYPES} 中的一个, 当前为 {dqn_type}'

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device
        self.wait_trade_close = wait_trade_close    
        self.replay_buffer = ReplayBuffer(buffer_size) if not wait_trade_close else ReplayBufferWaitClose(buffer_size)

        self.q_net, self.target_q_net = self.build_model()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)

    def state_dict(self):
        state = super().state_dict()  # 获取父类的状态
        state.update({
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'target_update': self.target_update,
            'count': self.count,
            'dqn_type': self.dqn_type,
            'device': self.device,
            'replay_buffer': self.replay_buffer,
        })

        # 模型参数
        self.state_dict_model(state, 'q_net', self.q_net)
        self.state_dict_model(state, 'target_q_net', self.target_q_net) 

        return state

    def build_model(self):
        q_net = dqn_network(self.features_extractor_class, self.features_extractor_kwargs, self.features_dim, self.net_arch, self.dqn_type).to(self.device)
        target_q_net = dqn_network(self.features_extractor_class, self.features_extractor_kwargs, self.features_dim, self.net_arch, self.dqn_type).to(self.device)
        # print(f'q_net: {id(q_net)}, target_q_net: {id(target_q_net)}')
        return q_net, target_q_net

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.from_numpy(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.from_numpy(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.from_numpy(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.from_numpy(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type in [DOUBLE_DQN, DD_DQN] :
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(
                1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    rl_training_ind = [
        'return_list',       # 验证集回合累计奖励
        'avg_return_list',  # 验证集回合平均长度奖励
        'episode_lens',     # 验证集回合长度
        'max_q_value_list' # 验证集最大Q值
    ]
    def val_test(self, env, data_type='val'):
        # 训练监控指标 / 每回合
        watch_data = {f'{i}_{data_type}': [] for i in self.rl_training_ind}
        max_q_value = 0
        episode_return = 0
        episode_len = 0

        env.set_data_type(data_type)
        # 10 次求平均
        for _ in range(10):
            state, info = env.reset()
            done = False
            while not done:
                action = self.take_action(state)
                max_q_value = self.max_q_value(
                    state) * 0.005 + max_q_value * 0.995  # 平滑处理
                next_state, reward, done1, done2, info = env.step(action)
                done = done1 or done2
                if info.get('close', False) and self.wait_trade_close:
                    # 更新评价指标
                    for k, v in info.items():
                        if k != 'close':
                            if k not in watch_data:
                                watch_data[f'{k}_{data_type}'] = []
                            watch_data[f'{k}_{data_type}'].append(v)
                state = next_state
                episode_return += reward
                episode_len += 1

        # 更新监控指标
        watch_data[f'return_list_{data_type}'].append(episode_return)
        watch_data[f'avg_return_list_{data_type}'].append(episode_return / episode_len)
        watch_data[f'episode_lens_{data_type}'].append(episode_len)
        watch_data[f'max_q_value_list_{data_type}'].append(max_q_value)

        # 返回均值
        for k in watch_data:
            watch_data[k] = np.mean(watch_data[k])
        return watch_data

    def package_root(self, watch_data):
        # 保存模型
        self.save(self.root)
        # 生成报告
        report_learning_process(watch_data, self.root)
        # 打包压缩
        zip_file = f'{self.root}.7z'
        if os.path.exists(zip_file):
            os.remove(zip_file)
        compress_folder(self.root, zip_file, 9, inplace=False)
        # 上传alist
        upload_folder = f'/train_data/'
        self.client.mkdir(upload_folder)
        self.client.upload(zip_file, upload_folder)

    def learn(self, train_title, env, num_episodes, minimal_size, batch_size, report_interval=50, test_interval=1000, update_interval=4):
        # 准备
        super().learn(train_title)

        # 训练监控指标 / 每回合
        watch_data = {i: [] for i in self.rl_training_ind}
        max_q_value = 0

        # 学习是否开始
        learning_start = False
        for i in range(num_episodes):
            episode_return = 0
            episode_len = 0 
            # 回合的评价指标
            episode_metrics = {}
            state, info = env.reset()
            done = False
            while not done:
                # 动作
                action = self.take_action(state)
                max_q_value = self.max_q_value(
                    state) * 0.005 + max_q_value * 0.995  # 平滑处理

                # 环境交互
                next_state, reward, done1, done2, info = env.step(action)
                done = done1 or done2

                # 添加到回放池
                self.replay_buffer.add(state, action, reward, next_state, done)

                # 如果 交易close 则需要回溯更新所有 reward 为最终close时的reward
                if info.get('close', False) and self.wait_trade_close:
                    self.replay_buffer.update_reward(reward)
                    # 更新评价指标
                    for k, v in info.items():
                        if k != 'close':
                            if k not in episode_metrics:
                                episode_metrics[k] = []
                                watch_data[k] = []
                            episode_metrics[k].append(v)
                    
                # 更新状态
                state = next_state
                episode_return += reward
                episode_len += 1

                # 更新网络
                if self.replay_buffer.size() > minimal_size and i % update_interval == 0:
                    if not learning_start:
                        learning_start = True
                        # 截断 max_q_value_list / return_list
                        # 之前都还每开始训练，记录无意义
                        max_q_value_list = max_q_value_list[-1:]
                        return_list = return_list[-1:]

                    b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(
                        batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    self.update(transition_dict)

            # 更新每回和的监控列表
            watch_data['return_list'].append(episode_return)
            watch_data['avg_return_list'].append(episode_return / episode_len)
            watch_data['episode_lens'].append(episode_len)
            watch_data['max_q_value_list'].append(max_q_value)
            # 更新每回合的平均评价指标
            for k, v in episode_metrics.items():
                watch_data[k].append(np.mean(v))

            # 验证和测试
            for data_type, interval in [('val', report_interval), ('test', test_interval)]:
                if (i + 1) % interval == 0 and learning_start:
                    watch_data_new = self.val_test(env, data_type=data_type)
                    for k in watch_data_new:
                        if k not in watch_data:
                            watch_data[k] = []
                        # 补齐长度
                        watch_data[k] += [watch_data_new[k]] * (len(watch_data['return_list']) - len(watch_data[k]))
                    self.package_root(watch_data)


if __name__ == '__main__':
    agent = DQN(
        action_dim=4,
        features_dim=4,
        features_extractor_class=test_features_extractor_class,
        features_extractor_kwargs={
            'state_dim': 8,
            'hidden_dim': 4
        },
        learning_rate=0.005,
        gamma=0.98,
        epsilon=0.01,
        target_update=10,
        device='cpu',
        dqn_type=DD_DQN
    )
    print(agent.q_net)
    print(agent.target_q_net)
    # print(f'q_net: {id(agent.q_net)}, target_q_net: {id(agent.target_q_net)}, q_net_features_extractor: {id(agent.q_net.features_extractor)}, target_q_net_features_extractor: {id(agent.target_q_net.features_extractor)}')
