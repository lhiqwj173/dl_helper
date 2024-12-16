import torch
import torch.nn.functional as F

from dl_helper.rl.base import BaseAgent, OffPolicyAgent

class c51_network(torch.nn.Module):
    def __init__(self, obs_shape, features_extractor_class, features_extractor_kwargs, features_dim, net_arch, n_atoms):
        """
        features_dim: features_extractor_class输出维度  + 3(symbol_id + 持仓 + 未实现收益率)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.n_atoms = n_atoms

        # 特征提取器
        self.features_extractor = features_extractor_class(**features_extractor_kwargs)
        
        # 添加Batch Normalization
        self.bn = torch.nn.BatchNorm1d(features_dim)
        self.dropout = torch.nn.Dropout(p=0.5)
        
        # 网络结构
        net_arch = net_arch['pi']
        self.fc_length = len(net_arch)
        
        if self.fc_length == 1:
            self.fc = torch.nn.Linear(features_dim, net_arch[0] * self.n_atoms)
        else:
            self.fc = torch.nn.ModuleList([torch.nn.Linear(features_dim, net_arch[0])])
            for i in range(1, self.fc_length):
                if i == self.fc_length - 1:
                    # 最后一层输出为 action_dim * n_atoms 个值
                    self.fc.append(torch.nn.Linear(net_arch[i - 1], net_arch[i] * self.n_atoms))
                else:
                    self.fc.append(torch.nn.Linear(net_arch[i - 1], net_arch[i]))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        先将x分成两个tensor
        lob: x[:, :-3]
        acc: x[:, -3:]
        """
        lob_data = x[:, :-3].view(-1, self.obs_shape[0], self.obs_shape[1])
        acc_data = x[:, -3:]

        feature = self.features_extractor(lob_data)
        feature = torch.cat([feature, acc_data], dim=1)
        
        # 应用Batch Normalization
        x = self.bn(feature)
        if self.fc_length > 1:
            for i in range(self.fc_length - 1):
                x = F.leaky_relu(self.fc[i](x))
                x = self.dropout(x)
            x = self.fc[-1](x)
        else:
            x = self.fc(x)

        # 将输出展平    
        x = x.view(-1, self.n_atoms)
        return F.softmax(x, dim=1)


class C51(OffPolicyAgent):

    def __init__(
        self,
        obs_shape,
        learning_rate,
        gamma,
        epsilon,
        target_update,

        # 基类参数
        buffer_size,
        train_title,
        action_dim,
        features_dim,
        features_extractor_class,
        features_extractor_kwargs=None,
        net_arch=None,

        n_atoms=51,
    ):
        """
        DQN
        
        Args:
            obs_shape: 观测空间维度
            learning_rate: 学习率
            gamma: TD误差折扣因子
            epsilon: epsilon-greedy策略参数
            target_update: 目标网络更新间隔
            n_atoms: 分位数数量

            基类参数
                buffer_size: 经验回放池大小
                train_title: 训练标题
                action_dim: 动作空间维度 
                features_dim: 特征维度
                features_extractor_class: 特征提取器类,必须提供
                features_extractor_kwargs=None: 特征提取器参数,可选
                net_arch=None: 网络架构参数,默认为一层mlp, 输入/输出维度为features_dim, action_dim
                    [action_dim] / dict(pi=[action_dim], vf=[action_dim]) 等价
        """
        super().__init__(buffer_size, train_title, action_dim, features_dim, features_extractor_class, features_extractor_kwargs, net_arch)

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.need_epsilon = True
        self.target_update = target_update
        self.count = 0
        self.n_atoms = n_atoms

        # 初始化网络
        self.build_model()
        self.optimizer = torch.optim.Adam(self.models['q_net'].parameters(),lr=learning_rate)
    ############################################################
    # TODO
    # 需要重写的函数
    #     build_model: 构建模型
    #     take_action(self, state): 根据状态选择动作
    #     _update(self, states, actions, rewards, next_states, dones, data_type): 更新模型
    #     sync_update_net_params_in_agent: 同步更新模型参数
    #     get_params_to_send: 获取需要上传的参数
    ############################################################

    def eval(self):
        self.need_epsilon = False

    def train(self):
        self.need_epsilon = True

    def build_model(self):
        q_net = dqn_network(self.obs_shape, self.features_extractor_class, self.features_extractor_kwargs, self.features_dim, self.net_arch, self.dqn_type)
        target_q_net = dqn_network(self.obs_shape, self.features_extractor_class, self.features_extractor_kwargs, self.features_dim, self.net_arch, self.dqn_type)
        self.models = {'q_net': q_net, 'target_q_net': target_q_net}

        self.models['q_net'].train()
        # 设置为eval模式
        self.models['target_q_net'].eval()

    def take_action(self, state):
        if self.need_epsilon and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            
            self.models['q_net'].eval()
            with torch.no_grad():
                action = self.models['q_net'](state).argmax().item()
            self.models['q_net'].train()
        return action

    def _update(self, states, actions, rewards, next_states, dones, data_type):
        # 计算当前Q值
        q_values = self.models['q_net'](states).gather(1, actions)
        
        # 计算目标Q值,不需要梯度
        with torch.no_grad():
            if self.dqn_type in [DOUBLE_DQN, DD_DQN]:
                # Double DQN: 用当前网络选择动作,目标网络评估动作价值
                max_action = self.models['q_net'](next_states).max(1)[1].view(-1, 1)
                max_next_q_values = self.models['target_q_net'](next_states).gather(1, max_action)
            else:
                # 普通DQN: 直接用目标网络计算最大Q值
                max_next_q_values = self.models['target_q_net'](next_states).max(1)[0].view(-1, 1)
            # 计算目标Q值
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # 计算TD误差
        td_error = torch.abs(q_targets - q_values).mean().item()
        
        # 计算损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # tracker 记录
        self.track_error(td_error, dqn_loss.item(), data_type)

        # 检查是否有nan/inf值
        if (torch.isnan(dqn_loss) or torch.isinf(dqn_loss) or 
            np.isnan(td_error) or np.isinf(td_error)):
            # 保存batch数据到pickle文件
            batch_data = {
                'states': states.cpu().numpy(),
                'actions': actions.cpu().numpy(),
                'rewards': rewards.cpu().numpy(), 
                'next_states': next_states.cpu().numpy(),
                'dones': dones.cpu().numpy(),
                'q_values': q_values.detach().cpu().numpy(),
                'q_targets': q_targets.detach().cpu().numpy(),
                'max_next_q_values': max_next_q_values.detach().cpu().numpy(),
                'dqn_loss': dqn_loss.item(),
                'td_error': td_error
            }
            
            # 保存到文件
            with open(f'nan_batch_{data_type}.pkl', 'wb') as f:
                pickle.dump(batch_data, f)
                
            error_msg = f'检测到NaN/Inf值,dqn_loss:{dqn_loss.item()},td_error:{td_error},batch数据已保存到nan_batch_{data_type}.pkl'
            raise ValueError(error_msg)
        
        if data_type == 'train':
            self.optimizer.zero_grad()
            dqn_loss.backward()
            self.optimizer.step()

            if self.count > 0 and self.count % self.target_update == 0:
                self.sync_update_net_params_in_agent()
            self.count += 1

    def sync_update_net_params_in_agent(self):
        self.models['target_q_net'].load_state_dict(self.models['q_net'].state_dict())

    def get_params_to_send(self):
        return self.models['q_net'].state_dict()
