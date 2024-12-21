import torch
import torch.nn.functional as F
import numpy as np
import pickle
import collections

from dl_helper.rl.base import BaseAgent, OffPolicyAgent
from dl_helper.rl.rl_env.lob_env import MEAN_SEC_BEFORE_CLOSE, STD_SEC_BEFORE_CLOSE

class c51_network(torch.nn.Module):
    def __init__(self, obs_shape, features_extractor_class, features_extractor_kwargs, features_dim, net_arch, n_atoms, v_min, v_max):
        """
        features_dim: features_extractor_class输出维度  + 4(symbol_id + 持仓 + 未实现收益率 + 距离收盘秒数)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        # 计算支持点
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # 特征提取器
        self.features_extractor = features_extractor_class(**features_extractor_kwargs)
        
        # 添加Batch Normalization
        self.bn = torch.nn.BatchNorm1d(features_dim)
        self.dropout = torch.nn.Dropout(p=0.5)
        
        # 网络结构
        net_arch = net_arch['pi']
        self.fc_length = len(net_arch)
        self.net_arch = net_arch
        
        # 最后一层输出为 action_dim(net_arch[-1]) * n_atoms 个值
        if self.fc_length == 1:
            self.fc = torch.nn.Linear(features_dim, net_arch[0] * self.n_atoms)
        else:
            self.fc = torch.nn.ModuleList([torch.nn.Linear(features_dim, net_arch[0])])
            for i in range(1, self.fc_length):
                if i == self.fc_length - 1:
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
        lob: x[:, :-4]
        acc: x[:, -4:]
        """
        lob_data = x[:, :-4].view(-1, self.obs_shape[0], self.obs_shape[1])
        acc_data = x[:, -4:]

        # 标准化 距离收盘秒数
        acc_data[:, 0] -= MEAN_SEC_BEFORE_CLOSE
        acc_data[:, 0] /= STD_SEC_BEFORE_CLOSE

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

        # 将输出重塑为(batch_size, action_dim, n_atoms)并应用softmax    
        x = x.view(-1, self.net_arch[-1], self.n_atoms)
        probs = F.softmax(x, dim=2)
        return probs

    def get_q_values(self, probs):
        """
        将概率分布转换为Q值
        """
        if self.support.device != probs.device:
            self.support = self.support.to(probs.device)

        q_values = torch.sum(probs * self.support.expand_as(probs), dim=2)
        return q_values

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
        train_buffer_class,
        train_title,
        action_dim,
        features_dim,
        features_extractor_class,
        features_extractor_kwargs=None,
        use_noisy=True,
        n_step=1,
        net_arch=None,

        n_atoms=51,
        v_min=-10,
        v_max=10,
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
            v_min: 值分布的最小值
            v_max: 值分布的最大值

            基类参数
                buffer_size: 经验回放池大小
                train_buffer_class: 经验回放池类,必须提供
                use_noisy: 是否使用噪声网络
                n_step: 多步学习的步数
                train_title: 训练标题
                action_dim: 动作空间维度 
                features_dim: 特征维度
                features_extractor_class: 特征提取器类,必须提供
                features_extractor_kwargs=None: 特征提取器参数,可选
                net_arch=None: 网络架构参数,默认为一层mlp, 输入/输出维度为features_dim, action_dim
                    [action_dim] / dict(pi=[action_dim], vf=[action_dim]) 等价
        """
        super().__init__(buffer_size, train_buffer_class, use_noisy, n_step, train_title, action_dim, features_dim, features_extractor_class, features_extractor_kwargs, net_arch)

        self.obs_shape = obs_shape
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # 预计算offset并移动到设备
        self.offset = None  # 初始化为None，第一次使用时再创建
        self.last_batch_size = None  # 用于检查batch size是否改变
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.need_epsilon = True
        self.target_update = target_update
        self.count = 0

        # 初始化网络
        self.build_model()
        self.optimizer = torch.optim.AdamW(
            self.models['q_net'].parameters(),
            lr=learning_rate,
        )
    ############################################################
    # 需要重写的函数
    #     _build_model: 构建模型
    #     _take_action(self, state): 根据状态选择动作
    #     _update(self, states, actions, rewards, next_states, dones, data_type, weights=None, n_step_rewards=None, n_step_next_states=None, n_step_dones=None): 更新模型
    #     sync_update_net_params_in_agent: 同步更新模型参数
    #     get_params_to_send: 获取需要上传的参数
    ############################################################
    def _build_model(self):
        q_net = c51_network(self.obs_shape, self.features_extractor_class,
                          self.features_extractor_kwargs, self.features_dim,
                          self.net_arch, self.n_atoms, self.v_min, self.v_max)
        target_q_net = c51_network(self.obs_shape, self.features_extractor_class,
                                 self.features_extractor_kwargs, self.features_dim,
                                 self.net_arch, self.n_atoms, self.v_min, self.v_max)

        self.models = {'q_net': q_net, 'target_q_net': target_q_net}
        self.models['q_net'].train()
        # 设置为eval模式
        self.models['target_q_net'].eval()

    def _take_action(self, state):
        self.models['q_net'].eval()
        with torch.no_grad():
            # 使用分布期望值选择动作
            probs = self.models['q_net'](state)
            q_values = self.models['q_net'].get_q_values(probs)
            action = q_values.argmax().item()
        self.models['q_net'].train()
        return action

    def _update(self, states, actions, rewards, next_states, dones, data_type, weights=None, n_step_rewards=None, n_step_next_states=None, n_step_dones=None):
        # 分布式Q学习更新
        batch_size = states.shape[0]

        # 只在batch_size改变或首次使用时重新计算offset
        if self.offset is None or batch_size != self.last_batch_size:
            self.offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).long().to(states.device)
            self.offset = self.offset.unsqueeze(1).expand(batch_size, self.n_atoms)
            self.last_batch_size = batch_size

        # 获取当前网络的分布预测
        current_probs = self.models['q_net'](states)

        with torch.no_grad():
            # 1. 计算单步目标分布
            next_probs_1 = self.models['target_q_net'](next_states)
            next_q_values_1 = self.models['q_net'].get_q_values(next_probs_1)
            next_actions_1 = next_q_values_1.argmax(1)
            target_probs_1 = next_probs_1[range(batch_size), next_actions_1]

            # 确保support在正确的设备上
            if self.support.device != rewards.device:
                self.support = self.support.to(rewards.device)

            # 计算单步目标分布
            tz1 = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * self.support
            tz1 = tz1.clamp(self.v_min, self.v_max)
            b1 = (tz1 - self.v_min) / self.delta_z
            l1 = b1.floor().long()
            u1 = b1.ceil().long()
            
            # 处理上下界相等的情况
            l1[(u1 > 0) * (l1 == u1)] -= 1
            u1[(l1 < (self.n_atoms - 1)) * (l1 == u1)] += 1

            # 2. 如果有n步数据，计算n步目标分布
            if n_step_rewards is not None:
                next_probs_n = self.models['target_q_net'](n_step_next_states)
                next_q_values_n = self.models['q_net'].get_q_values(next_probs_n)
                next_actions_n = next_q_values_n.argmax(1)
                target_probs_n = next_probs_n[range(batch_size), next_actions_n]

                # 计算n步目标分布
                tzn = n_step_rewards.unsqueeze(1) + (1 - n_step_dones.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support
                tzn = tzn.clamp(self.v_min, self.v_max)
                bn = (tzn - self.v_min) / self.delta_z
                ln = bn.floor().long()
                un = bn.ceil().long()
                
                # 处理上下界相等的情况
                ln[(un > 0) * (ln == un)] -= 1
                un[(ln < (self.n_atoms - 1)) * (ln == un)] += 1

                # 3. 计算两个目标分布
                target_dist_1 = torch.zeros_like(target_probs_1)
                target_dist_n = torch.zeros_like(target_probs_n)

                # 计算单步投影概率
                target_dist_1.view(-1).index_add_(
                    0, (l1 + self.offset).view(-1),
                    (target_probs_1 * (u1.float() - b1)).view(-1)
                )
                target_dist_1.view(-1).index_add_(
                    0, (u1 + self.offset).view(-1),
                    (target_probs_1 * (b1 - l1.float())).view(-1)
                )

                # 计算n步投影概率
                target_dist_n.view(-1).index_add_(
                    0, (ln + self.offset).view(-1),
                    (target_probs_n * (un.float() - bn)).view(-1)
                )
                target_dist_n.view(-1).index_add_(
                    0, (un + self.offset).view(-1),
                    (target_probs_n * (bn - ln.float())).view(-1)
                )

                # 4. 动态计算权重
                actions = actions.squeeze(-1)
                current_dist = current_probs[range(batch_size), actions]
                
                # 计算单步和n步的KL散度
                kl_div_1 = -(target_dist_1 * torch.log(current_dist + 1e-8)).sum(1)
                kl_div_n = -(target_dist_n * torch.log(current_dist + 1e-8)).sum(1)
                
                # 使用softmax计算动态权重
                kl_divs = torch.stack([kl_div_1, kl_div_n], dim=1)
                weights_soft = F.softmax(-kl_divs, dim=1)  # 负号使得误差越小权重越大
                
                # 最终目标分布是加权平均
                target_dist = (weights_soft[:, 0].unsqueeze(1) * target_dist_1 + 
                            weights_soft[:, 1].unsqueeze(1) * target_dist_n)
            else:
                # 如果没有n步数据，只使用单步目标
                target_dist = torch.zeros_like(target_probs_1)
                target_dist.view(-1).index_add_(
                    0, (l1 + self.offset).view(-1),
                    (target_probs_1 * (u1.float() - b1)).view(-1)
                )
                target_dist.view(-1).index_add_(
                    0, (u1 + self.offset).view(-1),
                    (target_probs_1 * (b1 - l1.float())).view(-1)
                )
        

        # 计算KL散度损失
        # 将actions压缩为一维
        actions = actions.squeeze(-1)  # 将shape从[256, 1]变为[256]
        current_dist = current_probs[range(batch_size), actions]

        # 计算KL散度 (对所有情况都使用相同的KL散度公式)
        kl_div = -(target_dist * torch.log(current_dist + 1e-8)).sum(1)

        td_error_for_update = None
        if weights is not None:
            # PER: 使用重要性采样权重加权KL散度
            loss = (weights * kl_div).mean()
            # 用于更新优先级的TD误差
            td_error_for_update = kl_div.detach().cpu().numpy()
            td_error = td_error_for_update.mean()
        else:
            # 普通回放池: 直接平均KL散度
            loss = kl_div.mean()
            td_error = kl_div.mean().item()

        # tracker 记录
        self.track_error(td_error, loss.item(), data_type)

        # 检查是否有nan/inf值
        if (torch.isnan(loss) or torch.isinf(loss) or 
            np.isnan(td_error) or np.isinf(td_error)):
            # 保存batch数据到pickle文件
            batch_data = {
                'states': states.cpu().numpy(),
                'actions': actions.cpu().numpy(),
                'rewards': rewards.cpu().numpy(), 
                'next_states': next_states.cpu().numpy(),
                'dones': dones.cpu().numpy(),
                'current_dist': current_dist.detach().cpu().numpy(),
                'target_dist': target_dist.detach().cpu().numpy(),
                'weights': weights.cpu().numpy() if weights is not None else None,
                'loss': loss.item(),
                'td_error': td_error
            }
            
            # 保存到文件
            with open(f'nan_batch_{data_type}.pkl', 'wb') as f:
                pickle.dump(batch_data, f)
                
            error_msg = f'检测到NaN/Inf值,loss:{loss.item()},td_error:{td_error},batch数据已保存到nan_batch_{data_type}.pkl'
            raise ValueError(error_msg)

        if data_type == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.count > 0 and self.count % self.target_update == 0:
                self.sync_update_net_params_in_agent()
            self.count += 1

        return td_error_for_update

    def sync_update_net_params_in_agent(self):
        self.models['target_q_net'].load_state_dict(self.models['q_net'].state_dict())

    def get_params_to_send(self):
        return self.models['q_net'].state_dict()