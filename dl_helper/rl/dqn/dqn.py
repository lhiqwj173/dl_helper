import os, time
import matplotlib.pyplot as plt
from py_ext.tool import log, get_log_file

import pickle   

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from dl_helper.rl.tracker import Tracker
from dl_helper.rl.rl_env.lob_env import data_producer, LOB_trade_env, ILLEGAL_REWARD
from dl_helper.rl.base import BaseAgent, OffPolicyAgent, BaseModel
from dl_helper.rl.socket_base import send_val_test_data
from dl_helper.train_param import match_num_processes, get_gpu_info
from dl_helper.trainer import notebook_launcher
from dl_helper.tool import upload_log_file
from py_ext.lzma import compress_folder
from py_ext.alist import alist

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

class dqn_network(BaseModel):
    def __init__(self, features_extractor_class, features_extractor_kwargs, features_dim, net_arch, dqn_type, need_reshape=None):
        """
        features_dim: features_extractor_class输出维度  + extra_features的维度
            log_env: features_extractor_class输出维度 + 4(symbol_id + 持仓 + 未实现收益率 + 距离市场关闭的秒数)
            breakout_env: features_extractor_class输出维度
        """
        super().__init__(features_extractor_class, features_extractor_kwargs, features_dim, need_reshape)

        self.dropout = torch.nn.Dropout(p=0.5)
        
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
            
        self.init_weights()

    def forward(self, x):
        # 特征提取
        feature = self.features_forward(x)

        x = feature
        if self.fc_a_length > 1:
            for i in range(self.fc_a_length - 1):
                x = F.leaky_relu(self.fc_a[i](x))
                x = self.dropout(x)
            x = self.fc_a[-1](x)
        else:
            x = self.fc_a(x)

        if self.fc_v is not None:
            v = self.fc_v(feature)
            x = v + x - x.mean(1).view(-1, 1)

        return x

class dqn_base(OffPolicyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sync_update_net_params_in_agent(self):
        self.models['target_q_net'].load_state_dict(self.models['q_net'].state_dict())

    def get_params_to_send(self):
        return self.models['q_net'].state_dict()

    def get_model_to_sync(self):
        return self.models['q_net']

class DQN(dqn_base):

    def __init__(
        self,
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

        need_reshape=None,
        dqn_type=VANILLA_DQN,
    ):
        """
        DQN
        
        Args:
            learning_rate: 学习率
            gamma: TD误差折扣因子
            epsilon: epsilon-greedy策略参数
            target_update: 目标网络更新间隔
            need_reshape: 需要reshape的维度, 例如(8, 10, 10)
            dqn_type=VANILLA_DQN: DQN类型

            基类参数
                buffer_size: 经验回放池大小
                train_buffer_class: 训练经验回放池类
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
        assert dqn_type in DQN_TYPES, f'dqn_type 必须是 {DQN_TYPES} 中的一个, 当前为 {dqn_type}'

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.need_reshape = need_reshape
        self.dqn_type = dqn_type

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
    #     _update(self, states, actions, rewards, next_states, dones, data_type, n_step_rewards=None, n_step_next_states=None, n_step_dones=None): 更新模型
    #     
    #     X(在dqn_base中已实现) get_model_to_sync: 获取需要同步的模型
    #     X(在dqn_base中已实现) sync_update_net_params_in_agent: 同步更新模型参数
    #     X(在dqn_base中已实现) get_params_to_send: 获取需要上传的参数
    ############################################################

    def _build_model(self):
        q_net = dqn_network(self.features_extractor_class, self.features_extractor_kwargs, self.features_dim, self.net_arch, self.dqn_type, self.need_reshape)
        target_q_net = dqn_network(self.features_extractor_class, self.features_extractor_kwargs, self.features_dim, self.net_arch, self.dqn_type, self.need_reshape)
        self.models = {'q_net': q_net, 'target_q_net': target_q_net}

        self.models['q_net'].train()
        # 设置为eval模式
        self.models['target_q_net'].eval()

    def _take_action(self, state):
        self.models['q_net'].eval()
        with torch.no_grad():
            action = self.models['q_net'](state).argmax().item()
        self.models['q_net'].train()
        return action

    def _update(self, states, actions, rewards, next_states, dones, data_type, weights=None, n_step_rewards=None, n_step_next_states=None, n_step_dones=None):
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
            one_step_q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        
            # 如果提供了n步数据，计算n步目标Q值
            if n_step_rewards is not None and n_step_next_states is not None and n_step_dones is not None:
                if self.dqn_type in [DOUBLE_DQN, DD_DQN]:
                    n_step_max_action = self.models['q_net'](n_step_next_states).max(1)[1].view(-1, 1)
                    n_step_max_next_q_values = self.models['target_q_net'](n_step_next_states).gather(1, n_step_max_action)
                else:
                    n_step_max_next_q_values = self.models['target_q_net'](n_step_next_states).max(1)[0].view(-1, 1)
                n_step_q_targets = n_step_rewards + (self.gamma ** self.n_step) * n_step_max_next_q_values * (1 - n_step_dones)

                # 动态调整单步和n步的权重
                # 使用TD误差的绝对值来调整权重：误差越大，越倾向于使用另一个估计
                with torch.no_grad():
                    one_step_td_error = torch.abs(q_values - one_step_q_targets)
                    n_step_td_error = torch.abs(q_values - n_step_q_targets)
                    
                    # 使用softmax计算权重
                    td_errors = torch.cat([one_step_td_error, n_step_td_error], dim=1)
                    weights_soft = F.softmax(-td_errors, dim=1)  # 负号使得误差越小权重越大
                    
                    # 最终目标Q值是加权平均
                    q_targets = (weights_soft[:, 0].view(-1, 1) * one_step_q_targets + 
                            weights_soft[:, 1].view(-1, 1) * n_step_q_targets)
            else:
                # 如果没有n步数据，只使用单步目标
                q_targets = one_step_q_targets

        # 如果提供了权重，使用重要性采样权重, 用于 PER buffer 更新优先级
        td_error_for_update = None
        if weights is not None:
            # 计算损失
            dqn_loss = (weights * (q_values - q_targets).pow(2)).mean()
            # 用于更新优先级的TD误差
            td_error_for_update = torch.abs(q_targets - q_values).detach().cpu().numpy()
            # 用于记录的TD误差
            td_error = td_error_for_update.mean()
        else:
            # 计算损失
            dqn_loss = nn.MSELoss()(q_values, q_targets)
            # 用于记录的TD误差
            td_error = torch.abs(q_targets - q_values).mean().item()
        
        # tracker 记录
        self.track_error(dqn_loss.item(), data_type)

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
                'weights': weights.cpu().numpy() if weights is not None else None,
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

            # 计算/记录重要性损失
            self.compute_importance_loss(dqn_loss)

            # 收集梯度
            self.collect_gradients(self.models['q_net'])

            if self.count > 0 and self.count % self.target_update == 0:
                self.sync_update_net_params_in_agent()
            self.count += 1

        return td_error_for_update


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
