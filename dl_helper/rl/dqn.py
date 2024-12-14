import os, time
import matplotlib.pyplot as plt
from py_ext.tool import log, get_log_file

import pickle   

import torch
import torch.nn.functional as F
import numpy as np

from dl_helper.rl.tracker import Tracker
from dl_helper.rl.rl_env.lob_env import data_producer, LOB_trade_env, ILLEGAL_REWARD
from dl_helper.rl.base import BaseAgent, OffPolicyAgent
from dl_helper.rl.socket_base import send_val_test_data
from dl_helper.rl.rl_utils import ReplayBuffer, ReplayBufferWaitClose
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

class dqn_network_0(torch.nn.Module):
    def __init__(self, obs_shape, features_extractor_class, features_extractor_kwargs, features_dim, net_arch, dqn_type):
        """
        features_dim: features_extractor_class输出维度  + 3(symbol_id + 持仓 + 未实现收益率)
        """
        super().__init__()
        self.obs_shape = obs_shape
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
        """
        先将x分成两个tensor
        lob: x[:, :-3]
        acc: x[:, -3:]
        """
        # -> batchsize, 100， 130
        lob_data = x[:, :-3].view(-1, self.obs_shape[0], self.obs_shape[1])
        acc_data = x[:, -3:]

        feature = self.features_extractor(lob_data)# -> batchsize, 3
        # concat acc
        feature = torch.cat([feature, acc_data], dim=1)

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

class dqn_network(torch.nn.Module):
    def __init__(self, obs_shape, features_extractor_class, features_extractor_kwargs, features_dim, net_arch, dqn_type):
        """
        features_dim: features_extractor_class输出维度  + 3(symbol_id + 持仓 + 未实现收益率)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.features_extractor = features_extractor_class(**features_extractor_kwargs)
        
        # 添加Batch Normalization
        self.bn = torch.nn.BatchNorm1d(features_dim)
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

class DQN(OffPolicyAgent):

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

        dqn_type=VANILLA_DQN,
    ):
        """
        DQN
        
        Args:
            obs_shape: 观测空间维度
            learning_rate: 学习率
            gamma: TD误差折扣因子
            epsilon: epsilon-greedy策略参数
            target_update: 目标网络更新间隔
            dqn_type=VANILLA_DQN: DQN类型

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
        assert dqn_type in DQN_TYPES, f'dqn_type 必须是 {DQN_TYPES} 中的一个, 当前为 {dqn_type}'

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.need_epsilon = True
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type

        # 初始化网络
        self.build_model()
        self.optimizer = torch.optim.Adam(self.models['q_net'].parameters(),lr=learning_rate)
    ############################################################
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

def run_val_test(val_test, rank, dqn, env):
    """根据val_test类型和rank决定是否执行及执行类型
    
    Args:
        val_test: 验证/测试类型 'val'/'test'/'all'
        rank: 进程rank
        dqn: DQN模型
        env: 环境
    """
    should_run = False
    test_type = None
    
    if val_test in ['val', 'test']:
        # val或test模式只在rank0执行
        if rank == 0:
            should_run = True
            test_type = val_test
    elif val_test == 'all':
        # all模式下rank0执行val,rank1执行test
        if rank == 0:
            should_run = True
            test_type = 'val'
        elif rank == 1:
            should_run = True
            test_type = 'test'
            
    if should_run:
        i = 0
        while True:
            log(f'{rank} {i} test {test_type} dataset...')
            i += 1

            # 同步最新参数
            # 拉取服务器的最新参数并更新
            dqn.update_params_from_server(env)

            log(f'{rank} {i} wait metrics for {test_type}')
            t = time.time()
            metrics = dqn.val_test(env, data_type=test_type)
            log(f'{rank} {i} metrics: {metrics}, cost: {time.time() - t:.2f}s')
            # 发送验证结果给服务器
            send_val_test_data(dqn.train_title, test_type, metrics)

def run_client_learning_device(rank, num_processes, data_folder, dqn, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step, simple_test=False, val_test='', enable_profiling=False):
    # 根据环境获取对应设备
    _run_device = get_gpu_info()
    if _run_device == 'TPU':  # 如果是TPU环境
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    elif _run_device in ['T4x2', 'P100']:
        device = torch.device(f'cuda:{rank}' if num_processes > 1 else 'cuda')
    else:
        device = torch.device('cpu')
    log(f'rank: {rank}, num_processes: {num_processes} device: {device}, run...')
    
    # 移动到设备
    dqn.to(device)
    dqn.tracker.set_rank(rank)
    
    # 初始化环境
    dp = data_producer(data_folder=data_folder, simple_test=simple_test, file_num=15 if enable_profiling else 0)
    env = LOB_trade_env(data_producer=dp)

    # 开始训练
    if val_test:
        # 验证/测试
        run_val_test(val_test, rank, dqn, env)
    else:
        log(f'{rank} learn...')
        dqn.learn(env, 5 if enable_profiling else num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step)

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
