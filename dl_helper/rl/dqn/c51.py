import torch
import torch.nn.functional as F
import numpy as np
import pickle
import collections

from dl_helper.rl.base import BaseAgent, OffPolicyAgent
from dl_helper.rl.dqn.dqn import ReplayBuffer  

class C51ReplayBuffer(ReplayBuffer):
    """C51算法的经验回放池"""
    def __init__(self, capacity, n_atoms=51):
        super().__init__(capacity)
        self.n_atoms = n_atoms
        # 更新数据类型以支持分布式奖励
        self.dtypes = [np.float32, np.int64, np.float32, np.float32, np.float32, np.float32]

    def add(self, state, action, reward, next_state, done, prob_dist=None):
        """
        添加经验到缓冲区
        prob_dist: 当前状态的奖励分布 (shape: n_atoms)
        """
        if prob_dist is None:
            # 如果没有提供分布，创建均匀分布
            prob_dist = np.ones(self.n_atoms) / self.n_atoms
        self.buffer.append((state, action, reward, next_state, done, prob_dist))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        return tuple(np.array([t[i] for t in transitions], dtype=self.dtypes[i]) 
                    for i in range(6))  # 注意这里改为6个元素

    def get(self, batch_size):
        n = min(batch_size, len(self.buffer))
        # 预分配列表空间
        transitions = []
        transitions.extend(self.buffer.popleft() for _ in range(n))
        # 预分配numpy数组
        return tuple(np.array([t[i] for t in transitions], dtype=self.dtypes[i])
                    for i in range(6))

class C51ReplayBufferWaitClose(C51ReplayBuffer):
    """支持延迟更新reward的C51回放池"""
    def __init__(self, capacity, n_atoms=51):
        super().__init__(capacity, n_atoms)
        self.buffer_temp = collections.deque()

    def add(self, state, action, reward, next_state, done, prob_dist=None):
        if prob_dist is None:
            prob_dist = np.ones(self.n_atoms) / self.n_atoms
        self.buffer_temp.append((state, action, reward, next_state, done, prob_dist))

    def update_reward(self, reward=None):
        if reward is not None:
            # 更新所有临时经验的reward，保持分布不变
            self.buffer_temp = collections.deque(
                (t[0], t[1], reward, t[3], t[4], t[5]) for t in self.buffer_temp
            )
        # 批量添加到buffer
        self.buffer.extend(self.buffer_temp)
        self.buffer_temp.clear()

    def reset(self):
        super().reset()
        self.buffer_temp.clear()

class c51_network(torch.nn.Module):
    def __init__(self, obs_shape, features_extractor_class, features_extractor_kwargs, features_dim, net_arch, n_atoms, v_min, v_max):
        """
        features_dim: features_extractor_class输出维度  + 3(symbol_id + 持仓 + 未实现收益率)
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

        # 将输出重塑为(batch_size, action_dim, n_atoms)并应用softmax    
        x = x.view(-1, self.net_arch[-1], self.n_atoms)
        probs = F.softmax(x, dim=2)
        return probs

    def get_q_values(self, probs):
        """
        将概率分布转换为Q值
        """
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
        train_title,
        action_dim,
        features_dim,
        features_extractor_class,
        features_extractor_kwargs=None,
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
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
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
    #     build_model: 构建模型
    #     take_action(self, state): 根据状态选择动作
    #     _update(self, states, actions, rewards, next_states, dones, data_type): 更新模型
    #     sync_update_net_params_in_agent: 同步更新模型参数
    #     get_params_to_send: 获取需要上传的参数
    ############################################################
    def init_replay_buffer(self):
        return C51ReplayBufferWaitClose(self.buffer_size)

    def eval(self):
        self.need_epsilon = False

    def train(self):
        self.need_epsilon = True

    def build_model(self):
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

    def take_action(self, state):
        if self.need_epsilon and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            
            self.models['q_net'].eval()
            with torch.no_grad():
                # 使用分布期望值选择动作
                probs = self.models['q_net'](state)
                q_values = self.models['q_net'].get_q_values(probs)
                action = q_values.argmax().item()
            self.models['q_net'].train()
        return action

    def _update(self, states, actions, rewards, next_states, dones, data_type):
        # 分布式Q学习更新
        batch_size = states.shape[0]

        # 获取当前网络的分布预测
        current_probs = self.models['q_net'](states)

        with torch.no_grad():
            # 获取目标网络的分布预测
            next_probs = self.models['target_q_net'](next_states)
            
            # 使用当前网络选择动作
            next_q_values = self.models['q_net'].get_q_values(next_probs)
            next_actions = next_q_values.argmax(1)
            
            # 获取目标分布
            target_probs = next_probs[range(batch_size), next_actions]
            
            # 计算投影的目标分布
            tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * self.support
            tz = tz.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # 处理上下界相等的情况
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) * (l == u)] += 1
            
            # 计算投影概率
            target_dist = torch.zeros_like(target_probs)
            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).long()
            offset = offset.unsqueeze(1).expand(batch_size, self.n_atoms)
            
            target_dist.view(-1).index_add_(
                0, (l + offset).view(-1),
                (target_probs * (u.float() - b)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (u + offset).view(-1),
                (target_probs * (b - l.float())).view(-1)
            )
        
        # 计算KL散度损失
        current_dist = current_probs[range(batch_size), actions]
        loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(1).mean()
        
        # 计算TD误差
        td_error = torch.abs(target_dist - current_dist).mean().item()
        
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

    def sync_update_net_params_in_agent(self):
        self.models['target_q_net'].load_state_dict(self.models['q_net'].state_dict())

    def get_params_to_send(self):
        return self.models['q_net'].state_dict()

    def learn(self, env, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step):
        """
        训练

        Args:
            env: 环境
            num_episodes: 训练回合数
            minimal_size: 最小训练次数
            batch_size: 批次大小
            sync_interval_learn_step:   同步参数间隔，会询问是否需要验证/测试
            learn_interval_step:        学习更新间隔                     
        """
        # 学习步数
        learn_step = 0

        # 拉取服务器的最新参数并更新
        self.update_params_from_server(env)

        # 学习是否开始
        for i in range(num_episodes):
            self.msg_head = f'[{self.device}][e{i}]'

            # 回合的评价指标
            state, info = env.reset()
            done = False
            step = 0
            while not done:
                step += 1

                # # 测试用
                # # 检查是否有nan/inf值
                # if np.argwhere(np.isnan(state)).any() or np.argwhere(np.isinf(state)).any():
                #     raise ValueError(f'检测到NaN/Inf值,state: {state}')

                # 动作
                action = self.take_action(state)
                # 更新跟踪器 动作
                self.tracker.update_action(action)

                # 环境交互
                next_state, reward, done1, done2, info = env.step(action)
                done = done1 or done2

                # 添加到回放池
                self.replay_buffer.add(state, action, reward, next_state, done)

                # 如果 交易close 则需要回溯更新所有 reward 为最终close时的reward
                if info.get('close', False):
                    self.replay_buffer.update_reward(reward if reward!=ILLEGAL_REWARD else None)

                    # 更新跟踪器 奖励
                    self.tracker.update_reward(reward)

                    # 更新跟踪器 非法/win/loss
                    if info['act_criteria'] == -1:
                        self.tracker.update_illegal()
                    elif info['act_criteria'] == 0:
                        self.tracker.update_win()
                    else:
                        self.tracker.update_loss_count()

                    # 更新评价指标
                    for k, v in info.items():
                        if k not in ['close', 'date_done', 'act_criteria']:
                            self.tracker.update_extra_metrics(k, v)
                
                # 更新跟踪器 日期文件完成, 需要更新
                if info.get('date_done', False):
                    self.tracker.day_end()

                # 更新状态
                state = next_state

                # 更新网络
                if self.replay_buffer.size() > minimal_size and step % learn_interval_step == 0:
                    # 学习经验
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

                    #################################
                    # 服务器通讯
                    #################################
                    learn_step += 1
                    need_train_back = False
                    if learn_step % sync_interval_learn_step == 0:
                        log(f'{self.msg_head} {learn_step} sync params')
                        # 同步最新参数
                        # 推送参数更新
                        self.push_params_to_server()
                        # 拉取服务器的最新参数并更新
                        self.update_params_from_server(env)

                        # 验证/测试
                        # 询问 服务器 是否需要 验证/测试
                        # 返回: 'val' / 'test' / 'no'
                        need_val_test_res = check_need_val_test(self.train_title)
                        if need_val_test_res != 'no':
                            test_type = need_val_test_res
                            need_train_back = True  
                            log(f'{self.msg_head} wait metrics for {test_type}')
                            t = time.time()
                            metrics = self.val_test(env, data_type=test_type)
                            log(f'{self.msg_head} metrics: {metrics}, cost: {time.time() - t:.2f}s')
                            # 发送验证结果给服务器
                            send_val_test_data(self.train_title, test_type, metrics)

                    #################################
                    # 服务器通讯
                    #################################
                    # 切换回训练模式
                    if need_train_back:
                        self.train()
                        env.set_data_type('train')
                        state, info = env.reset()
                        done = False
