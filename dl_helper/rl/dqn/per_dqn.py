import os, time
import matplotlib.pyplot as plt
from py_ext.tool import log, get_log_file

import pickle   
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from dl_helper.rl.tracker import Tracker
from dl_helper.rl.rl_env.lob_env import data_producer, LOB_trade_env, ILLEGAL_REWARD
from dl_helper.rl.dqn.dqn import DQN, VANILLA_DQN, DOUBLE_DQN, DD_DQN, ReplayBufferWaitClose
from dl_helper.rl.socket_base import get_net_params, send_net_updates, send_val_test_data, check_need_val_test
from dl_helper.train_param import match_num_processes, get_gpu_info
from dl_helper.trainer import notebook_launcher
from dl_helper.tool import upload_log_file
from py_ext.lzma import compress_folder
from py_ext.alist import alist

class SumTree:
    """
    SumTree数据结构，用于高效存储和采样
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # 修改初始化方式，使用 None 而不是 0
        self.data = np.array([None] * capacity, dtype=object)
        self.data_pointer = 0
        self.is_full = False

    def add(self, priority, data):
        """
        添加新的经验
        """
        if not isinstance(data, (tuple, list)) or len(data) != 5:
            pickle.dump((priority, data), open("error_SumTree_add_data.pkl", "wb"))
            raise ValueError(f"Invalid data format: expected tuple/list of length 5, got {data}({type(data)})")

        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.is_full = True
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        """
        更新优先级
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        根据优先级值获取叶子节点
        """
        parent_idx = 0
        
        # 添加输入验证
        if v < 0 or v > self.total_priority():
            raise ValueError(f"Invalid value v: {v}, total priority: {self.total_priority()}")
        
        # 计算有效的数据范围
        valid_start_idx = self.capacity - 1
        valid_end_idx = valid_start_idx + (self.data_pointer if not self.is_full else self.capacity)

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # 如果已经到达叶子层
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            # 如果当前节点在有效范围内，直接返回
            if parent_idx >= valid_start_idx:
                if parent_idx < valid_end_idx:
                    leaf_idx = parent_idx
                    break
                # 如果超出有效范围，选择最后一个有效节点
                leaf_idx = valid_end_idx - 1
                break

            # 正常的选择逻辑
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1

        # 确保 data_idx 在有效范围内
        if data_idx < 0:
            data_idx = 0
        elif data_idx >= self.data_pointer and not self.is_full:
            data_idx = self.data_pointer - 1
        elif data_idx >= self.capacity:
            data_idx = self.capacity - 1

        # 添加数据验证
        if self.data[data_idx] is None:
            raise ValueError(f"Accessed uninitialized data at index {data_idx}")

        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        """
        返回总优先级
        """
        return self.tree[0]

class PrioritizedReplayBuffer:
    """
    优先级经验回放
    """
    def __init__(
        self, 
        capacity=10000, 
        alpha=0.6,  # 决定优先级的指数
        beta=0.4,   # 重要性采样权重的初始值
        beta_increment_per_sampling=0.001,
        max_priority=1.0
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self._beta = beta
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.max_priority = max_priority
        self.epsilon = 1e-6  # 避免零优先级

        # 预定义数据类型
        self.dtypes = [np.float32, np.int64, np.float32, np.float32, np.float32]    

    def add(self, experience):
        """
        添加新的经验
        默认给最大优先级
        """
        # 验证经验数据格式
        if not isinstance(experience, tuple) or len(experience) != 5:
            pickle.dump(experience, open("error_PrioritizedReplayBuffer_add_experience.pkl", "wb"))
            raise ValueError(f"Invalid experience format: {experience}")

        max_priority = self.max_priority if not self.tree.is_full else self.tree.tree[0]
        self.tree.add(max_priority, experience)

    def sample(self, batch_size):
        """
        采样
        """
        if self.size() < batch_size:
            raise ValueError(f"Not enough samples in buffer. Current size: {self.size()}, requested: {batch_size}")

        batch = []
        batch_indices = []
        batch_priorities = []
        segment = self.tree.total_priority() / batch_size

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            value = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)
            
            # 添加类型检查
            if not isinstance(data, (tuple, list)) or len(data) != 5:
                print(f"Invalid data format at index {i}: {data}")
                # 保存错误现场
                error_info = {
                    'problematic_data': data,
                    'batch': batch,
                    'idx': idx,
                    'priority': priority
                }
                pickle.dump(error_info, open("error_debug.pkl", "wb"))
                raise ValueError(f"Expected tuple/list of length 5, got {type(data)} with value {data}")

            batch.append(data)
            batch_indices.append(idx)
            batch_priorities.append(priority)

        # 计算重要性采样权重
        probabilities = batch_priorities / self.tree.total_priority()
        weights = np.power(self.tree.capacity * probabilities, -self.beta)
        weights /= weights.max()

        # batch 内转为numpy数组
        try:
            batch = tuple(np.array([t[i] for t in batch], dtype=self.dtypes[i])
                    for i in range(5))
        except Exception as e:
            print(f"Error converting batch to numpy arrays: {str(e)}")
            print(f"Batch content: {batch}")
            pickle.dump(batch, open("error_batch.pkl", "wb"))
            raise e

        return batch, batch_indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """
        更新优先级
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            # 确保优先级非零
            priority = np.power(priority + self.epsilon, self.alpha)
            self.tree.update(idx, priority)

    def get(self, batch_size):
        # get 返回按顺序的batch, 在优先级经验回放中, 性能较差
        raise "should not use this function, use ReplayBufferWaitClose/ReplayBuffer get function instead"

    def size(self):
        """
        返回当前缓冲区中的经验数量
        """
        if self.tree.is_full:
            return self.capacity
        return self.tree.data_pointer

    def reset(self):
        """
        重置缓冲区
        """
        self.tree = SumTree(self.capacity)
        self.beta = self._beta   # 重置 beta 到初始值

class PrioritizedReplayBufferWaitClose(PrioritizedReplayBuffer):
    """
    支持延迟更新 reward 的优先级经验回放
    """
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, 
                 beta_increment_per_sampling=0.001, max_priority=1.0):
        super().__init__(capacity, alpha, beta, beta_increment_per_sampling, max_priority)
        self.temp_experiences = collections.deque()  # 临时存储经验
        self.temp_indices = collections.deque()      # 临时存储对应的树索引

    def add(self, state, action, reward, next_state, done):
        """
        临时存储经验
        """
        experience = (state, action, reward, next_state, done)
        self.temp_experiences.append(experience)

    def update_reward(self, reward=None):
        """
        更新 reward 并将临时经验转移到主缓冲区
        """
        if not self.temp_experiences:
            return
            
        if reward is not None:
            # 更新所有临时经验的 reward
            updated_experiences = collections.deque()
            for exp in self.temp_experiences:
                state, action, _, next_state, done = exp
                updated_experiences.append((state, action, reward, next_state, done))
            self.temp_experiences = updated_experiences

        # 将所有临时经验添加到主缓冲区
        for experience in self.temp_experiences:
            if not isinstance(experience, (tuple, list)) or len(experience) != 5:
                pickle.dump(experience, open("error_update_reward.pkl", "wb"))
                raise ValueError(f"Invalid experience format before adding to buffer: {experience}")
            super().add(experience)

        # 清空临时缓冲区
        self.temp_experiences.clear()
        self.temp_indices.clear()

    def reset(self):
        """
        重置缓冲区
        """
        super().reset()
        self.temp_experiences.clear()
        self.temp_indices.clear()

class PER_DQN(DQN):

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
        super().__init__(obs_shape, learning_rate, gamma, epsilon, target_update, buffer_size, train_title, action_dim, features_dim, features_extractor_class, features_extractor_kwargs, net_arch, dqn_type)
    
    def init_replay_buffer(self):
        return PrioritizedReplayBufferWaitClose(self.buffer_size)

    def _update(self, states, actions, rewards, next_states, dones, data_type, weights=None):
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
        
        # 用于更新优先级的TD误差
        td_error_for_update = torch.abs(q_targets - q_values).detach().cpu().numpy()

        # 用于记录的TD误差
        td_error = td_error_for_update.mean()
        
        # 如果提供了权重，使用重要性采样权重
        if weights is not None:
            dqn_loss = (weights * (q_values - q_targets).pow(2)).mean()
        else:
            dqn_loss = nn.MSELoss()(q_values, q_targets)

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
        
        return td_error_for_update

    def update(self, transition_dict, data_type='train', weights=None):
        states = torch.from_numpy(transition_dict['states']).to(self.device)
        actions = torch.from_numpy(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.from_numpy(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(transition_dict['next_states']).to(self.device)
        dones = torch.from_numpy(transition_dict['dones']).view(-1, 1).to(self.device)

        if weights is not None:
            weights = torch.from_numpy(weights).to(self.device)

        return self._update(states, actions, rewards, next_states, dones, data_type, weights)

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
                    # 添加调试信息
                    log(f"{self.msg_head} Tree status - is_full: {self.replay_buffer.tree.is_full}, data_pointer: {self.replay_buffer.tree.data_pointer}")
                    log(f"{self.msg_head} Total priority: {self.replay_buffer.tree.total_priority()}")
                    # 采样batch数据
                    (b_s, b_a, b_r, b_ns, b_d), indices, weights = self.replay_buffer.sample(batch_size)

                    # 学习经验
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns,
                        'dones': b_d
                    }
                    td_error_for_update = self.update(transition_dict, weights=weights)

                    # 更新优先级
                    self.replay_buffer.update_priorities(indices, td_error_for_update)

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

