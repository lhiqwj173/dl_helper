from tqdm import tqdm
import numpy as np
import torch
import collections
import random, pickle

def _get_n_step_info(n_step_buffer, gamma):
    """计算n步return"""
    reward, next_state, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]
        reward = r + gamma * reward * (1 - d)
        next_state = n_s if d else next_state
        done = d

    return reward, next_state, done

class ReplayBuffer:
    def __init__(self, capacity, n_step=1, gamma=0.99):
        self.buffer = collections.deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        # n步缓存
        self.n_step_buffer = collections.deque(maxlen=n_step) if n_step > 1 else None
        # 预定义数据类型
        self.dtypes = [np.float32, np.int64, np.float32, np.float32, np.float32]

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self.n_step > 1:
            self.n_step_buffer.append(transition)

            # 只有当n步缓存满了才添加到主缓存
            if len(self.n_step_buffer) == self.n_step:
                # 计算n步return
                n_reward, n_next_state, n_done = _get_n_step_info(self.n_step_buffer, self.gamma)
                state, action, reward, next_state, done = self.n_step_buffer[0]
                
                # 存储原始transition和n步信息
                self.buffer.append((
                    state, action, reward, next_state, done,  # 原始数据
                    n_reward, n_next_state, n_done  # n步数据
                ))
        else:
            self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]

        # 分离原始数据和n步数据
        states = np.array([t[0] for t in transitions], dtype=self.dtypes[0])
        actions = np.array([t[1] for t in transitions], dtype=self.dtypes[1])
        rewards = np.array([t[2] for t in transitions], dtype=self.dtypes[2])
        next_states = np.array([t[3] for t in transitions], dtype=self.dtypes[3])
        dones = np.array([t[4] for t in transitions], dtype=self.dtypes[4])

        n_rewards = None
        n_next_states = None
        n_dones = None
        # 如果使用n步学习，添加n步数据
        if self.n_step > 1:
            n_rewards = np.array([t[5] for t in transitions], dtype=self.dtypes[2])
            n_next_states = np.array([t[6] for t in transitions], dtype=self.dtypes[3])
            n_dones = np.array([t[7] for t in transitions], dtype=self.dtypes[4])

        return (states, actions, rewards, next_states, dones,
                n_rewards, n_next_states, n_dones)

    def get(self, batch_size):
        # 只针对1步 进行验证
        n = min(batch_size, len(self.buffer))
        # 预分配列表空间
        transitions = []
        transitions.extend(self.buffer.popleft() for _ in range(n))
        # 预分配numpy数组
        return tuple(np.array([t[i] for t in transitions], dtype=self.dtypes[i])
                    for i in range(5))

    def size(self):
        return len(self.buffer)

    def clear_n_step_buffer(self):
        if self.n_step > 1:
            self.n_step_buffer.clear()

    def reset(self):
        self.buffer.clear()
        self.clear_n_step_buffer()

class ReplayBufferWaitClose(ReplayBuffer):
    def __init__(self, capacity, n_step=1, gamma=0.99):
        super().__init__(capacity, n_step, gamma)
        # 使用deque替代list,提高append和extend性能
        self.buffer_temp = collections.deque()

    def add(self, state, action, reward, next_state, done):
        # 使用元组存储,减少内存使用
        self.buffer_temp.append((state, action, reward, next_state, done))

    def update_reward(self, reward=None):
        if reward is not None:
            # 使用列表推导式替代循环,性能更好
            self.buffer_temp = collections.deque(
                (t[0], t[1], reward, t[3], t[4]) for t in self.buffer_temp
            )

        if self.n_step > 1:
            # 使用父类add方法
            for t in self.buffer_temp:
                super().add(t[0], t[1], t[2], t[3], t[4])
            # 清空n步缓冲区
            self.clear_n_step_buffer()
        else:
            # 批量添加到buffer， 效率更高
            self.buffer.extend(self.buffer_temp)

        # 清空临时缓冲区
        self.buffer_temp.clear()

    def reset(self):
        super().reset()
        self.buffer_temp.clear()

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
        if not isinstance(data, (tuple, list)) or len(data) not in [5, 8]:
            pickle.dump((priority, data), open("error_SumTree_add_data.pkl", "wb"))
            raise ValueError(f"Invalid data format: expected tuple/list of length 5 or 8, got {data}({type(data)})")

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
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            # 如果到达叶子节点
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break

            # 否则继续向下搜索
            if v <= self.tree[left_child]:
                parent_idx = left_child
            else:
                v -= self.tree[left_child]
                parent_idx = right_child

        data_idx = leaf_idx - self.capacity + 1
        if self.data[data_idx] is None:
            raise ValueError("Trying to access empty data slot")
            
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
        max_priority=1.0,
        n_step=1,
        gamma=0.99
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self._beta = beta
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.max_priority = max_priority
        self.epsilon = 1e-6  # 避免零优先级

        self.n_step = n_step
        self.gamma = gamma
        # n步缓存
        self.n_step_buffer = collections.deque(maxlen=n_step) if n_step > 1 else None

        # 预定义数据类型
        self.dtypes = [np.float32, np.int64, np.float32, np.float32, np.float32]    

    def add(self, state, action, reward, next_state, done):
        """
        添加新的经验
        默认给最大优先级
        """
        transition = (state, action, reward, next_state, done)
        if self.n_step > 1:
            self.n_step_buffer.append(transition)

            # 只有当n步缓存满了才添加到主缓存
            if len(self.n_step_buffer) == self.n_step:
                # 计算n步return
                n_reward, n_next_state, n_done = _get_n_step_info(self.n_step_buffer, self.gamma)
                state, action, reward, next_state, done = self.n_step_buffer[0]
                
                # 存储原始transition和n步信息
                experience = (
                    state, action, reward, next_state, done,  # 原始数据
                    n_reward, n_next_state, n_done  # n步数据
                )
                max_priority = self.max_priority if not self.tree.is_full else self.tree.tree[0]
                self.tree.add(max_priority, experience)
        else:
            max_priority = self.max_priority if not self.tree.is_full else self.tree.tree[0]
            self.tree.add(max_priority, transition)

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
            
            batch.append(data)
            batch_indices.append(idx)
            batch_priorities.append(priority)

        # 计算重要性采样权重
        total_priority = self.tree.total_priority()
        # 确保总优先级不为零
        if total_priority == 0:
            probabilities = np.ones_like(batch_priorities) / len(batch_priorities)
        else:
            probabilities = batch_priorities / total_priority

        # 添加数值稳定性检查
        weights = np.zeros_like(probabilities)
        valid_probs = probabilities > 0
        if np.any(valid_probs):
            weights[valid_probs] = np.power(self.tree.capacity * probabilities[valid_probs], -self.beta)
            # 避免除以零，使用 np.maximum 确保分母不为零
            weights /= np.maximum(weights.max(), 1e-8)
        else:
            weights = np.ones_like(probabilities)  # 如果所有概率都为零，返回均匀权重

        # batch 内转为numpy数组
        try:
            # 原始数据
            states = np.array([t[0] for t in batch], dtype=self.dtypes[0])
            actions = np.array([t[1] for t in batch], dtype=self.dtypes[1])
            rewards = np.array([t[2] for t in batch], dtype=self.dtypes[2])
            next_states = np.array([t[3] for t in batch], dtype=self.dtypes[3])
            dones = np.array([t[4] for t in batch], dtype=self.dtypes[4])

            # n步数据
            n_rewards = None
            n_next_states = None
            n_dones = None
            if self.n_step > 1:
                n_rewards = np.array([t[5] for t in batch], dtype=self.dtypes[2])
                n_next_states = np.array([t[6] for t in batch], dtype=self.dtypes[3])
                n_dones = np.array([t[7] for t in batch], dtype=self.dtypes[4])

            # 合并数据  
            batch = (states, actions, rewards, next_states, dones,
                    n_rewards, n_next_states, n_dones)

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
            # 限制优先级范围,防止过大
            clipped_errors = np.minimum(priority, self.max_priority)
            # 确保优先级非零
            priority = np.power(clipped_errors + self.epsilon, self.alpha)
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

    def clear_n_step_buffer(self):
        if self.n_step > 1:
            self.n_step_buffer.clear()

    def reset(self):
        """
        重置缓冲区
        """
        self.tree = SumTree(self.capacity)
        self.beta = self._beta   # 重置 beta 到初始值
        self.clear_n_step_buffer()

class PrioritizedReplayBufferWaitClose(PrioritizedReplayBuffer):
    """
    支持延迟更新 reward 的优先级经验回放
    """
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, 
                 beta_increment_per_sampling=0.001, max_priority=1.0, n_step=1, gamma=0.99):
        super().__init__(capacity, alpha, beta, beta_increment_per_sampling, max_priority, n_step, gamma)
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

        # 清空n步缓冲区
        self.clear_n_step_buffer()
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

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                
def update_model_params(model, new_params, tau=0.005):
    """使用新参数软更新模型参数"""
    params = model.state_dict()
    for name, param in params.items():
        if name in new_params:
            # 确保新参数在同一设备上
            new_param = new_params[name]
            if new_param.device != param.device:
                new_param = new_param.to(param.device)
            param.copy_((1 - tau) * param + tau * new_param)
    return model

def calculate_importance_loss(loss: torch.Tensor) -> float:
    """计算更新的重要性权重
    
    Args:
        loss: 当前批次的损失值
    
    Returns:
        float: 重要性权重
    """
    # 基于loss值计算重要性
    importance = float(loss.item())
    # 归一化到[0, 1]范围
    importance = np.clip(importance / 10.0, 0, 1)
    return importance