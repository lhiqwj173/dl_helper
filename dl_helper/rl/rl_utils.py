from tqdm import tqdm
import numpy as np
import torch
import collections
import random, pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        # 预定义数据类型
        self.dtypes = [np.float32, np.int64, np.float32, np.float32, np.float32]

    def add(self, state, action, reward, next_state, done):
        # 直接使用元组存储,减少列表转换开销
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 使用numpy的random.choice替代random.sample,性能更好
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        # 预分配numpy数组
        return tuple(np.array([t[i] for t in transitions], dtype=self.dtypes[i]) 
                    for i in range(5))

    def get(self, batch_size):
        n = min(batch_size, len(self.buffer))
        # 预分配列表空间
        transitions = []
        transitions.extend(self.buffer.popleft() for _ in range(n))
        # 预分配numpy数组
        return tuple(np.array([t[i] for t in transitions], dtype=self.dtypes[i])
                    for i in range(5))

    def size(self):
        return len(self.buffer)

    def reset(self):
        self.buffer.clear()

class ReplayBufferWaitClose(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
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
        # 批量添加到buffer
        self.buffer.extend(self.buffer_temp)
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
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1


        # 添加边界检查
        if data_idx < 0 or data_idx >= self.capacity:
            raise ValueError(f"Invalid data_idx {data_idx}, capacity is {self.capacity}")

        # 添加数据存在性检查
        if not self.is_full and data_idx >= self.data_pointer:
            raise ValueError(f"Trying to access uninitialized data at index {data_idx}")

        # 添加数据验证
        data = self.data[data_idx]
        if data is None or (not isinstance(data, (tuple, list)) or len(data) != 5):
            # 错误数据: (8190, 0.0, None)
            # leaf_idx 8190, 
            # self.tree[leaf_idx] 0.0
            # self.data[data_idx] None
            pickle.dump((leaf_idx, self.tree[leaf_idx], self.data[data_idx]), open("error_SumTree_get_leaf.pkl", "wb"))
            raise ValueError(f"Invalid data at index {data_idx}: {data}")

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
