from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size):
        # 一次性采样,避免多次调用random.sample
        transitions = random.sample(self.buffer, batch_size)
        # 使用zip和numpy的一次性转换,避免多次转换
        return tuple(np.array(x, dtype=dtype) for x, dtype in 
                    zip(zip(*transitions), 
                        [np.float32, np.int64, np.float32, np.float32, np.float32]))

    def get(self, batch_size):
        # 获取实际可获取的数量
        n = min(batch_size, len(self.buffer))
        # 一次性获取所有数据并转换为numpy数组
        transitions = [self.buffer.popleft() for _ in range(n)]
        
        # 使用zip和numpy的一次性转换，避免多次转换
        return tuple(np.array(x, dtype=dtype) for x, dtype in 
                    zip(zip(*transitions), 
                        [np.float32, np.int64, np.float32, np.float32, np.float32]))

    def size(self): 
        return len(self.buffer)

    def reset(self):
        self.buffer.clear()

class ReplayBufferWaitClose(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.buffer_temp = []

    def add(self, state, action, reward, next_state, done): 
        # 先加入temp
        # 如果done 则将temp加入buffer
        self.buffer_temp.append([state, action, reward, next_state, done])

    def update_reward(self, reward=None):
        if not None is reward:
            # 修改 buffer_temp 中的所有 reawrd 为 reward
            for i in range(len(self.buffer_temp)):
                self.buffer_temp[i][2] = reward
        # 移动到buffer
        self.buffer.extend(self.buffer_temp)
        self.buffer_temp = []

    def reset(self):
        super().reset()
        self.buffer_temp = []

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
