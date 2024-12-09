import os, time
import matplotlib.pyplot as plt
from py_ext.tool import log, get_log_file

import torch
import torch.nn.functional as F
import numpy as np

from dl_helper.rl.dqn_tracker import DQNTracker
from dl_helper.rl.rl_env.lob_env import data_producer, LOB_trade_env
from dl_helper.rl.base import BaseAgent
from dl_helper.rl.net_center import get_net_params, send_net_updates, update_model_params, send_val_test_data, check_need_val_test
from dl_helper.rl.rl_utils import ReplayBuffer, ReplayBufferWaitClose
from dl_helper.train_param import match_num_processes, get_gpu_info
from dl_helper.trainer import notebook_launcher
from dl_helper.tool import upload_log_file
from py_ext.lzma import compress_folder
from py_ext.alist import alist

from threading import Lock
upload_lock = Lock()
last_upload_time = 0
UPLOAD_INTERVAL = 300  # 5分钟 = 300秒

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
    def __init__(self, obs_shape, features_extractor_class, features_extractor_kwargs, features_dim, net_arch, dqn_type):
        """
        features_dim: features_extractor_class输出维度  + 2(持仓 + 为实现收益率)
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

class DQN(BaseAgent):
    def __init__(self,
            obs_shape,
            action_dim,
            features_dim,
            features_extractor_class,
            learning_rate,
            gamma,
            epsilon,
            target_update,
            buffer_size,
            wait_trade_close = True,
            features_extractor_kwargs=None,
            net_arch=None,
            dqn_type=VANILLA_DQN,
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
            features_extractor_kwargs: 特征提取器参数,可选
            net_arch: 网络架构参数,可选
            dqn_type: DQN类型,可选 VANILLA_DQN/DOUBLE_DQN/DUELING_DQN/DD_DQN
        """
        super().__init__(action_dim, features_dim, features_extractor_class, features_extractor_kwargs, net_arch)   
        assert dqn_type in DQN_TYPES, f'dqn_type 必须是 {DQN_TYPES} 中的一个, 当前为 {dqn_type}'

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.need_epsilon = True,
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = None
        self.msg_head = f''
        self.buffer_size = buffer_size
        self.wait_trade_close = wait_trade_close    
        self.replay_buffer = ReplayBuffer(buffer_size) if not wait_trade_close else ReplayBufferWaitClose(buffer_size)
        self.q_net, self.target_q_net = self.build_model()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)

        # 跟踪器
        self.tracker = DQNTracker('learn', 10, action_dim)
        self.tracker_val_test = None

    def upload_log_file(self):
        global last_upload_time
        
        current_time = time.time()
        with upload_lock:
            # 检查是否距离上次上传已经超过5分钟
            if current_time - last_upload_time >= UPLOAD_INTERVAL:
                upload_log_file()
                last_upload_time = current_time
                log(f'{self.msg_head} Log file uploaded')
            else:
                remaining = UPLOAD_INTERVAL - (current_time - last_upload_time)
                log(f'{self.msg_head} Skip log upload, {remaining:.1f}s remaining')


    def to(self, device):
        self.device = device
        self.q_net = self.q_net.to(device)
        self.target_q_net = self.target_q_net.to(device)
        self.msg_head = f'[{device}]'

    def eval(self):
        self.need_epsilon = False

    def train(self):
        self.need_epsilon = True

    def state_dict(self):
        state = {}

        # 模型参数
        self.state_dict_model(state, 'q_net', self.q_net)
        self.state_dict_model(state, 'target_q_net', self.target_q_net) 
        return state

    def build_model(self):
        q_net = dqn_network(self.obs_shape, self.features_extractor_class, self.features_extractor_kwargs, self.features_dim, self.net_arch, self.dqn_type)
        target_q_net = dqn_network(self.obs_shape, self.features_extractor_class, self.features_extractor_kwargs, self.features_dim, self.net_arch, self.dqn_type)
        # print(f'q_net: {id(q_net)}, target_q_net: {id(target_q_net)}')
        return q_net, target_q_net

    def take_action(self, state):
        if np.random.random() < self.epsilon and self.need_epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        return self.q_net(state).max().item()

    def _update_target_q_net_params(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def update(self, transition_dict, data_type='train'):
        states = torch.from_numpy(transition_dict['states']).to(self.device)
        actions = torch.from_numpy(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.from_numpy(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(transition_dict['next_states']).to(self.device)
        dones = torch.from_numpy(transition_dict['dones']).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type in [DOUBLE_DQN, DD_DQN] :
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(
                1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1)
        q_targets = (rewards + self.gamma * max_next_q_values * (1 - dones))
        
        # tracker 记录
        tracker = self.tracker if data_type == 'train' else self.tracker_val_test

        # 计算TD误差
        td_error = torch.abs(q_targets - q_values).mean().item()
        tracker.update_td_error(td_error)
        
        # 计算损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        tracker.update_loss_value(dqn_loss.item())
        
        if data_type == 'train':
            self.optimizer.zero_grad()
            dqn_loss.backward()
            self.optimizer.step()

            if self.count > 0 and self.count % self.target_update == 0:
                self._update_target_q_net_params()
            self.count += 1

    def val_test(self, env, data_type='val'):
        # 验证模式
        self.eval()

        # 初始化跟踪器
        self.tracker_val_test = DQNTracker(10000, self.action_dim, title=data_type, rank=self.tracker.rank)

        env.set_data_type(data_type)

        # 回放池
        replay_buffer = ReplayBuffer(self.buffer_size) if not self.wait_trade_close else ReplayBufferWaitClose(self.buffer_size)

        log(f'{self.msg_head} {data_type} begin')
        state, info = env.reset()
        done = False
        while not done:
            action = self.take_action(state)
            # 更新跟踪器 动作
            self.tracker_val_test.update_action(action)
            next_state, reward, done1, done2, info = env.step(action)
            done = done1 or done2

            # 添加到回放池
            replay_buffer.add(state, action, reward, next_state, done)

            # 如果 交易close 则需要回溯更新所有 reward 为最终close时的reward
            if info.get('close', False):
                if self.wait_trade_close:
                    replay_buffer.update_reward(reward if reward>-1000 else None)
                # 更新跟踪器 奖励
                self.tracker_val_test.update_reward(reward)
                # 更新跟踪器 非法/win/loss
                if reward == -1000:
                    self.tracker_val_test.update_illegal()
                elif reward > 0:
                    self.tracker_val_test.update_win()
                else:
                    self.tracker_val_test.update_loss_count()
                # 更新评价指标
                for k, v in info.items():
                    if k not in ['close', 'date_done']:
                        self.tracker_val_test.update_extra_metrics(k, v)
            state = next_state

        # 超大batchsize计算error
        while replay_buffer.size() > 0:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.get(1024 * 1024)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            self.update(transition_dict, data_type=data_type)

        # 没有日期的限制，统计区间为全部数据
        # 只需要做一次的 day_end
        self.tracker_val_test.day_end()

        # 获取统计指标
        metrics = self.tracker_val_test.get_metrics()

        # 显式删除跟踪器对象及其内存
        del self.tracker_val_test
        self.tracker_val_test = None

        # 若是 test，上传预测数据文件到alist
        if data_type in ['val', 'test']:
            # 上传更新到alist
            client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
            # 上传文件夹
            upload_folder = f'/rl_learning_process/predict_{data_type}/'
            client.mkdir(upload_folder)
            client.upload(env.predict_file, upload_folder)

        return metrics

    def package_root(self, metrics):
        # 保存模型
        self.save(self.root)
        # 打包压缩
        zip_file = f'{self.root}.7z'
        if os.path.exists(zip_file):
            os.remove(zip_file)
        compress_folder(self.root, zip_file, 9, inplace=False)

    def update_params_from_server(self, env):
        new_params = get_net_params()
        if new_params:
            self.q_net = update_model_params(self.q_net, new_params, tau=1)
            self.target_q_net = update_model_params(self.target_q_net, new_params, tau=1)
            # 重置 buffer, 因为使用了最新的参数，之前的经验已经不正确
            self.replay_buffer.reset()
            log(f'{self.msg_head} replay_buffer reset > {self.replay_buffer.size()}')
            # 重置环境中的账户
            env.acc.reset()

    def push_params_to_server(self):
        # 更新目标网络参数
        self._update_target_q_net_params()
        # 上传参数 /上传学习监控指标
        send_net_updates(self.q_net.state_dict(), self.tracker.get_metrics())

    def learn(self, train_title, env, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step):
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
        # 准备
        super().learn(train_title)

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
                if step % 1000 == 0:
                    self.upload_log_file()

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
                    if self.wait_trade_close:
                        self.replay_buffer.update_reward(reward if reward>-1000 else None)
                    # 更新跟踪器 奖励
                    self.tracker.update_reward(reward)
                    # 更新跟踪器 非法/win/loss
                    if reward == -1000:
                        self.tracker.update_illegal()
                    elif reward > 0:
                        self.tracker.update_win()
                    else:
                        self.tracker.update_loss_count()

                    # 更新评价指标
                    for k, v in info.items():
                        if k not in ['close', 'date_done']:
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
                        need_val_test_res = check_need_val_test()
                        for test_type in ['val', 'test']:
                            if need_val_test_res[test_type]:
                                need_train_back = True  
                                log(f'{self.msg_head} wait metrics for {test_type}')
                                t = time.time()
                                metrics = self.val_test(env, data_type=test_type)
                                log(f'{self.msg_head} metrics: {metrics}, cost: {time.time() - t:.2f}s')
                                # 发送验证数据
                                send_val_test_data(test_type, metrics)

                        # 如果进行了验证/测试,上传日志
                        if any(need_val_test_res.values()):
                            self.upload_log_file()
                    #################################
                    # 服务器通讯
                    #################################

                    # 切换回训练模式
                    if need_train_back:
                        self.train()
                        env.set_data_type('train')
                        state, info = env.reset()
                        done = False

            log(f'{self.msg_head} done')
            self.upload_log_file()

def run_client_learning_device(rank, num_processes, train_title, data_folder, dqn, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step, simple_test=False, val_test=''):
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
    dp = data_producer(data_folder=data_folder, simple_test=simple_test)
    env = LOB_trade_env(data_producer=dp)

    # 开始训练
    if val_test:
        if rank == 0:
            log(f'{rank} {val_test} test...')
            t = time.time()
            metrics = dqn.val_test(env, data_type=val_test)
            log(f'metrics: \n{metrics}\n, cost: {time.time() - t:.2f}s')
    else:
        log(f'{rank} learn...')
        dqn.learn(train_title, env, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step)

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
