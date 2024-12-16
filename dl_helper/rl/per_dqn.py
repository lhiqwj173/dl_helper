import os, time
import matplotlib.pyplot as plt
from py_ext.tool import log, get_log_file

import pickle   
import torch
import torch.nn.functional as F
import numpy as np

from dl_helper.rl.tracker import Tracker
from dl_helper.rl.rl_env.lob_env import data_producer, LOB_trade_env, ILLEGAL_REWARD
from dl_helper.rl.dqn import DQN
from dl_helper.rl.rl_utils import ReplayBufferWaitClose, PrioritizedReplayBufferWaitClose
from dl_helper.train_param import match_num_processes, get_gpu_info
from dl_helper.trainer import notebook_launcher
from dl_helper.tool import upload_log_file
from py_ext.lzma import compress_folder
from py_ext.alist import alist

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
            weights = torch.FloatTensor(weights)
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

