import torch
import os
import time
import shutil
import numpy as np
from collections import defaultdict

from py_ext.tool import log, debug, get_log_folder, _get_caller_info
from py_ext.lzma import compress_folder, decompress
from py_ext.wechat import wx
from py_ext.alist import alist

from dl_helper.rl.rl_env.lob_env import ILLEGAL_REWARD
from dl_helper.rl.socket_base import get_net_params, send_val_test_data, check_need_val_test, request_client_id, send_accumulated_gradients
from dl_helper.rl.rl_utils import update_model_params, ReplayBuffer, ReplayBufferWaitClose, calculate_importance_loss,ExperimentHandler
from dl_helper.rl.tracker import Tracker
from dl_helper.rl.noisy import replace_linear_with_noisy

class BaseModel(torch.nn.Module):
    def __init__(self, features_extractor_class, features_extractor_kwargs, features_dim, need_reshape=None):
        super().__init__()
        self.features_dim = features_dim
        self.need_reshape = need_reshape
        self.extra_features = -1# forward时计算一次

        # 特征提取器
        self.features_extractor = features_extractor_class(**features_extractor_kwargs)

        # 添加Batch Normalization
        self.bn = torch.nn.BatchNorm1d(features_dim)
        
    def init_weights(self):
        """Initialize network weights using appropriate initialization schemes"""
        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                # Initialize weights using kaiming normal initialization
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                # Initialize batchnorm/groupnorm weights
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LSTM):
                # Initialize LSTM weights
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        torch.nn.init.constant_(param, 0)

    def features_forward(self, x):
        extra_x = None
        if self.need_reshape:
            assert len(x.shape) == 2, "if need_reshape, x must be 2D(batch_size, features_dim)"
            if self.extra_features == -1:
                total_items = np.prod(self.need_reshape)
                total_xs = np.prod(x.shape[1:])
                self.extra_features = total_xs - total_items
            
            main_x = x[:,:-self.extra_features].view(-1, *self.need_reshape)
            extra_x = x[:,-self.extra_features:]
        else:
            main_x = x

        # 特征提取
        feature = self.features_extractor(main_x)
        if extra_x is not None:
            feature = torch.cat([feature, extra_x], dim=1)

        # 应用Batch Normalization
        feature = self.bn(feature)

        return feature

class BaseAgent:
    def __init__(self,
        train_title,
        action_dim,
        features_dim,
        features_extractor_class,
        features_extractor_kwargs=None,
        net_arch=None,
    ):
        """Agent 基类
        
        Args:
            train_title: 训练标题
            action_dim: 动作空间维度 
            features_dim: 特征维度
            features_extractor_class: 特征提取器类,必须提供
            features_extractor_kwargs: 特征提取器参数,可选
            net_arch: 网络架构参数,默认为一层mlp, 输入/输出维度为features_dim, action_dim
                [action_dim] / dict(pi=[action_dim], vf=[action_dim]) 等价

        需要子类重写的函数
            build_model: 构建模型
            take_action: 根据状态选择动作
            _update(self, states, actions, rewards, next_states, dones, data_type): 更新模型
        """
        self.train_title = train_title

        if features_extractor_class is None:
            raise ValueError("必须提供特征提取器类 features_extractor_class")
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs or {}
        self.features_dim = features_dim
        self.net_arch = self.fix_net_arch(net_arch, action_dim)
        self.action_dim = action_dim

        # 初始化msg_head
        self.msg_head = f''

        # 初始化device
        self.device = None

        # 储存模型
        self.models = {}

        # 参数
        self.version = 0

        # 累积的梯度和重要性
        self.accumulated_grads = defaultdict(list)  # 存储每个参数的梯度列表
        self.importance_weights = []  # 存储每个batch的重要性

    def build_model(self):
        raise NotImplementedError

    def take_action(self, state):
        raise NotImplementedError

    def get_model_to_sync(self):
        raise NotImplementedError

    def _update(self, states, actions, rewards, next_states, dones, data_type, weights=None):
        raise NotImplementedError

    def fix_net_arch(self, net_arch, action_dim):
        if net_arch is None:
            return dict(pi=[action_dim], vf=[action_dim])
        elif isinstance(net_arch, list):
            return dict(pi=net_arch, vf=net_arch)
        elif isinstance(net_arch, dict):
            if 'pi' in net_arch and 'vf' in net_arch:
                return net_arch
            else:
                raise ValueError("net_arch 字典需包含 'pi' 和 'vf' 键")
        else:
            raise ValueError("net_arch 必须是列表或字典, 表示mlp每层的神经元个数")

    def update(self, transition_dict, data_type='train', weights=None):
        states = torch.from_numpy(transition_dict['states']).to(self.device)
        actions = torch.from_numpy(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.from_numpy(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(transition_dict['next_states']).to(self.device)
        dones = torch.from_numpy(transition_dict['dones']).view(-1, 1).to(self.device)

        # 供 PER回放池 使用
        if weights is not None:
            weights = torch.from_numpy(weights).to(self.device)

        return self._update(states, actions, rewards, next_states, dones, data_type, weights)

    def compute_importance_loss(self, loss):
        """计算重要性损失"""
        importance = calculate_importance_loss(loss)
        self.importance_weights.append(importance)

    def collect_gradients(self, model):
        """收集当前batch的梯度"""
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                # 转换为numpy并立即释放GPU内存
                grad_cpu = param.grad.detach().cpu().numpy()  # 直接转numpy
                self.accumulated_grads[param_name].append(grad_cpu)
    
    def produce_submit_accumulated_gradients(self, averaging_strategy='mean'):
        """生成并提交累积的梯度"""
        assert averaging_strategy in ['mean', 'weighted'], "averaging_strategy 必须是 'mean' 或 'weighted'"
        
        if len(self.accumulated_grads) == 0 or len(self.importance_weights) == 0:
            log(f"{self.msg_head} No gradients or importance weights accumulated")
            return {}, 0.0
        
        # 预处理：移除包含NaN/Inf的梯度
        valid_indices = []
        for i in range(len(next(iter(self.accumulated_grads.values())))):
            is_valid = True
            for grad_list in self.accumulated_grads.values():
                if np.any(np.isnan(grad_list[i])) or np.any(np.isinf(grad_list[i])):
                    is_valid = False
                    break
            if is_valid:
                valid_indices.append(i)
        
        if not valid_indices:
            log(f"{self.msg_head} Warning: All gradients are invalid (NaN/Inf)")
            return {}, 0.0
        
        # 只保留有效的梯度和对应的重要性权重
        filtered_grads = {
            name: [grad_list[i] for i in valid_indices]
            for name, grad_list in self.accumulated_grads.items()
        }
        filtered_weights = [self.importance_weights[i] for i in valid_indices]
        
        if len(valid_indices) < len(self.importance_weights):
            log(f"{self.msg_head} Filtered out {len(self.importance_weights) - len(valid_indices)} invalid gradients")
        
        if averaging_strategy == 'weighted':
            # 基于重要性权重的加权平均
            weights = np.array(filtered_weights)
            eps = 1e-8
            weights = np.clip(weights, eps, None)
            weights = weights / (weights.sum() + eps)
            
            averaged_grads = {}
            for name, grad_list in filtered_grads.items():
                stacked_grads = np.stack(grad_list, axis=0)
                averaged_grads[name] = np.average(stacked_grads, axis=0, weights=weights)
                
            averaged_importance = np.max(filtered_weights)
        else:
            # 简单平均
            averaged_grads = {
                name: np.mean(grad_list, axis=0)
                for name, grad_list in filtered_grads.items()
            }
            averaged_importance = np.mean(filtered_weights)
        
        # 最后的有效性检查
        for name, grad in averaged_grads.items():
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                log(f"{self.msg_head} Critical error: averaged gradients still contain NaN/Inf")
                return {}, 0.0
                
        # 清理累积状态
        self.accumulated_grads.clear()
        self.importance_weights.clear()
        
        return averaged_grads, averaged_importance

    def eval(self):
        for model in self.models.values():
            model.eval()

    def train(self):
        for model in self.models.values():
            model.train()

    def to(self, device):
        self.device = device
        if device != 'cpu':
            self.msg_head = f'[{device}]'
        else:
            self.msg_head = ''

        for model in self.models.values():
            model.to(device)

    def apply_new_params(self, new_params, tau=1):
        """
        应用新参数
        """
        if list(new_params.keys())[0] in self.models:
            # 分模型新参数
            for name, param in new_params.items():
                self.models[name] = update_model_params(self.models[name], param, tau=tau)
        else:
            # 全局新参数
            for name in self.models:
                self.models[name] = update_model_params(self.models[name], new_params, tau=tau)

    def state_dict(self):
        """只保存模型参数"""
        state_dict = {}
        for name, model in self.models.items():
            state_dict[name] = model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """
        加载模型参数
        
        Args:
            state_dict (dict): 包含模型参数的状态字典
        """
        
        # 兼容版本， 删除版本
        if 'version' in state_dict:
            del state_dict['version']

        # 兼容旧 state_dict
        model_key_suffix = "_state_dict@@@"
        is_old = False
        for key, value in state_dict.items():
            if key.endswith(model_key_suffix):
                model_name = key[:-len(model_key_suffix)]
                if model_name in self.models:
                    self.models[model_name].load_state_dict(value)
                    is_old = True


        # 加载模型参数
        if not is_old:
            for key, value in state_dict.items():
                self.models[key].load_state_dict(value)

    def save(self, root=''):
        """
        保存参数
        """
        torch.save(self.state_dict(), os.path.join(root, 'agent_data.pth'))

    def load(self, root=''):
        """
        加载参数
        """
        file = os.path.join(root, 'agent_data.pth')
        if os.path.exists(file):
            self.load_state_dict(torch.load(file))


class OffPolicyAgent(BaseAgent):
    def __init__(self, buffer_size, train_buffer_class, use_noisy, n_step, *args, **kwargs):
        """
        OffPolicyAgent 基类
        
        Args:
            buffer_size: 经验回放池大小
            train_buffer_class: 训练经验回放池类
            use_noisy: 是否使用噪声网络
            n_step: 多步学习的步数

            BaseAgent 参数
                train_title: 训练标题
                action_dim: 动作空间维度 
                features_dim: 特征维度
                features_extractor_class: 特征提取器类,必须提供
                features_extractor_kwargs=None: 特征提取器参数,可选
                net_arch=None: 网络架构参数,默认为一层mlp, 输入/输出维度为features_dim, action_dim
                    [action_dim] / dict(pi=[action_dim], vf=[action_dim]) 等价

        需要子类重写的函数
            _build_model: 构建模型
            take_action(self, state): 根据状态选择动作
            get_model_to_sync: 获取需要同步的模型
            _update(self, states, actions, rewards, next_states, dones, data_type, weights=None, n_step_rewards=None, n_step_next_states=None, n_step_dones=None): 更新模型
            sync_update_net_params_in_agent: 同步更新模型参数
            get_params_to_send: 获取需要上传的参数
        """
        super().__init__(*args, **kwargs)

        # 是否使用噪声网络
        self.use_noisy = use_noisy

        # 多步学习的步数
        self.n_step = n_step    

        # 离线经验回放池
        self.buffer_size = buffer_size
        self.train_buffer_class = train_buffer_class
        self.replay_buffer = self.train_buffer_class(self.buffer_size, n_step=self.n_step)

        # 跟踪器
        self.tracker = Tracker('learn', 10, action_space=self.action_dim)
        self.tracker_val_test = None

        # 是否是训练状态
        self.in_train = True

    def build_model(self):
        """构建模型并处理 noisy network 替换
        子类应该实现 _build_model 而不是 build_model"""
        # 调用子类的模型构建
        self._build_model()
        
        # 如果启用了 noisy network，替换所有线性层
        if self.use_noisy:
            for name, model in self.models.items():
                self.models[name] = replace_linear_with_noisy(model)
                log(f"Replaced linear layers with noisy layers in {name}")

    def _build_model(self):
        """子类需要实现此方法来构建具体的模型结构"""
        raise NotImplementedError

    def __call_subclass_take_action(self, state):
        """调用子类的 take_action"""
        state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        return self._take_action(state)   

    def take_action(self, state):
        if self.use_noisy:
            # 调用子类的 take_action
            return self.__call_subclass_take_action(state)
        else:
            if self.in_train and np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                # 调用子类的 take_action
                action = self.__call_subclass_take_action(state)   
        return action

    def _take_action(self, state):
        """子类需要实现此方法来实现具体的动作选择逻辑"""
        raise NotImplementedError

    def eval(self):
        self.in_train = False

    def train(self):
        self.in_train = True

    def sync_update_net_params_in_agent(self):
        raise NotImplementedError

    def get_params_to_send(self):
        raise NotImplementedError

    def update_params_from_server(self):
        params, self.version = get_net_params(self.train_title)
        log(f'{self.msg_head} update params from server, version: {self.version}')
        self.apply_new_params(params)

    def push_update_to_server(self):
        # 改用推送 累积的梯度
        send_accumulated_gradients(self.train_title, *self.produce_submit_accumulated_gradients(), self.version, self.tracker.get_metrics())

    def track_error(self, loss_value, data_type='train'):
        # tracker 记录
        tracker = self.tracker if data_type == 'train' else self.tracker_val_test
        # 计算损失
        tracker.update_loss_value(loss_value)

    def _update(self, states, actions, rewards, next_states, dones, data_type, weights=None, n_step_rewards=None, n_step_next_states=None, n_step_dones=None):
        raise NotImplementedError

    def update(self, transition_dict, data_type='train', weights=None):
        states = torch.from_numpy(transition_dict['states']).to(self.device)
        actions = torch.from_numpy(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.from_numpy(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(transition_dict['next_states']).to(self.device)
        dones = torch.from_numpy(transition_dict['dones']).view(-1, 1).to(self.device)

        # 计算 n-step return
        if 'n_step_rewards' in transition_dict:
            n_step_rewards = torch.from_numpy(transition_dict['n_step_rewards']).view(-1, 1).to(self.device)
            n_step_next_states = torch.from_numpy(transition_dict['n_step_next_states']).to(self.device)
            n_step_dones = torch.from_numpy(transition_dict['n_step_dones']).view(-1, 1).to(self.device)
        else:
            n_step_rewards = None
            n_step_next_states = None
            n_step_dones = None

        # 供 PER回放池 使用
        if weights is not None:
            weights = torch.from_numpy(weights).to(self.device)

        return self._update(states, actions, rewards, next_states, dones, data_type, weights, n_step_rewards, n_step_next_states, n_step_dones)

    def val_test(self, env, data_type='val'):
        # 验证模式
        self.eval()

        # 初始化跟踪器
        self.tracker_val_test = Tracker(data_type, 10000, rank=self.tracker.rank, action_space=self.action_dim)

        if hasattr(env, 'set_data_type'):
            env.set_data_type(data_type)

        # 回放池
        replay_buffer = ReplayBufferWaitClose(self.buffer_size) if env.need_wait_close() else ReplayBuffer(self.buffer_size)

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

            if env.need_wait_close() and info.get('close', False):
                # 需要wait close 时, 且 close为True时, 更新reward
                replay_buffer.update_reward(reward if reward!=ILLEGAL_REWARD else None)

            if (env.need_wait_close() and info.get('close', False)) or (not env.need_wait_close()):
                # 更新跟踪器 奖励
                self.tracker_val_test.update_reward(reward)

                # 更新跟踪器 非法/win/loss
                if info['act_criteria'] == -1:
                    self.tracker_val_test.update_illegal()
                elif info['act_criteria'] == 0:
                    self.tracker_val_test.update_win()
                else:
                    self.tracker_val_test.update_loss_count()

                # 更新评价指标
                for k, v in info.items():
                    if k not in env.no_need_track_info_item():
                        self.tracker_val_test.update_extra_metrics(k, v)

            # 更新跟踪器 日期文件完成, 需要更新
            if info.get('period_done', False):
                self.tracker_val_test.period_end()

            state = next_state

        log(f'{self.msg_head} {data_type} done, begin update...')

        # 超大batchsize计算error
        batch_size = 1024 * 1024
        while replay_buffer.size() > 0:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.get(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            self.update(transition_dict, data_type=data_type)
        log(f'{self.msg_head} {data_type} update done')

        # 获取统计指标
        metrics = self.tracker_val_test.get_metrics()

        # 显式删除跟踪器对象及其内存
        del self.tracker_val_test
        self.tracker_val_test = None

        # 若是 test，上传预测数据文件到alist
        if hasattr(env, 'need_upload_file'):
            if env.need_upload_file:
                # 上传更新到alist
                client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
                # 上传文件夹
                upload_folder = f'/rl_learning_process/{self.train_title}/file_{data_type}/'
                client.mkdir(upload_folder)
                client.upload(env.need_upload_file, upload_folder)

        log(f'{self.msg_head} {data_type} all done')
        return metrics

    def learn(self, env, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step, local=False):
        """
        训练

        Args:
            env: 环境
            num_episodes: 训练回合数
            minimal_size: 最小训练次数
            batch_size: 批次大小
            sync_interval_learn_step:   同步参数间隔，会询问是否需要验证/测试
            learn_interval_step:        学习更新间隔  
            local:                     是否是本地训练
        """
        # 学习步数
        learn_step = 0

        if not local:
            # 拉取服务器的最新参数并更新
            self.update_params_from_server()

        # 本地处理器
        local_handler = ExperimentHandler(self.train_title) if local else None

        # 学习是否开始
        for i in range(num_episodes):
            self.msg_head = f'[{self.device}][e{i}]'

            # 回合的评价指标
            state, info = env.reset()
            done = False
            step = 0
            while not done:
                step += 1
                log(f'{self.msg_head}{step} obs shape: {state.shape}')

                # 动作
                action = self.take_action(state)
                log(f'{self.msg_head}{step} action: {action}')
                # 更新跟踪器 动作
                self.tracker.update_action(action)

                # 环境交互
                next_state, reward, done1, done2, info = env.step(action)
                log(f'{self.msg_head}{step} reward: {reward}')
                done = done1 or done2

                # 添加到回放池
                self.replay_buffer.add(state, action, reward, next_state, done)

                if env.need_wait_close() and info.get('close', False):
                    # 需要wait close 时, 且 close为True时, 更新reward
                    self.replay_buffer.update_reward(reward if reward!=ILLEGAL_REWARD else None)

                if (env.need_wait_close() and info.get('close', False)) or (not env.need_wait_close()):
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
                        if k not in ['close', 'period_done', 'act_criteria']:
                            self.tracker.update_extra_metrics(k, v)
                
                # 更新跟踪器 日期文件完成, 需要更新
                if info.get('period_done', False):
                    self.tracker.period_end()

                # 更新状态
                state = next_state

                # 更新网络
                if self.replay_buffer.size() >= minimal_size and step % learn_interval_step == 0:
                    per_buffer = hasattr(self.replay_buffer, 'update_priorities')

                    if per_buffer:
                        # 添加调试信息
                        # log(f"{self.msg_head} Tree status - is_full: {self.replay_buffer.tree.is_full}, data_pointer: {self.replay_buffer.tree.data_pointer}")
                        # log(f"{self.msg_head} Total priority: {self.replay_buffer.tree.total_priority()}")
                        # 采样batch数据
                        (b_s, b_a, b_r, b_ns, b_d, step_r, step_ns, step_d), indices, weights = self.replay_buffer.sample(batch_size)

                    else:   
                        # 学习经验
                        b_s, b_a, b_r, b_ns, b_d, step_r, step_ns, step_d = self.replay_buffer.sample(batch_size)
                        weights = None

                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }

                    # 添加n步数据
                    if step_r is not None:
                        transition_dict['n_step_rewards'] = step_r
                        transition_dict['n_step_next_states'] = step_ns
                        transition_dict['n_step_dones'] = step_d

                    td_error_for_update = self.update(transition_dict, weights=weights)

                    if per_buffer:
                        # 更新优先级
                        self.replay_buffer.update_priorities(indices, td_error_for_update)  

                    learn_step += 1
                    need_train_back = False
                    if learn_step % sync_interval_learn_step == 0:
                        #################################
                        # 服务器通讯
                        #################################
                        if not local:
                            log(f'{self.msg_head} {learn_step} sync params')
                            # 同步最新参数
                            # 推送参数更新
                            self.push_update_to_server()
                            # 拉取服务器的最新参数并更新
                            self.update_params_from_server()
                        else:
                            local_handler.update_learn_metrics(self.tracker.get_metrics())

                        # 验证/测试
                        # 询问 服务器 是否需要 验证/测试
                        # 返回: 'val' / 'test' / 'no'
                        need_val_test_res = check_need_val_test(self.train_title) if not local else local_handler.check_need_val_test()
                        if need_val_test_res != 'no':
                            test_type = need_val_test_res
                            need_train_back = True  
                            log(f'{self.msg_head} wait metrics for {test_type}')
                            t = time.time()
                            metrics = self.val_test(env, data_type=test_type)
                            log(f'{self.msg_head} metrics: {metrics}, cost: {time.time() - t:.2f}s')
                            # 发送验证结果给服务器
                            send_val_test_data(self.train_title, test_type, metrics) if not local else local_handler.handle_val_test_data(self.train_title,test_type, metrics)
                        #################################
                        # 服务器通讯
                        #################################

                    # 切换回训练模式
                    if need_train_back:
                        self.train()
                        if hasattr(env, 'set_data_type'):
                            env.set_data_type('train')
                        state, info = env.reset()
                        done = False

                else:
                    if step % 100 == 0:
                        log(f'{self.msg_head} replay buffer size: {self.replay_buffer.size()}')
