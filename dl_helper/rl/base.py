import torch
import os
import time
import shutil

from py_ext.tool import log, debug, get_log_folder, _get_caller_info
from py_ext.lzma import compress_folder, decompress
from py_ext.wechat import wx
from dl_helper.tool import upload_log_file

from dl_helper.rl.rl_env.lob_env import ILLEGAL_REWARD
from dl_helper.rl.net_center import get_net_params, send_net_updates, update_model_params, send_val_test_data, check_need_val_test
from dl_helper.rl.rl_utils import ReplayBuffer, ReplayBufferWaitClose
from dl_helper.rl.tracker import Tracker

from threading import Lock
upload_lock = Lock()
last_upload_time = 0
UPLOAD_INTERVAL = 300  # 5分钟 = 300秒

class BaseAgent:
    def __init__(self,
                 action_dim,
                 features_dim,
                 features_extractor_class,
                 features_extractor_kwargs=None,
                 net_arch=None,
    ):
        """Agent 基类
        
        Args:
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
        if features_extractor_class is None:
            raise ValueError("必须提供特征提取器类 features_extractor_class")
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs or {}
        self.features_dim = features_dim

        if net_arch is None:
            self.net_arch = dict(pi=[action_dim], vf=[action_dim])
        elif isinstance(net_arch, list):
            self.net_arch = dict(pi=net_arch, vf=net_arch)
        elif isinstance(net_arch, dict):
            if 'pi' in net_arch and 'vf' in net_arch:
                self.net_arch = net_arch
            else:
                raise ValueError("net_arch 字典需包含 'pi' 和 'vf' 键")
        else:
            raise ValueError("net_arch 必须是列表或字典, 表示mlp每层的神经元个数")

        # 初始化msg_head
        self.msg_head = f''

        # 初始化device
        self.device = None

        # 储存模型
        self.models = {}

    def build_model(self):
        raise NotImplementedError

    def take_action(self, state):
        raise NotImplementedError

    def _update(self, states, actions, rewards, next_states, dones, data_type):
        raise NotImplementedError

    def update(self, transition_dict, data_type='train'):
        states = torch.from_numpy(transition_dict['states']).to(self.device)
        actions = torch.from_numpy(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.from_numpy(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(transition_dict['next_states']).to(self.device)
        dones = torch.from_numpy(transition_dict['dones']).view(-1, 1).to(self.device)

        return self._update(states, actions, rewards, next_states, dones, data_type)

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

    def upload_log_file(self):
        global last_upload_time
        
        current_time = time.time()
        with upload_lock:
            # 检查是否距离上次上传已经超过5分钟
            if current_time - last_upload_time >= UPLOAD_INTERVAL:
                upload_log_file()
                last_upload_time = current_time
                log(f'{self.msg_head} Log file uploaded')
            # else:
            #     remaining = UPLOAD_INTERVAL - (current_time - last_upload_time)
            #     log(f'{self.msg_head} Skip log upload, {remaining:.1f}s remaining')

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
    def __init__(self, buffer_size, *args, **kwargs):
        """
        OffPolicyAgent 基类
        
        Args:
            buffer_size: 经验回放池大小

            BaseAgent 参数
                action_dim: 动作空间维度 
                features_dim: 特征维度
                features_extractor_class: 特征提取器类,必须提供
                features_extractor_kwargs=None: 特征提取器参数,可选
                net_arch=None: 网络架构参数,默认为一层mlp, 输入/输出维度为features_dim, action_dim
                    [action_dim] / dict(pi=[action_dim], vf=[action_dim]) 等价

        需要子类重写的函数
            build_model: 构建模型
            take_action(self, state): 根据状态选择动作
            _update(self, states, actions, rewards, next_states, dones, data_type): 更新模型
            sync_update_net_params_in_agent: 同步更新模型参数
            get_params_to_send: 获取需要上传的参数
        """

        super().__init__(*args, **kwargs)

        # 离线经验回放池
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBufferWaitClose(buffer_size)

        # 跟踪器
        self.tracker = Tracker('learn', 10)
        self.tracker_val_test = None

    def sync_update_net_params_in_agent(self):
        raise NotImplementedError

    def get_params_to_send(self):
        raise NotImplementedError

    def update_params_from_server(self, env):
        new_params = get_net_params()
        if new_params:
            self.apply_new_params(new_params)
            # 重置 buffer, 因为使用了最新的参数，之前的经验已经不正确
            self.replay_buffer.reset()
            log(f'{self.msg_head} replay_buffer reset > {self.replay_buffer.size()}')
            # 重置环境中的账户
            env.acc.reset()

    def push_params_to_server(self):
        # 同步agent内部参数
        self.sync_update_net_params_in_agent()
        # 上传参数 /上传学习监控指标
        send_net_updates(self.get_params_to_send(), self.tracker.get_metrics())

    def track_error(self, td_error, loss_value, data_type='train'):
        # tracker 记录
        tracker = self.tracker if data_type == 'train' else self.tracker_val_test
        # 计算TD误差
        tracker.update_td_error(td_error)
        # 计算损失
        tracker.update_loss_value(loss_value)

    def val_test(self, env, data_type='val'):
        # 验证模式
        self.eval()

        # 初始化跟踪器
        self.tracker_val_test = Tracker(data_type, 10000, rank=self.tracker.rank)

        env.set_data_type(data_type)

        # 回放池
        replay_buffer = ReplayBufferWaitClose(self.buffer_size)

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
                replay_buffer.update_reward(reward if reward!=ILLEGAL_REWARD else None)
                # 更新跟踪器 奖励
                self.tracker_val_test.update_reward(reward)

                # 更新跟踪器 非法/win/loss
                if reward == ILLEGAL_REWARD:
                    self.tracker_val_test.update_illegal()
                elif info['total_return'] > 0:
                    self.tracker_val_test.update_win()
                else:
                    self.tracker_val_test.update_loss_count()

                # 更新评价指标
                for k, v in info.items():
                    if k not in ['close', 'date_done']:
                        self.tracker_val_test.update_extra_metrics(k, v)

            # 更新跟踪器 日期文件完成, 需要更新
            if info.get('date_done', False):
                self.tracker_val_test.day_end()

            state = next_state

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
                if step % 1000 == 0:
                    self.upload_log_file()

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

                # # 测试用
                # # 检查是否有nan/inf值
                # if np.argwhere(np.isnan(state)).any() or np.argwhere(np.isinf(state)).any():
                #     raise ValueError(f'检测到NaN/Inf值,state: {state}')

                # 添加到回放池
                self.replay_buffer.add(state, action, reward, next_state, done)

                # # 测试用
                # # 检查是否有nan/inf值
                # for d in self.replay_buffer.buffer_temp:
                #     state = d[0]
                #     if np.argwhere(np.isnan(state)).any() or np.argwhere(np.isinf(state)).any():
                #         raise ValueError(f'检测到NaN/Inf值,state: {state}')

                # 如果 交易close 则需要回溯更新所有 reward 为最终close时的reward
                if info.get('close', False):
                    self.replay_buffer.update_reward(reward if reward!=ILLEGAL_REWARD else None)

                    # # 测试用
                    # # 检查是否有nan/inf值
                    # for d in self.replay_buffer.buffer:
                    #     if np.argwhere(np.isnan(d[0])).any() or np.argwhere(np.isinf(d[0])).any():
                    #         raise ValueError(f'检测到NaN/Inf值,state: {d[0]}')

                    # 更新跟踪器 奖励
                    self.tracker.update_reward(reward)

                    # 更新跟踪器 非法/win/loss
                    if reward == ILLEGAL_REWARD:
                        self.tracker.update_illegal()
                    elif info['total_return'] > 0:
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
                        return

                        # 同步最新参数
                        # 推送参数更新
                        self.push_params_to_server()
                        # 拉取服务器的最新参数并更新
                        self.update_params_from_server(env)

                        # 验证/测试
                        # 询问 服务器 是否需要 验证/测试
                        need_val_test_res = check_need_val_test()
                        for test_type in ['val', 'test']:
                            if need_val_test_res[test_type]:
                                need_train_back = True  
                                log(f'{self.msg_head} wait metrics for {test_type}')
                                t = time.time()
                                metrics = self.val_test(env, data_type=test_type)
                                log(f'{self.msg_head} metrics: {metrics}, cost: {time.time() - t:.2f}s')
                                # 发送验证结果给服务器
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


