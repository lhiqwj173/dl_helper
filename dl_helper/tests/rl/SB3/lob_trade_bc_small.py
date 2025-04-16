# !pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz > /dev/null 2>&1
# !mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git > /dev/null 2>&1
# !cd /kaggle/working/3rd/dl_helper && pip install -e . > /dev/null 2>&1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecCheckNan
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger as imit_logger

import pandas as pd
import torch
import torch.nn as nn
import torch as th
from torch.optim.lr_scheduler import OneCycleLR, MultiplicativeLR
# th.autograd.set_detect_anomaly(True)
import time, pickle
import numpy as np
import random, psutil
import sys, os, shutil
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ['ALIST_USER'] = 'admin'
os.environ['ALIST_PWD'] = 'LHss6632673'

from py_ext.lzma import decompress, compress_folder
from py_ext.alist import alist
from py_ext.tool import init_logger,log
from py_ext.wechat import send_wx
from py_ext.datetime import beijing_time

from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_env.lob_trade.lob_expert import LobExpert_file
from dl_helper.rl.rl_utils import plot_bc_train_progress, CustomCheckpointCallback, check_gradients, cal_action_balance
from dl_helper.tool import report_memory_usage, in_windows
from dl_helper.train_folder_manager import TrainFolderManagerBC

from dl_helper.rl.custom_imitation_module.bc import BCWithLRScheduler

model_type = 'CnnPolicy'
# 'train' or 'test' or 'find_lr' or 'test_model' or 'test_transitions
# find_lr: 学习率从 1e-6 > 指数增长，限制总batch为150
# test_model: 使用相同的batch数据，测试模型拟合是否正常
# test_transitions: 测试可视化transitions
# 查找最大学习率
# df_progress = pd.read_csv('progress_all.csv')
# find_best_lr(df_progress.iloc[50:97]['bc/lr'], df_progress.iloc[50:97]['bc/loss'])
run_type = 'train'
# run_type = 'test_transitions'
_train_timesteps_list = [5e4, 1e6, 2.3e6]
_train_timesteps = _train_timesteps_list[1]

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == 'train':
            run_type = 'train'
        elif arg == 'find_lr':
            run_type = 'find_lr'
        elif arg == 'test':
            run_type = 'test'
        elif arg == 'test_model':
            run_type = 'test_model'

        elif arg == '0':
            _train_timesteps = _train_timesteps_list[0]
        elif arg == '1':
            _train_timesteps = _train_timesteps_list[1]
        elif arg == '2':
            _train_timesteps = _train_timesteps_list[2]

train_folder = train_title = f'20250415_lob_trade_bc_small_data_{_train_timesteps_list.index(_train_timesteps)}'
log_name = f'{train_title}_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)

# FOR DEBUG
# df_progress = pd.read_csv(r"C:\Users\lh\Downloads\progress_all (1).csv")
# plot_bc_train_progress(train_folder, df_progress=df_progress)

custom_logger = imit_logger.configure(
    folder=train_folder,
    format_strs=["csv", "stdout"],
)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

class ImprovedSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_output, _ = self.multihead_attn(x, x, x)  # 使用多头注意力
        # attn_output = self.dropout(attn_output)
        out = self.norm(x + attn_output)  # 残差连接 + LayerNorm
        return out

# 自定义特征提取器
# 参数量: 250795
class DeepLob(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space,
            features_dim: int = 64,
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims

        # 卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.Tanh(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.Tanh(),
            # nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
        )

        # Inception 模块
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
        )

        # LSTM 层 
        self.lstm = nn.LSTM(input_size=64*3, hidden_size=64, num_layers=1, batch_first=True)

        # 自注意力层
        # self.attention = SelfAttention(128)
        # self.improved_attention = ImprovedSelfAttention(64)

        # 静态特征处理
        self.static_net = nn.Sequential(
            nn.Linear(self.extra_input_dims, self.extra_input_dims * 4),
            nn.LayerNorm(self.extra_input_dims * 4),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(0.1),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + self.extra_input_dims * 4, 128),
            nn.LayerNorm(128),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(64, 64),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, features_dim),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 对 CNN 和全连接层应用 He 初始化
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 对 LSTM 应用特定初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # 输入到隐藏权重使用 He 初始化
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            elif 'weight_hh' in name:
                # 隐藏到隐藏权重使用正交初始化
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # 偏置初始化为零（可选：遗忘门偏置设为 1）
                nn.init.zeros_(param)
                # 如果需要鼓励遗忘门记住更多信息，可以设置：
                # bias_size = param.size(0)
                # param.data[bias_size//4:bias_size//2].fill_(1.0)  # 遗忘门偏置设为 1

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]# 最后一维是 日期信息， 不参与模型计算，用于专家透视未来数据
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)  # (B, 1, 10, 20)

        # 卷积块
        x = self.conv1(x)  # (B, 32, 28, 10)
        x = self.conv2(x)  # (B, 32, 26, 5)
        x = self.conv3(x)  # (B, 32, 24, 1)

        # Inception 模块
        x_inp1 = self.inp1(x)  # (B, 64, 24, 1)
        x_inp2 = self.inp2(x)  # (B, 64, 24, 1)
        x_inp3 = self.inp3(x)  # (B, 64, 24, 1)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)  # (B, 192, 24, 1)

        # 调整形状以适配 LSTM
        x = x.permute(0, 2, 1, 3).squeeze(3)  # (B, 24, 192)

        # LSTM 处理 取最后一个时间步
        lstm_out, _ = self.lstm(x)  # (B, 24, 64)

        # # # 自注意力
        # # attn_out = self.attention(lstm_out)  # (B, ?, 128)
        # # temporal_feat = attn_out[:, -1, :]  # 取最后一个时间步 (B, 128)
        # attn_out = self.improved_attention(lstm_out)
        # temporal_feat = attn_out.mean(dim=1)  # 平均池化

        # 取最后一个时间步
        temporal_feat = lstm_out[:, -1, :]  # (B, 64)

        # 静态特征处理
        static_out = self.static_net(extra_x)  # (B, self.extra_input_dims * 4)

        # 融合层
        fused_out = torch.cat([temporal_feat, static_out], dim=1)  # (B, 64 + self.extra_input_dims * 4)
        fused_out = self.fusion(fused_out)  # (B, features_dim)

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            raise ValueError("fused_out is nan or inf")

        return fused_out

model_config={
    # 自定义编码器参数  
    'input_dims' : (30, 20),
    'extra_input_dims' : 3,
    'features_dim' : 256,
}
env_config ={
    'data_type': 'train',# 训练/测试
    'his_len': 30,# 每个样本的 历史数据长度
    'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
    'train_folder': train_folder,
    'train_title': train_folder,
    'latest_dates': 100,# 只使用最近 100 个数据

    # 不使用数据增强
    'use_random_his_window': False,# 是否使用随机历史窗口
    'use_gaussian_noise_vol': False,# 是否使用高斯噪声
    'use_spread_add_small_limit_order': False,# 是否使用价差添加小单
}

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

# 配置 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=DeepLob,
    features_extractor_kwargs=model_config,
    net_arch = [128,64]
)

env_objs = []
def make_env():
    env = LOB_trade_env(env_config)
    env_objs.append(env)
    return RolloutInfoWrapper(env)

if run_type != 'test':

    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecCheckNan(env, raise_exception=True)  # 添加nan检查
    env = VecMonitor(env)  # 添加监控器

    # # 创建单个环境
    # env = LOB_trade_env(env_config)
    # vec_env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

    # 专家
    expert = LobExpert_file(pre_cache=True)

    model = PPO(
        model_type, 
        env, 
        verbose=1, 
        learning_rate=1e-3,
        ent_coef=0.2,
        gamma=0.97,
        policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
    )

    # 打印模型结构
    log("模型结构:")
    log(model.policy)
    log(f'参数量: {sum(p.numel() for p in model.policy.parameters())}')

    # test_x = env.observation_space.sample()
    # test_x = torch.from_numpy(test_x).unsqueeze(0)
    # log(test_x.shape)
    # test_x = test_x.float().to(device)
    # out = model.policy.features_extractor(test_x)
    # log(out.shape)
    # sys.exit()

    # 训练文件夹管理
    if not in_windows():
        train_folder_manager = TrainFolderManagerBC(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(model.policy)

    # 生成专家数据
    vec_env = env
    rng = np.random.default_rng()
    t = time.time()
    memory_usage = psutil.virtual_memory()
    # 训练数据
    train_timesteps = _train_timesteps if run_type=='train' else 4800 if run_type=='find_lr' else 500
    rollouts = rollout.rollout(
        expert,
        vec_env,
        # rollout.make_sample_until(min_timesteps=50000),
        # rollout.make_sample_until(min_timesteps=2e6 if run_type=='train' else 4800 if run_type=='find_lr' else 500),
        rollout.make_sample_until(min_timesteps=train_timesteps),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)
    # 验证数据
    for env in env_objs:
        env.val()
    rollouts_val = rollout.rollout(
        expert,
        vec_env,
        rollout.make_sample_until(min_timesteps=int(train_timesteps*0.2)),
        rng=rng,
    )
    transitions_val = rollout.flatten_trajectories(rollouts_val)
    memory_usage2 = psutil.virtual_memory()
    msg = ''
    cost_msg = f'生成专家数据耗时: {time.time() - t:.2f} 秒'
    log(cost_msg)
    msg += cost_msg + '\n'
    mem_pct_msg = f"CPU 内存占用：{memory_usage2.percent}% ({memory_usage2.used/1024**3:.3f}GB/{memory_usage2.total/1024**3:.3f}GB)"
    log(mem_pct_msg)
    msg += mem_pct_msg + '\n'
    mem_expert_msg = f"专家数据内存占用：{(memory_usage2.used - memory_usage.used)/1024**3:.3f}GB"
    log(mem_expert_msg)
    msg += mem_expert_msg + '\n'

    # 检查 transitions 样本均衡度
    label_balance = f'训练样本均衡度: {cal_action_balance(transitions)}'
    log(label_balance)
    msg += label_balance + '\n'
    label_balance = f'验证样本均衡度: {cal_action_balance(transitions_val)}'
    log(label_balance)
    msg += label_balance + '\n'
    send_wx(msg)
    # sys.exit()

    if run_type == 'test_transitions':
        # 测试可视化transitions
        # 初始化模型
        env_config['data_type'] = 'train'
        env_config['render_mode'] = 'human'
        env = LOB_trade_env(env_config)
        # 专家, 用于参考
        expert = LobExpert_file(pre_cache=False)
        for t in transitions:
            stop = env.set_state(t, expert)
            env.render()
            if stop:
                input('press any key to continue')
        sys.exit()

    total_epochs = 40 if run_type!='test_model' else 10000000000000000
    batch_size = 32
    max_lr = 0.022# find_best_lr
    batch_n = 2**5 if run_type=='train' else 1
    batch_n = 1
    total_steps = total_epochs * len(transitions) // (batch_size * batch_n)
    bc_trainer = BCWithLRScheduler(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        demonstrations_val=transitions_val,
        policy=model.policy,
        rng=rng,
        batch_size=batch_size * batch_n if run_type=='train' else batch_size,
        optimizer_kwargs={'lr': 1e-6} if run_type=='find_lr' else None,
        custom_logger=custom_logger,
        lr_scheduler_cls = OneCycleLR if run_type=='train' else MultiplicativeLR if run_type=='find_lr' else None,
        lr_scheduler_kwargs = {'max_lr':max_lr*batch_n, 'total_steps': total_steps} if run_type=='train' else {'lr_lambda': lambda epoch: 1.1},
    )

    env = env_objs[0]

    checkpoint_interval = 1 if run_type!='test_model' else 500
    begin = 0
    # 读取之前的i
    loop_i_file = os.path.join(train_folder, 'checkpoint', 'loop_i')
    if os.path.exists(loop_i_file):
        begin = int(open(loop_i_file, 'r').read()) + 1
    for i in range(begin, total_epochs // checkpoint_interval):
        _t = time.time()
        bc_trainer.policy.train()
        bc_trainer.train(
            n_epochs=checkpoint_interval,
            log_interval = 1 if run_type=='find_lr' else 500,
        )
        log(f'训练耗时: {time.time() - _t:.2f} 秒')

        # 检查梯度
        check_gradients(bc_trainer)

        # 验证模型
        _t = time.time()
        env.val()
        val_reward, _ = evaluate_policy(bc_trainer.policy, env)
        env.train()
        train_reward, _ = evaluate_policy(bc_trainer.policy, env)
        log(f"train_reward: {train_reward}, val_reward: {val_reward}, 验证耗时: {time.time() - _t:.2f} 秒")

        # 合并到 progress_all.csv
        progress_file = os.path.join(train_folder, f"progress.csv")
        progress_file_all = os.path.join(train_folder, f"progress_all.csv")
        if os.path.exists(progress_file_all):
            df_progress = pd.read_csv(progress_file_all)
        else:
            df_progress = pd.DataFrame()
        df_new = pd.read_csv(progress_file).iloc[len(df_progress):]
        df_new['bc/epoch'] += i * checkpoint_interval
        df_new['bc/mean_reward'] = np.nan
        df_new['bc/val_mean_reward'] = np.nan
        df_new['bc/mean_reward'].iloc[-1] = train_reward
        df_new['bc/val_mean_reward'].iloc[-1] = val_reward
        df_progress = pd.concat([df_progress, df_new])
        df_progress.ffill(inplace=True)
        df_progress.to_csv(progress_file_all, index=False)

        # 训练进度可视化
        try:
            plot_bc_train_progress(train_folder, df_progress=df_progress, title=train_title)
        except Exception as e:
            pickle.dump(df_progress, open('df_progress.pkl', 'wb'))
            log(f"训练进度可视化失败")
            raise e
        
        # 记录当前训练进度
        open(loop_i_file, 'w').write(str(i))
        
        # 保存模型
        bc_trainer.policy.save(os.path.join(train_folder, 'checkpoint', train_folder))

        # 上传
        if not in_windows():
            train_folder_manager.push()

        if run_type == 'find_lr':
            # 限制在 150 条 
            # 4800 / 32 = 150
            break
else:
    # test
    model_folder = rf'D:\code\dl_helper\dl_helper\tests\rl\SB3\{train_folder}'

    # 初始化模型
    env_config['data_type'] = 'test'
    env_config['render_mode'] = 'human'
    test_env = LOB_trade_env(env_config)
    test_env.test()
    model = PPO(
        model_type, 
        test_env, 
        verbose=1, 
        learning_rate=1e-3,
        ent_coef=0.2,
        gamma=0.97,
        policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
    )

    # 加载参数
    model.policy.load(model_folder)

    # 专家, 用于参考
    expert = LobExpert_file(pre_cache=False)
    
    # 测试
    rounds = 5
    rounds = 1
    for i in range(rounds):
        log('reset')
        seed = random.randint(0, 1000000)
        # seed = 646508
        obs, info = test_env.reset(seed)
        test_env.render()

        act = 1
        need_close = False
        while not need_close:
            action, _state = model.predict(obs, deterministic=True)
            expert.get_action(obs)
            expert.add_potential_data_to_env(test_env)

            obs, reward, terminated, truncated, info = test_env.step(action)
            test_env.render()
            need_close = terminated or truncated
            
        log(f'seed: {seed}')
        if rounds > 1:
            keep_play = input('keep play? (y)')
            if keep_play == 'y':
                continue
            else:
                break

    input('all done, press enter to close')
    test_env.close()