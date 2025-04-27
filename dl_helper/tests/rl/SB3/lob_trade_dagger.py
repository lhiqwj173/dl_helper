# !pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz > /dev/null 2>&1
# !mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git > /dev/null 2>&1
# !cd /kaggle/working/3rd/dl_helper && pip install -e . > /dev/null 2>&1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecCheckNan
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy

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

from dl_helper.rl.rl_env.lob_trade.lob_const import USE_CODES
from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_env.lob_trade.lob_expert import LobExpert_file
from dl_helper.rl.rl_utils import plot_bc_train_progress, CustomCheckpointCallback, check_gradients, cal_action_balance
from dl_helper.tool import report_memory_usage, in_windows
from dl_helper.train_folder_manager import TrainFolderManagerBC

from dl_helper.rl.custom_imitation_module.bc import BCWithLRScheduler
from dl_helper.rl.custom_imitation_module.rollout import rollouts_filter, combing_trajectories, load_trajectories
from dl_helper.rl.custom_imitation_module.dagger import SimpleDAggerTrainer

"""
使用 BC policy 进行 DAgger 训练
1. 初始的专家轨迹从文件中读取
2. 学习率使用 BC max_lr / 1e2, 手动调度调整
3. 需要跳过第0轮的训练
"""

model_type = 'CnnPolicy'
# 'train' or 'test' or 'test_model'
# test_model: 使用相同的batch数据，测试模型拟合是否正常

run_type = 'train'
# run_type = 'test'
# run_type = 'test_model'

#################################
# 命令行参数
arg_lr = None
arg_batch_n = None
arg_title = None
#################################
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == 'train':
            run_type = 'train'
        elif arg == 'test':
            run_type = 'test'
        elif arg == 'test_model':
            run_type = 'test_model'
        elif arg.startswith('lr='):
            arg_lr = float(arg.split('=')[1])
        elif arg.startswith('batch_n='):
            arg_batch_n = int(arg.split('=')[1])
        elif arg.startswith('title='):
            arg_title = arg.split('=')[1]

train_folder = train_title = f'20250422_lob_trade_dagger' \
    + ('' if arg_lr is None else f'_lr{arg_lr:2e}') \
        + ('' if arg_batch_n is None else f'_batch_n{arg_batch_n}') \
            + '' if arg_title is None else f'_{arg_title}'
            
log_name = f'{train_title}_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)

#################################
# 训练参数
checkpoint_interval = 1 if run_type!='test_model' else 500
batch_size = 32
max_lr = 5.5e-5# find_best_lr
batch_n = 2**7 if run_type=='train' else 1
batch_n = batch_n if arg_batch_n is None else arg_batch_n
lr = batch_n * max_lr / 1e2 if arg_lr is None else arg_lr
# 每轮新增的样本数量
each_round_train_num = batch_n * batch_size * (2**2)    
#################################

custom_logger = imit_logger.configure(
    folder=train_folder,
    format_strs=["csv", "stdout"],
)

# 自定义特征提取器
# 参数量: 176827
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
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),  # 添加 Dropout2d
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),  # 添加 Dropout2d
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),  # 添加 Dropout2d
        )

        # Inception 模块
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.ReLU(),
        )

        # LSTM 层 
        self.lstm = nn.LSTM(input_size=64*3, hidden_size=64, num_layers=1, batch_first=True)

        # id 嵌入层
        self.id_embedding = nn.Embedding(len(USE_CODES), 8)

        # 计算静态特征的输入维度
        static_input_dim = 8 + (self.extra_input_dims - 1)  # 嵌入维度 + 数值特征维度

        # 静态特征处理
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, static_input_dim * 4),
            nn.LayerNorm(static_input_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # 融合层
        fusion_input_dim = 64 + static_input_dim * 4  # LSTM 输出 64 维 + static_net 输出
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
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

        # 处理静态特征
        # extra_x 第1列是类别特征（整数索引）
        cat_feat = extra_x[:, 0].long()  # (B,)，转换为整数类型
        num_feat = extra_x[:, 1:]  # (B, self.extra_input_dims - 1)，数值特征

        # 嵌入类别特征
        embedded = self.id_embedding(cat_feat)  # (B, 8)

        # 拼接嵌入向量和数值特征
        static_input = torch.cat([embedded, num_feat], dim=1)  # (B, 8 + self.extra_input_dims - 1)

        # 静态特征处理
        static_out = self.static_net(static_input)  # (B, static_input_dim * 4)

        # 卷积块
        x = self.conv1(x)  # (B, 32, 28, 10)
        x = self.conv2(x)  # (B, 32, 26, 5)
        x = self.conv3(x)  # (B, 32, 24, 1)

        # Inception 模块
        x_inp1 = self.inp1(x)  # (B, 64, 24, 1)
        x_inp2 = self.inp2(x)  # (B, 64, 24, 1)
        x_inp3 = self.inp3(x)  # (B, 64, 24, 1)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)  # (B, 192, 24, 1)
        x = nn.Dropout2d(p=0.2)(x)  # 添加 Dropout2d

        # 调整形状以适配 LSTM
        x = x.permute(0, 2, 1, 3).squeeze(3)  # (B, 24, 192)

        # LSTM 处理 取最后一个时间步
        lstm_out, _ = self.lstm(x)  # (B, 24, 64)
        lstm_out = nn.Dropout(p=0.2)(lstm_out)  # 添加 Dropout
        # 取最后一个时间步
        temporal_feat = lstm_out[:, -1, :]  # (B, 64)

        # 融合层
        fused_out = torch.cat([temporal_feat, static_out], dim=1)  # (B, 64 + self.extra_input_dims * 4)
        fused_out = nn.Dropout(p=0.2)(fused_out)  # 添加 Dropout
        fused_out = self.fusion(fused_out)  # (B, features_dim)

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            raise ValueError("fused_out is nan or inf")

        return fused_out


model_config={
    # 自定义编码器参数  
    'input_dims' : (100, 20),
    'extra_input_dims' : 3,
    'features_dim' : 128,
}
env_config ={
    'data_type': 'train',# 训练/测试
    'his_len': 100,# 每个样本的 历史数据长度
    'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
    'train_folder': train_folder,
    'train_title': train_folder,

    # 使用数据增强
    'use_random_his_window': True,# 是否使用随机历史窗口
    'use_gaussian_noise_vol': True,# 是否使用高斯噪声
    'use_spread_add_small_limit_order': True,# 是否使用价差添加小单
}

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

# 配置 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=DeepLob,
    features_extractor_kwargs=model_config,
    net_arch = [64,32],
    activation_fn=nn.ReLU
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
    expert = LobExpert_file(env=env_objs[0], pre_cache=True)

    model = PPO(
        model_type, 
        env, 
        verbose=1, 
        learning_rate=1e-3,
        ent_coef=0.01,
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
    # test_x = test_x.float().to(model.policy.device)
    # out = model.policy.features_extractor(test_x)
    # log(out.shape)
    # sys.exit()

    vec_env = env

    memory_usage = psutil.virtual_memory()
    log(f"内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")

    # 生成验证数据 固定数量
    for env in env_objs:
        env.val()# 切换到验证模式
    rollouts_val = rollout.rollout(
        expert,
        vec_env,
        rollout.make_sample_until(min_timesteps=50_000 if run_type!='test_model' else 500),
        rng=np.random.default_rng(),
    )
    for env in env_objs:
        env.train()# 切换回训练模式
    transitions_val = rollout.flatten_trajectories(rollouts_val)
    memory_usage2 = psutil.virtual_memory()
    msg = ''
    mem_pct_msg = f"内存占用：{memory_usage2.percent}% ({memory_usage2.used/1024**3:.3f}GB/{memory_usage2.total/1024**3:.3f}GB)"
    log(mem_pct_msg)
    msg += mem_pct_msg + '\n'
    mem_expert_msg = f"专家数据内存占用：{(memory_usage2.used - memory_usage.used)/1024**3:.3f}GB"
    log(mem_expert_msg)
    msg += mem_expert_msg + '\n'

    # 检查 transitions 样本均衡度
    label_balance = f'验证样本均衡度: {cal_action_balance(transitions_val)}'
    log(label_balance)
    msg += label_balance + '\n'
    send_wx(msg)
    # sys.exit()

    bc_trainer = BCWithLRScheduler(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations_val = transitions_val,# 验证数据集
        policy=model.policy,
        rng=np.random.default_rng(),
        batch_size=batch_size * batch_n if run_type=='train' else batch_size,
        optimizer_kwargs={'lr': lr},
        custom_logger=custom_logger,
    )
    
    # 训练文件夹管理
    keep_on_train = False
    if not in_windows():
        train_folder_manager = TrainFolderManagerBC(train_folder)
        if train_folder_manager.exists():
            # 继续训练
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(bc_trainer)
            keep_on_train = True

    if not keep_on_train:
        # 新训练
        # 使用 BC policy 进行 DAgger 训练
        log(f"使用 BC policy 进行新训练")
        policy_folder = rf'/kaggle/input/pre-trained-policy' if not in_windows() else r"D:\L2_DATA_T0_ETF\train_data\RAW\DAGGER_pre_trained_policy"
        bc_trainer.load(policy_folder)

    # 初始化进度数据文件
    progress_file = os.path.join(train_folder, f"progress.csv")
    progress_file_all = os.path.join(train_folder, f"progress_all.csv")

    # 初始化 
    dagger_trainer = SimpleDAggerTrainer(
        venv=vec_env,
        scratch_dir=r'C:\Users\lh\Desktop\temp\temp_dagger' if in_windows() else r"/kaggle/temp",# 临时文件夹，不进行备份
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=np.random.default_rng(),
    )

    dagger_trainer.train(
        total_timesteps = 1_000_000_000_000_000_000, # 不受改参数限制，停止与否取决于训练效果

        train_folder = train_folder,
        train_title = train_title,
        train_folder_manager = train_folder_manager,
        eval_env = env_objs[0],
        progress_file = progress_file,
        progress_file_all = progress_file_all,
        
        rollout_round_min_timesteps = each_round_train_num if run_type!='test_model' else 500,
    )


else:
    # test
    model_folder = rf'D:\code\dl_helper\dl_helper\tests\rl\SB3\{train_folder}'
    # 加载 BC 训练的策略
    pretrained_policy = ActorCriticPolicy.load(model_folder)

    # 初始化模型
    # env_config['data_type'] = 'test'
    env_config['render_mode'] = 'human'
    test_env = LOB_trade_env(env_config)
    test_env.train()
    model = PPO(
        model_type, 
        test_env, 
        policy_kwargs=policy_kwargs if model_type == 'CnnPolicy' else None
    )

    # # 加载参数
    # model.policy.load_state_dict(pretrained_policy.state_dict())

    # 专家, 用于参考
    expert = LobExpert_file(pre_cache=False)
    
    # 测试
    rounds = 5
    # rounds = 1
    for i in range(rounds):
        log('reset')
        seed = random.randint(0, 1000000)
        seed = 477977
        seed = 195789
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