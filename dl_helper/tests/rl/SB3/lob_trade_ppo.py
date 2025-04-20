import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecCheckNan
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pandas as pd
import torch
import torch.nn as nn
import torch as th
th.autograd.set_detect_anomaly(True)
import pygame
import time
import numpy as np
import random
import sys, shutil
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ['ALIST_USER'] = 'admin'
os.environ['ALIST_PWD'] = 'LHss6632673'

from py_ext.lzma import decompress, compress_folder
from py_ext.alist import alist
from py_ext.tool import log, init_logger
from py_ext.datetime import beijing_time

from dl_helper.rl.rl_env.lob_trade.lob_env import LOB_trade_env
from dl_helper.rl.rl_env.lob_trade.lob_expert import LobExpert_file
from dl_helper.rl.rl_utils import CustomCheckpointCallback
from dl_helper.tool import report_memory_usage, in_windows
from dl_helper.train_folder_manager import TrainFolderManagerSB3

model_type = 'CnnPolicy'
# 'train' or 'test'
run_type = 'train'

train_folder = train_title = f'lob_trade_ppo'
os.makedirs(train_folder, exist_ok=True)
log_name = f'{train_title}_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)

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
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
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

        # 自注意力层
        # self.attention = SelfAttention(128)
        # self.improved_attention = ImprovedSelfAttention(64)

        # 静态特征处理
        self.static_net = nn.Sequential(
            nn.Linear(self.extra_input_dims, self.extra_input_dims * 4),
            nn.LayerNorm(self.extra_input_dims * 4),
            nn.ReLU(),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + self.extra_input_dims * 4, 128),
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
    return env

if run_type == 'train':
    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecCheckNan(env)  # 添加nan检查
    env = VecMonitor(env)  # 添加监控器

    model = PPO(
        model_type, 
        env, 
        ent_coef=0.01,
        verbose=1, 
        policy_kwargs=policy_kwargs
    )

    # 训练文件夹管理
    if not in_windows():
        train_folder_manager = TrainFolderManagerSB3(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(model, custom_objects= {"policy_kwargs": policy_kwargs})

    # 打印模型结构
    log("模型结构:")
    log(model.policy)
    log(f'参数量: {sum(p.numel() for p in model.policy.parameters())}')

    # test_x = env_objs[0].observation_space.sample()
    # test_x = torch.from_numpy(test_x).unsqueeze(0)
    # log(test_x.shape)
    # test_x = test_x.float().to(model.policy.device)
    # out = model.policy.features_extractor(test_x)
    # log(out.shape)
    # sys.exit()

    for i in range(10000000000000):
        model.learn(total_timesteps=50_000, callback=[checkpoint_callback])
        # model.save(os.path.join(train_folder, 'checkpoint', f"{train_folder}.zip"))
        model.save(train_folder_manager.check_point_file())

        # 打包文训练件夹，并上传到alist
        if not in_windows():
            train_folder_manager.push()

else:
    # test
    model_file = rf'D:\code\dl_helper\dl_helper\tests\rl\SB3\{train_folder}.zip'

    # 初始化模型
    env_config['data_type'] = 'test'
    env_config['render_mode'] = 'human'
    test_env = LOB_trade_env(env_config)
    test_env.test()
    model = PPO(
        model_type, 
        test_env, 
        ent_coef=0.01,
        verbose=1, 
        policy_kwargs=policy_kwargs
    )

    # 加载参数
    _model = model.load(model_file, custom_objects= {"policy_kwargs": policy_kwargs})
    policy_state_dict = _model.policy.state_dict()  
    model.policy.load_state_dict(policy_state_dict)  

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
            
            # 只作为参考
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