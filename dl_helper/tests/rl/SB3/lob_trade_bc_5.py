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
from torch.optim.lr_scheduler import MultiplicativeLR
from dl_helper.rl.custom_pytorch_module.lrscheduler import OneCycleLR
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
from dl_helper.idx_manager import get_idx
from dl_helper.train_folder_manager import TrainFolderManagerBC

from dl_helper.rl.custom_imitation_module.bc import BCWithLRScheduler
from dl_helper.rl.custom_imitation_module.dataset import LobTrajectoryDataset
from dl_helper.rl.custom_imitation_module.rollout import rollouts_filter, combing_trajectories, load_trajectories

model_type = 'CnnPolicy'
# 'train' or 'test' or 'find_lr' or 'test_model' or 'test_transitions
# find_lr: 学习率从 1e-6 > 指数增长，限制总batch为150
# test_model: 使用相同的batch数据，测试模型拟合是否正常
# test_transitions: 测试可视化transitions
# 查找最大学习率
"""
from dl_helper.rl.rl_utils import find_best_lr
df_progress = pd.read_csv('progress_all.csv')
find_best_lr(df_progress.iloc[50:97]['bc/lr'], df_progress.iloc[50:97]['bc/loss'])
"""
run_type = 'train'
# run_type = 'find_lr'
# run_type = 'test'
# run_type = 'test_transitions'
# run_type = 'bc_data'

#################################
# 命令行参数
arg_lr = None
arg_max_lr = None
arg_batch_n = None
arg_total_epochs = None
arg_l2_weight = None
arg_dropout = None
arg_amp = None
#################################
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
        elif arg == 'bc_data':
            run_type = 'bc_data'
        elif arg.startswith('lr='):
            arg_lr = float(arg.split('=')[1])
        elif arg.startswith('batch_n='):
            arg_batch_n = int(arg.split('=')[1])
        elif arg.startswith('total_epochs='):
            arg_total_epochs = int(arg.split('=')[1])
        elif arg.startswith('maxlr=') or arg.startswith('max_lr='):
            arg_max_lr = float(arg.split('=')[1])
        elif arg.startswith('l2_weight='):
            arg_l2_weight = float(arg.split('=')[1])
        elif arg.startswith('dropout='):
            arg_dropout = float(arg.split('=')[1])
        elif arg == 'amp':
            arg_amp = True

train_folder = train_title = f'20250508_lob_trade_bc_5' \
    + ('' if arg_lr is None else f'_lr{arg_lr:.0e}') \
        + ('' if arg_batch_n is None else f'_batch_n{arg_batch_n}') \
            + ('' if arg_total_epochs is None else f'_epochs{arg_total_epochs}') \
                + ('' if arg_max_lr is None else f'_maxlr{arg_max_lr:.0e}') \
                    + ('' if arg_l2_weight is None else f'_l2weight{arg_l2_weight:.0e}') \
                        + ('' if arg_dropout is None else f'_dropout{arg_dropout:.0e}') \
                            + ('' if arg_amp is None else f'_amp')
            
log_name = f'{train_title}_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)

#################################
# 训练参数
total_epochs = 1 if run_type=='find_lr' else 80 if run_type!='test_model' else 10000000000000000
total_epochs = total_epochs if arg_total_epochs is None else arg_total_epochs
checkpoint_interval = 1 if run_type!='test_model' else 500
batch_size = 32
max_lr = 3e-5 # find_best_lr
max_lr = arg_max_lr if arg_max_lr else max_lr
train_kaggle_batch_n = 2**7 if not arg_amp else 2**8
batch_n = train_kaggle_batch_n if (run_type=='train' and not in_windows()) else 1
batch_n = batch_n if arg_batch_n is None else arg_batch_n
#################################

custom_logger = imit_logger.configure(
    folder=train_folder,
    format_strs=["csv", "stdout"],
)

# 自定义特征提取器
# 参数量: 584455
class DeepLob_xlarge(BaseFeaturesExtractor):
    net_arch = [64,32]
    def __init__(
            self,
            observation_space,
            features_dim: int = 128,
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
            dropout_rate: float = arg_dropout,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.dropout_rate = dropout_rate

        # 卷积块1 - 添加LayerNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([32, input_dims[0], (input_dims[1] // 2)]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.LayerNorm([32, input_dims[0] - 1, (input_dims[1] // 2)]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.LayerNorm([32, input_dims[0] - 2, (input_dims[1] // 2)]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 卷积块2 - 添加LayerNorm
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([32, input_dims[0] - 2, (input_dims[1] // 4)]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.LayerNorm([32, input_dims[0] - 3, (input_dims[1] // 4)]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.LayerNorm([32, input_dims[0] - 4, (input_dims[1] // 4)]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 卷积块3 - 添加LayerNorm
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5)),
            nn.LayerNorm([32, input_dims[0] - 4, (input_dims[1] // 4) - 4]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.LayerNorm([32, input_dims[0] - 5, (input_dims[1] // 4) - 4]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.LayerNorm([32, input_dims[0] - 6, (input_dims[1] // 4) - 4]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )

        # Inception 模块 - 添加LayerNorm
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LayerNorm([64, input_dims[0] - 6, (input_dims[1] // 4) - 4]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.LayerNorm([64, input_dims[0] - 6, (input_dims[1] // 4) - 4]),  # 添加LayerNorm
            nn.ReLU(),
        )
        
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LayerNorm([64, input_dims[0] - 6, (input_dims[1] // 4) - 4]),  # 添加LayerNorm
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LayerNorm([64, input_dims[0] - 6, (input_dims[1] // 4) - 4]),  # 添加LayerNorm
            nn.ReLU(),
        )
        
        self.inp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LayerNorm([64, input_dims[0] - 6, (input_dims[1] // 4) - 4]),  # 添加LayerNorm
            nn.ReLU(),
        )

        # 计算LSTM输入维度
        # 通过推理得到卷积后的形状大小
        self.lstm_seq_len = input_dims[0] - 6  # 经过卷积块后的序列长度
        
        # LSTM 层 - 添加LayerNorm
        self.lstm = nn.LSTM(input_size=64*3, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm_norm = nn.LayerNorm([self.lstm_seq_len, 64])  # LSTM输出的LayerNorm

        # id 嵌入层
        self.id_embedding = nn.Embedding(5, 8)

        # 计算静态特征的输入维度
        static_input_dim = 8 + (self.extra_input_dims - 1)  # 嵌入维度 + 数值特征维度

        # 静态特征处理 - 已有LayerNorm，进一步优化结构
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, static_input_dim * 2),
            nn.LayerNorm(static_input_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
            nn.Linear(static_input_dim * 2, static_input_dim * 4),
            nn.LayerNorm(static_input_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
        )

        # 融合层 - 添加残差连接和额外的LayerNorm
        fusion_input_dim = 64 + static_input_dim * 4  # LSTM 输出 64 维 + static_net 输出
        self.fusion_pre = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 添加残差连接
        self.fusion_res = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )
        
        # 输出层
        self.fusion_out = nn.Linear(128, features_dim)

        self.dp = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()
        self.dp2d = nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 对 CNN 和全连接层应用 He 初始化
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # 对LayerNorm进行初始化
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
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
                # 偏置初始化为零，但将遗忘门偏置设为 1 以改善长期记忆
                bias_size = param.size(0)
                param.data.fill_(0.)
                param.data[bias_size//4:bias_size//2].fill_(1.0)  # 遗忘门偏置设为 1

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息，不参与模型计算，用于专家透视未来数据
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)  # (B, 1, 10, 20)

        # 处理静态特征
        cat_feat = extra_x[:, 0].long()  # (B,)，转换为整数类型
        num_feat = extra_x[:, 1:]  # (B, self.extra_input_dims - 1)，数值特征

        # 嵌入类别特征
        embedded = self.id_embedding(cat_feat)  # (B, 8)

        # 拼接嵌入向量和数值特征
        static_input = torch.cat([embedded, num_feat], dim=1)  # (B, 8 + self.extra_input_dims - 1)

        # 静态特征处理
        static_out = self.static_net(static_input)  # (B, static_input_dim * 4)

        # 卷积处理 - 添加梯度检查点以优化内存使用
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x)  

        # Inception 模块
        x_inp1 = self.inp1(x)  
        x_inp2 = self.inp2(x)  
        x_inp3 = self.inp3(x)  
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)  # (B, 192, seq_len, 1)
        x = self.dp2d(x)  # 添加 Dropout2d

        # 调整形状以适配 LSTM
        x = x.permute(0, 2, 1, 3).squeeze(3)  # (B, seq_len, 192)

        # LSTM 处理
        lstm_out, _ = self.lstm(x)  # (B, seq_len, 64)
        lstm_out = self.lstm_norm(lstm_out)  # 应用LayerNorm
        lstm_out = self.dp(lstm_out)  # 添加 Dropout
        # 取最后一个时间步
        temporal_feat = lstm_out[:, -1, :]  # (B, 64)

        # 融合层
        fused_input = torch.cat([temporal_feat, static_out], dim=1)  # (B, 64 + static_input_dim * 4)
        fused_out = self.fusion_pre(fused_input)  # (B, 128)
        
        # 添加残差连接
        residual = fused_out
        fused_out = self.fusion_res(fused_out)
        fused_out = fused_out + residual  # 残差连接
        
        # 输出层
        fused_out = self.fusion_out(fused_out)  # (B, features_dim)

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            # 保存出现问题的数据到本地
            log(f"发现 nan 或 inf 值, 保存数据到本地进行检查")
            model_sd = self.state_dict()
            # 确保在 CPU 上
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("fused_out is nan or inf")

        return fused_out

# 参数量: 391061
class DeepLob_large(BaseFeaturesExtractor):
    net_arch = [48, 24]
    def __init__(
            self,
            observation_space,
            features_dim: int = 64,
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
            dropout_rate: float = arg_dropout,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.dropout_rate = dropout_rate

        # 通道数量介于large和small之间
        conv1_out = 24  # large是32，small是16
        conv2_out = 28  # large是32，small是24
        conv3_out = 32  # 保持32通道
        inception_out = 48  # large是64，small是32

        # 卷积块1 - 保留三层卷积但减少通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([conv1_out, input_dims[0], (input_dims[1] // 2)]),
            nn.ReLU(),
            nn.Conv2d(conv1_out, conv1_out, kernel_size=(2, 1)),
            nn.LayerNorm([conv1_out, input_dims[0] - 1, (input_dims[1] // 2)]),
            nn.ReLU(),
            nn.Conv2d(conv1_out, conv1_out, kernel_size=(2, 1)),
            nn.LayerNorm([conv1_out, input_dims[0] - 2, (input_dims[1] // 2)]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 卷积块2 - 保留large的结构但减少通道数
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([conv2_out, input_dims[0] - 2, (input_dims[1] // 4)]),
            nn.ReLU(),
            nn.Conv2d(conv2_out, conv2_out, kernel_size=(2, 1)),
            nn.LayerNorm([conv2_out, input_dims[0] - 3, (input_dims[1] // 4)]),
            nn.ReLU(),
            nn.Conv2d(conv2_out, conv2_out, kernel_size=(2, 1)),
            nn.LayerNorm([conv2_out, input_dims[0] - 4, (input_dims[1] // 4)]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 卷积块3 - 介于large和small之间的复杂度
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv2_out, conv3_out, kernel_size=(1, 5)),
            nn.LayerNorm([conv3_out, input_dims[0] - 4, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
            nn.Conv2d(conv3_out, conv3_out, kernel_size=(2, 1)),
            nn.LayerNorm([conv3_out, input_dims[0] - 5, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )

        # Inception 模块 - 保留三个分支但减少通道数
        self.inp1 = nn.Sequential(
            nn.Conv2d(conv3_out, inception_out, kernel_size=(1, 1)),
            nn.LayerNorm([inception_out, input_dims[0] - 5, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
            nn.Conv2d(inception_out, inception_out, kernel_size=(3, 1), padding=(1, 0)),
            nn.LayerNorm([inception_out, input_dims[0] - 5, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
        )
        
        self.inp2 = nn.Sequential(
            nn.Conv2d(conv3_out, inception_out, kernel_size=(1, 1)),
            nn.LayerNorm([inception_out, input_dims[0] - 5, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
            nn.Conv2d(inception_out, inception_out, kernel_size=(5, 1), padding=(2, 0)),
            nn.LayerNorm([inception_out, input_dims[0] - 5, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
        )
        
        self.inp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(conv3_out, inception_out, kernel_size=(1, 1)),
            nn.LayerNorm([inception_out, input_dims[0] - 5, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
        )

        # 计算LSTM输入维度
        self.lstm_seq_len = input_dims[0] - 5  # 经过卷积块后的序列长度
        lstm_hidden_dim = 48  # large是64，small是32
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size=inception_out*3, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)
        self.lstm_norm = nn.LayerNorm([self.lstm_seq_len, lstm_hidden_dim])

        # id 嵌入层
        embedding_dim = 6  # large是8，small是4
        self.id_embedding = nn.Embedding(5, embedding_dim)

        # 计算静态特征的输入维度
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        static_hidden_dim = static_input_dim * 3  # 介于large和small之间

        # 静态特征处理网络 - 比small复杂但比large简单
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.LayerNorm(static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
            nn.Linear(static_hidden_dim, static_hidden_dim * 2),
            nn.LayerNorm(static_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
        )

        # 融合层 - 介于large的复杂残差结构和small的简单结构之间
        fusion_input_dim = lstm_hidden_dim + static_hidden_dim * 2
        fusion_hidden_dim = 96  # large是128，small是64
        
        self.fusion_pre = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 简化的残差连接
        self.fusion_out = nn.Linear(fusion_hidden_dim, features_dim)

        self.dp = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()
        self.dp2d = nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 对 CNN 和全连接层应用 He 初始化
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # 对LayerNorm进行初始化
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
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
                # 偏置初始化为零，但将遗忘门偏置设为 1 以改善长期记忆
                bias_size = param.size(0)
                param.data.fill_(0.)
                param.data[bias_size//4:bias_size//2].fill_(1.0)  # 遗忘门偏置设为 1

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息，不参与模型计算
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)  # (B, 1, 10, 20)

        # 处理静态特征
        cat_feat = extra_x[:, 0].long()  # (B,)，转换为整数类型
        num_feat = extra_x[:, 1:]  # (B, self.extra_input_dims - 1)，数值特征

        # 嵌入类别特征
        embedded = self.id_embedding(cat_feat)

        # 拼接嵌入向量和数值特征
        static_input = torch.cat([embedded, num_feat], dim=1)

        # 静态特征处理
        static_out = self.static_net(static_input)

        # 卷积处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Inception 模块
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = self.dp2d(x)

        # 调整形状以适配 LSTM
        x = x.permute(0, 2, 1, 3).squeeze(3)

        # LSTM 处理
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.dp(lstm_out)
        # 取最后一个时间步
        temporal_feat = lstm_out[:, -1, :]

        # 融合层
        fused_input = torch.cat([temporal_feat, static_out], dim=1)
        fused_out = self.fusion_pre(fused_input)
        fused_out = self.fusion_out(fused_out)

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            # 保存出现问题的数据到本地
            log(f"发现 nan 或 inf 值, 保存数据到本地进行检查")
            model_sd = self.state_dict()
            # 确保在 CPU 上
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("fused_out is nan or inf")

        return fused_out

# 参数量: 174931
class DeepLob(BaseFeaturesExtractor):
    net_arch = [32,16]
    def __init__(
            self,
            observation_space,
            features_dim: int = 32,  # 减少特征维度
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
            dropout_rate: float = arg_dropout,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.dropout_rate = dropout_rate

        # 减少通道数
        conv1_out = 16  # 从32减少到16
        conv2_out = 24  # 从32减少到24
        conv3_out = 32  # 保持32通道
        inception_out = 32  # 从64减少到32
        
        # 卷积块1 - 添加LayerNorm但减少通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([conv1_out, input_dims[0], (input_dims[1] // 2)]),
            nn.ReLU(),
            nn.Conv2d(conv1_out, conv1_out, kernel_size=(2, 1)),
            nn.LayerNorm([conv1_out, input_dims[0] - 1, (input_dims[1] // 2)]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 卷积块2 - 减少层数和通道数
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([conv2_out, input_dims[0] - 1, (input_dims[1] // 4)]),
            nn.ReLU(),
            nn.Conv2d(conv2_out, conv2_out, kernel_size=(2, 1)),
            nn.LayerNorm([conv2_out, input_dims[0] - 2, (input_dims[1] // 4)]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 卷积块3 - 简化结构
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv2_out, conv3_out, kernel_size=(1, 5)),
            nn.LayerNorm([conv3_out, input_dims[0] - 2, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )

        # 简化的Inception模块 - 只保留两个分支
        self.inp1 = nn.Sequential(
            nn.Conv2d(conv3_out, inception_out, kernel_size=(1, 1)),
            nn.LayerNorm([inception_out, input_dims[0] - 2, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
            nn.Conv2d(inception_out, inception_out, kernel_size=(3, 1), padding=(1, 0)),
            nn.LayerNorm([inception_out, input_dims[0] - 2, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
        )
        
        self.inp2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(conv3_out, inception_out, kernel_size=(1, 1)),
            nn.LayerNorm([inception_out, input_dims[0] - 2, (input_dims[1] // 4) - 4]),
            nn.ReLU(),
        )

        # 计算LSTM输入维度 - 由于减少了一些卷积层，序列长度变化
        self.lstm_seq_len = input_dims[0] - 2  # 更新后的序列长度
        lstm_hidden_dim = 32  # 从64减少到32
        
        # LSTM层 - 减少隐藏单元
        self.lstm = nn.LSTM(input_size=inception_out*2, hidden_size=lstm_hidden_dim, 
                           num_layers=1, batch_first=True)
        self.lstm_norm = nn.LayerNorm([self.lstm_seq_len, lstm_hidden_dim])

        # ID嵌入层 - 减少嵌入维度
        embedding_dim = 4  # 从8减少到4
        self.id_embedding = nn.Embedding(5, embedding_dim)

        # 计算静态特征的输入维度
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        static_hidden_dim = static_input_dim * 2  # 减少隐藏层大小

        # 简化的静态特征处理网络
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.LayerNorm(static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
        )

        # 融合层 - 简化结构
        fusion_input_dim = lstm_hidden_dim + static_hidden_dim
        fusion_hidden_dim = 64  # 从128减少到64
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
            nn.Linear(fusion_hidden_dim, features_dim),
        )

        self.dp = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()
        self.dp2d = nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 对CNN和全连接层应用He初始化
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # LayerNorm初始化
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # LSTM特定初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                bias_size = param.size(0)
                param.data.fill_(0.)
                param.data[bias_size//4:bias_size//2].fill_(1.0)  # 遗忘门偏置设为1

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)

        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]

        # 嵌入类别特征
        embedded = self.id_embedding(cat_feat)

        # 拼接嵌入向量和数值特征
        static_input = torch.cat([embedded, num_feat], dim=1)

        # 静态特征处理
        static_out = self.static_net(static_input)

        # 卷积处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 简化的Inception模块 - 只有两个分支
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x = torch.cat((x_inp1, x_inp2), dim=1)
        x = self.dp2d(x)

        # 调整形状以适配LSTM
        x = x.permute(0, 2, 1, 3).squeeze(3)

        # LSTM处理
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.dp(lstm_out)
        # 取最后一个时间步
        temporal_feat = lstm_out[:, -1, :]

        # 融合层
        fused_input = torch.cat([temporal_feat, static_out], dim=1)
        fused_out = self.fusion(fused_input)

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            # 保存出现问题的数据到本地
            log(f"发现 nan 或 inf 值, 保存数据到本地进行检查")
            model_sd = self.state_dict()
            # 确保在 CPU 上
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("fused_out is nan or inf")

        return fused_out

# 参数量: 70703
class DeepLob_mid(BaseFeaturesExtractor):
    net_arch = [16,16]
    def __init__(
            self,
            observation_space,
            features_dim: int = 16,  # 进一步减少特征维度
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
            dropout_rate: float = arg_dropout,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.dropout_rate = dropout_rate

        # 进一步减少通道数和层数
        conv1_out = 8   # 从16减少到8
        conv2_out = 16  # 从24减少到16
        lstm_hidden = 16  # 从32减少到16
        
        # 第一卷积块 - 简化为单层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([conv1_out, input_dims[0], input_dims[1] // 2]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 第二卷积块 - 简化为单层
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([conv2_out, input_dims[0], input_dims[1] // 4]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),
        )
        
        # 移除第三个卷积块，直接连接到特征提取
        # 使用单个特征提取器替代Inception模块
        self.feature_extract = nn.Sequential(
            nn.Conv2d(conv2_out, lstm_hidden*2, kernel_size=(2, 1), padding=(1, 0)),
            nn.LayerNorm([lstm_hidden*2, input_dims[0], input_dims[1] // 4]),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((input_dims[0] // 2, 1))
        )

        # 简化LSTM为GRU来减少参数
        self.rnn = nn.GRU(
            input_size=lstm_hidden*2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # 减少嵌入维度
        embedding_dim = 4
        self.id_embedding = nn.Embedding(5, embedding_dim)

        # 简化静态特征处理
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
        )

        # 简化融合层
        fusion_input_dim = lstm_hidden*2  # LSTM输出 + 静态特征
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim*2),
            nn.LayerNorm(features_dim*2),
            nn.ReLU(),
            nn.Linear(features_dim*2, features_dim),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 简化的初始化 - 只应用于重要的参数
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)

        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]

        # 嵌入类别特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = self.static_net(static_input)

        # 卷积处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.feature_extract(x)

        # 调整形状以适配RNN
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(1))  # (B, seq_len, features)

        # RNN处理
        rnn_out, _ = self.rnn(x)
        
        # 取最后一个时间步或使用全局平均池化
        rnn_feat = rnn_out[:, -1, :]
        
        # 融合静态特征和时序特征
        fused_features = torch.cat([rnn_feat, static_out], dim=1)
        output = self.fusion(fused_features)

        # 数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            # 保存出现问题的数据到本地
            log(f"发现 nan 或 inf 值, 保存数据到本地进行检查")
            model_sd = self.state_dict()
            # 确保在 CPU 上
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("fused_out is nan or inf")

        return output

# 参数量: 34206
class DeepLob_small(BaseFeaturesExtractor):
    net_arch = [10, 10]
    def __init__(
            self,
            observation_space,
            features_dim: int = 10,  # 轻量特征维度
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
            dropout_rate: float = arg_dropout,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.dropout_rate = dropout_rate

        # 轻量化的通道数设计
        conv1_out = 6  # 减少第一层通道数
        conv2_out = 10 # 减少第二层通道数
        hidden_dim = 10 # 隐藏层维度
        
        # 使用分组卷积减少参数量
        self.conv_net = nn.Sequential(
            # 第一层卷积 - 正常卷积
            nn.Conv2d(1, conv1_out, kernel_size=(1, 2), stride=(1, 2)),
            nn.LayerNorm([conv1_out, input_dims[0], input_dims[1] // 2]),
            nn.ReLU(),
            
            # 第二层卷积 - 分组卷积减少参数
            nn.Conv2d(conv1_out, conv2_out, kernel_size=(1, 2), stride=(1, 2), groups=2),
            nn.LayerNorm([conv2_out, input_dims[0], input_dims[1] // 4]),
            nn.ReLU(),
            
            # 特征提取层 - 分组卷积并融合两种感受野
            nn.Conv2d(conv2_out, hidden_dim, kernel_size=(3, 1), padding=(1, 0), groups=2),
            nn.LayerNorm([hidden_dim, input_dims[0], input_dims[1] // 4]),
            nn.ReLU(),
            
            # 池化层
            nn.AdaptiveMaxPool2d((input_dims[0] // 2, 1))
        )
        
        # 使用轻量级时序处理 - 用1D卷积替代GRU/LSTM
        self.temporal_net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=2),
            nn.LayerNorm([hidden_dim, input_dims[0] // 2]),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # 全局池化
        )
        
        # 轻量级嵌入
        embedding_dim = 3
        self.id_embedding = nn.Embedding(5, embedding_dim)
        
        # 静态特征处理 - 简化
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, features_dim),
            nn.LayerNorm(features_dim),
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)

        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]

        # 嵌入类别特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = self.static_net(static_input)

        # 卷积特征提取
        batch_size = x.size(0)
        x = self.conv_net(x)
        
        # 调整形状以适配时序处理
        x = x.view(batch_size, x.size(1), -1)  # (B, channels, seq_len)
        
        # 时序特征提取
        temporal_out = self.temporal_net(x).squeeze(-1)  # (B, hidden_dim)
        
        # 融合静态特征和时序特征
        fused_features = torch.cat([temporal_out, static_out], dim=1)
        output = self.fusion(fused_features)

        # 数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            # 保存出现问题的数据到本地
            log(f"发现 nan 或 inf 值, 保存数据到本地进行检查")
            model_sd = self.state_dict()
            # 确保在 CPU 上
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("fused_out is nan or inf")

        return output

# 参数量: 12146
class DeepLob_lite(BaseFeaturesExtractor):
    net_arch = [10, 8]  # 中等大小的网络架构 
    def __init__(
            self,
            observation_space,
            features_dim: int = 8,  # 中等特征维度
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
            dropout_rate: float = arg_dropout,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.dropout_rate = dropout_rate

        # 中等通道数设计 - 确保所有通道数都能被分组数整除
        conv1_out = 6  # 和tiny一样
        conv2_out = 8  # 介于small和tiny之间
        hidden_dim = 10  # 确保能被2整除，用于分组卷积
        
        # 使用分组卷积减少参数量
        self.conv_net = nn.Sequential(
            # 第一层卷积 - 轻量卷积
            nn.Conv2d(1, conv1_out, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.LayerNorm([conv1_out, (input_dims[0] + 1) // 2, (input_dims[1] + 1) // 2]),
            nn.ReLU(),
            
            # 第二层卷积 - 分组卷积减少参数
            nn.Conv2d(conv1_out, conv2_out, kernel_size=(2, 2), stride=(2, 2), 
                     padding=(1, 1), groups=2),  # 6 / 2 = 3 个通道/组, 8个输出通道
            nn.LayerNorm([conv2_out, (input_dims[0] + 3) // 4, (input_dims[1] + 3) // 4]),
            nn.ReLU(),
            
            # 特征提取层 - 使用小卷积核和分组卷积
            nn.Conv2d(conv2_out, hidden_dim, kernel_size=(2, 1), padding=(1, 0), groups=2),  # 8 / 2 = 4 通道/组, 10个输出通道
            nn.LayerNorm([hidden_dim, (input_dims[0] + 5) // 4, (input_dims[1] + 3) // 4]),
            nn.ReLU(),
            
            # 池化层
            nn.AdaptiveMaxPool2d((input_dims[0] // 4, 1))
        )
        
        # 使用轻量级时序处理 - 确保分组数能整除通道数
        self.temporal_net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=2),  # 10 / 2 = 5 通道/组
            nn.LayerNorm([hidden_dim, input_dims[0] // 4]),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 嵌入维度
        embedding_dim = 3  # 与small相同
        self.id_embedding = nn.Embedding(5, embedding_dim)
        
        # 静态特征处理
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)

        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]

        # 嵌入类别特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = self.static_net(static_input)

        # 卷积特征提取
        batch_size = x.size(0)
        x = self.conv_net(x)
        
        # 调整形状以适配时序处理
        x = x.view(batch_size, x.size(1), -1)  # (B, channels, seq_len)
        
        # 时序特征提取
        temporal_out = self.temporal_net(x).squeeze(-1)  # (B, hidden_dim)
        
        # 融合静态特征和时序特征
        fused_features = torch.cat([temporal_out, static_out], dim=1)
        output = self.fusion(fused_features)

        # 数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            # 保存出现问题的数据到本地
            log(f"发现 nan 或 inf 值, 保存数据到本地进行检查")
            model_sd = self.state_dict()
            # 确保在 CPU 上
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("fused_out is nan or inf")

        return output

# 参数量: 8827
class DeepLob_tiny(BaseFeaturesExtractor):
    net_arch = [8,8]
    def __init__(
            self,
            observation_space,
            features_dim: int = 8,  # 极度减少特征维度
            input_dims: tuple = (10, 20),
            extra_input_dims: int = 3,
            dropout_rate: float = arg_dropout,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.dropout_rate = dropout_rate

        # 最小化通道数
        conv1_out = 6   # 极小通道数
        hidden_dim = 8  # 极小隐藏维度
        
        # 统一使用分组卷积来减少参数
        self.feature_extractor = nn.Sequential(
            # 第一层：降采样 + 特征提取
            nn.Conv2d(1, conv1_out, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.LayerNorm([conv1_out, (input_dims[0] + 1) // 2, (input_dims[1] + 1) // 2]),
            nn.ReLU(),
            
            # 第二层：进一步降采样 + 特征集成
            nn.Conv2d(conv1_out, hidden_dim, kernel_size=(2, 2), stride=(2, 2), 
                     padding=(1, 1), groups=2),  # 使用分组卷积减少参数
            nn.LayerNorm([hidden_dim, (input_dims[0] + 3) // 4, (input_dims[1] + 3) // 4]),
            nn.ReLU(),
            
            # 全局池化层
            nn.AdaptiveAvgPool2d(1),  # 输出 (B, hidden_dim, 1, 1)
        )
        
        # 极简化ID嵌入
        embedding_dim = 2  # 最小嵌入维度
        self.id_embedding = nn.Embedding(5, embedding_dim)
        
        # 极简静态特征处理
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Linear(static_input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 将时序处理简化为轻量级TCN (Temporal Convolutional Network)
        # 代替RNN降低参数量
        temporal_in = hidden_dim
        self.temporal_net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1, groups=1),
            nn.LayerNorm([hidden_dim, temporal_in]),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # 全局池化到单一特征向量
        )
        
        # 极简融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        
        # 简化初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # 最小化初始化 - 只应用于关键参数
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)  # 使用Xavier初始化适合小网络
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息
        extra_x = use_obs[:, -self.extra_input_dims:]
        x = use_obs[:, :-self.extra_input_dims].reshape(-1, 1, *self.input_dims)

        # 处理静态特征 - 简化流程
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]
        
        # 嵌入并处理静态特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = self.norm1(F.relu(self.static_net(static_input)))

        # CNN特征提取 - 简化为单一管道
        batch_size = x.size(0)
        conv_features = self.feature_extractor(x)
        conv_features = conv_features.view(batch_size, -1)
        
        # 时序处理 - 使用1D卷积替代RNN
        temporal_input = conv_features.unsqueeze(1)  # 添加通道维度 (B, 1, features)
        temporal_out = self.temporal_net(temporal_input).squeeze(-1)  # (B, hidden_dim)
        
        # 简化融合
        combined = torch.cat([temporal_out, static_out], dim=1)
        output = self.fusion(combined)
        
        # 数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            # 保存出现问题的数据到本地
            log(f"发现 nan 或 inf 值, 保存数据到本地进行检查")
            model_sd = self.state_dict()
            # 确保在 CPU 上
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("fused_out is nan or inf")

        return output

model_cls = DeepLob_xlarge

model_config={
    # 自定义编码器参数  
    'input_dims' : (100, 20),
    'extra_input_dims' : 3,
}
data_config = {
    'his_len': 100,# 每个样本的 历史数据长度
    'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
}
env_config ={
    'data_type': 'train',# 训练/测试
    'his_len': data_config['his_len'],# 每个样本的 历史数据长度
    'need_cols': data_config['need_cols'],
    'train_folder': train_folder,
    'train_title': train_folder,

    # 使用数据增强
    'use_random_his_window': False,# 是否使用随机历史窗口
    'use_gaussian_noise_vol': False,# 是否使用高斯噪声
    'use_spread_add_small_limit_order': False,# 是否使用价差添加小单
}

checkpoint_callback = CustomCheckpointCallback(train_folder=train_folder)

# 配置 policy_kwargs
policy_kwargs = dict(
    features_extractor_class=model_cls,
    features_extractor_kwargs=model_config,
    net_arch = model_cls.net_arch,
    activation_fn=nn.ReLU
)

env_objs = []
def make_env():
    env = LOB_trade_env(env_config)
    env_objs.append(env)
    return RolloutInfoWrapper(env)

if run_type != 'test':

    # 专家
    expert = LobExpert_file(pre_cache=True)

    if run_type == 'bc_data':
        # 初始化开始时间
        start_time = time.time()
        n_hours = 11.75  # 设置时间阈值（小时）
        # n_hours = 1  # 设置时间阈值（小时）
        
        # 请求获取文件名id
        id = get_idx('time') if not in_windows() else 0
        zip_file = f'transitions_{id}.7z'
        t_folder = f'transitions'
        os.makedirs(t_folder, exist_ok=True)

        def produce_bc_data(idx):
            # 创建单个环境
            env = LOB_trade_env(env_config)
            vec_env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

            file_name = f'{t_folder}/{id}_{idx}.pkl'

            # 生成训练数据用
            f = rollouts_filter()

            while True:
                # 生成专家数据
                rng = np.random.default_rng()
                
                # 训练数据
                rollouts = rollout.rollout(
                    expert,
                    vec_env,
                    rollout.make_sample_until(min_timesteps=1e4),
                    rng=rng,
                )
                f.add_rollouts(rollouts)

                # 检查总运行时间（转换为小时）
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours > n_hours:
                    break

            # 保存数据
            parts_dict = f.get_parts_dict()
            with open(file_name, 'wb') as f:
                pickle.dump(parts_dict, f)

        # 启动 4 个进程并行生成数据
        import multiprocessing
        processes = []
        for i in range(4):
            process = multiprocessing.Process(target=produce_bc_data, args=(i,))
            processes.append(process)
            process.start()
        # 等待所有进程结束
        for process in processes:
            process.join()

        log(f"总运行时间超过 {n_hours} 小时，上传数据并退出")
        t = time.time()
        from py_ext.alist import alist
        from py_ext.lzma import compress_folder
        compress_folder(t_folder, zip_file, level=9)
        client = alist(os.environ['ALIST_USER'], os.environ['ALIST_PWD'])
        client.upload(zip_file, '/bc_train_data/')
        msg = f"运行时间: {((time.time() - start_time) / 3600):.2f} 小时, {zip_file} 上传数据耗时: {time.time() - t:.2f} 秒"
        send_wx(msg)
        sys.exit()

    # 创建并行环境（4 个环境）
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecCheckNan(env, raise_exception=True)  # 添加nan检查
    env = VecMonitor(env)  # 添加监控器
    vec_env = env

    model = PPO(
        model_type, 
        vec_env, 
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

    # ###############################################
    # # 模型检查 1. 验证模式输出是否相同
    # test_x = env.observation_space.sample()
    # test_x = torch.from_numpy(test_x).unsqueeze(0)
    # test_x[:, -4] = 0#symbol_id 0 - 4
    # log(test_x.shape)
    # test_x = test_x.float().to(model.policy.device)
    # model.policy.eval()
    # out1 = model.policy.features_extractor(test_x)
    # out2 = model.policy.features_extractor(test_x)
    # log(f'验证模式输出是否相同: {torch.allclose(out1, out2)}')
    # log(out1.shape)
    # ###############################################
    # ###############################################
    # # 模型检查 2. 不同batch之间的数据是否污染
    # test_batch_size = 10
    # test_batch_obs = np.array([env.observation_space.sample() for _ in range(test_batch_size)])
    # test_batch_obs[:, -4] = 0#symbol_id 0 - 4
    # # 转换为 tensor 格式并确保需要计算梯度
    # test_batch_obs = torch.tensor(test_batch_obs, dtype=torch.float32, requires_grad=True, device=model.policy.device)
    # # 通过模型的特征提取器得到输出
    # model.policy.train()
    # out = model.policy.features_extractor(test_batch_obs)
    # # 计算第一个样本的损失
    # loss = out[0].sum()
    # loss.backward()
    # # 检查梯度，仅第一个样本的梯度应该非零
    # assert test_batch_obs.grad is not None, "Gradient should not be None"
    # assert test_batch_obs.grad[0].sum() != 0, "the first sample should have a gradient"
    # assert test_batch_obs.grad[1:].sum() == 0, "Only the first sample should have a gradient"
    # ###############################################
    # sys.exit()

    # 遍历读取训练数据
    if not in_windows():
        data_folder = rf'/kaggle/input/bc-train-data/'
    else:
        data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data'
    data_set = LobTrajectoryDataset(data_folder, data_config=data_config)
    log(f"训练数据样本数: {len(data_set)}")

    # 生成验证数据
    rng = np.random.default_rng()
    for env in env_objs:
        env.val()
    rollouts_val = rollout.rollout(
        expert,
        vec_env,
        rollout.make_sample_until(min_timesteps=50_000 if not in_windows() else 500),
        rng=rng,
    )
    transitions_val = rollout.flatten_trajectories(rollouts_val)

    memory_usage = psutil.virtual_memory()
    log(f"内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")

    total_steps = total_epochs * len(data_set) // (batch_size * batch_n)
    bc_trainer = BCWithLRScheduler(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=model.policy,
        rng=np.random.default_rng(),
        train_dataset=data_set,
        demonstrations_val=transitions_val,
        batch_size=batch_size * batch_n if run_type=='train' else batch_size,
        l2_weight=0 if not arg_l2_weight else arg_l2_weight,
        optimizer_kwargs={'lr': 1e-7} if run_type=='find_lr' else {'lr': arg_lr} if arg_lr else None,
        custom_logger=custom_logger,
        lr_scheduler_cls = OneCycleLR if (run_type=='train' and not arg_lr) \
            else MultiplicativeLR if run_type=='find_lr' \
                else None,
        lr_scheduler_kwargs = {'max_lr':max_lr*batch_n, 'total_steps': total_steps} if (run_type=='train' and not arg_lr) \
            else {'lr_lambda': lambda epoch: 1.05} if run_type=='find_lr' \
                else None,
        use_mixed_precision=True if arg_amp else False,
    )
    
    # 训练文件夹管理
    if not in_windows():
        train_folder_manager = TrainFolderManagerBC(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            train_folder_manager.load_checkpoint(bc_trainer)

    # 初始化进度数据文件
    progress_file = os.path.join(train_folder, f"progress.csv")
    progress_file_all = os.path.join(train_folder, f"progress_all.csv")
    if os.path.exists(progress_file_all):
        df_progress = pd.read_csv(progress_file_all)
    else:
        df_progress = pd.DataFrame()

    env = env_objs[0]
    begin = bc_trainer.train_loop_idx
    log(f'begin: {begin}')
    for i in range(begin, total_epochs // checkpoint_interval):
        log(f'第 {i} 次训练')

        _t = time.time()
        bc_trainer.train(
            n_epochs=checkpoint_interval,
            log_interval = 1 if run_type=='find_lr' else 500,
        )
        log(f'训练耗时: {time.time() - _t:.2f} 秒')

        # 验证模型
        _t = time.time()
        env.val()
        val_reward, _ = evaluate_policy(bc_trainer.policy, env)
        env.train()
        train_reward, _ = evaluate_policy(bc_trainer.policy, env)
        log(f"train_reward: {train_reward}, val_reward: {val_reward}, 验证耗时: {time.time() - _t:.2f} 秒")

        # 合并到 progress_all.csv
        latest_ts = df_progress.iloc[-1]['timestamp'] if len(df_progress) > 0 else 0
        df_new = pd.read_csv(progress_file)
        df_new = df_new.loc[df_new['timestamp'] > latest_ts, :]
        df_new['bc/epoch'] += i * checkpoint_interval
        df_new['bc/mean_reward'] = np.nan
        df_new['bc/val_mean_reward'] = np.nan
        df_new.loc[df_new.index[-1], 'bc/mean_reward'] = train_reward
        df_new.loc[df_new.index[-1], 'bc/val_mean_reward'] = val_reward
        df_progress = pd.concat([df_progress, df_new]).reset_index(drop=True)
        df_progress.ffill(inplace=True)
        df_progress.to_csv(progress_file_all, index=False)

        # 当前点是否是最优的 checkpoint
        # 使用 recall 判断
        if 'bc/recall' in list(df_progress):
            bset_recall = df_progress['bc/recall'].max()
            best_epoch = df_progress.loc[df_progress['bc/recall'] == bset_recall, 'bc/epoch'].values[0]
            is_best = df_progress.iloc[-1]['bc/epoch'] == best_epoch
        else:
            is_best = False

        # 训练进度可视化
        try:
            plot_bc_train_progress(train_folder, df_progress=df_progress, title=train_title)
        except Exception as e:
            pickle.dump(df_progress, open('df_progress.pkl', 'wb'))
            log(f"训练进度可视化失败")
            raise e

        if not in_windows():
            # 保存模型
            train_folder_manager.checkpoint(bc_trainer, best=is_best)

        if run_type == 'find_lr':
            # 只运行一个 epoch
            break


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