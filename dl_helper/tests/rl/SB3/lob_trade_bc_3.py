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
from dl_helper.idx_manager import get_idx
from dl_helper.train_folder_manager import TrainFolderManagerBC

from dl_helper.rl.custom_imitation_module.bc import BCWithLRScheduler
from dl_helper.rl.custom_imitation_module.dataset import TrajectoryDataset
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

# train_folder = train_title = f'20250429_lob_trade_bc_3' \
train_folder = train_title = f'20250429_lob_trade_bc_3' \
    + ('' if arg_lr is None else f'_lr{arg_lr:.0e}') \
        + ('' if arg_batch_n is None else f'_batch_n{arg_batch_n}') \
            + ('' if arg_total_epochs is None else f'_epochs{arg_total_epochs}') \
                + ('' if arg_max_lr is None else f'_maxlr{arg_max_lr:.0e}') \
                    + ('' if arg_l2_weight is None else f'_l2weight{arg_l2_weight:.0e}') \
                        + ('' if arg_dropout is None else f'_dropout{arg_dropout:.0e}')
            
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
batch_n = 2**7 if run_type=='train' else 1
batch_n = batch_n if arg_batch_n is None else arg_batch_n
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
            dropout_rate: float = arg_dropout,
    ):
        super().__init__(observation_space, features_dim)

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.dropout_rate = dropout_rate

        # 卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),  # 添加 Dropout2d
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),  # 添加 Dropout2d
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity(),  # 添加 Dropout2d
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
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
        )

        # 融合层
        fusion_input_dim = 64 + static_input_dim * 4  # LSTM 输出 64 维 + static_net 输出
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )

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
        x = self.dp2d(x)  # 添加 Dropout2d

        # 调整形状以适配 LSTM
        x = x.permute(0, 2, 1, 3).squeeze(3)  # (B, 24, 192)

        # LSTM 处理 取最后一个时间步
        lstm_out, _ = self.lstm(x)  # (B, 24, 64)
        lstm_out = self.dp(lstm_out)  # 添加 Dropout
        # 取最后一个时间步
        temporal_feat = lstm_out[:, -1, :]  # (B, 64)

        # 融合层
        fused_out = torch.cat([temporal_feat, static_out], dim=1)  # (B, 64 + self.extra_input_dims * 4)
        fused_out = self.dp(fused_out)  # 添加 Dropout
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
    expert = LobExpert_file(pre_cache=True)

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
    # model.policy.eval()
    # out1 = model.policy.features_extractor(test_x)
    # out2 = model.policy.features_extractor(test_x)
    # log(f'验证模式输出是否相同: {torch.allclose(out1, out2)}')
    # log(out1.shape)
    # sys.exit()

    vec_env = env

    if run_type == 'bc_data':
        # 生成训练数据用
        f = rollouts_filter()
        # 此处耗时约10min
        # 初始化开始时间
        start_time = time.time()
        n_hours = 11.5  # 设置时间阈值（小时）
        n_hours = 1  # 设置时间阈值（小时）
        # 请求获取文件名id
        id = get_idx('time') if not in_windows() else 0
        file_name = f'transitions_{id}.pkl'
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
            transitions = f.flatten_trajectories()

            # 写入临时文件
            temp_file_path = 'transitions_temp.pkl'  # 临时文件名
            with open(temp_file_path, 'wb') as temp_file:
                pickle.dump(transitions, temp_file)
                temp_file.flush()  # 确保数据写入文件系统缓存
                os.fsync(temp_file.fileno())  # 强制将缓存数据写入磁盘（可选）
            
            # 原子性地重命名临时文件为目标文件
            os.replace(temp_file_path, file_name)

            # 检查总运行时间（转换为小时）
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours > n_hours:
                log(f"总运行时间超过 {n_hours} 小时，上传数据并退出")
                t = time.time()
                from py_ext.alist import alist
                from py_ext.lzma import compress
                zip_file = file_name + '.7z'
                compress(file_name, level=9)
                client = alist(os.environ['ALIST_USER'], os.environ['ALIST_PWD'])
                client.upload(zip_file, '/bc_train_data/')
                msg = f"运行时间: {elapsed_hours:.2f} 小时, {file_name} 上传数据耗时: {time.time() - t:.2f} 秒"
                send_wx(msg)
                sys.exit()
    
    memory_usage = psutil.virtual_memory()
    log(f"内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")

    # 生成验证数据
    rng = np.random.default_rng()
    for env in env_objs:
        env.val()
    rollouts_val = rollout.rollout(
        expert,
        vec_env,
        rollout.make_sample_until(min_timesteps=50_000),
        rng=rng,
    )
    transitions_val = rollout.flatten_trajectories(rollouts_val)

    # 遍历读取训练数据
    if not in_windows():
        data_folder = [
            rf'/kaggle/input/pre-trained-policy-2/',# kaggle 命名失误
            rf'/kaggle/input/lob-bc-train-data-filted-3/',
        ]
    else:
        data_folder = r'D:\L2_DATA_T0_ETF\train_data\RAW\BC_train_data'
    data_set = TrajectoryDataset(data_folder)
    log(f"训练数据样本数: {len(data_set)}")
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

    data_set.stop()

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