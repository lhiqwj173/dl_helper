import os
import torch

from dl_helper.rl.dqn import DQN, VANILLA_DQN, DOUBLE_DQN, DUELING_DQN, DD_DQN
from dl_helper.rl.rl_env.lob_env import data_producer, LOB_trade_env
from dl_helper.models.binctabl import m_bin_ctabl_fix_shape

lr = 1e-2
num_episodes = 5000
hidden_dim = 128
gamma = 0.98
epsilon = 0.5
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64
seed = 42
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

t1, t2, t3, t4 = [100, 30, 10, 1]
d1, d2, d3, d4 = [130, 60, 30, 3]
features_extractor_kwargs = {'d2': d2, 'd1': d1, 't1': t1, 't2': t2, 'd3': d3, 't3': t3, 'd4': d4, 't4': t4}

if __name__ == '__main__':
    dqn = DQN(
        obs_shape=(100, 130),
        action_dim=3,
        features_dim=d4+2,
        features_extractor_class=m_bin_ctabl_fix_shape,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        target_update=target_update,
        buffer_size=buffer_size,
        device=device,
        wait_trade_close = True,
        features_extractor_kwargs=features_extractor_kwargs,
        net_arch=None,
        dqn_type=VANILLA_DQN,
        sync_alist=False
    )

    # # 训练数据
    # input_folder = r'/kaggle/input'
    # # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
    # data_folder_name = os.listdir(input_folder)[0]
    # data_folder = os.path.join(input_folder, data_folder_name)

    os.environ['ALIST_USER'] = 'admin'
    os.environ['ALIST_PWD'] = 'LHss6632673'
    data_folder = r'D:\L2_DATA_T0_ETF\train_data\RL_combine_data_test'

    data_producer = data_producer(data_folder=data_folder)
    env = LOB_trade_env(data_producer=data_producer)

    # 开始训练
    dqn.learn('rl_test', env, num_episodes, minimal_size, batch_size)

