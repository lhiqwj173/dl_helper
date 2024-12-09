import os, sys
import torch

from py_ext.tool import log, init_logger
import requests, socket
try:
    ip = requests.get('https://api.ipify.org').text
except:
    # 如果获取外网IP失败,使用内网IP作为备选
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
init_logger(f'{ip}', level='INFO')

from dl_helper.rl.dqn import DQN, VANILLA_DQN, DOUBLE_DQN, DUELING_DQN, DD_DQN, run_client_learning_device
from dl_helper.rl.net_center import run_param_center
from dl_helper.rl.run import run_client_learning
from dl_helper.models.binctabl import m_bin_ctabl_fix_shape
from dl_helper.train_param import in_kaggle

# 训练参数
train_title = 'rl_test'
lr = 1e-4
num_episodes = 5000
hidden_dim = 128
gamma = 0.98
epsilon = 0.5
target_update = 50
buffer_size = 5000
minimal_size = 3000
batch_size = 64
sync_interval_learn_step=150
learn_interval_step=4

# 快速测试
simple_test = False

# val_test
val_test = ''

t1, t2, t3, t4 = [100, 30, 10, 1]
d1, d2, d3, d4 = [130, 60, 30, 7]
features_extractor_kwargs = {'d2': d2, 'd1': d1, 't1': t1, 't2': t2, 'd3': d3, 't3': t3, 'd4': d4, 't4': t4}

if __name__ == '__main__':

    # 检查是否有命令行参数
    is_server = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('train_title=') or arg.startswith('title='):
                train_title = arg.split('=')[1]
            elif arg.startswith('lr='):
                lr = float(arg.split('=')[1])
            elif arg.startswith('num_episodes='):
                num_episodes = int(arg.split('=')[1])
            elif arg.startswith('hidden_dim='):
                hidden_dim = int(arg.split('=')[1])
            elif arg.startswith('gamma='):
                gamma = float(arg.split('=')[1])
            elif arg.startswith('epsilon='):
                epsilon = float(arg.split('=')[1])
            elif arg.startswith('target_update='):
                target_update = int(arg.split('=')[1])
            elif arg.startswith('buffer_size='):
                buffer_size = int(arg.split('=')[1])
            elif arg.startswith('minimal_size='):
                minimal_size = int(arg.split('=')[1])
            elif arg.startswith('batch_size='):
                batch_size = int(arg.split('=')[1])
            elif arg.startswith('sync_interval_learn_step='):
                sync_interval_learn_step = int(arg.split('=')[1])
            elif arg.startswith('learn_interval_step='):
                learn_interval_step = int(arg.split('=')[1])
            elif arg == 'server':
                is_server = True
            elif arg == 'client':
                is_server = False
            elif arg == 'simple_test':
                simple_test = True
                print('simple_test')
            elif arg.startswith('test_val='):
                val_test = arg.split('=')[1]

    dqn = DQN(
        obs_shape=(100, 130),
        action_dim=3,
        features_dim=d4+3,
        features_extractor_class=m_bin_ctabl_fix_shape,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        target_update=target_update,
        buffer_size=buffer_size,
        wait_trade_close = True,
        features_extractor_kwargs=features_extractor_kwargs,
        net_arch=[6, 3],
        dqn_type=DD_DQN,
    )

    if not is_server:
        # 训练数据
        if in_kaggle:
            input_folder = r'/kaggle/input'
            # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
            data_folder_name = os.listdir(input_folder)[0]
            data_folder = os.path.join(input_folder, data_folder_name)
        else:
            data_folder = r'D:\L2_DATA_T0_ETF\train_data\RL_combine_data_test'

        args = (train_title, data_folder, dqn, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step)
        kwargs = {'simple_test': simple_test, 'val_test': val_test}
        run_client_learning(run_client_learning_device, args, kwargs)
    else:
        # 服务端
        run_param_center(dqn, simple_test=simple_test)
