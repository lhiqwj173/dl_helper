import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cProfile
import pstats
from datetime import datetime
import threading

from py_ext.tool import log

from dl_helper.rl.dqn.dqn import VANILLA_DQN, DOUBLE_DQN, DUELING_DQN, DD_DQN
from dl_helper.rl.dqn.c51 import C51 
from dl_helper.rl.net_center import add_train_title_item
from dl_helper.rl.run import run_client_learning, run_client_learning_device_breakout
from dl_helper.rl.rl_utils import ReplayBufferWaitClose, PrioritizedReplayBuffer
from dl_helper.train_param import in_kaggle
from dl_helper.tool import keep_upload_log_file, init_logger_by_ip

# 计算经过卷积后的特征图大小
def conv2d_size_out(size, kernel_size, stride):
    return ((size - (kernel_size - 1) - 1) // stride) + 1

class cnn_breakout(nn.Module):
    def __init__(self):
        super(cnn_breakout, self).__init__()
        
        # 修改卷积层参数以减小输出维度
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4)  # 增加stride到4
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)  # 增加stride到2

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x

# 训练参数
train_title = 'C51_breakout_20241226'
lr = 1e-4
num_episodes = 5000
hidden_dim = 128
gamma = 0.98
epsilon = 0.5
target_update = 50
buffer_size = 3000
minimal_size = 3000
batch_size = 256
sync_interval_learn_step=150
learn_interval_step=4

# 初始化日志
init_logger_by_ip(train_title)

# 快速测试
simple_test = False

# val_test
val_test = ''

# 性能分析参数
enable_profiling = False
profile_stats_n = 30  # 显示前N个耗时函数
profile_output_dir = 'profile_results'  # 性能分析结果保存目录

features_extractor_kwargs = {}

if __name__ == '__main__':
    # 命令行参数文档
    help_doc = """
    命令行参数说明:
    
    训练相关参数:
        train_title=<str>           训练标题
        lr=<float>                   学习率, 默认1e-4开始
        num_episodes=<int>           训练回合数, 默认5000
        hidden_dim=<int>            隐藏层维度, 默认128
        gamma=<float>               折扣因子, 默认0.98
        epsilon=<float>             探索率, 默认0.5
        target_update=<int>         目标网络更新频率, 默认50
        buffer_size=<int>           经验回放池大小, 默认5000
        minimal_size=<int>          最小训练样本数, 默认3000
        batch_size=<int>            批次大小, 默认64
        sync_interval_learn_step=<int>  同步参数间隔, 默认150
        learn_interval_step=<int>   学习更新间隔, 默认4
    
    运行模式:
        server                      以服务端模式运行
        client                      以客户端模式运行(默认)
        simple_test                 启用简单测试模式
        test_val=<val/test/all>     验证/测试模式
    
    性能分析:
        profile                     启用性能分析
        profile_stats_n=<int>       显示前N个耗时函数, 默认30
        profile_output_dir=<str>    性能分析结果保存目录, 默认'profile_results'
    
    使用示例:
        python script.py lr=0.001 num_episodes=1000 server
    """

    # 检查是否有命令行参数
    is_server = False
    if len(sys.argv) > 1:
        # 如果输入help,打印帮助文档并退出
        if 'help' in sys.argv[1:]:
            print(help_doc)
            sys.exit(0)
            
        for arg in sys.argv[1:]:
            if arg.startswith('train_title='):
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
                # val/test/all
                val_test = arg.split('=')[1]
            elif arg == 'profile':
                enable_profiling = True
            elif arg.startswith('profile_stats_n='):
                profile_stats_n = int(arg.split('=')[1])
            elif arg.startswith('profile_output_dir='):
                profile_output_dir = arg.split('=')[1]

    # 使用新的卷积参数计算输出尺寸
    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(161, 8, 4), 4, 4), 3, 2)
    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(144, 8, 4), 4, 4), 3, 2)
    # 特征提取层输出维度
    features_dim = convw * convh * 64

    # 动作维度
    action_dim = 4

    # 根据参数决定是否启用性能分析
    profiler = None
    start_time = None
    if enable_profiling:
        # 创建性能分析结果保存目录
        os.makedirs(profile_output_dir, exist_ok=True)
        
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()

    agent_class = C51
    agent_kwargs = {
        'learning_rate': lr,
        'gamma': gamma,
        'epsilon': epsilon,
        'target_update': target_update,
        'buffer_size': buffer_size,
        'train_buffer_class': PrioritizedReplayBuffer,
        'use_noisy': True,
        'n_step': 1,
        'train_title': train_title,
        'action_dim': action_dim,
        'features_dim': features_dim,
        'features_extractor_class': cnn_breakout,
        'features_extractor_kwargs': features_extractor_kwargs,
        'net_arch': [512, action_dim],
    }

    if not is_server:
        # 保持上传日志文件
        upload_thread = threading.Thread(target=keep_upload_log_file, args=(train_title,), daemon=True)
        upload_thread.start()

        # 初始化agrnt
        agent = agent_class(**agent_kwargs)

        args = (agent, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step)
        kwargs = {'simple_test': simple_test, 'val_test': val_test, 'enable_profiling': enable_profiling}
        run_client_learning(run_client_learning_device_breakout, args, kwargs)
    else:
        # 服务端
        add_train_title_item(train_title, agent_class, agent_kwargs, simple_test)

    # 如果启用了性能分析，输出并保存结果
    if enable_profiling:
        profiler.disable()
        end_time = time.time()
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建结果文件名
        stats_file = os.path.join(profile_output_dir, f'profile_stats_{timestamp}.txt')
        
        # 打开文件并重定向stdout
        with open(stats_file, 'w') as f:
            # 记录总运行时间
            total_time = end_time - start_time
            f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
            
            # 创建性能分析报告
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')  # 按累计时间排序
            stats.print_stats(profile_stats_n)  # 显示前N个耗时最多的函数
            
            # 保存调用关系图
            stats.print_callers()
            stats.print_callees()
        
        # 同时在控制台显示结果
        log(f"Total execution time: {total_time:.2f} seconds")
        log(f"Profile results saved to: {stats_file}")
        
        # 创建二进制统计文件以供后续分析
        stats.dump_stats(os.path.join(profile_output_dir, f'profile_stats_{timestamp}.prof'))
