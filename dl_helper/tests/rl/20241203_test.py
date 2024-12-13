import os, sys
import torch
import time
import cProfile
import pstats
from datetime import datetime
import threading

from py_ext.tool import log

from dl_helper.rl.dqn import DQN, VANILLA_DQN, DOUBLE_DQN, DUELING_DQN, DD_DQN, run_client_learning_device
from dl_helper.rl.net_center import run_param_center
from dl_helper.rl.run import run_client_learning
from dl_helper.models.binctabl import m_bin_ctabl_fix_shape
from dl_helper.train_param import in_kaggle
from dl_helper.tool import keep_upload_log_file, init_logger_by_ip

# 训练参数
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

# 性能分析参数
enable_profiling = False
profile_stats_n = 30  # 显示前N个耗时函数
profile_output_dir = 'profile_results'  # 性能分析结果保存目录

t1, t2, t3, t4 = [100, 30, 10, 1]
d1, d2, d3, d4 = [130, 60, 30, 7]
features_extractor_kwargs = {'d2': d2, 'd1': d1, 't1': t1, 't2': t2, 'd3': d3, 't3': t3, 'd4': d4, 't4': t4}

if __name__ == '__main__':
    # 初始化日志
    init_logger_by_ip()

    # 保持上传日志文件
    upload_thread = threading.Thread(target=keep_upload_log_file, daemon=True)
    upload_thread.start()

    # 检查是否有命令行参数
    is_server = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('lr='):
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
            elif arg == 'profile':
                enable_profiling = True
            elif arg.startswith('profile_stats_n='):
                profile_stats_n = int(arg.split('=')[1])
            elif arg.startswith('profile_output_dir='):
                profile_output_dir = arg.split('=')[1]

    # 根据参数决定是否启用性能分析
    profiler = None
    start_time = None
    if enable_profiling:
        # 创建性能分析结果保存目录
        os.makedirs(profile_output_dir, exist_ok=True)
        
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()

    dqn = DQN(
        obs_shape=(100, 130),
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        target_update=target_update,

        # 基类参数
        buffer_size=buffer_size,
        action_dim=3,
        features_dim=d4+3,
        features_extractor_class=m_bin_ctabl_fix_shape,
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

        args = (data_folder, dqn, num_episodes, minimal_size, batch_size, sync_interval_learn_step, learn_interval_step)
        kwargs = {'simple_test': simple_test, 'val_test': val_test, 'enable_profiling': enable_profiling}
        run_client_learning(run_client_learning_device, args, kwargs)
    else:
        # 服务端
        run_param_center(dqn, simple_test=simple_test)

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
