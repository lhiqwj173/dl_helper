import os
import sys
import shutil
import subprocess 

def run(need_clear):
    if need_clear:
        print('清理 /root/alist_data/rl_learning_process 目录下的所有文件')
        # 清理 /root/alist_data/rl_learning_process 目录下的所有文件
        shutil.rmtree('/root/alist_data/rl_learning_process')
        os.makedirs('/root/alist_data/rl_learning_process')

    code_folder = r'/root/code/dl_helper'

    # 训练服务
    print('注册训练服务')
    for py_file in ['20241203_dddqn.py', '20241216_per_dqn.py', '20241217_c51.py']:
        path = os.path.join(code_folder, 'dl_helper', 'tests', 'rl', py_file)
        subprocess.run(['python3', path, 'server'], check=True)

    # 运行服务端
    print('运行服务端')
    # D:\code\dl_helper\dl_helper\rl\net_center.py
    subprocess.run(['python3', '/root/code/dl_helper/dl_helper/rl/net_center.py'], check=True)

if __name__ == '__main__':
    need_clear = False

    # 获取命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'clear':
        need_clear = True
    
    run(need_clear)
