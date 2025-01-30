import os
import sys
import shutil
import subprocess 

from dl_helper.rl.rl_utils import rl_folder
from dl_helper.train_param import is_kaggle

def run(need_clear):
    if need_clear:
        print('需要清理...')
        if os.path.exists(rl_folder):
            print('清理 rl_learning 目录下的所有文件')
            shutil.rmtree(rl_folder)
            os.makedirs(rl_folder)
        else:
            # 创建目录
            os.makedirs(rl_folder)

    code_folder = r'/root/code/dl_helper' if not is_kaggle() else r'/kaggle/working/3rd/dl_helper'

    # 训练服务
    print('注册训练服务')
    # for py_file in ['20241203_dddqn.py', '20241216_per_dqn.py', '20241217_c51.py']:
    # for py_file in ['RLlib/20250110_breakout.py']:
    for py_file in ['RLlib/20250130_cartpole.py']:
        path = os.path.join(code_folder, 'dl_helper', 'tests', 'rl', py_file)
        subprocess.run(['python3', path, 'server'], check=True)

    # 运行服务端
    print('运行服务端')
    # D:\code\dl_helper\dl_helper\rl\net_center.py
    subprocess.run(['python3', f'{code_folder}/dl_helper/rl/net_center.py'], check=True)

if __name__ == '__main__':
    need_clear = False

    # 获取命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'clear':
        need_clear = True
    
    run(need_clear)
