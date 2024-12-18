import os
import shutil
import subprocess 

def run():
    # 清理 /root/alist_data/rl_learning_process 目录下的所有文件
    shutil.rmtree('/root/alist_data/rl_learning_process')
    os.makedirs('/root/alist_data/rl_learning_process')
    
    # 训练服务
    for py_file in ['20241203_dddqn.py', '20241216_per_dqn.py', '20241217_c51.py']:
        path = os.path.join(os.path.dirname(__file__), py_file)
        subprocess.run(f'python3 {path} server', check=True)

    # 运行服务端
    # D:\code\dl_helper\dl_helper\rl\net_center.py
    subprocess.run(f'python3 /root/code/dl_helper/dl_helper/rl/net_center.py', check=True)

if __name__ == '__main__':
    run()
