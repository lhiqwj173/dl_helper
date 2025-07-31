import subprocess
import importlib.metadata
import os
import sys

if '__main__' == __name__:

    install_dl_helper = True
    if len(sys.argv) > 1 and sys.argv[1] == 'not_install_dl_helper':
        install_dl_helper = False

    # !mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git
    cmd = 'mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git'
    # cmd = 'mkdir 3rd && cd 3rd && git clone -b ps https://github.com/lhiqwj173/dl_helper.git'
    subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
    if install_dl_helper:
        # !cd /kaggle/working/3rd/dl_helper && pip install -e .
        cmd = 'cd /kaggle/working/3rd/dl_helper && pip install -e .'
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

        root = r'/kaggle/working/'
        sys.path.append(os.path.join(root, '3rd', 'dl_helper'))

    # 安装 py_ext
    # pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz
    cmd = 'cd /kaggle/working/3rd && wget https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz && tar -xzvf py_ext-1.0.0.tar.gz && cd py_ext-1.0.0 && python setup.py install'
    # cmd = 'cd /kaggle/working/3rd && wget https://raw.githubusercontent.com/lhiqwj173/dl_helper/ps/py_ext-1.0.0.tar.gz && tar -xzvf py_ext-1.0.0.tar.gz && cd py_ext-1.0.0 && python setup.py install'
    subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

    for cmd in [
            'pip install --upgrade torch ray[rllib]',
            'pip install --upgrade torchvision',
            'pip install accelerate==1.3.0',
            'pip install imitation',
            'pip install --upgrade stable-baselines3',
            'pip install shimmy',
            'pip install uvloop',
            'pip install dill',
            'pip install ale_py',
            'pip install tensorboardX',
            'pip install lz4',
            'pip install gymnasium[accept-rom-license,atari]==1.0.0',
            'pip install MoviePy',
            'pip install df2img',
            'pip install imgkit',
            'pip install loguru',
            'pip install seaborn',
            'pip install einops',
            'pip install torchmetrics',
            'pip install requests_toolbelt',
            'pip install torchstat',
            'pip install torchinfo',
            'pip install dataframe_image',
        ]:
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
