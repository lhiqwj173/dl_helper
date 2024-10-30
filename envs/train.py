import subprocess
import importlib.metadata
import os
import sys

KAGGLE, COLAB = range(2)
ENV =  KAGGLE if any(key.startswith("KAGGLE") for key in os.environ.keys()) else COLAB

package_name = 'accelerate'
try:
    # 检查包是否安装
    distribution = importlib.metadata.distribution(package_name)
    version = distribution.version
    print(f"The package {package_name} is installed. Version: {version}")
    
    if version != '0.32.0.dev0':
        print(f"update package {package_name} -> 0.32.0.dev0")

        # 版本不正确 重新安装
        # !pip uninstall accelerate -y
        # !git clone https://github.com/lhiqwj173/accelerate.git
        # !cd accelerate && git checkout fix-save_state-bug-with-MpDeviceLoaderWrapper-object && pip install .
        cmd = "pip uninstall accelerate -y"
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
        cmd = "git clone https://github.com/lhiqwj173/accelerate.git"
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
        cmd = 'cd accelerate && git checkout fix-save_state-bug-with-MpDeviceLoaderWrapper-object && pip install .'
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
    
except importlib.metadata.PackageNotFoundError:
    print(f"The package {package_name} is not installed.")
    print(f"install ...")
    # !git clone https://github.com/lhiqwj173/accelerate.git
    # !cd accelerate && git checkout fix-save_state-bug-with-MpDeviceLoaderWrapper-object && pip install .
    cmd = 'git clone https://github.com/lhiqwj173/accelerate.git'
    subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
    cmd = 'cd accelerate && git checkout fix-save_state-bug-with-MpDeviceLoaderWrapper-object && pip install .'
    subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

root = r'/kaggle/working/' if ENV==KAGGLE else r'/content/'
if os.path.exists(os.path.join(root, '3rd')):
    if ENV==KAGGLE:
        #!cd /kaggle/working/3rd/dl_helper && git pull
        cmd = 'cd /kaggle/working/3rd/dl_helper && git pull'
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
    else:
        # !cd /content/3rd/dl_helper && git pull
        cmd = 'cd /content/3rd/dl_helper && git pull'
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
else:
    # !mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git
    cmd = 'mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git'
    subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
    if ENV==KAGGLE:
        # !cd /kaggle/working/3rd/dl_helper && pip install -e .
        cmd = 'cd /kaggle/working/3rd/dl_helper && pip install -e .'
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
        ##  !cd /kaggle/working/3rd/dl_helper && git checkout 5b178e9 && pip install -e .
        # cmd = 'cd /kaggle/working/3rd/dl_helper && git checkout 5b178e9 && pip install -e .'
        # subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
    else:
        # !cd /content/3rd/dl_helper && pip install -e .
        cmd = 'cd /content/3rd/dl_helper && pip install -e .'
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

    # !pip install "pandas<2.0.0"
    # !pip install pip install df2img
    # !pip install loguru
    # !pip install einops
    # !pip install dill
    # !pip install torchinfo
    # !pip install telethon # 弃用
    # !pip install torchmetrics
    # !pip install pympler # 弃用
    # !pip install requests_toolbelt
    # !pip install torchstat
    # !pip install torchinfo
    # # !pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0-py3-none-any.whl
    # !pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz
    for cmd in [
            'pip install "pandas<2.0.0"',
            'pip install df2img',
            'pip install loguru',
            'pip install einops',
            'pip install dill',
            'pip install torchinfo',
            'pip install torchmetrics',
            'pip install requests_toolbelt',
            'pip install torchstat',
            'pip install torchinfo',
            'pip install https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz',
        ]:
        subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

    sys.path.append(os.path.join(root, '3rd', 'dl_helper'))