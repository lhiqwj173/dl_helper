import subprocess
import importlib.metadata
import os
import sys

# !mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git
cmd = 'mkdir 3rd && cd 3rd && git clone https://github.com/lhiqwj173/dl_helper.git'
subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

# !cd /kaggle/working/3rd/dl_helper && pip install -e .
cmd = 'cd /kaggle/working/3rd/dl_helper && pip install -e .'
subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

# # !wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb
# # !cp wkhtmltox_0.12.6-1.bionic_amd64.deb /usr/bin
# # !sudo apt install /usr/bin/wkhtmltox_0.12.6-1.bionic_amd64.deb -y
# cmd = 'wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb'
# subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
# cmd = 'cp wkhtmltox_0.12.6-1.bionic_amd64.deb /usr/bin'
# subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
# cmd ='sudo apt install /usr/bin/wkhtmltox_0.12.6-1.bionic_amd64.deb -y'
# subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

# 安装 py_ext
cmd = 'cd /kaggle/working/3rd && wget https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz && tar -xzvf py_ext-1.0.0.tar.gz && cd py_ext-1.0.0 && python setup.py install'
subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

for cmd in [
        'pip install autogluon',
        'pip install df2img',
        'pip install imgkit',
        'pip install torchstat',
    ]:
    subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)

root = r'/kaggle/working/'
sys.path.append(os.path.join(root, '3rd', 'dl_helper'))