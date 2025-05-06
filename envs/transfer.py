import subprocess
import importlib.metadata
import os
import sys

if '__main__' == __name__:

    # 安装 py_ext
    cmd = 'wget https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/py_ext-1.0.0.tar.gz && tar -xzvf py_ext-1.0.0.tar.gz && cd py_ext-1.0.0 && python setup.py install'
    subprocess.call(cmd, shell=True)#, stdout=subprocess.DEVNULL)
