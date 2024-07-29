import subprocess
import importlib.metadata

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
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
        cmd = "git clone https://github.com/lhiqwj173/accelerate.git"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
        cmd = 'cd accelerate && git checkout fix-save_state-bug-with-MpDeviceLoaderWrapper-object && pip install .'
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
    
except importlib.metadata.PackageNotFoundError:
    print(f"The package {package_name} is not installed.")
    print(f"install ...")
    # !git clone https://github.com/lhiqwj173/accelerate.git
    # !cd accelerate && git checkout fix-save_state-bug-with-MpDeviceLoaderWrapper-object && pip install .
    cmd = 'git clone https://github.com/lhiqwj173/accelerate.git'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
    cmd = 'cd accelerate && git checkout fix-save_state-bug-with-MpDeviceLoaderWrapper-object && pip install .'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)