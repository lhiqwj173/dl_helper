import subprocess
import importlib.metadata
import os
import sys

if '__main__' == __name__:
    download_cmds = [
        'wget https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/envs/frpc',
        'wget https://raw.githubusercontent.com/lhiqwj173/dl_helper/master/envs/frpc.toml',
        'chmod +x frpc',
    ]

    for cmd in download_cmds:
        subprocess.call(cmd, shell=True)

    # frpc -c frpc.toml
    subprocess.Popen(["frpc", "-c", "frpc.toml"], start_new_session=True)
