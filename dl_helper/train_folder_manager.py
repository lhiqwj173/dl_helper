"""
管理训练文件夹
1. 训练开始时, 拉取alist训练文件夹
2. 训练过程/结束, 打包上传alist
"""
import os, shutil, pickle
from py_ext.alist import alist
from py_ext.tool import log
from py_ext.wechat import send_wx
from py_ext.lzma import decompress, compress_folder

ALIST_UPLOAD_FOLDER = 'rl_learning_process'

class TrainFolderManager:
    def __init__(self, train_folder):
        self.train_folder = train_folder

        abs_train_folder = os.path.abspath(train_folder)
        if '\\' in abs_train_folder:
            self.train_title = abs_train_folder.split('\\')[-1]
        else:
            self.train_title = abs_train_folder.split('/')[-1]

        # 创建检查点保存目录
        self.checkpoint_folder = os.path.join(os.path.abspath(train_folder), 'checkpoint')
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        # 检查是否有训练记录
        # 若无，则需要尝试拉取训练最新记录
        if not self.exists():
            self.pull()

    def exists(self):
        """
        检查是否存在训练记录
        """
        return os.path.exists(os.path.join(self.checkpoint_folder, 'rllib_checkpoint.json'))

    def load_checkpoint(self, algo, only_params=True):
        """
        加载检查点
        """
        if only_params:
            # 获取模型参数
            # 加载文件内容
            module_state = pickle.load(open(os.path.join(self.checkpoint_folder, 'learner_group', 'learner', 'rl_module', 'default_policy', 'module_state.pt'), 'rb'))
            optimizer_state = pickle.load(open(os.path.join(self.checkpoint_folder, 'learner_group', 'learner', 'state.pkl'), 'rb'))['optimizer']
            # 组装state
            state = {'learner_group':{'learner':{
                'rl_module':{'default_policy': module_state},
                'optimizer': optimizer_state
            }}}
            algo.set_state(state)
        else:
            algo.restore_from_path(self.checkpoint_folder)

    def pull(self):
        """
        拉取训练最新记录, 并解压覆盖到训练文件夹
        """
        client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
        try:
            _file = f'alist/{self.train_title}.7z'
            # 下载文件
            download_folder = f'/{ALIST_UPLOAD_FOLDER}/'
            client.download(f'{download_folder}{self.train_title}.7z', 'alist/')
            log(f'download {_file}')
        except:
            pass

        if os.path.exists(_file):
            # 解压文件
            decompress(_file)
            log(f'decompress {_file}')
            # move 
            folder = os.path.join('/kaggle/working/alist', self.train_title, 'checkpoint')
            log(f'checkpoint folder {folder}')
            if os.path.exists(folder):
                send_wx(f'[{self.train_title}] 使用alist缓存文件继续训练')
                log(f"使用alist缓存文件继续训练")
                # 覆盖到训练文件夹
                shutil.copytree(os.path.join('/kaggle/working/alist', self.train_title), self.train_folder, dirs_exist_ok=True)
        else:
            os.makedirs(self.train_folder, exist_ok=True)

    def push(self):
        """
        打包并上传, 覆盖alist最新记录
        """
        zip_file = f'{self.train_title}.7z'
        if os.path.exists(zip_file):
            os.remove(zip_file)
        compress_folder(self.train_folder, zip_file, 9, inplace=False)
        log('compress_folder done')

        # 上传更新到alist
        client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
        # 上传文件夹
        upload_folder = f'/{ALIST_UPLOAD_FOLDER}/'
        client.mkdir(upload_folder)
        client.upload(zip_file, upload_folder)
        log('upload done')
