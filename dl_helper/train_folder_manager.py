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

from dl_helper.tool import print_directory_tree

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

        # 创建最佳点保存目录
        self.best_checkpoint_folder = os.path.join(os.path.abspath(train_folder), 'best_checkpoint')
        os.makedirs(self.best_checkpoint_folder, exist_ok=True)

        # 检查是否有训练记录
        # 若无，则需要尝试拉取训练最新记录
        if not self.exists():
            self.pull()

    def check_point_file(self):
        """ 检查点文件 """
        self.checkpoint_folder

    def exists(self):
        """
        检查是否存在训练记录
        """
        return os.path.exists(os.path.join(self.checkpoint_folder, 'rllib_checkpoint.json'))

    def load_checkpoint(self, algo, only_params=False):
        """
        加载检查点
        """
        need_optimizer_state = False
        if not only_params:
            # TODO restore_from_path 出现bug，等待修复
            # algo.restore_from_path(self.checkpoint_folder)
            only_params = True
            # need_optimizer_state = True

        if only_params:
            # 获取模型参数
            # 加载文件内容
            module_state_folder = os.path.join(self.checkpoint_folder, 'learner_group', 'learner', 'rl_module', 'default_policy')
            file = [i for i in os.listdir(module_state_folder) if 'module_state' in i]
            if len(file) == 0:
                raise ValueError(f'{module_state_folder} 中没有找到 module_state 文件')
            module_state = pickle.load(open(os.path.join(module_state_folder, file[0]), 'rb'))
            if need_optimizer_state:
                optimizer_state = pickle.load(open(os.path.join(self.checkpoint_folder, 'learner_group', 'learner', 'state.pkl'), 'rb'))['optimizer']
                # 组装state
                state = {'learner_group':{'learner':{
                    'rl_module':{'default_policy': module_state},
                    'optimizer': optimizer_state
                }}}
            else:
                state = {'learner_group':{'learner':{
                    'rl_module':{'default_policy': module_state},
                }}}
            algo.set_state(state)

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

                # print_directory_tree(self.train_folder, log_func=log)
                # 覆盖到训练文件夹
                shutil.copytree(os.path.join('/kaggle/working/alist', self.train_title), self.train_folder, dirs_exist_ok=True)
                # print_directory_tree(self.train_folder, log_func=log)
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


class TrainFolderManagerBC(TrainFolderManager):

    def check_point_file(self):
        """ 检查点文件 """
        return os.path.join(self.checkpoint_folder, "policy")

    def exists(self):
        """
        检查是否存在训练记录
        """
        return os.path.exists(self.check_point_file())
    
    def checkpoint(self, bc_trainer, best=False):
        """
        保存检查点
        """
        # 保存检查点
        bc_trainer.save(self.checkpoint_folder)
        # 保存最佳检查点
        if best:
            # 复制到最佳检查点文件夹
            if os.path.exists(self.best_checkpoint_folder):
                shutil.rmtree(self.best_checkpoint_folder)
            shutil.copytree(self.checkpoint_folder, self.best_checkpoint_folder)
            log(f'保存最佳检查点到 {self.best_checkpoint_folder}')
        # 推送
        self.push()

    def load_checkpoint(self, bc_trainer):
        """
        加载检查点
        """
        bc_trainer.load(self.checkpoint_folder)
        log(f'加载完毕')

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

                # BC 需要删除 progress.csv
                progress_file = os.path.join('/kaggle/working/alist', self.train_title, f"progress.csv") 
                if os.path.exists(progress_file):
                    os.remove(progress_file)

                # print_directory_tree(self.train_folder, log_func=log)
                # 覆盖到训练文件夹
                shutil.copytree(os.path.join('/kaggle/working/alist', self.train_title), self.train_folder, dirs_exist_ok=True)
                # print_directory_tree(self.train_folder, log_func=log)
        else:
            os.makedirs(self.train_folder, exist_ok=True)

class TrainFolderManagerSB3(TrainFolderManager):

    def check_point_file(self):
        """ 检查点文件 """
        return os.path.join(self.checkpoint_folder, f"{self.train_folder}.zip")

    def exists(self):
        """
        检查是否存在训练记录
        """
        return os.path.exists(self.check_point_file())

    def load_checkpoint(self, model, custom_objects=None):
        """
        加载检查点
        """
        _model = model.load(self.check_point_file(), custom_objects= custom_objects)
        policy_state_dict = _model.policy.state_dict()  
        model.policy.load_state_dict(policy_state_dict)  


class TrainFolderManagerOptuna(TrainFolderManager):
    def check_point_file(self):
        """ 检查点文件 """
        return os.path.join(self.checkpoint_folder, 'optuna_study.pkl')

    def exists(self):
        """
        检查是否存在训练记录
        """
        return os.path.exists(self.check_point_file())

    def load_checkpoint(self):
        """
        加载检查点
        """
        # 加载现有的 study
        with open(self.check_point_file(), 'rb') as f:
            study = pickle.load(f)
        return study
