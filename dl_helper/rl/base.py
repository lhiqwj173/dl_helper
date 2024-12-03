import torch
import os
import shutil

from py_ext.tool import log, debug, get_log_folder, _get_caller_info
from py_ext.lzma import compress_folder, decompress
from py_ext.wechat import wx
from py_ext.alist import alist

class BaseAgent:
    def __init__(self,
                 action_dim,
                 features_dim,
                 features_extractor_class,
                 features_extractor_kwargs=None,
                 net_arch=None,
                 sync_alist=True,
    ):
        """Agent 基类
        
        Args:
            action_dim: 动作空间维度 
            features_dim: 特征维度
            features_extractor_class: 特征提取器类,必须提供
            features_extractor_kwargs: 特征提取器参数,可选
            net_arch: 网络架构参数,默认为一层mlp, 输入/输出维度为features_dim, action_dim
                [action_dim] / dict(pi=[action_dim], vf=[action_dim]) 等价
        """
        if features_extractor_class is None:
            raise ValueError("必须提供特征提取器类 features_extractor_class")
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs or {}
        self.features_dim = features_dim

        if net_arch is None:
            self.net_arch = dict(pi=[action_dim], vf=[action_dim])
        elif isinstance(net_arch, list):
            self.net_arch = dict(pi=net_arch, vf=net_arch)
        elif isinstance(net_arch, dict):
            if 'pi' in net_arch and 'vf' in net_arch:
                self.net_arch = net_arch
            else:
                raise ValueError("net_arch 字典需包含 'pi' 和 'vf' 键")
        else:
            raise ValueError("net_arch 必须是列表或字典, 表示mlp每层的神经元个数")

        self.sync_alist = sync_alist
        self.client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD')) if sync_alist else None

    def build_model(self):
        """构建Q网络,子类需要实现具体的网络结构"""
        raise NotImplementedError

    model_key_suffix = "_state_dict@@@"
    def state_dict_model(self, state_dict, model_key, model_obj):
        """增加key后缀， 添加到state_dict中"""
        state_dict[model_key + self.model_key_suffix] = model_obj.state_dict()

    def state_dict(self):
        """返回需要保存的状态字典
        
        Returns:
            dict: 包含模型参数等需要保存的状态
        """
        # 只保存关键参数
        return {
            "features_extractor_class": self.features_extractor_class,
            "features_extractor_kwargs": self.features_extractor_kwargs,
            "features_dim": self.features_dim,
            "net_arch": self.net_arch
        }

    def load_state_dict(self, state_dict):
        """加载保存的状态
        
        Args:
            state_dict (dict): 包含模型参数和配置的状态字典
        """
        for key, value in state_dict.items():
            if key.endswith(self.model_key_suffix): 
                # 获取原始属性名（去掉_state_dict后缀）
                attr_name = key[:-len(self.model_key_suffix)]
                if hasattr(self, attr_name):
                    # 加载模型参数
                    getattr(self, attr_name).load_state_dict(value)
            else:
                # 加载普通配置
                setattr(self, key, value)

    def save(self, root):
        """保存参数
        """
        torch.save(self.state_dict(), os.path.join(root, 'agent_data.pth'))

    def load(self, root):
        """加载参数
        """
        self.load_state_dict(torch.load(os.path.join(root, 'agent_data.pth')))

    def learn(self, train_title):
        self.root = f'{train_title}'
        if not self.sync_alist:
            os.makedirs(self.root, exist_ok=True)
            return

        try:
            # 下载
            _file = f'alist/{train_title}.7z'
            # 下载文件
            download_folder = f'/train_data/'
            self.client.download(f'{download_folder}{train_title}.7z', 'alist/')
            # 解压文件
            decompress(_file)
            # 移动
            shutil.copytree(os.path.join('alist', train_title), self.root, dirs_exist_ok=True)
            # 读取训练参数
            self.load(self.root)
        except Exception as e:
            print(f'下载失败: {e}')
            os.makedirs(self.root, exist_ok=True)
        