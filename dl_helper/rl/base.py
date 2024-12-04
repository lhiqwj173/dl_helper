import torch
import os
import shutil

from py_ext.tool import log, debug, get_log_folder, _get_caller_info
from py_ext.lzma import compress_folder, decompress
from py_ext.wechat import wx

class BaseAgent:
    def __init__(self,
                 action_dim,
                 features_dim,
                 features_extractor_class,
                 features_extractor_kwargs=None,
                 net_arch=None,
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

    def build_model(self):
        """构建Q网络,子类需要实现具体的网络结构"""
        raise NotImplementedError

    model_key_suffix = "_state_dict@@@"
    def state_dict_model(self, state_dict, model_key, model_obj):
        """增加key后缀， 添加到state_dict中"""
        state_dict[model_key + self.model_key_suffix] = model_obj.state_dict()

    def state_dict(self):
        """只保存模型参数"""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """
        加载模型参数
        
        Args:
            state_dict (dict): 包含模型参数的状态字典
        """
        for key, value in state_dict.items():
            if key.endswith(self.model_key_suffix): 
                # 获取原始属性名（去掉_state_dict后缀）
                attr_name = key[:-len(self.model_key_suffix)]
                if hasattr(self, attr_name):
                    # 加载模型参数
                    getattr(self, attr_name).load_state_dict(value)

    def save(self, root=''):
        """
        保存参数
        """
        torch.save(self.state_dict(), os.path.join(root, 'agent_data.pth'))

    def load(self, root=''):
        """
        加载参数
        """
        file = os.path.join(root, 'agent_data.pth')
        if os.path.exists(file):
            self.load_state_dict(torch.load(file))

    def learn(self, train_title):
        self.root = f'{train_title}'
        os.makedirs(self.root, exist_ok=True)