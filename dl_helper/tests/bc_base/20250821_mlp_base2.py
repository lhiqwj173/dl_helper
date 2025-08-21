import sys, torch, os, pickle
from torch.nn.init import xavier_uniform_, zeros_, normal_
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import functools
from math import prod
from itertools import product
import numpy as np

from accelerate.utils import set_seed

from dl_helper.rl.custom_imitation_module.dataset import LobTrajectoryDataset
from dl_helper.tester import test_base
from dl_helper.train_param import Params, match_num_processes
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.base import transform
from dl_helper.trainer import run
from dl_helper.tool import model_params_num, check_dependencies, run_dependency_check_without_bn
"""
订单簿 bc 数据集
目标: 
    1. 考察容量更深的模型对性能的影响
        MLPPureModelSmall   模型参数量: 4514
        MLPPureModel        模型参数量: 13122
        MLPPureModelLarge   模型参数量: 50882

结论: 

"""

class MLPPureModel(nn.Module):
    """
    极简纯MLP模型：仅由一个MLP主干构成，处理拼接后的时序与静态特征。
    - 取消 StaticFeatureProcessor
    - 取消 fusion_net
    - 保留原始参数名和接口，便于替换
    """
    def __init__(
        self,
        # 时间序列参数
        num_ts_features: int,
        time_steps: int,
        ts_mlp_hidden_dims: tuple = (128, 64),
        # 静态特征参数（仅用于计算维度）
        num_static_features: int = 3,
        # 输出参数
        output_dim: int = 2,
        # 正则化参数
        use_regularization: bool = False,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # === 保留所有原始参数名 ===
        self.num_ts_features = num_ts_features
        self.time_steps = time_steps
        self.num_static_features = num_static_features
        self.output_dim = output_dim
        self.use_regularization = use_regularization
        self.dropout_rate = dropout_rate if use_regularization else 0.0

        # === 计算总输入维度 ===
        self.input_dim = num_ts_features * time_steps + num_static_features

        # === 单一MLP主干（从 input_dim 直接到 output_dim）===
        layers = []
        in_dim = self.input_dim

        for h_dim in ts_mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if self.use_regularization:
                layers.append(nn.Dropout(self.dropout_rate))
            in_dim = h_dim

        # 最后一层：直接映射到输出
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 去掉最后一列
        ts_len = self.num_ts_features * self.time_steps
        x = observations[:, :ts_len + self.num_static_features]  # [B, input_dim]
        return self.mlp(x)

class MLPPureModelSmall(MLPPureModel):
    def __init__(self, **kwargs):
        super().__init__(
            ts_mlp_hidden_dims=(64, 32),      # 更小的隐藏层
            **kwargs
        )

class MLPPureModelLarge(MLPPureModel):
    def __init__(self, **kwargs):
        super().__init__(
            ts_mlp_hidden_dims=(256, 128, 64),  # 更深更宽
            **kwargs
        )

# 简单的数据结构
his_len = 30
base_features = [item for i in range(1) for item in [f'BASE卖{i+1}量', f'BASE买{i+1}量']]
base_features = []
ext_features = [
    "EXT_ofi",
]
data_config = {
    'his_len': his_len,# 每个样本的 历史数据长度
    'need_cols':  base_features + ext_features,
}

class test(test_base):
    title_base = '20250821_pure'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params_kwargs['y_n'] = 2
        self.params_kwargs['classify'] = True
        self.params_kwargs['no_better_stop'] = 0
        self.params_kwargs['batch_n'] = 128
        self.params_kwargs['epochs'] = 200
        self.params_kwargs['learning_rate'] = 3e-4
        self.params_kwargs['no_better_stop'] = 0
        self.params_kwargs['label_smoothing'] = 0

        args = []
        for i in range(5):
            for model_cls in [MLPPureModelSmall, MLPPureModel, MLPPureModelLarge]:
                args.append((model_cls, i))

        self.model_cls, self.seed = args[self.idx]
        self.base_data_folder = r'/kaggle/input/bc-train-data-20250818'
        self.params_kwargs['seed'] = self.seed

        # 实例化 参数对象
        self.para = Params(
            **self.params_kwargs
        )

        # 准备数据集
        data_dict_folder = os.path.join(self.base_data_folder, 'data_dict')
        train_split_rng = np.random.default_rng(seed=self.seed)
        self.train_dataset = LobTrajectoryDataset(
            data_folder= data_dict_folder, 
            input_zero=input_indepent, 
            sample_num_limit= None if not overfit else 5, 
            data_config = data_config, 
            base_data_folder=os.path.join(self.base_data_folder, 'train_data'),
            split_rng=train_split_rng,
            use_data_file_num=200,
        )
        self.val_dataset = LobTrajectoryDataset(data_folder=data_dict_folder, data_config = data_config,base_data_folder=os.path.join(self.base_data_folder, 'train_data'), data_type='val', use_data_file_num=200)
        self.test_dataset = LobTrajectoryDataset(data_folder=data_dict_folder, data_config = data_config,base_data_folder=os.path.join(self.base_data_folder, 'train_data'), data_type='test', use_data_file_num=200)
    
    def get_title_suffix(self):
        """获取后缀"""
        res = f'{self.model_cls.__name__}_seed{self.seed}'

        if input_indepent:
            res += '_input_indepent'

        if overfit:
            res += '_overfit'

        return res

    def get_model(self):
        return self.model_cls(
            num_ts_features=len(ext_features) + len(base_features),
            time_steps=his_len,
            num_static_features=3,
        )
    
    def get_data(self, _type, data_sample_getter_func=None):
        if _type == 'train':
            return DataLoader(dataset=self.train_dataset, batch_size=self.para.batch_size, shuffle=True, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'val':
            return DataLoader(dataset=self.val_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'test':
            return DataLoader(dataset=self.test_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        
if '__main__' == __name__:
    # 全局训练参数
    input_indepent = False# 训练无关输入（全0）的模型
    test_init_loss = False# 验证初始化损失
    check_data_sample_balance = False # 检查 train/val/test 样本均衡
    overfit = False # 小样本过拟合测试
    need_check_dependencies = False # 检查梯度计算依赖关系

    for arg in sys.argv[1:]:
        if arg == 'test_init_loss':
            test_init_loss = True
        elif arg == 'input_indepent':
            input_indepent = True
        elif arg == 'check_data_sample_balance':
            check_data_sample_balance = True
        elif arg == 'overfit':
            overfit = True
        elif arg == "check_dependencies":
            need_check_dependencies = True

    # ################################
    # # 测试模型
    # ################################
    x = torch.randn(10, his_len*(len(ext_features) + len(base_features))+4)
    x[:, -4] = 0
    for model_cls in [MLPPureModelSmall, MLPPureModel, MLPPureModelLarge]:
        model = model_cls(
            num_ts_features=len(ext_features) + len(base_features),
            time_steps=his_len,
        )
        model(x)
        print(f"{model_cls.__name__} 模型参数量: {model_params_num(model)}")

    # # ################################
    # # # 验证初始化损失 == log(C)
    # # ################################
    # if test_init_loss:
    #     from tqdm import tqdm
    #     init_losses = []
    #     for i in tqdm(range(10)):
    #         model = MLPPureModel(
    #             num_ts_features=len(ext_features) + len(base_features),
    #             time_steps=his_len,
    #         )
    #         num_classes = 2
    #         criterion = nn.CrossEntropyLoss()
    #         batchsize = 128
    #         x = torch.randn(batchsize, his_len*(len(ext_features) + len(base_features))+4)
    #         x[:, -4] = 0
    #         y = torch.randint(0, num_classes, (batchsize,))  # 随机标签
    #         # 前向传播
    #         outputs = model(x)
    #         loss = criterion(outputs, y)
    #         init_losses.append(loss.item())
    #     print(init_losses)
    #     print(f"Initial loss: { np.mean(init_losses)}")
    #     print(f"Expected loss: {torch.log(torch.tensor(num_classes)).item()}")

    # elif need_check_dependencies:
    #     x = torch.randn(10, his_len*(len(ext_features) + len(base_features))+4)
    #     x[:, -4] = 0
    #     run_dependency_check_without_bn(model, x, 3)

    # else:
    #     # 开始训练
    #     run(
    #         test, 
    #     )
