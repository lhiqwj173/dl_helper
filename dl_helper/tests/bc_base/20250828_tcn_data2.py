import sys, torch, os, pickle, math
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
特征: EXT_total_ofi | EXT_ofi_level_1 | EXT_ofi_imbalance | EXT_log_ret_mid_price | EXT_log_ret_micro_price
标签: deeplob
模型: TimeSeriesStaticModelx16

目标: 
    专注于 train_loss
    观察 deeplob / bc 标签 在420days数据集 训练效果

结论: 

"""
class StaticFeatureProcessor(nn.Module):
    """
    静态特征处理模块
    用于处理包含类别特征和数值特征的静态输入。
    增强功能：当 num_categories 为 1 时，将忽略类别特征，只处理数值特征。
    """
    def __init__(
        self,
        num_categories: int = 10,
        embedding_dim: int = 16,
        num_features: int = 2,
        output_dim: int = 32,
        static_hidden: tuple = (64, 32),
        use_regularization: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # 参数验证
        if num_categories < 1: raise ValueError("num_categories必须大于等于1")
        if num_categories > 1 and embedding_dim <= 0: raise ValueError("当num_categories > 1时，embedding_dim必须大于0")
        if num_features < 0: raise ValueError("num_features不能为负数")
        if num_categories == 1 and num_features == 0: raise ValueError("当num_categories为1时, num_features必须大于0, 否则没有有效输入")
        if output_dim <= 0: raise ValueError("output_dim必须大于0")
        if not (0 <= dropout_rate < 1): raise ValueError("dropout_rate必须在[0, 1)范围内")
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.output_dim = output_dim
        self.use_regularization = use_regularization
        self.dropout_rate = dropout_rate if use_regularization else 0.0
        
        static_input_dim = 0
        
        # 类别特征嵌入层 (仅当类别数 > 1 时创建)
        if self.num_categories > 1:
            self.id_embedding = nn.Embedding(num_categories, embedding_dim)
            static_input_dim += embedding_dim
        else:
            self.id_embedding = None

        # 数值特征预处理层
        if num_features > 0:
            self.num_feature_norm = nn.BatchNorm1d(num_features)
            static_input_dim += num_features
        
        # 静态特征融合网络
        layers = []
        in_dim = static_input_dim
        
        for i, h_dim in enumerate(static_hidden):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.static_net = nn.Sequential(*layers)
        
    def forward(self, static_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            static_features: 静态特征张量，形状为 [batch_size, total_features]
                           其中第一个特征是类别特征(整数)，其余为数值特征
        
        Returns:
            处理后的静态特征，形状为 [batch_size, output_dim]
        """
        if static_features.size(1) != (1 + self.num_features):
            raise ValueError(f"输入特征维度不匹配，期望 {1 + self.num_features}，实际 {static_features.size(1)}")
            
        # 当类别数 > 1 时，处理类别特征和数值特征
        if self.num_categories > 1:
            cat_feat = static_features[:, 0].long()
            if torch.any(cat_feat < 0) or torch.any(cat_feat >= self.num_categories):
                raise ValueError(f"类别特征值必须在[0, {self.num_categories-1}]范围内")
            
            embedded = self.id_embedding(cat_feat)
            
            if self.num_features > 0:
                num_feat = static_features[:, 1:]
                num_feat = self.num_feature_norm(num_feat)
                static_input = torch.cat([embedded, num_feat], dim=1)
            else:
                static_input = embedded
        
        # 当类别数 == 1 时，忽略类别特征，只处理数值特征
        else:
            if self.num_features > 0:
                # 忽略第一个类别特征，只取后面的数值特征
                num_feat = static_features[:, 1:]
                static_input = self.num_feature_norm(num_feat)
            else:
                # 这种情况在__init__中已经被阻止，但为了代码健壮性可以保留
                raise RuntimeError("当num_categories为1时，必须有数值特征。")

        output = self.static_net(static_input)
        return output
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        """
        return self.output_dim
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        获取嵌入层权重，用于可视化或分析
        """
        if self.id_embedding is None:
            raise AttributeError("当 num_categories=1 时，模型没有嵌入层。")
        return self.id_embedding.weight.detach()
    
    def freeze_embedding(self):
        """
        冻结嵌入层参数，防止训练时更新
        """
        if self.id_embedding is None:
            # 如果没有嵌入层，则静默返回
            return
        for param in self.id_embedding.parameters():
            param.requires_grad = False
    
    def unfreeze_embedding(self):
        """
        解冻嵌入层参数，允许训练时更新
        """
        if self.id_embedding is None:
            # 如果没有嵌入层，则静默返回
            return
        for param in self.id_embedding.parameters():
            param.requires_grad = True

class TemporalBlock(nn.Module):
    """TCN的基本时间块，包含因果卷积、残差连接和正则化"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, use_regularization=True):
        super(TemporalBlock, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(n_outputs),                                                  # BN层固定存在
            nn.ReLU(),
            nn.Dropout(dropout) if use_regularization else nn.Identity(),               # Dropout受use_regularization控制
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(n_outputs),                                                  # BN层固定存在
            nn.ReLU(),
            nn.Dropout(dropout) if use_regularization else nn.Identity()                # Dropout受use_regularization控制
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播，包含残差连接"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """移除卷积后右侧的填充，保持因果性"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    """时间卷积网络 (TCN)"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, use_regularization=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀率呈指数增长
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout, use_regularization=use_regularization
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FusionNetwork(nn.Module):
    """
    一个可配置的融合网络，用于合并来自不同来源的特征。
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hiddens: tuple = (64, 32),
        use_regularization: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hiddens:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if use_regularization and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 模型参数量: 86539
class TimeSeriesStaticModel(nn.Module):
    """
    一个结合了TCN处理时间序列特征和MLP处理静态特征的混合模型。

    该模型包含两个并行的处理流：
    1. 时间序列流：使用时间卷积网络（TCN）来捕捉时间序列数据中的时序依赖关系。
    2. 静态特征流：使用一个多层感知机（MLP），包括嵌入层，来处理不随时间变化的静态特征。

    最后，两个流的输出被拼接在一起，通过一个融合网络（Fusion Network）来产生最终的预测。
    """
    def __init__(
        self,
        # 时间序列参数
        num_ts_features: int,
        time_steps: int,
        tcn_channels: list = [64, 64, 64, 32, 32],
        tcn_kernel_size: int = 3,
        # 静态特征参数
        num_static_features: int = 3, # 1个类别特征 + 2个数值特征
        static_num_categories: int = 1,# 默认不使用类别特征
        static_embedding_dim: int = 16,
        static_output_dim: int = 32,
        static_hidden_dims: tuple = (64, 32),
        # 融合网络参数
        fusion_hidden_dims: tuple = None,
        # 输出参数
        output_dim: int = 2,
        # 正则化参数
        use_regularization: bool = False,
        dropout_rate: float = 0.2,
    ):
        """
        初始化模型。

        Args:
            num_ts_features (int): 每个时间步的时间序列特征数量。
            time_steps (int): 时间序列的长度。
            tcn_channels (list): TCN中每个时间块的输出通道数列表。
            tcn_kernel_size (int): TCN中卷积层的核大小。
            num_static_features (int): 静态特征的总数（类别+数值）。
            static_num_categories (int): 静态数据中类别特征的类别总数。
            static_embedding_dim (int): 类别特征的嵌入维度。
            static_output_dim (int): 静态特征处理器处理后的输出维度。
            static_hidden_dims (tuple): 静态特征处理器中隐藏层的维度。
            fusion_hidden_dims (tuple): 融合网络中隐藏层的维度。如果为None，则自动计算。
            output_dim (int): 模型的最终输出维度。
            use_regularization (bool): 是否启用Dropout正则化。
            dropout_rate (float): TCN和融合网络中使用的dropout比率。
        """
        super().__init__()
        
        # --- 保存维度信息 ---
        self.num_ts_features = num_ts_features
        self.time_steps = time_steps
        self.num_static_features = num_static_features
        self.output_dim = output_dim
        self.use_regularization = use_regularization
        
        # --- 1. 静态特征处理流 ---
        self.static_processor = StaticFeatureProcessor(
            num_categories=static_num_categories,
            embedding_dim=static_embedding_dim,
            num_features=num_static_features - 1, # 减去1个类别特征
            output_dim=static_output_dim,
            static_hidden=static_hidden_dims,
            use_regularization=use_regularization,
            dropout_rate=dropout_rate,
        )

        # --- 2. 时间序列特征处理流 (TCN) ---
        self.ts_tcn = TemporalConvNet(
            num_inputs=self.num_ts_features,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout_rate,
            use_regularization=self.use_regularization
        )
        tcn_output_dim = tcn_channels[-1]

        # --- 3. 融合网络 ---
        fusion_input_dim = tcn_output_dim + static_output_dim
        if fusion_hidden_dims is None:
            # 自动计算隐藏层维度
            fusion_hidden_dims = ((fusion_input_dim + output_dim) // 2,)

        self.fusion_net = FusionNetwork(
            input_dim=fusion_input_dim,
            output_dim=self.output_dim,
            hiddens=fusion_hidden_dims,
            use_regularization=self.use_regularization,
            dropout_rate=dropout_rate,
        )
    
        # --- 权重初始化 ---
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        健壮的权重初始化函数，遵循最佳实践。
        """
        # 最终分类/回归层的权重和偏置初始化为0，有助于稳定训练初期
        if hasattr(self, 'fusion_net') and hasattr(self.fusion_net, 'net') and module == self.fusion_net.net[-1]:
            init.constant_(module.weight, 0)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        # 对线性和卷积层使用Kaiming初始化 (适用于ReLU)
        elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)
        # 对BatchNorm层使用标准初始化
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        # 对嵌入层使用Xavier初始化
        elif isinstance(module, nn.Embedding):
            init.xavier_uniform_(module.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            observations (torch.Tensor): 形状为 [batch_size, total_features] 的输入张量。
                特征排列顺序应为: [时间序列特征 (T * F) | 静态特征 (S)]
        
        Returns:
            torch.Tensor: 模型的输出，形状为 [batch_size, output_dim]。
        """
        batch_size = observations.size(0)

        # --- 1. 输入特征分割 ---
        ts_data_len = self.time_steps * self.num_ts_features
        
        # 提取时间序列数据
        x_ts = observations[:, :ts_data_len]
        # 提取静态数据 最后一个日期数据，不在训练中使用
        x_static = observations[:, ts_data_len:-1]

        # --- 2. 时间序列流处理 ---
        # Reshape: [B, T * F] -> [B, T, F]
        h_ts = x_ts.view(batch_size, self.time_steps, self.num_ts_features)
        # Permute: [B, T, F] -> [B, F, T] 以匹配Conv1d的输入 (N, C_in, L_in)
        h_ts = h_ts.permute(0, 2, 1)
        
        # 通过TCN网络
        h_ts = self.ts_tcn(h_ts)
        
        # 取序列最后一个时间步的输出作为该序列的表示
        # h_ts shape: [B, tcn_output_dim, T] -> [B, tcn_output_dim]
        h_ts = h_ts[:, :, -1]
        
        # --- 3. 静态特征流处理 ---
        h_static = self.static_processor(x_static) # [B, static_output_dim]
        
        # --- 4. 特征融合 ---
        # 将两个流的输出拼接起来
        fusion_input = torch.cat([h_ts, h_static], dim=1)
        
        # 通过融合网络得到最终输出
        output = self.fusion_net(fusion_input)
        
        return output

class TimeSeriesStaticModelx8(TimeSeriesStaticModel):
    """参数量约为原始模型八倍的版本"""
    def __init__(self, *args, **kwargs):
        factor = math.sqrt(8)  # ≈2.828, 使得总参数量 ~8x
        # 获取原始参数
        tcn_channels = kwargs.pop('tcn_channels', [64, 64, 64, 32, 32])
        static_embedding_dim = kwargs.pop('static_embedding_dim', 16)
        static_output_dim = kwargs.pop('static_output_dim', 32)
        static_hidden_dims = kwargs.pop('static_hidden_dims', (64, 32))

        # 按比例放大（向上取整更稳定）
        kwargs['tcn_channels'] = [int(round(c * factor)) for c in tcn_channels]
        kwargs['static_embedding_dim'] = int(round(static_embedding_dim * factor))
        kwargs['static_output_dim'] = int(round(static_output_dim * factor))
        kwargs['static_hidden_dims'] = tuple(int(round(d * factor)) for d in static_hidden_dims)
        kwargs['fusion_hidden_dims'] = None  # 让融合层自动重新计算维度

        super().__init__(*args, **kwargs)

class TimeSeriesStaticModelx16(TimeSeriesStaticModel):
    """参数量约为原始模型十六倍的版本"""
    def __init__(self, *args, **kwargs):
        factor = 4  # sqrt(16) = 4
        # 获取原始参数
        tcn_channels = kwargs.pop('tcn_channels', [64, 64, 64, 32, 32])
        static_embedding_dim = kwargs.pop('static_embedding_dim', 16)
        static_output_dim = kwargs.pop('static_output_dim', 32)
        static_hidden_dims = kwargs.pop('static_hidden_dims', (64, 32))

        # 按比例放大
        kwargs['tcn_channels'] = [int(c * factor) for c in tcn_channels]
        kwargs['static_embedding_dim'] = int(static_embedding_dim * factor)
        kwargs['static_output_dim'] = int(static_output_dim * factor)
        kwargs['static_hidden_dims'] = tuple(int(d * factor) for d in static_hidden_dims)
        kwargs['fusion_hidden_dims'] = None

        super().__init__(*args, **kwargs)

# 简单的数据结构
his_len = 30
base_features = [item for i in range(1) for item in [f'BASE卖{i+1}量', f'BASE买{i+1}量']]
base_features = []
ext_features = [
    "EXT_total_ofi",
    "EXT_ofi_level_1",
    "EXT_ofi_imbalance",
    "EXT_log_ret_mid_price",
    "EXT_log_ret_micro_price",
]
data_config = {
    'his_len': his_len,# 每个样本的 历史数据长度
    'need_cols':  base_features + ext_features,
}

class test(test_base):
    title_base = '20250828_data'
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
            for model_cls in [TimeSeriesStaticModelx8]:
                for use_data_file_num in [420]:
                    for data_folder in [
                        '/kaggle/input/bc-train-data-20250828-all/BC_train_data_20250828_bc',
                        '/kaggle/input/bc-train-data-20250828-all/BC_train_data_20250828_deeplob'
                    ]:
                        args.append((model_cls, i, use_data_file_num, data_folder))

        self.model_cls, self.seed, self.use_data_file_num, self.base_data_folder = args[self.idx]
        self.params_kwargs['seed'] = self.seed

        # 实例化 参数对象
        self.para = Params(
            **self.params_kwargs
        )

        # 准备数据集
        self.data_dict_folder = os.path.join(self.base_data_folder, 'data_dict')
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def get_dataloaders(self):
        """返回需要测试的数据集"""
        # 共 381 个日期
        test_batch_num = 50     
        data_sets = [
            LobTrajectoryDataset(
                data_folder=self.data_dict_folder, 
                data_config = data_config,
                base_data_folder=os.path.join(self.base_data_folder, 'train_data'), 
                data_type='test', 
                use_data_begin=i,
                use_data_end=i+test_batch_num
            )

            for i in range(0, 381, test_batch_num)
        ]

        data_loaders = [
            DataLoader(
                dataset=data_set,
                batch_size=self.para.batch_size,
                shuffle=False,
                num_workers=4//match_num_processes(),
                pin_memory=True,
            )

            for data_set in data_sets
        ]

        return data_loaders
    
    def get_models(self):
        """返回需要测试的模型"""
        model = self.model_cls(
            num_ts_features=len(ext_features) + len(base_features),
            time_steps=his_len,
            num_static_features=3,
        )

        # 加载模型
        from safetensors.torch import load_file
        model.load_state_dict(load_file(r"/kaggle/input/data200/pytorch/default/1/model.safetensors"))

        return [model]

    def get_title_suffix(self):
        """获取后缀"""
        # res = f'{self.model_cls.__name__}_seed{self.seed}'
        # res = f'{self.use_data_file_num}_seed{self.seed}'
        res = f'{os.path.basename(self.base_data_folder).split("_")[-1]}_{self.use_data_file_num}_seed{self.seed}'

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
        if self.test_dataset is None:
            train_split_rng = np.random.default_rng(seed=self.seed)
            self.train_dataset = LobTrajectoryDataset(
                data_folder= self.data_dict_folder, 
                input_zero=input_indepent, 
                sample_num_limit= None if not overfit else 5, 
                data_config = data_config, 
                base_data_folder=os.path.join(self.base_data_folder, 'train_data'),
                split_rng=train_split_rng,
                use_data_file_num=self.use_data_file_num,
            )
            self.val_dataset = LobTrajectoryDataset(
                data_folder=self.data_dict_folder, 
                data_config = data_config,
                base_data_folder=os.path.join(self.base_data_folder, 'train_data'), 
                data_type='val', 
                use_data_file_num=self.use_data_file_num)
            self.test_dataset = LobTrajectoryDataset(
                data_folder=self.data_dict_folder, 
                data_config = data_config,
                base_data_folder=os.path.join(self.base_data_folder, 'train_data'), 
                data_type='test', 
                use_data_file_num=self.use_data_file_num)

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
    mode='normal'

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
        elif arg.startswith('mode='):
            mode = arg.split('=')[1]

    # ################################
    # # 测试模型
    # ################################
    x = torch.randn(10, his_len*(len(ext_features) + len(base_features))+4)
    x[:, -4] = 0
    for model_cls in [TimeSeriesStaticModelx8, TimeSeriesStaticModelx16]:
        model = model_cls(
            num_ts_features=len(ext_features) + len(base_features),
            time_steps=his_len,
        )
        # print(model)
        model(x)
        print(f"{model_cls.__name__} 模型参数量: {model_params_num(model)}")

    # ################################
    # # 验证初始化损失 == log(C)
    # ################################
    if test_init_loss:
        from tqdm import tqdms
        init_losses = []
        for i in tqdm(range(10)):
            model = TimeSeriesStaticModelx16(
                num_ts_features=len(ext_features) + len(base_features),
                time_steps=his_len,
            )
            num_classes = 2
            criterion = nn.CrossEntropyLoss()
            batchsize = 128
            x = torch.randn(batchsize, his_len*(len(ext_features) + len(base_features))+4)
            x[:, -4] = 0
            y = torch.randint(0, num_classes, (batchsize,))  # 随机标签
            # 前向传播
            outputs = model(x)
            loss = criterion(outputs, y)
            init_losses.append(loss.item())
        print(init_losses)
        print(f"Initial loss: { np.mean(init_losses)}")
        print(f"Expected loss: {torch.log(torch.tensor(num_classes)).item()}")

    elif need_check_dependencies:
        x = torch.randn(10, his_len*(len(ext_features) + len(base_features))+4)
        x[:, -4] = 0
        run_dependency_check_without_bn(model, x, 3)

    else:
        # 开始训练
        run(
            test, 
            mode=mode,
        )
