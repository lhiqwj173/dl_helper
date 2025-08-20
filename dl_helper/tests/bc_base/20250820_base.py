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
    1. 考察不同 nomove 阈值的数据集的表现情况
    2. 考察新的标的范围对模型性能的影响
结论: 

    1. nomove 30 为最优，性能随着阈值减小单调递增
       同时阈值越小，训练数据越少，相同参数的模型越容易拟合，这个结论可能没有什么意义
       TODO 还需要考察各个数据集的最优参数模型（train_f1无显著提升）的val_f1性能对比

                                                        train_f1	val_f1	cost
        train_title			
        20250817_base_P100_DeepLOB_v2_final_nomove30	0.920442	0.648930	12.07h
        20250817_base_P100_DeepLOB_v2_final_nomove50	0.906923	0.638214	14.863999999999999h
        20250817_base_P100_DeepLOB_v2_final_nomove70	0.897084	0.620636	16.922h

    2. new_codes 训练样本 1(810372)@1(810372) 1(1079480)@1(1079480) 1(1292891)@1(1292891)
       old_codes 训练样本 1(330787)@1(330787)

       指标上来看 新标的范围 性能不如 旧的
       
                                                        train_f1	val_f1	cost
        train_title_old_codes		
        20250811_el_P100_DeepLOB_v2_ExtraLarge_final	0.992900	0.795387	8.42h
        20250811_s&l_P100_DeepLOB_v2_Large_final	    0.991325	0.779560	6.388h
        20250518_base_P100_DeepLOB_v2_final	            0.986987	0.851849	5.28h
        20250811_base_P100_DeepLOB_v2_final	            0.986273	0.776328	4.904999999999999h
        20250811_s&l_P100_DeepLOB_v2_Small_final	    0.973233	0.760379	4.3420000000000005h

"""
class StaticFeatureProcessor(nn.Module):
    """
    静态特征处理模块
    用于处理包含类别特征和数值特征的静态输入
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
        if num_categories <= 0: raise ValueError("num_categories必须大于0")
        if embedding_dim <= 0: raise ValueError("embedding_dim必须大于0")
        if num_features < 0: raise ValueError("num_features不能为负数")
        if output_dim <= 0: raise ValueError("output_dim必须大于0")
        if not (0 <= dropout_rate < 1): raise ValueError("dropout_rate必须在[0, 1)范围内")
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.output_dim = output_dim
        self.use_regularization = use_regularization
        self.dropout_rate = dropout_rate if use_regularization else 0.0
        
        # 类别特征嵌入层
        self.id_embedding = nn.Embedding(num_categories, embedding_dim)
        
        # 数值特征预处理层 (BatchNorm现在是默认组件)
        if num_features > 0:
            self.num_feature_norm = nn.BatchNorm1d(num_features)
        
        # 静态特征融合网络
        static_input_dim = embedding_dim + num_features
        layers = []
        in_dim = static_input_dim
        
        for i, h_dim in enumerate(static_hidden):
            layers.append(nn.Linear(in_dim, h_dim))
            # 优化: BatchNorm层在激活函数之前
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            # Dropout仅在启用正则化时添加
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            in_dim = h_dim
        
        # 最后的输出层（不加激活函数和dropout）
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
        # 输入验证
        if static_features.size(1) != (1 + self.num_features):
            raise ValueError(f"输入特征维度不匹配，期望 {1 + self.num_features}，实际 {static_features.size(1)}")
        
        # 分离类别特征和数值特征
        cat_feat = static_features[:, 0].long()
        
        # 验证类别特征的有效性
        if torch.any(cat_feat < 0) or torch.any(cat_feat >= self.num_categories):
            raise ValueError(f"类别特征值必须在[0, {self.num_categories-1}]范围内")
        
        # 对类别特征进行嵌入
        embedded = self.id_embedding(cat_feat)
        
        # 处理数值特征
        if self.num_features > 0:
            num_feat = static_features[:, 1:]
            
            # # 检查数值特征中的异常值
            # if torch.isnan(num_feat).any() or torch.isinf(num_feat).any():
            #     raise ValueError("数值特征中包含NaN或Inf值")
            
            # 应用批标准化（如果启用）
            num_feat = self.num_feature_norm(num_feat)
            
            # 拼接嵌入特征和数值特征
            static_input = torch.cat([embedded, num_feat], dim=1)
        else:
            # 如果没有数值特征，只使用嵌入特征
            static_input = embedded
        
        # 通过静态特征处理网络
        output = self.static_net(static_input)
        
        # # 输出验证
        # if torch.isnan(output).any() or torch.isinf(output).any():
        #     raise ValueError("模型输出包含NaN或Inf值")
        
        return output
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        Returns:
            输出特征的维度
        """
        return self.output_dim
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        获取嵌入层权重，用于可视化或分析
        
        Returns:
            嵌入层权重张量
        """
        return self.id_embedding.weight.detach()
    
    def freeze_embedding(self):
        """
        冻结嵌入层参数，防止训练时更新
        """
        for param in self.id_embedding.parameters():
            param.requires_grad = False
    
    def unfreeze_embedding(self):
        """
        解冻嵌入层参数，允许训练时更新
        """
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

# 模型参数量: 33606
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
        tcn_channels: list = [64, 32],
        tcn_kernel_size: int = 3,
        # 静态特征参数
        num_static_features: int = 3, # 1个类别特征 + 2个数值特征
        static_num_categories: int = 10,
        static_embedding_dim: int = 16,
        static_output_dim: int = 32,
        static_hidden_dims: tuple = (64, 32),
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
            dropout=dropout_rate if self.use_regularization else 0.0,
            use_regularization=self.use_regularization
        )
        tcn_output_dim = tcn_channels[-1]

        # --- 3. 融合网络 ---
        fusion_input_dim = tcn_output_dim + static_output_dim
        fusion_hidden_dim = (fusion_input_dim + output_dim) // 2
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if self.use_regularization else nn.Identity(),
            nn.Linear(fusion_hidden_dim, self.output_dim)
        )
    
        # --- 权重初始化 ---
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        健壮的权重初始化函数，遵循最佳实践。
        """
        # 最终分类/回归层的权重和偏置初始化为0，有助于稳定训练初期
        if hasattr(self, 'fusion_net') and module == self.fusion_net[-1]:
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

# 模型参数量: 13302
class TimeSeriesStaticModelSmall(TimeSeriesStaticModel):
    """
    (小版本) TimeSeriesStaticModel的轻量级变体，参数量约减少50%。
    """
    def __init__(self, num_ts_features: int, time_steps: int, num_static_features: int, **kwargs):
        # 定义小模型的默认参数
        model_params = dict(
            tcn_channels=[32, 16],
            tcn_kernel_size=3,
            static_output_dim=16,
            static_hidden_dims=(32, 16),
            output_dim=2,
            use_regularization=False,
        )
        # 允许用户通过kwargs覆盖默认值
        model_params.update(kwargs)
        
        super().__init__(
            num_ts_features=num_ts_features,
            time_steps=time_steps,
            num_static_features=num_static_features,
            **model_params
        )

# 模型参数量: 75750
class TimeSeriesStaticModelLarge(TimeSeriesStaticModel):
    """
    (大版本) TimeSeriesStaticModel的高容量变体，参数量约增加50%。
    """
    def __init__(self, num_ts_features: int, time_steps: int, num_static_features: int, **kwargs):
        # 定义大模型的默认参数
        model_params = dict(
            tcn_channels=[128, 64],
            tcn_kernel_size=3,
            static_output_dim=32,
            static_hidden_dims=(64, 32),
            output_dim=2,
            use_regularization=False,
        )
        # 允许用户通过kwargs覆盖默认值
        model_params.update(kwargs)

        super().__init__(
            num_ts_features=num_ts_features,
            time_steps=time_steps,
            num_static_features=num_static_features,
            **model_params
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
    title_base = '20250820_tcn_base'
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
            for model_cls in [TimeSeriesStaticModel, TimeSeriesStaticModelSmall, TimeSeriesStaticModelLarge]:
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
    for model_cls in [TimeSeriesStaticModel, TimeSeriesStaticModelSmall, TimeSeriesStaticModelLarge]:
        model = model_cls(
            num_ts_features=len(ext_features) + len(base_features),
            time_steps=his_len,
            num_static_features=3,
        )
        print(model)
        print(f"{model_cls.__name__} 模型参数量: {model_params_num(model)}")

    # ################################
    # # 验证初始化损失 == log(C)
    # ################################
    if test_init_loss:
        from tqdm import tqdm
        init_losses = []
        for i in tqdm(range(10)):
            model = TimeSeriesStaticModel(
                num_ts_features=len(ext_features) + len(base_features),
                time_steps=his_len,
                tcn_channels=[64, 32],
                tcn_kernel_size=3,
                num_static_features=3,
                static_num_categories=10,
                static_embedding_dim=16,
                static_output_dim=32,
                static_hidden_dims=(64, 32),
                output_dim=2,
                use_regularization=False,
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
        )
