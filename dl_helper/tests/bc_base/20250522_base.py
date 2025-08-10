import sys, torch, os, pickle
from torch.nn.init import xavier_uniform_, zeros_, normal_
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
from dl_helper.rl.rl_env.lob_trade.lob_const import DATA_FOLDER
from dl_helper.tester import test_base
from dl_helper.train_param import Params, match_num_processes
from dl_helper.scheduler import OneCycle_fast
from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.base import transform
from dl_helper.trainer import run
from dl_helper.tool import model_params_num

"""
订单簿 bc 数据集
目标: tcn 替换的 DeepLOB 模型，作为基准
结论: 

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

class DeepLOB(nn.Module):
    """
    DeepLOB模型，使用TCN替代LSTM进行序列建模
    模型参数量: 333842
    """
    def __init__(
            self, 
            input_dims: tuple = (50, 4),  # 时间维度和特征维度
            extra_input_dims: int = 3,
            output_dim=2,
            use_regularization=False,
            preserve_seq_len: bool = True  # 新增：是否在CNN中保留时序长度
        ):
        super().__init__()
        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        self.use_regularization = use_regularization
        self.y_len = output_dim

        static_out_dim = 32
        self.static_feature_processor = StaticFeatureProcessor(
            num_categories=5,
            embedding_dim=16,
            num_features=2,
            output_dim=static_out_dim,
            static_hidden=(64, 32),
            use_regularization=use_regularization,
        )

        # 卷积块1：使用LeakyReLU激活函数
        # 用于提取每个买卖档位价量之间的关系
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
        )
        
        # 卷积块2：使用Tanh激活函数
        # 用于提取每个档位 买卖盘之间的关系
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(32), nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32), nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32), nn.Tanh(),
        )
        
        # 卷积块3：使用LeakyReLU激活函数
        # 用于处理多个档位之间的关系
        if input_dims[1] > 4:
            # 如果特征维度大于4，意味着有多个档位的数据
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
                nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
                nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
                nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
            )
        else:
            self.conv3 = None
        
        # Inception模块1：3x1卷积分支
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01),
        )
        
        # Inception模块2：5x1卷积分支
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01),
        )
        
        # Inception模块3：最大池化分支
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01),
        )
        
        # TCN网络
        # TCN的通道数配置：从192输入逐渐减少到64
        tcn_channels = [128, 96, 64]
        dropout_rate = 0.2 if self.use_regularization else 0.0
        self.tcn = TemporalConvNet(num_inputs=192, 
                                   num_channels=tcn_channels, 
                                   kernel_size=3, 
                                   dropout=dropout_rate,
                                   use_regularization=self.use_regularization)
        
        # 融合网络
        fusion_dim = 64 + static_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, self.y_len)
        )
        
        # 如果使用正则化，添加dropout层
        if self.use_regularization:
            self.dropout_final = nn.Dropout(0.3)
    
        # 应用统一的权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        优化: 统一、集中的权重初始化函数。
        该函数会被self.apply()递归地应用到所有子模块。
        """
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            xavier_uniform_(module.weight)

    def forward(self, observations):
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息, 训练时不使用
        extra_x = use_obs[:, -self.extra_input_dims:]
        # 直接将原始时间序列数据重塑为TCN所需的格式
        # 从 [..., time*features+extra] 到 [..., time, features]
        batch_size = observations.size(0)
        time_dim, feature_dim = self.input_dims

        # 提取非静态特征部分并重塑
        x = use_obs[:, :-self.extra_input_dims].reshape(batch_size, time_dim, feature_dim)
        # 增加一个维度 [batch_size, time_length, num_rows] -> [batch_size, 1, time_length, num_rows]
        x = x.unsqueeze(1)# [B, 1, 50, 4]

        # 处理静态特征
        static_out = self.static_feature_processor(extra_x)

        # 卷积特征提取
        x = self.conv1(x)# [B, 32, 44, 2]
        x = self.conv2(x)# [B, 32, 38, 1]
        if self.conv3 is not None:
            x = self.conv3(x)
        
        # Inception模块进行多尺度特征提取
        x_inp1 = self.inp1(x)  # 3x1卷积分支 [B, 64, 38, 1]
        x_inp2 = self.inp2(x)  # 5x1卷积分支 [B, 64, 38, 1]
        x_inp3 = self.inp3(x)  # 池化分支 [B, 64, 38, 1]
        
        # 连接所有Inception分支 (batch_size, 192, height, width)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1) # [B, 192, 38, 1]
        
        # 调整维度适配TCN：(batch_size, channels, sequence_length)
        # 将高度维度作为序列长度，宽度维度压缩到通道中
        x = x.permute(0, 2, 1, 3)  # (batch_size, height, channels, width)[B, 38, 192, 1]
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2] * x.shape[3]))  # (batch_size, seq_len, features)[B, 38, 192]
        x = x.permute(0, 2, 1)  # (batch_size, features, seq_len) - TCN需要的格式[B, 192, 38]
        
        # TCN进行时序建模
        x = self.tcn(x)  # (batch_size, 64, seq_len)[B, 64, 38]
        
        # 取最后一个时间步的输出
        x = x[:, :, -1]  # (batch_size, 64)[B, 64]

        # 融合静态特征
        x = torch.cat((x, static_out), dim=1)# [B, 96]
        
        # 应用最终的dropout（如果启用正则化）
        if self.use_regularization:
            x = self.dropout_final(x)
        
        # 融合网络
        output = self.fusion(x)# [B, 2]
        
        # 数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            model_sd = self.state_dict()
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("output contains nan or inf")
        
        return output

class DeepLOB_v2(nn.Module):
    """
    DeepLOB模型 v2 版本
    - 增加了对订单簿延伸特征（如mid-price, imbalance等）的处理流。
    - 采用多流架构：LOB流、延伸特征流、静态特征流，最后进行融合。
    模型参数量: 342610
    """
    def __init__(
            self,
            # --- 新增的维度参数 ---
            num_lob_levels: int = 1,              # LOB的档位数 (例如1档)
            num_extension_features: int = 5,      # 延伸特征的数量 (例如mid_price, mid_vol, spread, imbalance)
            # --- 原有参数 ---
            time_steps: int = 50,                 # 时间序列长度
            static_input_dims: int = 3,           # 静态特征数量 (id, time, position)
            output_dim: int = 2,                  # 输出维度 (0/1分类)
            use_regularization: bool = False,
        ):
        super().__init__()
        
        # --- 维度信息 ---
        self.time_steps = time_steps
        self.num_lob_levels = num_lob_levels
        self.lob_feature_dim = 4 # (ask_price, ask_vol, bid_price, bid_vol)
        self.num_extension_features = num_extension_features
        self.static_input_dims = static_input_dims
        self.y_len = output_dim
        self.use_regularization = use_regularization
        
        # --- 1. 静态特征处理流 ---
        static_out_dim = 32
        self.static_feature_processor = StaticFeatureProcessor(
            num_categories=10,
            embedding_dim=16,
            num_features=2, # time_to_close, has_position
            output_dim=static_out_dim,
            static_hidden=(64, 32),
            use_regularization=use_regularization,
        )

        # --- 2. LOB特征处理流 (基本不变) ---
        # 卷积块1
        self.lob_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
        )
        
        # 卷积块2
        self.lob_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(32), nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32), nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32), nn.Tanh(),
        )
        
        # 卷积块3 (处理多档位)
        # 输入特征维度是 lob_feature_dim * num_lob_levels / 2 (conv1) / 2 (conv2) = num_lob_levels
        if self.num_lob_levels > 1:
            self.lob_conv3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, self.num_lob_levels)),
                nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)), 
                nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
                nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
            )
        else:
            self.lob_conv3 = None
            
        # Inception模块 (不变)
        self.inp1 = self._build_inception_branch(64, (3,1))
        self.inp2 = self._build_inception_branch(64, (5,1))
        self.inp3 = self._build_inception_pool_branch(64)
        
        # LOB流的TCN
        lob_tcn_input_dim = 64 * 3  # 3个Inception分支
        lob_tcn_channels = [128, 96, 64]
        lob_tcn_output_dim = lob_tcn_channels[-1]
        dropout_rate = 0.2 if self.use_regularization else 0.0
        self.lob_tcn = TemporalConvNet(
            num_inputs=lob_tcn_input_dim, 
            num_channels=lob_tcn_channels, 
            kernel_size=3, 
            dropout=dropout_rate,
            use_regularization=self.use_regularization
        )
        
        # --- 3. 新增: 延伸特征处理流 (使用TCN) ---
        ext_tcn_channels = [32, 32] # 可以根据延伸特征的复杂性调整
        ext_tcn_output_dim = ext_tcn_channels[-1]
        self.extension_tcn = TemporalConvNet(
            num_inputs=self.num_extension_features,
            num_channels=ext_tcn_channels,
            kernel_size=3,
            dropout=dropout_rate,
            use_regularization=self.use_regularization
        )
        self.dropout_final = nn.Dropout(0.3) if self.use_regularization else None
        
        # --- 4. 融合网络 ---
        # 融合来自三个流的特征
        fusion_input_dim = lob_tcn_output_dim + ext_tcn_output_dim + static_out_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3) if self.use_regularization else nn.Identity(),
            nn.Linear(64, self.y_len)
        )
    
        self.apply(self._init_weights)

    def _build_inception_branch(self, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(1,1), padding='same'),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.01),
        )

    def _build_inception_pool_branch(self, out_channels):
        return nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(1,1), padding='same'),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.01),
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            xavier_uniform_(module.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            observations (torch.Tensor): 形状为 [batch_size, total_features]。
                特征排列顺序应为: 
                [LOB数据 (50*4*n) | 延伸特征 (50*m) | 静态特征 (3)]
        """
        batch_size = observations.size(0)

        # --- 输入特征分割 ---
        lob_data_len = self.time_steps * self.lob_feature_dim * self.num_lob_levels
        ext_data_len = self.time_steps * self.num_extension_features
        
        # 1. LOB数据
        x_lob = observations[:, :lob_data_len]
        x_lob = x_lob.reshape(batch_size, self.time_steps, self.lob_feature_dim * self.num_lob_levels)
        
        # 2. 延伸特征数据
        x_ext = observations[:, lob_data_len : lob_data_len + ext_data_len]
        x_ext = x_ext.reshape(batch_size, self.time_steps, self.num_extension_features)
        
        # 3. 静态特征数据
        x_static = observations[:, lob_data_len + ext_data_len: lob_data_len + ext_data_len + self.static_input_dims]

        # --- 流 1: LOB 特征处理 ---
        # [B, T, F] -> [B, 1, T, F] for Conv2d
        h_lob = x_lob.unsqueeze(1)
        h_lob = self.lob_conv1(h_lob)
        h_lob = self.lob_conv2(h_lob)
        if self.lob_conv3 is not None:
            h_lob = self.lob_conv3(h_lob)
        
        # Inception 模块
        h_lob_inp1 = self.inp1(h_lob)
        h_lob_inp2 = self.inp2(h_lob)
        h_lob_inp3 = self.inp3(h_lob)
        h_lob = torch.cat((h_lob_inp1, h_lob_inp2, h_lob_inp3), dim=1) # [B, 192, H, W]
        
        # 调整维度以适应TCN [B, C, T]
        h_lob = h_lob.permute(0, 2, 1, 3)
        h_lob = torch.reshape(h_lob, (batch_size, h_lob.shape[1], -1)) # [B, T_new, C_new]
        h_lob = h_lob.permute(0, 2, 1) # [B, C_new, T_new]
        
        # LOB TCN
        h_lob = self.lob_tcn(h_lob)
        h_lob = h_lob[:, :, -1] # 取序列最后一个时间步的输出 [B, lob_tcn_output_dim]
        
        # --- 流 2: 延伸特征处理 ---
        # [B, T, M] -> [B, M, T] for TCN
        h_ext = x_ext.permute(0, 2, 1) 
        h_ext = self.extension_tcn(h_ext)
        h_ext = h_ext[:, :, -1] # 取序列最后一个时间步的输出 [B, ext_tcn_output_dim]
        
        # --- 流 3: 静态特征处理 ---
        h_static = self.static_feature_processor(x_static) # [B, static_out_dim]
        
        # --- 4. 特征融合 ---
        # 将三个流的输出拼接起来
        fusion_input = torch.cat([h_lob, h_ext, h_static], dim=1)
        
        if self.use_regularization:
            fusion_input = self.dropout_final(fusion_input)
            
        output = self.fusion_net(fusion_input)
        
        # 数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            model_sd = self.state_dict()
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            with open('debug_nan_data.pkl', 'wb') as f:
                pickle.dump({
                    'state_dict': model_sd,
                    'observations': observations.detach().cpu(),
                }, f)
            raise ValueError("output contains nan or inf")
        
        return output

# 简单的数据结构
his_len = 50
base_features = [item for i in range(1) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']]
ext_features = [
    "距离收盘秒数",
    "EXT_micro_price",
    "EXT_volatility_5",
    "EXT_pressure_imbalance",
    'EXT_volatility_wap_5',
    'EXT_relative_spread_l1',
    'EXT_weighted_mid_price_v2',
    'EXT_depth_imbalance',
    'EXT_ofi',
]
data_config = {
    'his_len': his_len,# 每个样本的 历史数据长度
    'need_cols':  base_features + ext_features,
}

class test(test_base):
    title_base = '20250518_base'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params_kwargs['y_n'] = 2
        self.params_kwargs['classify'] = True
        self.params_kwargs['no_better_stop'] = 0
        self.params_kwargs['batch_n'] = 64
        self.params_kwargs['epochs'] = 200
        self.params_kwargs['learning_rate'] = 3e-4
        self.params_kwargs['no_better_stop'] = 0
        self.params_kwargs['label_smoothing'] = 0

        seeds = range(5)
        self.model_cls = DeepLOB_v2
        self.seed = seeds[self.idx]
        self.params_kwargs['seed'] = self.seed

        # 实例化 参数对象
        self.para = Params(
            **self.params_kwargs
        )

        # 准备数据集
        data_dict_folder = os.path.join(os.path.dirname(DATA_FOLDER), 'data_dict')
        train_split_rng = np.random.default_rng(seed=self.seed)
        self.train_dataset = LobTrajectoryDataset(data_folder= data_dict_folder, data_config = data_config, train_split_rng=train_split_rng)
        self.val_dataset = LobTrajectoryDataset(data_folder= data_dict_folder, data_config = data_config, data_type='val')
        self.test_dataset = LobTrajectoryDataset(data_folder= data_dict_folder, data_config = data_config, data_type='test')
    
    def get_title_suffix(self):
        """获取后缀"""
        return f'{self.model_cls.__name__}_seed{self.seed}'

    def get_model(self):
        return self.model_cls(
            num_lob_levels=1,
            num_extension_features=len(ext_features),
            time_steps=his_len,
            static_input_dims=3,
            output_dim=2,
            use_regularization=False,
        )
    
    def get_data(self, _type, data_sample_getter_func=None):
        if _type == 'train':
            return DataLoader(dataset=self.train_dataset, batch_size=self.para.batch_size, shuffle=True, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'val':
            return DataLoader(dataset=self.val_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'test':
            return DataLoader(dataset=self.test_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        
if '__main__' == __name__:
    # 测试模型
    model = DeepLOB_v2(
        num_lob_levels=1,
        num_extension_features=len(ext_features),
        time_steps=his_len,
        static_input_dims=3,
        output_dim=2,
        use_regularization=False,
    )
    x = torch.randn(10, his_len*(len(ext_features) + len(base_features))+4)
    x[:, -4] = 0
    print(model(x).shape)
    print(f"模型参数量: {model_params_num(model)}")

    # input_folder = r'/kaggle/input'
    # # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
    # data_folder_name = os.listdir(input_folder)[0]
    # data_folder = os.path.join(input_folder, data_folder_name)

    # run(
    #     test, 
    # )