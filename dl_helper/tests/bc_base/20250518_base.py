import sys, torch, os, pickle
from torch.nn.init import xavier_uniform_, zeros_
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import functools
from math import prod
from itertools import product

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
目标: 检验模型复杂度对性能的正向提升
结论: 
    TCNLob_4 训练爆炸
    
    TCNLob_3 最优
    train_loss	train_acc	train_f1	train_recall
    0.629298	0.654283	0.654325667	0.654357333

"""
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        # 裁剪 padding 部分以确保因果性
        return x[:, :, :-self.padding]

# 简单的 tcn 网络
# 模型参数量: 10221
class TCNLob_0(nn.Module):
    def __init__(
            self,
            input_dims: tuple = (10, 4),  # 时间维度和特征维度
            extra_input_dims: int = 3,
            output_dim=2,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        
        # 核心模型参数
        hidden_dim = 26

        # 嵌入维度
        embedding_dim = 8
        self.id_embedding = nn.Embedding(5, embedding_dim)
        
        # 静态特征处理
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Linear(static_input_dim, hidden_dim//2)
        
        # 计算输入特征维度 - 将原始输入直接展平为TCN输入
        # 输入形状为 [batch, time, features]
        self.tcn_input_dim = input_dims[1]  # 特征维度
        
        # 纯TCN架构替代原来的CNN feature extractor
        # 定义扩张率，增加感受野
        dilations = [1, 2, 4]
        
        # TCN块 - 每个块包含一个带扩张的1D卷积和残差连接
        self.tcn_blocks = nn.ModuleList()
        
        # 第一层 - 输入投影层
        self.input_proj = nn.Conv1d(self.tcn_input_dim, hidden_dim, kernel_size=1)
        
        # 多层TCN块，使用扩张卷积增加感受野
        for dilation in dilations:
            block = nn.Sequential(
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
            )
            self.tcn_blocks.append(block)
            
        # 全局池化层
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 融合网络
        fusion_dim = hidden_dim + hidden_dim//2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, output_dim)
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化模型的权重和偏置
        - 嵌入层、卷积层和线性层权重使用Xavier初始化
        - 偏置初始化为0
        """
        # 嵌入层初始化
        xavier_uniform_(self.id_embedding.weight)

        # 输入投影层
        xavier_uniform_(self.input_proj.weight)
        zeros_(self.input_proj.bias)

        # TCN块中的卷积层
        for block in self.tcn_blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        zeros_(layer.bias)

        # 静态特征处理层
        xavier_uniform_(self.static_net.weight)
        zeros_(self.static_net.bias)

        # 融合层
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息
        extra_x = use_obs[:, -self.extra_input_dims:]
        
        # 直接将原始时间序列数据重塑为TCN所需的格式
        # 从 [..., time*features+extra] 到 [..., time, features]
        batch_size = observations.size(0)
        time_dim, feature_dim = self.input_dims
        
        # 提取非静态特征部分并重塑
        x = use_obs[:, :-self.extra_input_dims].reshape(batch_size, time_dim, feature_dim)
        
        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]
        
        # 嵌入并处理静态特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = F.relu(self.static_net(static_input))
        
        # 重新排列时间序列数据为TCN所需的格式: [batch, features, time]
        # 从 [batch, time, features] 到 [batch, features, time]
        x = x.transpose(1, 2)
        
        # 应用输入投影
        x = self.input_proj(x)
        
        # 应用TCN块，带残差连接
        for tcn_block in self.tcn_blocks:
            residual = x
            x = tcn_block(x)
            x = x + residual  # 残差连接
        
        # 全局池化得到固定维度输出
        temporal_out = self.global_pool(x).squeeze(-1)
        
        # 融合静态和时序特征
        combined = torch.cat([temporal_out, static_out], dim=1)
        output = self.fusion(combined)
        
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

# 模型参数量: 15306
class TCNLob_1(nn.Module):
    def __init__(
            self,
            input_dims: tuple = (10, 4),  # 时间维度和特征维度
            extra_input_dims: int = 3,
            output_dim=2,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        
        # 核心模型参数
        hidden_dim = 32

        # 嵌入维度
        embedding_dim = 8
        self.id_embedding = nn.Embedding(5, embedding_dim)
        
        # 静态特征处理
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Linear(static_input_dim, hidden_dim//2)
        
        # 计算输入特征维度 - 将原始输入直接展平为TCN输入
        # 输入形状为 [batch, time, features]
        self.tcn_input_dim = input_dims[1]  # 特征维度
        
        # 纯TCN架构替代原来的CNN feature extractor
        # 定义扩张率，增加感受野
        dilations = [1, 2, 4]
        
        # TCN块 - 每个块包含一个带扩张的1D卷积和残差连接
        self.tcn_blocks = nn.ModuleList()
        
        # 第一层 - 输入投影层
        self.input_proj = nn.Conv1d(self.tcn_input_dim, hidden_dim, kernel_size=1)
        
        # 多层TCN块，使用扩张卷积增加感受野
        for dilation in dilations:
            block = nn.Sequential(
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
            )
            self.tcn_blocks.append(block)
            
        # 全局池化层
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 融合网络
        fusion_dim = hidden_dim + hidden_dim//2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, output_dim)
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化模型的权重和偏置
        - 嵌入层、卷积层和线性层权重使用Xavier初始化
        - 偏置初始化为0
        """
        # 嵌入层初始化
        xavier_uniform_(self.id_embedding.weight)

        # 输入投影层
        xavier_uniform_(self.input_proj.weight)
        zeros_(self.input_proj.bias)

        # TCN块中的卷积层
        for block in self.tcn_blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        zeros_(layer.bias)

        # 静态特征处理层
        xavier_uniform_(self.static_net.weight)
        zeros_(self.static_net.bias)

        # 融合层
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息
        extra_x = use_obs[:, -self.extra_input_dims:]
        
        # 直接将原始时间序列数据重塑为TCN所需的格式
        # 从 [..., time*features+extra] 到 [..., time, features]
        batch_size = observations.size(0)
        time_dim, feature_dim = self.input_dims
        
        # 提取非静态特征部分并重塑
        x = use_obs[:, :-self.extra_input_dims].reshape(batch_size, time_dim, feature_dim)
        
        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]
        
        # 嵌入并处理静态特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = F.relu(self.static_net(static_input))
        
        # 重新排列时间序列数据为TCN所需的格式: [batch, features, time]
        # 从 [batch, time, features] 到 [batch, features, time]
        x = x.transpose(1, 2)
        
        # 应用输入投影
        x = self.input_proj(x)
        
        # 应用TCN块，带残差连接
        for tcn_block in self.tcn_blocks:
            residual = x
            x = tcn_block(x)
            x = x + residual  # 残差连接
        
        # 全局池化得到固定维度输出
        temporal_out = self.global_pool(x).squeeze(-1)
        
        # 融合静态和时序特征
        combined = torch.cat([temporal_out, static_out], dim=1)
        output = self.fusion(combined)
        
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

# 模型参数量: 33882
class TCNLob_2(nn.Module):
    def __init__(
            self,
            input_dims: tuple = (10, 4),  # 时间维度和特征维度
            extra_input_dims: int = 3,
            output_dim=2,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        
        # 核心模型参数
        hidden_dim = 48

        # 嵌入维度
        embedding_dim = 8
        self.id_embedding = nn.Embedding(5, embedding_dim)
        
        # 静态特征处理
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Linear(static_input_dim, hidden_dim//2)
        
        # 计算输入特征维度 - 将原始输入直接展平为TCN输入
        # 输入形状为 [batch, time, features]
        self.tcn_input_dim = input_dims[1]  # 特征维度
        
        # 纯TCN架构替代原来的CNN feature extractor
        # 定义扩张率，增加感受野
        dilations = [1, 2, 4]
        
        # TCN块 - 每个块包含一个带扩张的1D卷积和残差连接
        self.tcn_blocks = nn.ModuleList()
        
        # 第一层 - 输入投影层
        self.input_proj = nn.Conv1d(self.tcn_input_dim, hidden_dim, kernel_size=1)
        
        # 多层TCN块，使用扩张卷积增加感受野
        for dilation in dilations:
            block = nn.Sequential(
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
            )
            self.tcn_blocks.append(block)
            
        # 全局池化层
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 融合网络
        fusion_dim = hidden_dim + hidden_dim//2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, output_dim)
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化模型的权重和偏置
        - 嵌入层、卷积层和线性层权重使用Xavier初始化
        - 偏置初始化为0
        """
        # 嵌入层初始化
        xavier_uniform_(self.id_embedding.weight)

        # 输入投影层
        xavier_uniform_(self.input_proj.weight)
        zeros_(self.input_proj.bias)

        # TCN块中的卷积层
        for block in self.tcn_blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        zeros_(layer.bias)

        # 静态特征处理层
        xavier_uniform_(self.static_net.weight)
        zeros_(self.static_net.bias)

        # 融合层
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]  # 最后一维是日期信息
        extra_x = use_obs[:, -self.extra_input_dims:]
        
        # 直接将原始时间序列数据重塑为TCN所需的格式
        # 从 [..., time*features+extra] 到 [..., time, features]
        batch_size = observations.size(0)
        time_dim, feature_dim = self.input_dims
        
        # 提取非静态特征部分并重塑
        x = use_obs[:, :-self.extra_input_dims].reshape(batch_size, time_dim, feature_dim)
        
        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]
        
        # 嵌入并处理静态特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = F.relu(self.static_net(static_input))
        
        # 重新排列时间序列数据为TCN所需的格式: [batch, features, time]
        # 从 [batch, time, features] 到 [batch, features, time]
        x = x.transpose(1, 2)
        
        # 应用输入投影
        x = self.input_proj(x)
        
        # 应用TCN块，带残差连接
        for tcn_block in self.tcn_blocks:
            residual = x
            x = tcn_block(x)
            x = x + residual  # 残差连接
        
        # 全局池化得到固定维度输出
        temporal_out = self.global_pool(x).squeeze(-1)
        
        # 融合静态和时序特征
        combined = torch.cat([temporal_out, static_out], dim=1)
        output = self.fusion(combined)
        
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

# 模型参数量: 220762
class TCNLob_3(nn.Module):
    def __init__(
            self,
            input_dims: tuple = (10, 4),  # 时间维度和特征维度
            extra_input_dims: int = 3,
            output_dim=2,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        
        # 核心模型参数 - 增加隐藏维度
        hidden_dim = 96  # 从48增加到96

        # 嵌入维度增加
        embedding_dim = 16  # 从8增加到16
        self.id_embedding = nn.Embedding(5, embedding_dim)
        
        # 静态特征处理 - 增加更深的网络
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)  # 增加一层
        )
        
        # 计算输入特征维度
        self.tcn_input_dim = input_dims[1]
        
        # 增加更多扩张率，增强时序建模能力
        dilations = [1, 2, 4, 8]  # 增加一个扩张率
        
        # TCN块
        self.tcn_blocks = nn.ModuleList()
        
        # 输入投影层
        self.input_proj = nn.Conv1d(self.tcn_input_dim, hidden_dim, kernel_size=1)
        
        # 多层TCN块，每个块增加更多层
        for dilation in dilations:
            block = nn.Sequential(
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),  # 增加一个卷积层
                nn.ReLU(),
            )
            self.tcn_blocks.append(block)
            
        # 全局池化层
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 融合网络 - 增加深度
        fusion_dim = hidden_dim + hidden_dim//2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim//2),  # 增加一层
            nn.ReLU(),
            nn.Linear(fusion_dim//2, output_dim)
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """初始化模型的权重和偏置"""
        # 嵌入层初始化
        xavier_uniform_(self.id_embedding.weight)

        # 输入投影层
        xavier_uniform_(self.input_proj.weight)
        zeros_(self.input_proj.bias)

        # TCN块中的卷积层
        for block in self.tcn_blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        zeros_(layer.bias)

        # 静态特征处理层
        for layer in self.static_net:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                zeros_(layer.bias)

        # 融合层
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]
        extra_x = use_obs[:, -self.extra_input_dims:]
        
        # 重塑时间序列数据
        batch_size = observations.size(0)
        time_dim, feature_dim = self.input_dims
        x = use_obs[:, :-self.extra_input_dims].reshape(batch_size, time_dim, feature_dim)
        
        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]
        
        # 嵌入并处理静态特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = self.static_net(static_input)
        
        # 重新排列时间序列数据: [batch, features, time]
        x = x.transpose(1, 2)
        
        # 应用输入投影
        x = self.input_proj(x)
        
        # 应用TCN块，带残差连接
        for tcn_block in self.tcn_blocks:
            residual = x
            x = tcn_block(x)
            x = x + residual
        
        # 全局池化
        temporal_out = self.global_pool(x).squeeze(-1)
        
        # 融合静态和时序特征
        combined = torch.cat([temporal_out, static_out], dim=1)
        output = self.fusion(combined)
        
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

# 模型参数量: 1203093
class TCNLob_4(nn.Module):
    def __init__(
            self,
            input_dims: tuple = (10, 4),  # 时间维度和特征维度
            extra_input_dims: int = 3,
            output_dim=2,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.extra_input_dims = extra_input_dims
        
        # 核心模型参数 - 大幅增加隐藏维度
        hidden_dim = 144  # 从48增加到144

        # 嵌入维度大幅增加
        embedding_dim = 32  # 从8增加到32
        self.id_embedding = nn.Embedding(5, embedding_dim)
        
        # 静态特征处理 - 更深的网络
        static_input_dim = embedding_dim + (self.extra_input_dims - 1)
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2)  # 三层深度
        )
        
        # 计算输入特征维度
        self.tcn_input_dim = input_dims[1]
        
        # 更复杂的扩张率序列
        dilations = [1, 2, 4, 8, 16]  # 增加更多扩张率
        
        # TCN块
        self.tcn_blocks = nn.ModuleList()
        
        # 输入投影层，使用更大的核
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.tcn_input_dim, hidden_dim//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=1)
        )
        
        # 多层TCN块，每个块包含更多层
        for i, dilation in enumerate(dilations):
            # 每个TCN块包含更多卷积层
            block = nn.Sequential(
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=1),  # 额外的因果卷积
                nn.ReLU(),
            )
            self.tcn_blocks.append(block)
        
        # 多尺度特征提取
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]  # 不同大小的卷积核
        ])
        
        # 全局池化层
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 融合网络 - 显著增加深度
        temporal_feat_dim = hidden_dim + 3 * (hidden_dim//4)  # 包含多尺度特征
        fusion_dim = temporal_feat_dim + hidden_dim//2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim//2),
            nn.ReLU(),
            nn.Linear(fusion_dim//2, fusion_dim//4),
            nn.ReLU(),
            nn.Linear(fusion_dim//4, output_dim)
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """初始化模型的权重和偏置"""
        # 嵌入层初始化
        xavier_uniform_(self.id_embedding.weight)

        # 输入投影层
        for layer in self.input_proj:
            if isinstance(layer, nn.Conv1d):
                xavier_uniform_(layer.weight)
                zeros_(layer.bias)

        # TCN块中的卷积层
        for block in self.tcn_blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        zeros_(layer.bias)

        # 多尺度卷积
        for conv in self.multi_scale_conv:
            xavier_uniform_(conv.weight)
            zeros_(conv.bias)

        # 静态特征处理层
        for layer in self.static_net:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                zeros_(layer.bias)

        # 融合层
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 分离时间序列和静态特征
        use_obs = observations[:, :-1]
        extra_x = use_obs[:, -self.extra_input_dims:]
        
        # 重塑时间序列数据
        batch_size = observations.size(0)
        time_dim, feature_dim = self.input_dims
        x = use_obs[:, :-self.extra_input_dims].reshape(batch_size, time_dim, feature_dim)
        
        # 处理静态特征
        cat_feat = extra_x[:, 0].long()
        num_feat = extra_x[:, 1:]
        
        # 嵌入并处理静态特征
        embedded = self.id_embedding(cat_feat)
        static_input = torch.cat([embedded, num_feat], dim=1)
        static_out = self.static_net(static_input)
        
        # 重新排列时间序列数据: [batch, features, time]
        x = x.transpose(1, 2)
        
        # 应用输入投影
        x = self.input_proj(x)
        
        # 应用TCN块，带残差连接
        for tcn_block in self.tcn_blocks:
            residual = x
            x = tcn_block(x)
            x = x + residual
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            multi_scale_features.append(self.global_pool(conv(x)).squeeze(-1))
        
        # 全局池化主特征
        temporal_main = self.global_pool(x).squeeze(-1)
        
        # 组合所有时序特征
        temporal_out = torch.cat([temporal_main] + multi_scale_features, dim=1)
        
        # 融合静态和时序特征
        combined = torch.cat([temporal_out, static_out], dim=1)
        output = self.fusion(combined)
        
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
# 10 * 4 的矩阵
data_config = {
    'his_len': 10,# 每个样本的 历史数据长度
    'need_cols': [item for i in range(1) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
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
        self.params_kwargs['learning_rate'] = 1e-4

        # 每个模型 3 个随机种子
        # 共 5 * 3 = 15 个实验
        model_clss = [TCNLob_0, TCNLob_1, TCNLob_2, TCNLob_3, TCNLob_4]
        seeds = range(2)
        train_args = list(product(model_clss, seeds))
        for _model_cls in model_clss:
            train_args.append((_model_cls, 2))

        self.model_cls, self.seed = train_args[self.idx]
        self.params_kwargs['seed'] = self.seed

        # 实例化 参数对象
        self.para = Params(
            **self.params_kwargs
        )

        # 准备数据集
        data_dict_folder = os.path.join(os.path.dirname(DATA_FOLDER), 'data_dict')
        self.train_dataset = LobTrajectoryDataset(data_folder= data_dict_folder, data_config = data_config)
        self.val_dataset = LobTrajectoryDataset(data_folder= data_dict_folder, data_config = data_config, data_type='val')
        self.test_dataset = LobTrajectoryDataset(data_folder= data_dict_folder, data_config = data_config, data_type='test')
    
    def get_title_suffix(self):
        """获取后缀"""
        return f'{self.model_cls.__name__}_seed{self.seed}'

    def get_model(self):
        return self.model_cls()
    
    def get_data(self, _type, data_sample_getter_func=None):
        if _type == 'train':
            return DataLoader(dataset=self.train_dataset, batch_size=self.para.batch_size, shuffle=True, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'val':
            return DataLoader(dataset=self.val_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        elif _type == 'test':
            return DataLoader(dataset=self.test_dataset, batch_size=self.para.batch_size, shuffle=False, num_workers=4//match_num_processes(), pin_memory=True)
        
if '__main__' == __name__:
    # model = MlpLob()
    # x = torch.randn(10, 44)
    # x[:, -4] = 0
    # print(model(x).shape)
    # print(f"模型参数量: {model_params_num(model)}")

    # model = TCNLob_3()
    # x = torch.randn(10, 44)
    # x[:, -4] = 0
    # print(model(x).shape)
    # print(f"模型参数量: {model_params_num(model)}")

    # model = TCNLob_4()
    # x = torch.randn(10, 44)
    # x[:, -4] = 0
    # print(model(x).shape)
    # print(f"模型参数量: {model_params_num(model)}")

    # input_folder = r'/kaggle/input'
    # # input_folder = r'C:\Users\lh\Desktop\temp\test_train_data'
    # data_folder_name = os.listdir(input_folder)[0]
    # data_folder = os.path.join(input_folder, data_folder_name)

    run(
        test, 
    )