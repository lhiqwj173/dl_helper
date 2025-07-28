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
目标: tcn 替换的 deeplob 模型，作为基准
结论: 

"""
class StaticFeatureProcessor(nn.Module):
    """
    静态特征处理模块
    用于处理包含类别特征和数值特征的静态输入
    """
    def __init__(
        self, 
        num_categories: int = 5,           # 类别特征的数量（嵌入词汇表大小）
        embedding_dim: int = 16,           # 嵌入维度
        num_features: int = 2,             # 数值特征的数量
        output_dim: int = 32,              # 输出维度
        static_hidden: tuple = (64, 32),   # 静态特征融合网络的隐藏层维度
        use_regularization: bool = False,   # 是否使用正则化组件
        dropout_rate: float = 0.1,         # Dropout概率（仅当use_regularization=True时生效）
    ):
        """
        初始化静态特征处理器
        
        Args:
            num_categories: 类别特征的数量，用于嵌入层的词汇表大小
            embedding_dim: 嵌入向量的维度
            num_features: 数值特征的数量
            output_dim: 输出特征的维度
            static_hidden: 静态特征融合网络的隐藏层维度
            use_regularization: 是否使用正则化组件（统一控制dropout和batch_norm）
            dropout_rate: Dropout概率，仅当use_regularization=True时生效
        """
        super().__init__()
        
        # 参数验证
        if num_categories <= 0:
            raise ValueError("num_categories必须大于0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim必须大于0")
        if num_features < 0:
            raise ValueError("num_features不能为负数")
        if output_dim <= 0:
            raise ValueError("output_dim必须大于0")
        if not (0 <= dropout_rate < 1):
            raise ValueError("dropout_rate必须在[0, 1)范围内")
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.output_dim = output_dim
        self.use_regularization = use_regularization
        # 只有在启用正则化时才应用这些参数
        self.dropout_rate = dropout_rate if use_regularization else 0.0
        self.batch_norm = True if use_regularization else False
        
        # 类别特征嵌入层
        self.id_embedding = nn.Embedding(num_categories, embedding_dim)
        
        # 数值特征预处理层（仅在启用正则化时使用BatchNorm）
        if num_features > 0:
            self.num_feature_norm = nn.BatchNorm1d(num_features) if self.batch_norm else nn.Identity()
        
        # 静态特征融合网络
        # 输入维度 = 嵌入维度 + 数值特征维度
        static_input_dim = embedding_dim + num_features
        
        # 构建静态特征处理网络
        layers = []
        in_dim = static_input_dim
        
        for i, out_dim in enumerate(static_hidden):
            layers.append(nn.Linear(in_dim, out_dim))
            
            # 添加批标准化（仅在启用正则化时）
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            
            layers.append(nn.ReLU())
            
            # 添加Dropout（仅在启用正则化时）
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            in_dim = out_dim
        
        # 最后的输出层（不加激活函数和dropout）
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.static_net = nn.Sequential(*layers)
        
        # 初始化权重
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        初始化模型权重
        - 嵌入层使用Xavier均匀初始化
        - 线性层使用Xavier均匀初始化，偏置初始化为0
        """
        # 嵌入层权重初始化
        xavier_uniform_(self.id_embedding.weight)
        
        # 遍历静态网络中的所有层进行初始化
        for module in self.static_net.modules():
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                # 批标准化层初始化
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
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
        self.use_regularization = use_regularization
        
        # 第一个因果卷积层
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # 移除右侧填充以保持因果性
        self.relu1 = nn.ReLU()
        if self.use_regularization:
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.BatchNorm1d(n_outputs)
        
        # 第二个因果卷积层
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        if self.use_regularization:
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.BatchNorm1d(n_outputs)

        # 构建网络序列
        if self.use_regularization:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.norm1, self.relu1, self.dropout1,
                                   self.conv2, self.chomp2, self.norm2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                   self.conv2, self.chomp2, self.relu2)
        
        # 残差连接的降采样层
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

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
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout, use_regularization=use_regularization)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class deeplob(nn.Module):
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
        conv1_layers = [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
        ]
        if self.use_regularization:
            conv1_layers.append(nn.BatchNorm2d(32))
        conv1_layers.extend([
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
        ])
        if self.use_regularization:
            conv1_layers.append(nn.BatchNorm2d(32))
        conv1_layers.extend([
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
        ])
        if self.use_regularization:
            conv1_layers.append(nn.BatchNorm2d(32))
        self.conv1 = nn.Sequential(*conv1_layers)
        
        # 卷积块2：使用Tanh激活函数
        # 用于提取每个档位 买卖盘之间的关系
        conv2_layers = [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
        ]
        if self.use_regularization:
            conv2_layers.append(nn.BatchNorm2d(32))
        conv2_layers.extend([
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
        ])
        if self.use_regularization:
            conv2_layers.append(nn.BatchNorm2d(32))
        conv2_layers.extend([
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
        ])
        if self.use_regularization:
            conv2_layers.append(nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(*conv2_layers)
        
        # 卷积块3：使用LeakyReLU激活函数
        # 用于处理多个档位之间的关系
        if input_dims[1] > 4:
            # 如果特征维度大于4，意味着有多个档位的数据
            conv3_layers = [
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
                nn.LeakyReLU(negative_slope=0.01),
            ]
            if self.use_regularization:
                conv3_layers.append(nn.BatchNorm2d(32))
            conv3_layers.extend([
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
                nn.LeakyReLU(negative_slope=0.01),
            ])
            if self.use_regularization:
                conv3_layers.append(nn.BatchNorm2d(32))
            conv3_layers.extend([
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
                nn.LeakyReLU(negative_slope=0.01),
            ])
            if self.use_regularization:
                conv3_layers.append(nn.BatchNorm2d(32))
            self.conv3 = nn.Sequential(*conv3_layers)
        else:
            self.conv3 = None
        
        # Inception模块1：3x1卷积分支
        inp1_layers = [
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        ]
        if self.use_regularization:
            inp1_layers.append(nn.BatchNorm2d(64))
        inp1_layers.extend([
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        ])
        if self.use_regularization:
            inp1_layers.append(nn.BatchNorm2d(64))
        self.inp1 = nn.Sequential(*inp1_layers)
        
        # Inception模块2：5x1卷积分支
        inp2_layers = [
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        ]
        if self.use_regularization:
            inp2_layers.append(nn.BatchNorm2d(64))
        inp2_layers.extend([
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        ])
        if self.use_regularization:
            inp2_layers.append(nn.BatchNorm2d(64))
        self.inp2 = nn.Sequential(*inp2_layers)
        
        # Inception模块3：最大池化分支
        inp3_layers = [
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        ]
        if self.use_regularization:
            inp3_layers.append(nn.BatchNorm2d(64))
        self.inp3 = nn.Sequential(*inp3_layers)
        
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
            nn.ReLU(),
            nn.Linear(fusion_dim, self.y_len)
        )
        
        # 如果使用正则化，添加dropout层
        if self.use_regularization:
            self.dropout_final = nn.Dropout(0.3)
    
        # 应用统一的权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """统一的权重初始化函数"""
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            xavier_uniform_(module.weight)
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        # TCN的特殊初始化可以单独处理
        if isinstance(module, TemporalBlock):
            normal_(module.net[0].weight, 0, 0.01) # conv1
            normal_(module.net[4 if self.use_regularization else 2].weight, 0, 0.01) # conv2
            if module.downsample is not None:
                normal_(module.downsample.weight, 0, 0.01)

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

# 简单的数据结构
# 50 * 4 的矩阵
data_config = {
    'his_len': 50,# 每个样本的 历史数据长度
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

        seeds = range(5)
        self.model_cls = deeplob
        self.seed = seeds[self.idx]
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
    # 测试模型
    model = deeplob()
    x = torch.randn(10, 50*4+4)
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