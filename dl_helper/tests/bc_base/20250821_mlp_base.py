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
    1. 考察容量更深的模型(简单mlp)对性能的影响
        MLPStaticModelSmall     模型参数量: 7971
        MLPStaticModel          模型参数量: 21675
        MLPStaticModelLarge     模型参数量: 65683

        

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

# 模型参数量: 21675
class MLPStaticModel(nn.Module):
    """
    一个结合了MLP处理时间序列特征和另一个MLP处理静态特征的混合模型。

    该模型包含两个并行的处理流：
    1. 时间序列流：使用一个多层感知机（MLP）来处理扁平化的时间序列数据。
    2. 静态特征流：使用一个专门的MLP（StaticFeatureProcessor），包括嵌入层，
       来处理不随时间变化的静态特征。

    最后，两个流的输出被拼接在一起，通过一个融合网络（Fusion Network）来产生最终的预测。
    """
    def __init__(
        self,
        # 时间序列参数
        num_ts_features: int,
        time_steps: int,
        ts_mlp_hidden_dims: tuple = (128, 64),
        ts_mlp_output_dim: int = 32,
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
            ts_mlp_hidden_dims (tuple): 处理时间序列的MLP的隐藏层维度。
            ts_mlp_output_dim (int): 处理时间序列的MLP的输出维度。
            num_static_features (int): 静态特征的总数（类别+数值）。
            static_num_categories (int): 静态数据中类别特征的类别总数。
            static_embedding_dim (int): 类别特征的嵌入维度。
            static_output_dim (int): 静态特征处理器处理后的输出维度。
            static_hidden_dims (tuple): 静态特征处理器中隐藏层的维度。
            output_dim (int): 模型的最终输出维度。
            use_regularization (bool): 是否启用Dropout正则化。
            dropout_rate (float): MLP和融合网络中使用的dropout比率。
        """
        super().__init__()
        
        # --- 保存维度信息 ---
        self.num_ts_features = num_ts_features
        self.time_steps = time_steps
        self.num_static_features = num_static_features
        self.output_dim = output_dim
        self.use_regularization = use_regularization
        
        # --- 1. 静态特征处理流 (保持不变) ---
        self.static_processor = StaticFeatureProcessor(
            num_categories=static_num_categories,
            embedding_dim=static_embedding_dim,
            num_features=num_static_features - 1, # 减去1个类别特征
            output_dim=static_output_dim,
            static_hidden=static_hidden_dims,
            use_regularization=use_regularization,
            dropout_rate=dropout_rate,
        )

        # --- 2. 时间序列特征处理流 (修改为MLP) ---
        ts_input_dim = self.num_ts_features * self.time_steps
        ts_layers = []
        in_dim = ts_input_dim
        
        for h_dim in ts_mlp_hidden_dims:
            ts_layers.append(nn.Linear(in_dim, h_dim))
            ts_layers.append(nn.BatchNorm1d(h_dim))
            ts_layers.append(nn.ReLU())
            if self.use_regularization:
                ts_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        
        # MLP的输出层
        ts_layers.append(nn.Linear(in_dim, ts_mlp_output_dim))
        self.ts_mlp = nn.Sequential(*ts_layers)

        # --- 3. 融合网络 ---
        fusion_input_dim = ts_mlp_output_dim + static_output_dim
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
        if hasattr(self, 'fusion_net') and module == self.fusion_net[-1]:
            init.constant_(module.weight, 0)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
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
        # --- 1. 输入特征分割 ---
        ts_data_len = self.time_steps * self.num_ts_features
        
        # 提取时间序列数据 (已经是扁平化的)
        x_ts = observations[:, :ts_data_len]
        # 提取静态数据 (假设最后一个特征是日期等非输入特征)
        x_static = observations[:, ts_data_len:-1]

        # --- 2. 时间序列流处理 (MLP) ---
        # 直接将扁平化的时间序列数据送入MLP
        h_ts = self.ts_mlp(x_ts)  # [B, ts_mlp_output_dim]
        
        # --- 3. 静态特征流处理 ---
        h_static = self.static_processor(x_static) # [B, static_output_dim]
        
        # --- 4. 特征融合 ---
        # 将两个流的输出拼接起来
        fusion_input = torch.cat([h_ts, h_static], dim=1)
        
        # 通过融合网络得到最终输出
        output = self.fusion_net(fusion_input)
        
        return output

# --- 版本 2: 更小的模型 ---
# 型参数量: 7971
class MLPStaticModelSmall(MLPStaticModel):
    """
    MLPStaticModel的小型版本，参数量约为标准版的50%。
    通过减少MLP隐藏层和嵌入层的维度来实现。
    """
    def __init__(self, **kwargs):
        # 覆盖默认参数以减小模型尺寸
        super().__init__(
            ts_mlp_hidden_dims=(64, 32),
            ts_mlp_output_dim=32,
            static_embedding_dim=8,
            static_output_dim=16,
            static_hidden_dims=(32, 16),
            **kwargs  # 允许用户覆盖这些小型化设置
        )

# --- 版本 3: 更大的模型 ---
# 模型参数量: 65683
class MLPStaticModelLarge(MLPStaticModel):
    """
    MLPStaticModel的大型版本，参数量约为标准版的150%。
    通过增加MLP隐藏层和嵌入层的维度来实现。
    """
    def __init__(self, **kwargs):
        # 覆盖默认参数以增大模型尺寸
        super().__init__(
            ts_mlp_hidden_dims=(256, 128, 64),
            ts_mlp_output_dim=32,
            static_embedding_dim=24,
            static_output_dim=48,
            static_hidden_dims=(96, 48),
            **kwargs  # 允许用户覆盖这些大型化设置
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
    title_base = '20250821'
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
            for model_cls in [MLPStaticModelSmall, MLPStaticModel, MLPStaticModelLarge]:
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
    for model_cls in [MLPStaticModelSmall, MLPStaticModel, MLPStaticModelLarge]:
        model = model_cls(
            num_ts_features=len(ext_features) + len(base_features),
            time_steps=his_len,
        )
        model(x)
        print(f"{model_cls.__name__} 模型参数量: {model_params_num(model)}")

    # ################################
    # # 验证初始化损失 == log(C)
    # ################################
    if test_init_loss:
        from tqdm import tqdm
        init_losses = []
        for i in tqdm(range(10)):
            model = MLPStaticModel(
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
        )
