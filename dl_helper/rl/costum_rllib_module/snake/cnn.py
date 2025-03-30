import pickle
import functools
from typing import Tuple, List, Union
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.encoder import TorchModel, Encoder
from dataclasses import dataclass
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.columns import Columns

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CNNEncoder(TorchModel, Encoder):
    def __init__(self, config):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.input_dims = config.input_dims
        self.hidden_sizes = config.hidden_sizes
        self.output_dims = config.output_dims
        self.need_layer_norm = config.need_layer_norm
        if config.active_func == "relu":
            active_func_class = nn.ReLU
        elif config.active_func == "tanh":
            active_func_class = nn.Tanh
        else:
            raise ValueError(f"不支持的激活函数: {config.active_func}")

        # 假设形状为(C, H, W)
        self.in_channels, self.height, self.width = self.input_dims

        convs = []
        in_channels = self.in_channels
        for i, hidden_size in enumerate(self.hidden_sizes):
            convs.extend([
                # 使用更大的卷积核，步幅减小为1，保留更多空间信息
                nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_size) if self.need_layer_norm else nn.Identity(),
                active_func_class(),
            ])
            in_channels = hidden_size

        self.convs = nn.Sequential(*convs)
            
        # 计算卷积后的特征图大小
        with torch.no_grad():
            # 假设输入是 (batch_size, channels, height, width)
            dummy_input = torch.zeros(1, *self.input_dims)
            dummy_output = self.convs(dummy_input)
            self.flatten_size = int(np.prod(dummy_output.shape[1:]))  # 计算展平后的特征维度
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.flatten_size, self.output_dims[0]),
            nn.LayerNorm(self.output_dims[0]) if self.need_layer_norm else nn.Identity(),
            active_func_class(),
        )

        self.error_count = 0

    def _forward(self, inputs: dict, **kwargs) -> dict:
        # for debug
        pickle.dump(inputs, open(f'inputs_{self.error_count}.pkl', 'wb'))

        x = inputs[Columns.OBS] # 输入的obs是(B, 1, 10, 10)
        
        # 通过卷积层
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 通过全连接层
        x = self.output_layer(x)

        return {ENCODER_OUT: x}

@dataclass
class CNNEncoderConfig(ModelConfig):
    """
    output_dims函数 返回编码器输出的维度，用于其他构造 head模型 的输入
    """
    # 默认: 1, 10, 10
    input_dims: Union[int] = (1, 10, 10)
    hidden_sizes: Tuple[int] = (32, 64)
    need_layer_norm: bool = True
    _output_dims: int = 24
    always_check_shapes: bool = True
    active_func: str = "relu"

    def build(self, framework: str = "torch"):
        if framework == "torch":
            return CNNEncoder(self)

        else:
            raise ValueError(f'only torch ModelConfig')

    @property
    def output_dims(self):
        """Read-only `output_dims` are inferred automatically from other settings."""
        return (int(self._output_dims),)# 注意返回的是维度，不是int

class CNNPPOCatalog(PPOCatalog):
    """
    - 重写 _determine_components_hook 生成配置
    """
    def _determine_components_hook(self):
        # 获取输入参数
        input_dims = tuple(self._model_config_dict["input_dims"])   
        hidden_sizes = self._model_config_dict["hidden_sizes"]
        output_dims = self._model_config_dict["output_dims"]
        need_layer_norm = self._model_config_dict["need_layer_norm"]
        active_func = self._model_config_dict["active_func"]
        # 生成配置
        self._encoder_config = CNNEncoderConfig(input_dims=input_dims, hidden_sizes=hidden_sizes, need_layer_norm=need_layer_norm, _output_dims=output_dims, active_func=active_func)

        # 不变
        # Create a function that can be called when framework is known to retrieve the
        # class type for action distributions
        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space, action_space=self.action_space
        )

        # 不变
        # The dimensions of the latent vector that is output by the encoder and fed
        # to the heads.
        self.latent_dims = self._encoder_config.output_dims

if __name__ == "__main__":
    
    net = CNNEncoderConfig().build()
    print(net)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {total_params}")# 模型参数量: 25176

    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 10, 10)  # PyTorch格式
    print(input_tensor.shape)

    # 前向传播
    output = net({Columns.OBS: input_tensor})# 与默认编码器一致输入/输出
    print(output)