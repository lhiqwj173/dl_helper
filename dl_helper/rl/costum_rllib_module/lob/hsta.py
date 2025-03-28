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

class HSTAEncoder(TorchModel, Encoder):
    def __init__(self, config):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.input_dims = config.input_dims
        self.feature_per_step = self.input_dims[1]
        self.extra_input_dims = config.extra_input_dims
        self.output_dims = config._output_dims
        self.split_index = np.prod(self.input_dims)

        # 层次化时空注意力
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.feature_per_step,
            num_heads=4,
            dropout=0.1
        )
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.feature_per_step,
                nhead=4,
                dim_feedforward=64,
                dropout=0.1
            ),
            num_layers=3
        )
        
        # 静态特征处理
        self.static_net = nn.Sequential(
            nn.Linear(self.extra_input_dims, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_per_step * 2 + 16, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.output_dims)
        )

        self.error_count = 0

    def _forward(self, inputs: dict, **kwargs) -> dict:
        # for debug
        pickle.dump(inputs, open(f'inputs_{self.error_count}.pkl', 'wb'))

        extra_x = inputs[Columns.OBS][:,self.split_index:]
        x = inputs[Columns.OBS][:,:self.split_index].reshape(-1, *self.input_dims)

        # 空间注意力（特征维度）
        spatial_in = x.permute(1, 0, 2)  # [10,B,20]
        spatial_attn_out, _ = self.spatial_attn(spatial_in, spatial_in, spatial_in)
        spatial_attn_out = spatial_attn_out.permute(1, 0, 2)  # [B,10,20]

        # 时间注意力
        temporal_in = x.permute(1, 0, 2)  # [10,B,20]
        temporal_attn_out = self.temporal_attn(temporal_in)
        temporal_attn_out = temporal_attn_out.permute(1, 0, 2)  # [B,10,20]

        # 时空特征融合
        fused_time = torch.cat([
            spatial_attn_out.mean(dim=1),  # [B,20]
            temporal_attn_out.mean(dim=1)   # [B,20]
        ], dim=1)  # [B,40]

        # 静态特征处理
        static_out = self.static_net(extra_x)  # [B,16]

        # 融合层
        fused_out = torch.cat([fused_time, static_out], dim=1)  # [B,56]
        fused_out = self.fusion(fused_out)  # [B,self.output_dims]

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            self.error_count += 1

        return {ENCODER_OUT: fused_out}

@dataclass
class HSTAEncoderConfig(ModelConfig):
    """
    output_dims函数 返回编码器输出的维度，用于其他构造 head模型 的输入
    """
    # 默认: his_len = 100, cols = 40
    input_dims: Union[List[int], Tuple[int]] = (100, 40)
    extra_input_dims: int = 4
    _output_dims: int = 6
    always_check_shapes: bool = True

    def build(self, framework: str = "torch"):
        if framework == "torch":
            return HSTAEncoder(self)

        else:
            raise ValueError(f'only torch ModelConfig')

    @property
    def output_dims(self):
        """Read-only `output_dims` are inferred automatically from other settings."""
        return (int(self._output_dims),)# 注意返回的是维度，不是int

class HSTAPPOCatalog(PPOCatalog):
    """
    - 重写 _determine_components_hook 生成配置
    """
    def _determine_components_hook(self):
        # 获取输入参数 可设置参数 ds / ts
        input_dims = tuple(self._model_config_dict["input_dims"])   
        extra_input_dims = self._model_config_dict["extra_input_dims"]
        output_dims = self._model_config_dict["output_dims"]
        # 生成配置
        self._encoder_config = HSTAEncoderConfig(input_dims=input_dims, extra_input_dims=extra_input_dims, _output_dims=output_dims)

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
    net = HSTAEncoderConfig(input_dims=(10, 20), extra_input_dims=4, _output_dims=6).build()
    print(net)

    # 模型参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {total_params}")# 模型参数量: 17090

    batch_size = 2
    input_tensor = torch.randn(batch_size, 20*10+4)  # PyTorch格式
    print(input_tensor.shape)

    # 前向传播
    output = net({Columns.OBS: input_tensor})# 与默认编码器一致输入/输出
    print(output)