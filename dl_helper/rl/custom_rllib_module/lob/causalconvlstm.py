import pickle
import functools
from typing import Tuple, List, Union
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.encoder import TorchModel, Encoder
from dataclasses import dataclass
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.columns import Columns
from ray.rllib.examples.rl_modules.classes.intrinsic_curiosity_model_rlm import IntrinsicCuriosityModel
from ray.rllib.models.utils import get_activation_fn

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CausalConvLSTMEncoder(TorchModel, Encoder):
    def __init__(self, config):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.input_dims = config.input_dims
        self.feature_per_step = self.input_dims[1]
        self.extra_input_dims = config.extra_input_dims
        self.output_dims = config._output_dims
        self.split_index = np.prod(self.input_dims)

        # 因果卷积网络
        self.causal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.feature_per_step,
                out_channels=64,
                kernel_size=3,
                padding=2,  # 保持时间维度不变
                dilation=2
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=4,
                dilation=4
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True,
            bidirectional=False
        )
        
        # 静态特征处理
        self.static_net = nn.Sequential(
            nn.Linear(self.extra_input_dims, self.extra_input_dims * 4),
            nn.LayerNorm(self.extra_input_dims * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + self.extra_input_dims * 4, 32),
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

        # 因果卷积处理
        conv_in = x.permute(0, 2, 1)  # [B,20,10]
        conv_out = self.causal_conv(conv_in)     # [B,32,10]
        conv_out = conv_out.permute(0, 2, 1)     # [B,10,32]

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(conv_out)
        temporal_feat = lstm_out[:, -1, :]  # 取最后一个时间步 [B,64]

        # 静态特征处理
        static_out = self.static_net(extra_x)  # [B,16]

        # 融合层
        fused_out = torch.cat([temporal_feat, static_out], dim=1)  # [B,80]
        fused_out = self.fusion(fused_out)  # [B,self.output_dims]

        # 数值检查
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            self.error_count += 1

        return {ENCODER_OUT: fused_out}

@dataclass
class CausalConvLSTMEncoderConfig(ModelConfig):
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
            return CausalConvLSTMEncoder(self)

        else:
            raise ValueError(f'only torch ModelConfig')

    @property
    def output_dims(self):
        """Read-only `output_dims` are inferred automatically from other settings."""
        return (int(self._output_dims),)# 注意返回的是维度，不是int

class CausalConvLSTMPPOCatalog(PPOCatalog):
    """
    - 重写 _determine_components_hook 生成配置
    """
    def _determine_components_hook(self):
        # 获取输入参数
        input_dims = tuple(self._model_config_dict["input_dims"])   
        extra_input_dims = self._model_config_dict["extra_input_dims"]
        output_dims = self._model_config_dict["output_dims"]
        # 生成配置
        self._encoder_config = CausalConvLSTMEncoderConfig(input_dims=input_dims, extra_input_dims=extra_input_dims, _output_dims=output_dims)

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

class CausalConvLSTMIntrinsicCuriosityModel(IntrinsicCuriosityModel):
    def setup(self):
        # Get the ICM achitecture settings from the `model_config` attribute:
        cfg = self.model_config

        # 使用与主模型一致的特征提取
        # 获取输入参数
        input_dims = tuple(cfg["input_dims"])   
        extra_input_dims = cfg["extra_input_dims"]
        output_dims = cfg["output_dims"]
        # 生成配置
        self._feature_config = CausalConvLSTMEncoderConfig(input_dims=input_dims, extra_input_dims=extra_input_dims, _output_dims=output_dims)
        self._feature_net = self._feature_config.build()

        # Build the inverse model (predicting the action between two observations).
        layers = []
        dense_layers = cfg.get("inverse_net_hiddens", (256,))
        # `in_size` is 2x the feature dim.
        in_size = output_dims * 2
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("inverse_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["inverse_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (action space).
        layers.append(nn.Linear(in_size, self.action_space.n))
        self._inverse_net = nn.Sequential(*layers)

        # Build the forward model (predicting the next observation from current one and
        # action).
        layers = []
        dense_layers = cfg.get("forward_net_hiddens", (256,))
        # `in_size` is the feature dim + action space (one-hot).
        in_size = output_dims + self.action_space.n
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("forward_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["forward_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (feature dimension).
        layers.append(nn.Linear(in_size, output_dims))
        self._forward_net = nn.Sequential(*layers)



if __name__ == "__main__":
    net = CausalConvLSTMEncoderConfig(input_dims=(10, 20), extra_input_dims=4, _output_dims=6).build()
    print(net)

    # 模型参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {total_params}")# 模型参数量: 38326

    batch_size = 2
    input_tensor = torch.randn(batch_size, 20*10+4)  # PyTorch格式
    print(input_tensor.shape)

    # 前向传播
    output = net({Columns.OBS: input_tensor})# 与默认编码器一致输入/输出
    print(output)