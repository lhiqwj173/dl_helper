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

class MLPEncoder(TorchModel, Encoder):
    def __init__(self, config):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.input_dims = config.input_dims
        self.hidden_sizes = config.hidden_sizes
        self.output_dims = config.output_dims
        input_size = np.prod(self.input_dims)

        # 使用层归一化来提高模型性能
        layers = []
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            ])
            input_size = hidden_size

        self.layers = nn.Sequential(*layers)

        self.output_layer = nn.Sequential(
            nn.Linear(input_size, self.output_dims[0]),
            nn.LayerNorm(self.output_dims[0]),
            nn.ReLU(),
        )

        self.error_count = 0

    def _forward(self, inputs: dict, **kwargs) -> dict:
        # for debug
        pickle.dump(inputs, open(f'inputs_{self.error_count}.pkl', 'wb'))

        x = inputs[Columns.OBS]
        x = self.layers(x)
        x = self.output_layer(x)
        return {ENCODER_OUT: x}

@dataclass
class MLPEncoderConfig(ModelConfig):
    """
    output_dims函数 返回编码器输出的维度，用于其他构造 head模型 的输入
    """
    # 默认: 10, 10
    input_dims: Union[int] = (10, 10)
    hidden_sizes: Tuple[int] = (128, 128)
    _output_dims: int = 24
    always_check_shapes: bool = True

    def build(self, framework: str = "torch"):
        if framework == "torch":
            return MLPEncoder(self)

        else:
            raise ValueError(f'only torch ModelConfig')

    @property
    def output_dims(self):
        """Read-only `output_dims` are inferred automatically from other settings."""
        return (int(self._output_dims),)# 注意返回的是维度，不是int

class MLPPPOCatalog(PPOCatalog):
    """
    - 重写 _determine_components_hook 生成配置
    """
    def _determine_components_hook(self):
        # 获取输入参数
        input_dims = tuple(self._model_config_dict["input_dims"])   
        hidden_sizes = self._model_config_dict["hidden_sizes"]
        output_dims = self._model_config_dict["output_dims"]
        # 生成配置
        self._encoder_config = MLPEncoderConfig(input_dims=input_dims, hidden_sizes=hidden_sizes, _output_dims=output_dims)

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
    net = MLPEncoderConfig().build()
    print(net)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {total_params}")# 模型参数量: 33048

    batch_size = 2
    input_tensor = torch.randn(batch_size, 10*10)  # PyTorch格式
    print(input_tensor.shape)

    # 前向传播
    output = net({Columns.OBS: input_tensor})# 与默认编码器一致输入/输出
    print(output)