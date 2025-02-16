import functools
from typing import Tuple, List, Union
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.encoder import TorchModel, Encoder
from dataclasses import dataclass
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.columns import Columns
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dl_helper.models.binctabl import BiN, BL_layer, TABL_layer
from dl_helper.rl.rl_env.lob_env import LOB_trade_env

import pickle

class BinCtablEncoder(TorchModel, Encoder):
    def __init__(self, config):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        d1, d2, d3, d4 = config.ds
        t1, t2, t3, t4 = config.ts

        self.input_dims = config.input_dims
        self.extra_input_dims = config.extra_input_dims
        self.split_index = np.prod(self.input_dims)

        self.BiN = BiN(d1, t1)
        self.BL = BL_layer(d2, d1, t1, t2)
        self.BL2 = BL_layer(d3, d2, t2, t3)
        self.TABL = TABL_layer(d4, d3, t3, t4)
        self.dropout = nn.Dropout(0.1)

        self.error_count = 0

    def _forward(self, inputs: dict, **kwargs) -> dict:
        # for debug
        pickle.dump(inputs, open(f'inputs_{self.error_count}.pkl', 'wb'))

        extra_x = inputs[Columns.OBS][:,self.split_index:]
        x = inputs[Columns.OBS][:,:self.split_index].reshape(-1, *self.input_dims)
        x = self.BiN(x)

        with torch.no_grad():
            self.max_norm_(self.BL.W1)
            self.max_norm_(self.BL.W2)
            self.max_norm_(self.BL2.W1)
            self.max_norm_(self.BL2.W2)
            self.max_norm_(self.TABL.W1)
            self.max_norm_(self.TABL.W)
            self.max_norm_(self.TABL.W2)

        x = self.BL(x)
        x = self.dropout(x)

        x = self.BL2(x)
        x = self.dropout(x)

        x = self.TABL(x)

        x = torch.squeeze(x,dim=2)# 保留batch维度
        x = torch.cat([x, extra_x], dim=1)

        # 数值检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            self.error_count += 1

        return {ENCODER_OUT: x}

    def max_norm_(self, p):
        norm = torch.linalg.matrix_norm(p.data)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        p.data.mul_(torch.where(norm > 10.0, desired / (1e-8 + norm), torch.tensor(1., device=p.data.device)) )
  

@dataclass
class BinCtablEncoderConfig(ModelConfig):
    """
    output_dims函数 返回编码器输出的维度，用于其他构造 head模型 的输入
    """
    # 默认: his_len = 100, cols = 40
    input_dims: Union[List[int], Tuple[int]] = (40, 100)
    extra_input_dims: int = 4
    always_check_shapes: bool = True
    ds: tuple = (40, 60, 120, 3)
    ts: tuple = (100, 40, 12, 1)

    def build(self, framework: str = "torch"):
        if framework == "torch":
            return BinCtablEncoder(self)

        else:
            raise ValueError(f'only torch ModelConfig')

    @property
    def output_dims(self):
        """Read-only `output_dims` are inferred automatically from other settings."""
        return (int(self.ds[-1]+self.extra_input_dims),)# 注意返回的是维度，不是int


class lob_PPOCatalog(PPOCatalog):
    """
    - 重写 _determine_components_hook 生成配置
    """
    def _determine_components_hook(self):
        # 获取输入参数 可设置参数 ds / ts
        ds = tuple(self._model_config_dict["ds"])
        ts = tuple(self._model_config_dict["ts"])
        input_dims = tuple(self._model_config_dict["input_dims"])   
        extra_input_dims = self._model_config_dict["extra_input_dims"]
        # 生成配置
        self._encoder_config = BinCtablEncoderConfig(input_dims=input_dims, extra_input_dims=extra_input_dims, ds=ds, ts=ts)

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

def _get_env(env):
    while not isinstance(env, LOB_trade_env):
        env = env.unwrapped
    return env

class LobCallbacks(DefaultCallbacks):
    def on_evaluate_start(self, *args, **kwargs):
        print('on_evaluate_start')
        # 获取 eval_env_runner
        algo = kwargs['algorithm'] 
        if algo.eval_env_runner_group is None:
            eval_env_runner = algo.env_runner_group.local_env_runner
        else:
            eval_env_runner = algo.eval_env_runner
        # 切换环境到 val模式
        for env in eval_env_runner.env.unwrapped.envs:
            _env = _get_env(env)
            _env.val()

    def on_evaluate_end(self, *args, **kwargs):
        print('on_evaluate_end')
        # 获取 eval_env_runner
        algo = kwargs['algorithm'] 
        if algo.eval_env_runner_group is None:
            eval_env_runner = algo.env_runner_group.local_env_runner
            # 只有本地 eval_env_runner 需要切换回 train模式
            for env in eval_env_runner.env.unwrapped.envs:
                _env = _get_env(env)
                _env.train()

if __name__ == "__main__":
    net = BinCtablEncoderConfig(input_dims=(20, 10), extra_input_dims=4, ds=(20, 40, 40, 3), ts=(10, 6, 3, 1)).build()
    print(net)

    batch_size = 2
    input_tensor = torch.randn(batch_size, 20*10+4)  # PyTorch格式
    print(input_tensor.shape)

    # 前向传播
    output = net({Columns.OBS: input_tensor})# 与默认编码器一致输入/输出
    print(output)
