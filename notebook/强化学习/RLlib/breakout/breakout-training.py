from pprint import pprint
import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStack,
    TransformObservation
)
import ale_py
gym.register_envs(ale_py)
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

torch, nn = try_import_torch()

# 创建检查点保存目录
checkpoint_base_dir = "breakout_checkpoints"
os.makedirs(checkpoint_base_dir, exist_ok=True)

# 自定义 Atari 环境包装器
def make_atari_env(config):
    env = gym.make("ALE/Breakout-v5", **config)
    # 标准的 Atari 预处理
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False
    )
    # 堆叠4帧
    env = FrameStack(env, 4)
    return env

register_env("breakout_preprocessed", make_atari_env)

# 自定义CNN模型（与之前相同）
class BreakoutCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        
        conv_out_size = self._get_conv_out_size(obs_space.shape)
        
        self.fc_layers = nn.Sequential(
            SlimFC(conv_out_size, 512),
            nn.ReLU(),
            SlimFC(512, num_outputs)
        )
        
        self.value_branch = nn.Sequential(
            SlimFC(conv_out_size, 512),
            nn.ReLU(),
            SlimFC(512, 1)
        )
        
        self._features = None
        
    def _get_conv_out_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        conv_out = self.conv_layers(dummy_input)
        return int(np.prod(conv_out.shape[1:]))
        
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float() / 255.0  # 归一化
        x = x.permute(0, 3, 1, 2)  # 调整维度顺序
        x = self.conv_layers(x)
        self._features = x.reshape(x.shape[0], -1)
        logits = self.fc_layers(self._features)
        return logits, state
    
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.value_branch(self._features).squeeze(1)

# 注册自定义模型
ModelCatalog.register_custom_model("breakout_cnn", BreakoutCNN)

# 配置
config = (
    PPOConfig()
    .environment("breakout_preprocessed", env_config={
        "render_mode": None,
        "frameskip": 1,  # 因为我们在预处理中处理帧跳过
    })
    # .api_stack( enable_rl_module_and_learner=False )
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .framework("torch")
    .env_runners(num_env_runners=4)
    .resources(num_gpus=2)  # 设置使用2个GPU
    .training(
        model={
            "custom_model": "breakout_cnn",
            "custom_model_config": {},
        },
    )
    .evaluation(
        evaluation_interval=50,
        evaluation_duration=10,
    )
)

# 构建算法
algo = config.build()

# 训练循环
rounds = 100
for i in range(rounds):
    print(f"\nTraining iteration {i+1}/{rounds}")
    result = algo.train()
    
    metrics_to_print = {
        "episode_reward_mean": result["episode_reward_mean"],
        "episode_reward_max": result["episode_reward_max"],
        "episode_len_mean": result["episode_len_mean"],
        "episodes_this_iter": result["episodes_this_iter"],
    }
    pprint(metrics_to_print)
    
    if (i + 1) % 5 == 0:
        checkpoint_dir = algo.save_to_path(
            os.path.join(checkpoint_base_dir, f"checkpoint_{i+1}")
        )
        print(f"Checkpoint saved in directory {checkpoint_dir}")


# 保存最终模型
final_checkpoint = algo.save_to_path(
    os.path.join(checkpoint_base_dir, "final_checkpoint")
)
print(f"Final checkpoint saved in directory {final_checkpoint}")

# 清理
algo.stop()