import torch
import torch.nn.functional as F

from dl_helper.rl.base import BaseAgent, OffPolicyAgent

class sac_network(torch.nn.Module):
    def __init__(self, obs_shape, features_extractor_class, features_extractor_kwargs, features_dim, net_arch):
        """
        features_dim: features_extractor_class输出维度  + 2(持仓 + 为实现收益率)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.features_extractor = features_extractor_class(
            **features_extractor_kwargs
        )

        # 剩余部分
        net_arch = net_arch['pi']

        self.fc_a_length = len(net_arch)
        if self.fc_a_length == 1:
            self.fc_a = torch.nn.Linear(features_dim, net_arch[0])
        else:
            self.fc_a = torch.nn.ModuleList([torch.nn.Linear(features_dim, net_arch[0])])
            for i in range(1, self.fc_a_length):
                self.fc_a.append(torch.nn.Linear(net_arch[i - 1], net_arch[i]))

        self.fc_v = None
        if dqn_type in [DUELING_DQN, DD_DQN]:
            self.fc_v = torch.nn.Linear(features_dim, 1)

    def forward(self, x):
        """
        先将x分成两个tensor
        lob: x[:, :-3]
        acc: x[:, -3:]
        """
        # -> batchsize, 100， 130
        lob_data = x[:, :-3].view(-1, self.obs_shape[0], self.obs_shape[1])
        acc_data = x[:, -3:]

        feature = self.features_extractor(lob_data)# -> batchsize, 3
        # concat acc
        feature = torch.cat([feature, acc_data], dim=1)

        x = feature
        if self.fc_a_length > 1:
            for i in range(self.fc_a_length - 1):
                x = F.relu(self.fc_a[i](x))
            x = self.fc_a[-1](x)
        else:
            x = self.fc_a(x)

        if self.fc_v is not None:
            v = self.fc_v(feature)
            x = v + x - x.mean(1).view(-1, 1)  # Q值由V值和A值计算得到

        return x


class SAC(OffPolicyAgent):

    def __init__(
        self,
        obs_shape,
        learning_rate,
        gamma,
        epsilon,
        target_update,

        # 基类参数
        buffer_size,
        action_dim,
        features_dim,
        features_extractor_class,
        features_extractor_kwargs=None,
        net_arch=None,
    ):
        """
        SAC
        
        Args:
            obs_shape: 观测空间维度
            learning_rate: 学习率
            gamma: TD误差折扣因子
            epsilon: epsilon-greedy策略参数
            target_update: 目标网络更新间隔

            基类参数
                buffer_size: 经验回放池大小
                action_dim: 动作空间维度 
                features_dim: 特征维度
                features_extractor_class: 特征提取器类,必须提供
                features_extractor_kwargs=None: 特征提取器参数,可选
                net_arch=None: 网络架构参数,默认为一层mlp, 输入/输出维度为features_dim, action_dim
                  