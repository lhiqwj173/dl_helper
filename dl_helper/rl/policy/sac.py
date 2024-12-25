import torch
import torch.nn.functional as F

from dl_helper.rl.base import BaseAgent, OffPolicyAgent

class sac_network(torch.nn.Module):
    def __init__(self, obs_shape, features_extractor_class, features_extractor_kwargs, features_dim, net_arch):
        """
        features_dim: features_extractor_class输出维度  + 4(symbol_id + 持仓 + 未实现收益率 + 距离收盘秒数)
        net_arch : dict(pi=[action_dim], vf=[action_dim])
        """
        super().__init__()
        self.obs_shape = obs_shape

        # 特征提取器
        self.features_extractor = features_extractor_class(
            **features_extractor_kwargs
        )

        # 策略网络/q网络
        name_dict = {'pi': 'policy', 'vf': 'q'}
        for key in ['pi', 'vf']:
            _net = net_arch[key]
            _head_length = len(_net)
            _name = f'{name_dict[key]}_head'
            if _head_length == 1:
                setattr(self, _name, torch.nn.Linear(features_dim, _net[0]))
            else:
                _head = torch.nn.ModuleList([torch.nn.Linear(features_dim, _net[0])])
                for i in range(1, _head_length):
                    # 添加激活函数  
                    _head.append(torch.nn.ReLU())
                    _head.append(torch.nn.Linear(_net[i - 1], _net[i]))
                setattr(self, _name, _head)

        # 价值网络
        _net = net_arch['vf']
        _head_length = len(_net)
        if _head_length == 1:
            setattr(self, 'value_head', torch.nn.Linear(features_dim, 1))
        else:
            _net = [features_dim] + _net
            # 将最后一个维度改为1
            _net[-1] = 1
            _head = torch.nn.ModuleList()
            for i in range(_head_length):
                _head.append(torch.nn.Linear(_net[i], _net[i + 1]))
                if i < _head_length - 1:   
                    _head.append(torch.nn.ReLU())
            setattr(self, 'value_head', _head)
        

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
        if _head_length > 1:
            for i in range(_head_length - 1):
                x = F.relu(self.policy_head[i](x))
            x = self.policy_head[-1](x)
        else:
            x = self.policy_head(x)

        if self.fc_v is not None:
            v = self.fc_v(feature)
            x = v + x - x.mean(1).view(-1, 1)  # Q值由V值和A值计算得到

        return x


class SAC(OffPolicyAgent):

    def __init__(
        self,

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
            基类参数
                buffer_size: 经验回放池大小
                action_dim: 动作空间维度 
                features_dim: 特征维度
                features_extractor_class: 特征提取器类,必须提供
                features_extractor_kwargs=None: 特征提取器参数,可选
                net_arch=None: 网络架构参数,默认为一层mlp, 输入/输出维度为features_dim, action_dim
        """
        super().__init__(buffer_size, action_dim, features_dim, features_extractor_class, features_extractor_kwargs, net_arch)

    ############################################################
    # 需要重写的函数
    #     _build_model: 构建模型
    #     _take_action(self, state): 根据状态选择动作
    #     _update(self, states, actions, rewards, next_states, dones, data_type, n_step_rewards=None, n_step_next_states=None, n_step_dones=None): 更新模型
    #     get_model_to_sync: 获取需要同步的模型
    #     sync_update_net_params_in_agent: 同步更新模型参数
    #     get_params_to_send: 获取需要上传的参数
    ############################################################


if __name__ == '__main__':

    from dl_helper.models.binctabl import m_bin_ctabl_fix_shape

    t1, t2, t3, t4 = [100, 30, 10, 1]
    d1, d2, d3, d4 = [130, 60, 30, 7]
    features_extractor_kwargs = {'d2': d2, 'd1': d1, 't1': t1, 't2': t2, 'd3': d3, 't3': t3, 'd4': d4, 't4': t4}

    net = sac_network(
        obs_shape=(100, 130), 
        features_extractor_class=m_bin_ctabl_fix_shape, 
        features_extractor_kwargs=features_extractor_kwargs, 
        features_dim=d4 + 4, 
        net_arch={'pi': [6, 3], 'vf': [6, 3]}
    )
    print(net)

