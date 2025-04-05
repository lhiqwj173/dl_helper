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

