import torch
import torch.nn as nn

"""
随机模型
"""
class m_dummy(nn.Module):
    @classmethod
    def model_name(cls):
        return "dummy"

    def __init__(self, y_len):
        self.y_len = y_len

    def forward(self, x):
        return torch.randn(x.shape[0], self.y_len)