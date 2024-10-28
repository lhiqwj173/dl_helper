import torch
import torch.nn as nn

class m_meta(nn.Module):
    @classmethod
    def model_name(cls):
        return "meta"

    def __init__(self, input_len, y_len):
        super(m_meta, self).__init__()
        self.fc = nn.Linear(input_len, y_len)

    def forward(self, x):
        return self.fc(x)

class m_meta_level2(nn.Module):
    @classmethod
    def model_name(cls):
        return "meta_level2"

    def __init__(self, input_len, y_len):
        super(m_meta_level2, self).__init__()

        middle_size = input_len // 2

        self.fc0 = nn.Linear(input_len, middle_size)
        self.relu = nn.ReLU()          # ReLU激活函数
        self.fc1 = nn.Linear(middle_size, y_len)

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        return x

