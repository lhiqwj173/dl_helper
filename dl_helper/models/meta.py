import torch
import torch.nn as nn

class m_meta(nn.Module):
    @classmethod
    def model_name(cls):
        return "meta"

    def __init__(self, input_len, y_len):
        super(m_meta, self).__init__()
        self.linear = nn.Linear(input_len, y_len)

    def forward(self, x):
        return self.fc(x)


        