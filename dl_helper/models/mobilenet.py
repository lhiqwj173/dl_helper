import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(
            normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(
            normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


"""
Total params: 605,456
FLOPs: 15.24M
"""
class m_mobilenet(nn.Module):
    @classmethod
    def model_name(cls):
        return "mobilenet"

    def __init__(self, y_len, use_trade_data=True):
        super().__init__()
        self.use_trade_data = use_trade_data

        self.pk_stem = None
        self.trade_stem = None
        if use_trade_data:
            self.pk_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=4, out_channels=8,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=8, out_channels=16,
                        kernel_size=(1, 10)),
                LayerNorm(16, eps=1e-6, data_format="channels_first")
            )

            self.trade_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4,
                        kernel_size=(1, 3), stride=(1, 3)),
                nn.Conv2d(in_channels=4, out_channels=8,
                        kernel_size=(1, 2), stride=(1, 2)),
                LayerNorm(8, eps=1e-6, data_format="channels_first")
            )

        else:
            self.pk_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=6, out_channels=12,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=12, out_channels=24,
                        kernel_size=(1, 10)),
                LayerNorm(24, eps=1e-6, data_format="channels_first")
            ) 
 
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
 
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
 
        self.model = nn.Sequential(
            conv_dw( 24,  48, 1),# [1, 48, 70, 1]
            conv_dw( 48, 96, 2),# [1, 96, 35, 1]
            conv_dw(96, 192, 1),# [1, 192, 35, 1]
            
            conv_dw(192, 192, 1),# [1, 192, 18, 1]
            conv_dw(192, 192, 1),
            conv_dw(192, 192, 1),
            conv_dw(192, 192, 1),
            conv_dw(192, 192, 1),
            
            conv_dw(192, 384, 2),# [1, 384, 9, 1]
            conv_dw(384, 768, 1),# [1, 768, 9, 1]
            
            nn.AdaptiveAvgPool2d((1, 1)),# [1, 768, 1, 1]
        )
        self.fc = nn.Linear(768, y_len)
 
    def forward(self, combine_x):
        # 盘口数据
        x = combine_x[:, :, :, :40]  # torch.Size([1, 1, 70, 40])
        x = self.pk_stem(x)  # torch.Size([1, 16, 70, 1])

        # 成交数据
        if self.use_trade_data:
            x_2 = combine_x[:, :, :, 40:]  # torch.Size([1, 1, 70, 6])
            x_2 = self.trade_stem(x_2)  # torch.Size([1, 8, 70, 1])

            # 合并
            x = torch.cat((x, x_2), dim=1)# torch.Size([1, 24, 70, 1])

        x = self.model(x)
        x = x.view(-1, 768)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    
    from torchinfo import summary   
    from thop import profile
    from thop import clever_format

    device = 'cuda'

    model = m_mobilenet(y_len=2, use_trade_data=False)
    print(model.model_name())
    print(model)

    summary(model, (1, 1, 70, 40), device=device)

    model = model.to(device)
    input = torch.randn((1, 1, 70, 46)).to(device)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params])
    print(f"FLOPs: {flops} Params: {params}")