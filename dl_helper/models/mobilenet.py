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

class stem(nn.Module):
    """
    normal 类:
        'ln': LayerNorm
        'bn': BatchNorm2d
    """
    def __init__(self, use_trade_data, normal='ln'):
        super().__init__()

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
                LayerNorm(16, eps=1e-6, data_format="channels_first") if normal=='ln' else nn.BatchNorm2d(16),
            )

            self.trade_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4,
                        kernel_size=(1, 3), stride=(1, 3)),
                nn.Conv2d(in_channels=4, out_channels=8,
                        kernel_size=(1, 2), stride=(1, 2)),
                LayerNorm(8, eps=1e-6, data_format="channels_first")  if normal=='ln' else nn.BatchNorm2d(16),
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
 
    def forward(self, combine_x):
        # 盘口数据
        x = combine_x[:, :, :, :40]  # torch.Size([1, 1, 70, 40])
        x = self.pk_stem(x)  # torch.Size([1, 16, 70, 1])

        # 成交数据
        if not None is self.trade_stem:
            x_2 = combine_x[:, :, :, 40:]  # torch.Size([1, 1, 70, 6])
            x_2 = self.trade_stem(x_2)  # torch.Size([1, 8, 70, 1])

            # 合并
            x = torch.cat((x, x_2), dim=1)# torch.Size([1, 24, 70, 1])

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

        # 合并特征
        self.stem = stem(use_trade_data)

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
 
    def forward(self, x):
        x = self.stem(x)
        x = self.model(x)
        x = x.view(-1, 768)
        x = self.fc(x)
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []                          # 定义层列表
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


"""
alpha=0.2
Total params: 117,889
FLOPs: 1.34M

alpha=0.4

alpha=0.6
Total params: 765,761
FLOPs: 5.89M
"""
class m_mobilenet_v2(nn.Module):
    def __init__(self, y_len, alpha=1.0, round_nearest=8, use_trade_data=True):
        super().__init__()

        # 合并特征
        self.stem = stem(use_trade_data, normal='bn')

        block = InvertedResidual
        input_channel = 24          # 将卷积核个数调整到最接近8的整数倍数
        last_channel = _make_divisible(640 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, y_len)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)         # 初始化均值为0
                nn.init.zeros_(m.bias)          # 初始化方差为1
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @classmethod
    def model_name(cls):
        return "mobilenet_v2"


if __name__ == "__main__":
    
    from torchinfo import summary   
    from thop import profile
    from thop import clever_format

    device = 'cuda'

    # model = m_mobilenet(y_len=3, use_trade_data=False)
    model = m_mobilenet_v2(y_len=3, alpha=0.4, use_trade_data=False)
    print(model.model_name())
    print(model)

    summary(model, (1, 1, 70, 40), device=device)

    model = model.to(device)
    input = torch.randn((1, 1, 70, 46)).to(device)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params])
    print(f"FLOPs: {flops} Params: {params}")