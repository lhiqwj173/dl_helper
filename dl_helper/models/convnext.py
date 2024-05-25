import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem import stem

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

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

class Residual(nn.Module):  # @save
    def __init__(self, num_channels, dp_rate):
        super().__init__()
        mid_channels = num_channels * 4
        # depthwise conv
        self.conv0 = nn.Conv2d(num_channels, num_channels,kernel_size=(7, 1), padding=(3, 0), stride=(1, 1), groups=num_channels)
        self.norm = LayerNorm(num_channels, eps=1e-6, data_format="channels_last")
        self.conv1 = nn.Linear(num_channels, mid_channels)
        self.conv2 = nn.Linear(mid_channels, num_channels)

        self.gamma = nn.Parameter(1e-6 * torch.ones((num_channels,)),requires_grad=True)
        self.grn = GRN(mid_channels)
        self.cbam = CBAMLayer(num_channels)

        self.drop_path = DropPath(dp_rate)

    def forward(self, X):
        Y = self.conv0(X)
        Y = Y.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        Y = self.norm(Y)
        Y = F.relu(self.conv1(Y))
        Y = self.conv2(Y)
        Y = Y.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        # 注意力
        Y = self.cbam(Y)

        Y = X + self.drop_path(Y)
        return Y

        # Y = self.conv0(X)
        # Y = Y.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        # Y = self.norm(Y)
        # Y = F.relu(self.conv1(Y))
        # Y = self.grn(Y)
        # Y = self.conv2(Y)
        # Y = Y.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        # Y = X + self.drop_path(Y)
        # return Y

        # Y = self.conv0(X)
        # Y = Y.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        # Y = self.norm(Y)
        # Y = F.relu(self.conv1(Y))
        # Y = self.conv2(Y)
        # Y = self.gamma * Y
        # Y = Y.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        # Y = X + self.drop_path(Y)
        # return Y

def make_block(layer_n, in_channels, out_channels, dp_rate):
    layers = []
    if out_channels > in_channels:
        # 降维
        layers.append(LayerNorm(in_channels, eps=1e-6, data_format="channels_first"))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(2, 1), stride=(2, 1)))

    for i in range(layer_n):
        layers.append(Residual(out_channels, dp_rate))
    return nn.Sequential(*layers)

class ConvNeXt_block(nn.Module):
    def __init__(self, y_len, in_channel, layer_list, channel_list, drop_path_rate=0.0):
        super().__init__()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, len(channel_list))]

        self.stages = nn.ModuleList()
        for layer_n, out_channel, dp_rate in zip(layer_list,channel_list, dp_rates):
            self.stages.append(make_block(layer_n, in_channel, out_channel, dp_rate))
            in_channel = out_channel

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.f = nn.Flatten()
        self.ln = nn.LayerNorm(channel_list[-1], eps=1e-6)
        self.l = nn.Linear(channel_list[-1], y_len)

    def forward(self, X):
        for stage in self.stages:
            X = stage(X)

        X = self.global_avg_pool(X)
        return self.l(self.ln(X.mean([-2, -1])))

"""
1, 1, 70, 46
width_ratio=0.3 
Total params: 247,965
FLOPs: 4.71M

width_ratio=0.7 
Total params: 769,317
"""
class m_convnext(nn.Module):
    def __init__(self, y_len, width_ratio=0.3, layer_ratio=1, use_trade_data=True, use_pk_data=True):        
        super().__init__()

        self.y_len = y_len

        self.stem = stem(use_trade_data, use_pk_data)

        channel_list = [24]
        for i in range(3):
            channel_list.append(int((1*width_ratio + 1) * channel_list[-1]))
        print(f'channel_list: {channel_list}')

        self.block = ConvNeXt_block(
            self.y_len, 24, 
            [int(i*layer_ratio) for i in [3,3,9,3]],
            channel_list,
            0.3
        )

    def forward(self, combine_x):
        x = self.stem(combine_x)
        return self.block(x)

    @classmethod
    def model_name(cls):
        return "convnext"

if __name__ == "__main__":
    
    from torchinfo import summary   
    from thop import profile
    from thop import clever_format

    device = 'cuda'

    model = m_convnext(y_len=2, width_ratio=0.7, use_trade_data=False, use_pk_data=True)
    print(model.model_name())
    print(model)

    summary(model, (1, 1, 70, 46), device=device)

    model = model.to(device)
    input = torch.randn((1, 1, 70, 46)).to(device)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params])
    print(f"FLOPs: {flops} Params: {params}")