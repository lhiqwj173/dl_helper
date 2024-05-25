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
    def __init__(self, use_trade_data, use_pk_data, stem_alpha=1, normal='ln'):
        assert use_trade_data or use_pk_data

        super().__init__()

        self.out_channel = int(24*stem_alpha)

        self.pk_stem = None
        self.trade_stem = None
        if use_trade_data and use_pk_data:
            # 同时使用盘口数据和成交数据
            # 24 = 16 + 8
            # 1, 4, 8, 16
            pk_out_channel = int(self.out_channel*(2/3))
            pk_cs = [pk_out_channel//4, pk_out_channel//2, pk_out_channel]
            self.pk_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=pk_cs[0],
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=pk_cs[0], out_channels=pk_cs[1],
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=pk_cs[1], out_channels=pk_cs[2],
                        kernel_size=(1, 10)),
                LayerNorm(pk_cs[2], eps=1e-6, data_format="channels_first") if normal=='ln' else nn.BatchNorm2d(pk_cs[2]),
            )

            # 1, 4, 8
            trade_out_channel = self.out_channel - pk_out_channel
            trade_cs = [trade_out_channel//2, trade_out_channel]
            self.trade_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=trade_cs[0],
                        kernel_size=(1, 3), stride=(1, 3)),
                nn.Conv2d(in_channels=trade_cs[0], out_channels=trade_cs[1],
                        kernel_size=(1, 2), stride=(1, 2)),
                LayerNorm(trade_cs[1], eps=1e-6, data_format="channels_first")  if normal=='ln' else nn.BatchNorm2d(trade_cs[1]),
            )

        elif use_pk_data:
            # 仅使用盘口数据
            # 1, 6, 12, 24
            cs = [self.out_channel//4, self.out_channel//2, self.out_channel]
            self.pk_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=cs[0],
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=cs[0], out_channels=cs[1],
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=cs[1], out_channels=cs[2],
                        kernel_size=(1, 10)),
                LayerNorm(cs[2], eps=1e-6, data_format="channels_first") if normal=='ln' else nn.BatchNorm2d(cs[2]),
            ) 
        elif use_trade_data:
            # 仅使用交易数据
            # 1, 12, 24
            cs = [self.out_channel//2, self.out_channel]
            self.trade_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=cs[0],
                        kernel_size=(1, 3), stride=(1, 3)),
                nn.Conv2d(in_channels=cs[0], out_channels=cs[1],
                        kernel_size=(1, 2), stride=(1, 2)),
                LayerNorm(cs[1], eps=1e-6, data_format="channels_first") if normal=='ln' else nn.BatchNorm2d(cs[1]),
            )
 
    def forward(self, combine_x):
        # 盘口数据
        x_1 = None
        if not None is self.pk_stem:
            x_1 = combine_x[:, :, :, :40]  # torch.Size([1, 1, 70, 40])
            x_1 = self.pk_stem(x_1)  # torch.Size([1, 16, 70, 1])

        # 成交数据
        x_2 = None
        if not None is self.trade_stem:
            x_2 = combine_x[:, :, :, 40:]  # torch.Size([1, 1, 70, 6])
            x_2 = self.trade_stem(x_2)  # torch.Size([1, 8, 70, 1])

        if x_1 is None:
            return x_2
        elif x_2 is None:
            return x_1
        else:
            # 合并
            return torch.cat((x_1, x_2), dim=1)# torch.Size([1, 24, 70, 1])

class stem_same_channel(nn.Module):
    """
    全部使用 24*stem_alpha 相同的通道数
    normal 类:
        'ln': LayerNorm
        'bn': BatchNorm2d
    """
    def __init__(self, use_trade_data, use_pk_data, stem_alpha=1, normal='ln'):
        assert use_trade_data or use_pk_data

        super().__init__()

        self.out_channel = int(24*stem_alpha)

        self.pk_stem = None
        self.trade_stem = None
        if use_trade_data and use_pk_data:
            # 同时使用盘口数据和成交数据
            pk_out_channel = int(self.out_channel*(2/3))
            self.pk_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=pk_out_channel,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=pk_out_channel, out_channels=pk_out_channel,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=pk_out_channel, out_channels=pk_out_channel,
                        kernel_size=(1, 10)),
                LayerNorm(pk_out_channel, eps=1e-6, data_format="channels_first") if normal=='ln' else nn.BatchNorm2d(pk_out_channel),
            )

            # 1, 4, 8
            trade_out_channel = self.out_channel - pk_out_channel
            self.trade_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=trade_out_channel,
                        kernel_size=(1, 3), stride=(1, 3)),
                nn.Conv2d(in_channels=trade_out_channel, out_channels=trade_out_channel,
                        kernel_size=(1, 2), stride=(1, 2)),
                LayerNorm(trade_out_channel, eps=1e-6, data_format="channels_first")  if normal=='ln' else nn.BatchNorm2d(trade_out_channel),
            )

        elif use_pk_data:
            # 仅使用盘口数据
            self.pk_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=self.out_channel,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel,
                        kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel,
                        kernel_size=(1, 10)),
                LayerNorm(self.out_channel, eps=1e-6, data_format="channels_first") if normal=='ln' else nn.BatchNorm2d(self.out_channel),
            ) 
        elif use_trade_data:
            # 仅使用交易数据
            self.trade_stem = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=self.out_channel,
                        kernel_size=(1, 3), stride=(1, 3)),
                nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel,
                        kernel_size=(1, 2), stride=(1, 2)),
                LayerNorm(self.out_channel, eps=1e-6, data_format="channels_first") if normal=='ln' else nn.BatchNorm2d(self.out_channel),
            )

    def forward(self, combine_x):
        # 盘口数据
        x_1 = None
        if not None is self.pk_stem:
            x_1 = combine_x[:, :, :, :40]  # torch.Size([1, 1, 70, 40])
            x_1 = self.pk_stem(x_1)  # torch.Size([1, 16, 70, 1])

        # 成交数据
        x_2 = None
        if not None is self.trade_stem:
            x_2 = combine_x[:, :, :, 40:]  # torch.Size([1, 1, 70, 6])
            x_2 = self.trade_stem(x_2)  # torch.Size([1, 8, 70, 1])

        if x_1 is None:
            return x_2
        elif x_2 is None:
            return x_1
        else:
            # 合并
            return torch.cat((x_1, x_2), dim=1)# torch.Size([1, 24, 70, 1])

