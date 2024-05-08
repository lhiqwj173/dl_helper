import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# B：batch size
# M：多变量序列的变量数
# L：过去序列的长度
# T: 预测序列的长度
# N: 分Patch后Patch的个数
# D：每个变量的通道数
# P：kernel size of embedding layer
# S：stride of embedding layer

class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2048):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=D, 
            kernel_size=P, 
            stride=S
            )

    def forward(self, x):
        # x: [B, M, L]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
        x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
        x_pad = F.pad(
            x,
            pad=(0, self.P-self.S),
            mode='replicate'
            )  # [B*M, 1, L] -> [B*M, 1, L+P-S]
        
        x_emb = self.conv(x_pad)  # [B*M, 1, L+P-S] -> [B*M, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*M, D, N] -> [B, M, D, N]

        return x_emb  # x_emb: [B, M, D, N]


class ConvFFN(nn.Module):
    def __init__(self, M, D, r, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        groups_num = M if one else D
        self.pw_con1 = nn.Conv1d(
            in_channels=M*D, 
            out_channels=r*M*D, 
            kernel_size=1,
            groups=groups_num
            )
        self.pw_con2 = nn.Conv1d(
            in_channels=r*M*D, 
            out_channels=M*D, 
            kernel_size=1,
            groups=groups_num
            )

    def forward(self, x):
        # x: [B, M*D, N]
        x = self.pw_con2(F.gelu(self.pw_con1(x)))
        return x  # x: [B, M*D, N]


class ModernTCNBlock(nn.Module):
    def __init__(self, M, D, kernel_size, r):
        super(ModernTCNBlock, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.dw_conv = nn.Conv1d(
            in_channels=M*D, 
            out_channels=M*D, 
            kernel_size=kernel_size,
            groups=M*D,
            padding='same'
            )  
        self.bn = nn.BatchNorm1d(M*D)
        self.conv_ffn1 = ConvFFN(M, D, r, one=True)
        self.conv_ffn2 = ConvFFN(M, D, r, one=False)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        D = x_emb.shape[-2]
        x = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.dw_conv(x)                                   # [B, M*D, N] -> [B, M*D, N]
        x = self.bn(x)                                        # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_ffn1(x)                                 # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,2,1,3)                                # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]

        x = self.conv_ffn2(x)                                 # [B, D*M, N] -> [B, D*M, N]

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]

        out = x + x_emb

        return out  # out: [B, M, D, N]

"""
D=16 M=40
Total params: 250,387
FLOPs: 3.75M

D=24 M=40
Total params: 406,299
FLOPs: 6.15M 

D=32 M=40
Total params: 582,691
FLOPs: 8.90M

D=16 M=46 num_layers=2
Total params: 305,587
FLOPs: 4.62M

D=16 M=46 num_layers=4
Total params: 573,491
FLOPs: 9.10M 

D=16 M=46 num_layers=6
Total params: 841,395
FLOPs: 13.58M

D=24 M=46
Total params: 493,707
FLOPs: 7.53M

D=32 M=46
Total params: 705,379
FLOPs: 10.84M
"""
class m_moderntcn(nn.Module):
    """
    # y_len 分类类别数
    # M 多变量序列的变量数
    # L 过去序列的长度
    # T 预测序列的长度 用于回归
    # D 每个变量的通道数
    # P kernel size of embedding layer
    # S stride of embedding layer
    """
    def __init__(self, y_len, M=46, L=70, T=0, D=16, P=8, S=4, kernel_size=51, r=1, num_layers=2, dropout=0.1, use_trade_data=True):
        super().__init__()

        self.use_trade_data = use_trade_data

        # 深度分离卷积负责捕获时域关系
        self.num_layers = num_layers
        N = L // S
        self.embed_layer = Embedding(P, S, D)
        self.backbone = nn.ModuleList([ModernTCNBlock(M, D, kernel_size, r) for _ in range(num_layers)])

        self.act_class = F.gelu
        self.class_dropout = nn.Dropout(dropout)

        self.head = nn.Linear(D*N, T) if T > 0 else nn.Linear(M*D*N, y_len)

    def forward(self, x):
        # 不使用交易数据
        if not self.use_trade_data:
            x = x[:, :, :, :40]

        x = rearrange(x, 'b c l m -> b (c m) l')  # [B, C, L, M] -> [B, C*M, L] C=1

        # x: [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]

        # # 回归
        # # Flatten
        # z = rearrange(x_emb, 'b m d n -> b m (d n)')  # [B, M, D, N] -> [B, M, D*N]
        # pred = self.head(z)  # [B, M, D*N] -> [B, M, T]
        # return pred  # out: [B, M, T]

        # # 分类
        x = self.act_class(x_emb)
        x = self.class_dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.head(x) # [B, M*D*N] -> [B, y_len]

        return x

    @classmethod
    def model_name(cls):
        return "moderntcn"


if __name__ == "__main__":
    
    from torchinfo import summary   
    from thop import profile
    from thop import clever_format

    device = 'cuda'

    model = m_moderntcn(3, 46, 70, num_layers=6, use_trade_data=True)# 三分类, 40个变量, 过去70个时间步, 不使用交易数据
    print(model.model_name())
    print(model)

    summary(model, (1, 1, 70, 46), device=device)

    model = model.to(device)
    input = torch.randn((1, 1, 70, 46)).to(device)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params])
    print(f"FLOPs: {flops} Params: {params}")