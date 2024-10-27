import math
import torch
import torch.nn as nn


"""
Total params: 50,671,651
Trainable params: 50,671,651
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 71.82

模型太过于复杂
"""

class m_cnn(nn.Module):
    @classmethod
    def model_name(cls):
        return "cnn"

    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, (3,1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, (1,1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, (3,1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, (1,1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv2d(128, 256, (3,1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, (1,1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(negative_slope=0.01),
            
        )
        
        #fully connected layers
        self.linear1 = nn.Linear(8192, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, self.y_len)

        #activation
        self.activation = nn.LeakyReLU(negative_slope=0.01)
      
    def forward(self, x):
      #print(x.shape)
      x = self.conv1(x)
      #print(x.shape)
      x = self.conv2(x)
      #print(x.shape)
      x = self.conv3(x)
      #print(x.shape)
      x = self.inp1(x)
       
      x = torch.flatten(x, 1)
      x = self.activation(self.linear1(x))
      #print(x.shape)
      x = self.activation(self.linear2(x))
      #print(x.shape)
      out = self.linear3(x)
      
      return out

if __name__ == "__main__":
    
    from torchinfo import summary   
    from thop import profile
    from thop import clever_format

    device = 'cuda'

    model = m_cnn(3)
    summary(model, (1, 1, 50, 40), device=device)

    # model = model.to(device)
    # input = torch.randn((1, 1, 50, 40)).to(device)
    # output = model(input)

    # # 导出模型为ONNX格式
    # onnx_path = "deeplob.onnx"
    # torch.onnx.export(model, input, onnx_path)

    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params])
    # print(f"FLOPs: {flops} Params: {params}")