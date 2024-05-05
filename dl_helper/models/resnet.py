import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # logger.debug(i, (kernel_size-1) * dilation_size, dilation_size)
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class inception_resnet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        inception_in = channels
        inception_out = inception_in // 4
        # inception moduels  
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=inception_in, out_channels=inception_out, kernel_size=(1,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_out),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=inception_out, out_channels=inception_in, kernel_size=(3,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_in),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=inception_in, out_channels=inception_out, kernel_size=(1,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_out),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=inception_out, out_channels=inception_in, kernel_size=(5,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_in),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=inception_in, out_channels=inception_in, kernel_size=(1,1), padding='same', bias=False),
            nn.BatchNorm2d(inception_in),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.conv1_1 = nn.Conv2d(inception_in, inception_in*3,kernel_size=1,padding='same')

    def forward(self, x):
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x_inp = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = self.conv1_1(x) 
        return F.relu(x_inp+x)
