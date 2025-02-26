
import torch
import torch.nn as nn
import torchsummary
from torch.nn import init
from torch import nn, einsum
import numpy as np
import torch.nn.functional as F




def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
# from torchstat import stat

class CBAM_Module(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM_Module, self).__init__()

        # channel attention  H,W=1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True
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
class FenZhiBlock(nn.Module):
    def __init__(self, in_channals,out_channals,dilation_rate):
        super(FenZhiBlock,self).__init__()
        # The first module, dilated convolution, does not change the size of the feature map, and the features of the feature map remain unchanged
        self.conv1 = nn.Conv2d(in_channels=in_channals, out_channels=out_channals, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(out_channals)
        self.relu1 = nn.ReLU()

        # The second one is regular convolution, which does not change the size
        self.conv2 = nn.Conv2d(in_channels=out_channals, out_channels=out_channals, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channals)
        self.relu2 = nn.ReLU()

        # CBAM attention module
        self.cbam = CBAM_Module(channel=out_channals)

    def forward(self, x):
        # Apply first convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply second convolutional block
        x2 = self.conv2(x)
        x2 = self.bn2(x)
        x2 = self.relu2(x)

        out = x+x2

        # Apply CBAM attention

        out = self.cbam(out)

        return out




class Denseasppblock(nn.Module):
    def __init__(self, in_channels,out_channels,inter_channals=32,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(Denseasppblock, self).__init__()
        inter_channals = 32
        self.aspp_3 = FenZhiBlock(in_channels, inter_channals, 1)
        self.aspp_6 = FenZhiBlock(in_channels + inter_channals * 1,inter_channals, 2)
        self.aspp_12 = FenZhiBlock(in_channels + inter_channals * 2, inter_channals, 3)
        self.aspp_18 = FenZhiBlock(in_channels + inter_channals * 3, inter_channals, 4)
        self.aspp_24 = FenZhiBlock(in_channels + inter_channals * 4, inter_channals, 5)
        self.in_channels = in_channels
        self.conv_change = nn.Conv2d(in_channels=in_channels+inter_channals*5,out_channels=out_channels, kernel_size=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        # 448 - 256
        x = self.conv_change(x)

        x = self.pool(x)

        return x



if __name__ == '__main__':

    in_chan = 256
    out_chan = 512

    net = Denseasppblock(in_chan, out_chan).cuda()



    torchsummary.summary(net, input_size=(256, 128, 128), device="cuda")
    print(1)

