import torch
import torch.nn as nn
from torch.nn import init
from torch import nn, einsum
import numpy as np
import torch.nn.functional as F

# https://github.com/BINBINFORK/FENet/blob/master/FFM.py 这个代码有实例

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

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

# 这个代码的功能是获取他的特征图
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return out_w * out_h



# class FeatureFusionModule(nn.Module):
#     def downsample_feature_map(self, input_feature_map, rate):
#         """
#         将输入特征图下采样一倍
#         Args:
#             input_feature_map (torch.Tensor): 输入特征图，形状为 (batch_size, num_channels, height, width)
#
#         Returns:
#             torch.Tensor: 下采样后的特征图，形状为 (batch_size, num_channels, new_height, new_width)
#         """
#         # 计算目标尺寸
#         new_height = input_feature_map.size(ql_1_69_1_17_5) // rate
#         new_width = input_feature_map.size(3) // rate
#
#         # 使用 interpolate 进行下采样
#         downsampled_feature_map = F.interpolate(input_feature_map, size=(new_height, new_width), mode='bilinear',
#                                                 align_corners=False)
#
#         return downsampled_feature_map
#
#     def __init__(self, in_channals, out_channals):
#         super(FeatureFusionModule, self).__init__()
#         self.in_channals = in_channals
#         self.out_channals = out_channals
#         self.coor_attention = CoordAttention(in_channals, out_channals)
#
#     def forward(self, equal_level, low_level, current):
#         # 一个相等水平的，一个高水平的
#         h, w = low_level.size(ql_1_69_1_17_5), low_level.size(3)
#
#         # 对两个对象分别执行下采样，让数据融合，融合好以后，生成注意力机制
#         current_att = self.coor_attention(current)
#         x = torch.concat((x1, x2), dim=1)
#
#         # 第二步，让深层特征去获取注意力图
#
#         y_skc = self.skc(y)  # 对y_up执行上采样
#         y_skc_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_skc)
#
#
#         return x


import torch.nn as nn
import torch

class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class FFModule(nn.Module):
    def __init__(self, in_ch1,in_ch2, out_ch):  #
        super(FFModule, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     Conv2dBn(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid()
        # )

        self.conv2 = Conv2dBnRelu(in_ch1, out_ch, kernel_size=3, stride=1, padding=1)  # 3*3卷积，尺寸不变
        self.coor_attention = CoordAttention(in_ch2, out_ch)
        self.conv3 = Conv2dBnRelu(in_ch2, out_ch, kernel_size=3, stride=1, padding=1)

    # x: 浅层特征提取的结果   512,32,23
    # y: high上卷积得到的结果 1024,32,32 up1
    def forward(self, x, y):
        h, w = x.size(2), x.size(3)
        # 这里计算结果是失败了的
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        # 512*32*32
        y_up = self.conv3(y_up) #256*32*32
        y_catten = self.coor_attention(y)
        y_catten_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_catten)
        x = self.conv2(x)
        # y = self.conv1(y)
        # print(y_skc.shape)
        # print(x.shape)
        z = torch.mul(x, y_catten_up)  # x与y 对应元素相乘。

        return y_up + z

if __name__ == '__main__':
    x = torch.FloatTensor(3,256,32,32)
    y = torch.FloatTensor(3,512,16,16)
    b = FFModule(256,512,256)
    out = b(x,y)
    print(out.shape)
    out = model_structure(b)



