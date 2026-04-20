""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AFFModule(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=32, r=4):
        super(AFFModule, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 【8， 32， 288， 288】
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        # wei = self.sigmoid(xlg)
        wei = self.sigmoid(xa)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class AFFModule_ADD(nn.Module):
    def __init__(self, channels=32, r=4):
        super(AFFModule_ADD, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        wei = self.sigmoid(xa)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class AFFModule_G(nn.Module):
    '''
    多特征融合 AFF  ONLY GLOBAL
    '''

    def __init__(self, channels=32, r=4):
        super(AFFModule_G, self).__init__()
        inter_channels = int(channels // r)

        # 【8， 32， 288， 288】
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xg = self.global_att(xa)
        wei = self.sigmoid(xg)
        return 2 * x * wei + 2 * residual * (1 - wei)

class AFFModule_L(nn.Module):
    '''
    多特征融合 AFF only local
    '''

    def __init__(self, channels=32, r=4):
        super(AFFModule_L, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        wei = self.sigmoid(xl)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class ModifiedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        修改后的深度可分离卷积模块的初始化

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小，默认为3
        :param stride: 卷积步长，默认为1
        :param padding: 卷积填充大小，默认为1
        """
        super(ModifiedDepthwiseSeparableConv, self).__init__()
        # 深度卷积部分
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=in_channels)
        # 原始逐点卷积部分
        self.pointwise_conv_original = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 用于处理通道作差后的逐点卷积部分
        self.pointwise_conv_diff = nn.Conv2d(in_channels-1, out_channels, kernel_size=1)
        # 注意力特征融合模块
        self.aff_module = AFFModule(out_channels)
        # self.aff_module = AFFModule_G(out_channels)
        # self.aff_module = AFFModule_L(out_channels)

    def forward(self, x):
        """
        前向传播函数

        :param x: 输入张量，形状为 (batch_size, in_channels, height, width)
        :return: 输出张量，形状为 (batch_size, out_channels, height, width)
        """
        depthwise_out = self.depthwise_conv(x)

        # 对深度卷积结果按通道作差
        channel_diff = depthwise_out[:, 1:, :, :] - depthwise_out[:, :-1, :, :]

        # 通过新的逐点卷积处理通道作差后的结果
        pointwise_diff_out = self.pointwise_conv_diff(channel_diff)

        # 通过原始逐点卷积处理深度卷积结果
        pointwise_original_out = self.pointwise_conv_original(depthwise_out)

        # 融合两个逐点卷积的结果ADD
        # final_out = pointwise_original_out + pointwise_diff_out
        # 使用注意力特征融合（AFF）模块融合两个逐点卷积的结果
        final_out = self.aff_module(pointwise_original_out, pointwise_diff_out)
        return final_out

class DownModifiedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        修改后的深度可分离卷积模块的初始化

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小，默认为3
        :param stride: 卷积步长，默认为1
        :param padding: 卷积填充大小，默认为1
        """
        super(DownModifiedDepthwiseSeparableConv, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ModifiedDepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        return self.maxpool_conv(x)