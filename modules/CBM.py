import torch
import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CBM','CGFE']



class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv_BN(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale






class CGFE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'],
                 no_spatial=False, num_feature_levels=2):
        super(CGFE, self).__init__()
        # 初始化参数
        self.num_feat = num_feature_levels  # 存储特征层的数量
        self.ChannelGate = ChannelGate(in_channels, reduction_ratio, pool_types)  # 初始化通道注意力模块
        self.no_spatial = no_spatial  # 控制是否使用空间注意力
        if not no_spatial:
            self.SpatialGate = SpatialGate()  # 初始化空间注意力模块

        # 用于多尺度特征生成的卷积层
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, CBM_feature, x):
        # CBM_feature：来自 CMM 模块的密度图
        # x：输入的另一特征图
        feats = []  # 用于存储每层增强后的特征
        bs, c, h, w = x.shape  # 获取输入特征图的形状

        # 计算空间注意力
        if not self.no_spatial:
            c2 = self.SpatialGate(x)  # 使用当前输入特征图x计算空间注意力权重
            x = x * c2  # 将特征图与空间注意力权重相乘，增强空间信息

        # 计算通道注意力
        c1 = self.ChannelGate(x)  # 计算通道注意力权重
        x = x * c1  # 将特征图与通道注意力权重相乘，增强通道信息

        # 对特征图进行下采样生成多尺度特征
        feat1 = self.conv1(x)  # 第一尺度
        feat2 = self.conv2(feat1)  # 第二尺度
        feat3 = self.conv3(feat2)  # 第三尺度

        feats.extend([feat1, feat2, feat3])  # 将多尺度特征图添加到列表中

        # 返回所有尺度的特征图
        x_out = torch.cat(feats, 1)  # 将所有特征图在通道维度上拼接
        return x_out  # 返回拼接后的输出







class CBM(nn.Module):
    def __init__(self, in_channels=16, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBM, self).__init__()
        self.CBM_cfg = [32, 32, 32, 16, 16, 16]
        self.in_channels = 32
        self.conv1 = nn.Conv2d(in_channels,self.in_channels, kernel_size=1)  # 输入通道设为128，适应P2的输入
        self.CBM = make_layers(self.CBM_cfg, in_channels=self.in_channels, d_rate=2)  # 使用膨胀卷积层生成密度特征



        self.ChannelGate = ChannelGate(in_channels, reduction_ratio, pool_types)  # 初始化通道注意力模块
        self.SpatialGate = SpatialGate()  # 初始化空间注意力模块

    def forward(self, features):
        # print('进入的features:',features.shape)
        # 输入特征图：features，形状为 (batch_size, channels, height, width)
        x = self.conv1(features)  #
        CBM_feature = self.CBM(x)  # 生成密度特征


        # 计算空间注意力
        c2 = self.SpatialGate(features)  # 使用当前输入特征图计算空间注意力权重
        features = CBM_feature * c2  # 将特征图与空间注意力权重相乘，增强空间信息

        # 计算通道注意力
        c1 = self.ChannelGate(features)  # 计算通道注意力权重
        features = features * c1  # 将特征图与通道注意力权重相乘，增强通道信息
        # print('出来的features:',features.shape)

        return  features  # 返回拼接后的输出



def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    layers = []
    for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)









