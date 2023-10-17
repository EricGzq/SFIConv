#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhiqing Guo
# Pytorch Implementation of SFIConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fractions import Fraction
import math

# -----frequency modules----- #
class MCSConv(nn.Module):
    def __init__(self, in_planes):
        super(MCSConv, self).__init__()
        self.in_planes = in_planes
        # initialization the weight
        self.const_weight = nn.Parameter(torch.randn(size=[in_planes, 1, 5, 5]), requires_grad=True)
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        
        for i in range(in_planes):
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, 2, 2] = -1.0
        
        # get MCSConv kernel
        kernel = torch.FloatTensor(self.const_weight).expand(in_planes, 1, 5, 5)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(in_planes), requires_grad=False) 
    
    def forward(self, x):
        #out = F.conv2d(x, self.weight, self.bias, stride=1, padding=2)
        out = F.conv2d(x, self.weight, stride=1, padding=2, groups=self.in_planes)
        return out


# -----basic functions of SFIConv----- #
class FirstSFIConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstSFIConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.AvgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.s2s = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.f2f = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        # frequency modules
        self.MCSConv = MCSConv(in_channels)
        
    def forward(self, x):
        if self.stride ==2:
            x = self.AvgPool(x)
        
        # spatial domain
        X_s = self.s2s(x)
        
        # frequency domain
        X_f = self.MCSConv(x)   # Multichannel Constrained Separable Conv
        
        X_f = self.f2f(X_f)
        #X_f = F.interpolate(X_f, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)

        return X_s, X_f


class SFIConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(SFIConv, self).__init__()
        kernel_size = kernel_size[0]
        self.AvgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        #self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha <= 1, "Alpha should be in the interval from 0 to 1."
        self.alpha = alpha
        
        
        self.f2f = None if alpha == 0 else \
                   torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, math.ceil(alpha * groups), bias)
        self.f2s = None if alpha == 0 or alpha == 1 or self.is_dw else \
                   torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.s2f = None if alpha == 1 or alpha == 0 or self.is_dw else \
                   torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.s2s = None if alpha == 1 else \
                   torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, math.ceil(groups - alpha * groups), bias)

    def forward(self, x):
        X_s, X_f = x

        if self.stride ==2:
            X_s, X_f = self.AvgPool(X_s), self.AvgPool(X_f)

        X_s2s = self.s2s(X_s)
        X_f2s = self.f2s(X_f) if not self.is_dw else None
        
        X_f2f = self.f2f(X_f)
        X_s2f = self.s2f(X_s) if not self.is_dw else None
        
        #X_f2s = F.interpolate(X_f2s, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        #X_s2f = F.interpolate(X_s2f, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
        X_s = X_s2s + X_f2s if X_f2s is not None else X_s2s
        X_f = X_s2f + X_f2f if X_s2f is not None else X_f2f

        return X_s, X_f


class LastSFIConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastSFIConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.AvgPool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.s2s = torch.nn.Conv2d(in_channels - int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.f2f = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)

        #self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        

    def forward(self, x):
        X_s, X_f = x

        if self.stride ==2:
            X_s, X_f = self.AvgPool(X_s), self.AvgPool(X_f)

        X_s2s = self.s2s(X_s)
        X_f2f = self.f2f(X_f)
        
        #X_f2f = F.interpolate(X_f2f, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        X_out = X_s2s + X_f2f
        
        return X_out


# -----SFIConv functions used in backbone----- #
class SFIConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SFIConvBR, self).__init__()
        self.conv = SFIConv(in_channels,out_channels,kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(int(out_channels*(1-alpha)))
        self.bn_f = norm_layer(int(out_channels*alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s, x_f = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        x_f = self.relu(self.bn_f(x_f))
        return x_s, x_f


class SFIConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SFIConvB, self).__init__()
        self.conv = SFIConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                               groups, bias)
        self.bn_s = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_f = norm_layer(int(out_channels * alpha))

    def forward(self, x):
        x_s, x_f = self.conv(x)
        x_s = self.bn_s(x_s)
        x_f = self.bn_f(x_f)
        return x_s, x_f


class FirstSFIConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstSFIConvBR, self).__init__()
        self.conv = FirstSFIConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_s = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_f = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s, x_f = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        x_f = self.relu(self.bn_f(x_f))
        return x_s, x_f


class LastSFIConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastSFIConvBR, self).__init__()
        self.conv = LastSFIConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        return x_s


class FirstSFIConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FirstSFIConvB, self).__init__()
        self.conv = FirstSFIConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_f = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s, x_f = self.conv(x)
        x_s = self.bn_s(x_s)
        x_f = self.bn_f(x_f)
        return x_s, x_f


class LastSFIConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastSFIConvB, self).__init__()
        self.conv = LastSFIConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s = self.conv(x)
        x_s = self.bn_s(x_s)
        return x_s


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------- #
    # test FirstMYconv
    i = torch.Tensor(1, 3, 32, 32).cuda()
    FirstMYconv = FirstSFIConv(kernel_size=(3, 3), in_channels=3, out_channels=32, alpha=0.5).cuda()
    x_out, y_out = FirstMYconv(i)
    print("FirstMYconv: ", x_out.size(), y_out.size())
    
    # test MYconv
    MYconv = SFIConv(kernel_size=(3,3), in_channels=32, out_channels=64, bias=False, stride=2, alpha=0.5).cuda()
    i = x_out, y_out
    x_out, y_out = MYconv(i)
    print("MYconv: ", x_out.size(), y_out.size())
    
    MYconv_b = SFIConvB(in_channels=64, out_channels=64, alpha=0.5).cuda()
    i = x_out, y_out
    x_out, y_out = MYconv_b(i)
    print("MYconv_b:",x_out.size(),y_out.size())
    
    # test LastMYconv
    LastMYconv = LastSFIConv(kernel_size=(3, 3), in_channels=64, out_channels=256, alpha=0.5).cuda()
    i = x_out, y_out
    out = LastMYconv(i)
    print("LastMYconv: ", out.size())
    # ----------------------------------------------------------------------------------------------------- #

