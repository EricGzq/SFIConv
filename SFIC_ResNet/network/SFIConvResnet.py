#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhiqing Guo
# Pytorch Implementation of SFIConv-Resnet

import torch.nn as nn
from network.SFIConv import *

__all__ = ['SFIresnet26','SFIresnet50','SFIresnet101','SFIresnet152']



def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 conv"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=stride, bias=False, padding=0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None,First=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.first = First
        if self.first:
            self.SFIC1 = FirstSFIConvBR(inplanes, width, kernel_size=(1, 1),norm_layer=norm_layer,padding=0)
        else:
            self.SFIC1 = SFIConvBR(inplanes, width, kernel_size=(1,1),norm_layer=norm_layer,padding=0)

        self.SFIC2 = SFIConvBR(width, width, kernel_size=(3,3), stride=stride, groups=groups, norm_layer=norm_layer)

        self.SFIC3 = SFIConvB(width, planes * self.expansion, kernel_size=(1,1), norm_layer=norm_layer,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        if self.first:
            x_s_res, x_f_res = self.SFIC1(x)
            x_s, x_f = self.SFIC2((x_s_res, x_f_res))
        else:
            x_s_res, x_f_res = x
            x_s, x_f = self.SFIC1((x_s_res,x_f_res))
            x_s, x_f = self.SFIC2((x_s, x_f))

        x_s, x_f = self.SFIC3((x_s, x_f))

        if self.downsample is not None:
            x_s_res, x_f_res = self.downsample((x_s_res,x_f_res))

        x_s += x_s_res
        x_f += x_f_res

        x_s = self.relu(x_s)
        x_f = self.relu(x_f)

        return x_s, x_f

class BottleneckLast(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckLast, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Last means the end of two branch
        self.SFIC1 = SFIConvBR(inplanes, width,kernel_size=(1,1),padding=0)
        self.SFIC2 = SFIConvBR(width, width, kernel_size=(3, 3), stride=stride, groups=groups, norm_layer=norm_layer)
        self.SFIC3 = LastSFIConvB(width, planes * self.expansion, kernel_size=(1, 1), norm_layer=norm_layer, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):

        x_s_res, x_f_res = x
        x_s, x_f = self.SFIC1((x_s_res, x_f_res))

        x_s, x_f = self.SFIC2((x_s, x_f))
        x_s = self.SFIC3((x_s, x_f))

        if self.downsample is not None:
            x_s_res = self.downsample((x_s_res, x_f_res))

        x_s += x_s_res
        x_s = self.relu(x_s)

        return x_s

class BottleneckOrigin(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckOrigin, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SFIConvResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(SFIConvResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, First=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_last_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SFIConvB(in_channels=self.inplanes,out_channels=planes * block.expansion, kernel_size=(1,1), stride=stride, padding=0)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_last_layer(self, block, planes, blocks, stride=1, norm_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                LastSFIConvB(in_channels=self.inplanes,out_channels=planes * block.expansion, kernel_size=(1,1), stride=stride, padding=0)
            )

        layers = []
        layers.append(BottleneckLast(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(BottleneckOrigin(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_s, x_f = self.layer1(x)
        x_s, x_f = self.layer2((x_s,x_f))
        x_s, x_f = self.layer3((x_s,x_f))
        # print(x_s.size(), x_f.size())
        x_s = self.layer4((x_s,x_f))
        x = self.avgpool(x_s)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def SFIresnet26(pretrained=False, **kwargs):
    """Constructs a ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SFIConvResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model

def SFIresnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SFIConvResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def SFIresnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SFIConvResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def SFIresnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SFIConvResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    model = SFIresnet50(num_classes=2).cuda()
    print(model)
    i = torch.Tensor(1,3,256,256).cuda()
    y= model(i)
    print(y.size())
