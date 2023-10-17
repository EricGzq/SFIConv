'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
import torch
import torch.nn as nn
import torchvision.models as models

import network.resnet as resnet
import network.SFIConvResnet as SFIConvResnet

# Load pretrained model
#rgb_stream = resnet.resnet26(pretrained=False)
rgb_stream = SFIConvResnet.SFIresnet26(pretrained=False)

# Remove the last layer of ResNet (i.e. FC layer)
rgb_extract_feature = nn.Sequential(*list(rgb_stream.children())[:-1])

class MainNet(nn.Module):
    def __init__(self, num_classes=None):
        print('num_classes:',num_classes)
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        
        self.rgb_extract_feature = rgb_extract_feature
        self.fc = nn.Linear(2048, num_classes)
            
    def forward(self, rgb_data):
        # extract features
        output = self.rgb_extract_feature(rgb_data)
        
        final_out = torch.flatten(output, 1)
        final_out = self.fc(final_out)
        
        return final_out

