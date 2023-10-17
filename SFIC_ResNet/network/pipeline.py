'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
import torch
import numpy as np
from torchstat import stat
from thop import profile, clever_format
from ptflops import get_model_complexity_info

def params_count(model):
    '''
    Compute the parameters.
    '''
    return np.sum([p.numel() for p in model.parameters()]).item()


def cal_params_thop(model, tensor):
    '''
    Using thop to compute the parameters, FLOPs
    tensor: torch.randn(1, 3, 256, 256)
    '''
    flops, params = profile(model, inputs = (tensor, ))
    flops, params = clever_format([flops, params], '%.3f')
    return flops, params

def cal_params_ptflops(model, shape):
    '''
    Using ptflops to compute the parameters, FLOPs
    shape: (3, 256, 256)
    '''
    with torch.cuda.device(0):
        #model = models.resnet50()
        flops, params = get_model_complexity_info(model, shape, as_strings=True, print_per_layer_stat=True, verbose=True)
        #print('{:<30}  {:<8}'.format('Computational complexity (FLOPs): ', flops))
        #print('{:<30}  {:<8}'.format('Number of parameters (Params): ', params))
    return flops, params



