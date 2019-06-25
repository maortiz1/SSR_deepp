import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable


class ResBlock(nn.Module):
    #initialization of resbloc
    def __init__(self, inplanes, planes,res_scale=1):
        super(ResBlock,self).__init__
        
        layers=[]
        #building up layers on resbloc
        for i in range(2):
            layers.append(nn.Conv3d(inplanes,planes,kernel_size=3))
            if i == 0:
                layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)
        self.res_scale=res_scale
    #residual block forward model
    def forward(self,x):
        res = self.body(x).mul(self.res_scale) + x
        return res


#n_resnlocks : number of residual blocks
# planes: #of features
#inplanes: Z
#scale:upsampling
#kernel_size 
#
class ResNET(nn.Module):
    def __init__(self, n_resblocks, planes, inplanes,scale,res_scale=1,kernel_size=3):
        super(ResNET,self).__init__()
        m_head = [nn.Conv3d(inplanes,planes,kernel_size)]
        m_body = [ResBlock(inplanes,planes,res_scale) for i in range(n_resblocks)]
        #tail upsampling block
        m_tail_up =[nn.Conv3D(inplanes,planes*scale,kernel_size),nn.PixelShuffle(scale)]
        self.

    


