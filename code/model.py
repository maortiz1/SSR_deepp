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
    def __init__(self, n_resblocks,scale,output_size,res_scale=1,kernel_size=3):
        super(ResNET,self).__init__()
        m_head = [nn.Conv3d(64,64,kernel_size)]
        m_body = [ResBlock(64,64,res_scale) for i in range(n_resblocks)]
        #tail upsampling block
        m_tail_up =[nn.Conv3d(64,64*scale,kernel_size),nn.LeakyReLU(),nn.Upsample(size=output_size,mode='trilinear')]
    
        self.head=nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail_up)

    def forward(self,x):#Forward model for Residual Network 

        x=self.head(x)

        res = self.body(x)
        res+=x

        x=self.tail(res)
        return x
    def load_state_dict(self,state_dict, strict = True):
        own_state = self.state_dict()
        for name,param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param=param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') ==-1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
                    elif strict:
                        if name.find('tail')==-1:
                            raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
        

    


