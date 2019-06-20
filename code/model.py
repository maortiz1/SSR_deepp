import torch
import torch.nn as nn
import torch.nn.functional as nn
from torch.autograd  import Variable
import math 
from functools import partial



class ResBlock(nn.Module):
    #initialization of resbloc
    def __init__(self, inplanes, planes,res_scale=1):
        super(ResBlock,self).__init__
        
        layers=[]
        #building up layers on resbloc
        for i in range(2):
            layers.append(nn.Conv3d(inplanes,planes,kernel_size=3))
            if i == 0:
                layers.append(nn.Relu(inplace=True))
        self.body = nn.Sequential(*layers)
        self.res_scale=1
    #residual block forward model
    def forward(self,x):
        res = self.body(x).mul(self.res_scale) + x
        return res



class ResNET(nn.Module):
    def __init__(self, args, ):
        super(ResNET,self).__init__()
    


class upSamplingBlock(nn.Module):
    def __init__(self, n_blocs, ):
        return super().__init__(*args, **kwargs)




