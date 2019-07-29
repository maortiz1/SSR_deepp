import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class Unet3D(nn.Module):
    def __init__(self,in_channels=1,kernel_size=3,depth = 4):
        self.padding= (kernel_size//2)
        self.padding2=(2//2)
        super(Unet3D,self).__init__()

        self.cb0 = Block_Contracting(self.padding,in_channels,16,kernel_size=kernel_size)
       
        self.cb1 = Block_Contracting(self.padding,16,32,kernel_size=kernel_size)
        self.pooling1 = nn.MaxPool3d(kernel_size=(2, 2, 2),stride=2,padding= self.padding2)
      
        self.cb2 = Block_Contracting(self.padding,32,64,kernel_size=kernel_size)
        self.pooling2 = nn.MaxPool3d(kernel_size=(2, 2, 2),stride=2,padding= self.padding2)
        
        self.cb3 = Block_Contracting(self.padding,64,128,kernel_size=kernel_size)
        self.pooling3 = nn.MaxPool3d(kernel_size=(2, 2, 2),stride=2,padding= self.padding2)
      
        self.cb4 = Block_Contracting(self.padding,128,256,kernel_size=kernel_size)
        self.pooling4 = nn.MaxPool3d(kernel_size=(2, 2, 2),stride=2,padding= self.padding2)
       
        self.cb5 =Block_Contracting(self.padding,256,512,kernel_size=kernel_size)

    	#Expansive Path

        self.eb1 = Block_Expansive(self.padding2,512,512)
        self.cb6 = Block_Contracting(self.padding,512+256,256,kernel_size=kernel_size)
        

        self.eb2 = Block_Expansive(self.padding2,256,256)
        self.cb7 = Block_Contracting(self.padding,256+128,128,kernel_size=kernel_size)

        self.eb3 = Block_Expansive(self.padding2,128,128)
        self.cb8 = Block_Contracting(self.padding,128+64,64,kernel_size=kernel_size)

        self.eb4 = Block_Expansive(self.padding2,64,64,kernel_size=2)
        self.cb9 = Block_Contracting(self.padding,64+32,32,kernel_size=kernel_size)

        self.cb10 = Block_Contracting(self.padding,16+32,16,kernel_size=kernel_size)
        self.cb11 = Block_Contracting(self.padding,16,1,kernel_size=kernel_size)



    def forward(self,x):
        cb0 = self.cb0(x)

        cb1 = self.cb1(cb0)
        pooling1 = self.pooling1(cb1)

        cb2 = self.cb2(pooling1)
        pooling2 = self.pooling2(cb2)
     
        cb3 = self.cb3(pooling2)
        pooling3 = self.pooling3(cb3)
        
        cb4 = self.cb4(pooling3)
        pooling4 = self.pooling4(cb4)
        
        cb5 = self.cb5(pooling4)    

        eb1 = self.eb1(cb5)
        print(eb1.shape)
         print(cb4.shape)
        eb1 = torch.cat((eb1,cb4),1)        
        cb6 = self.cb6(eb1)
      
        eb2 = self.eb2(cb6)   
        eb2 = torch.cat((eb2,cb3),1) 
        cb7 = self.cb7(eb2)      

        eb3 = self.eb3(cb7)
        eb3 = torch.cat((eb3,cb2),1)
        cb8 = self.cb8(eb3)
      
        eb4 = self.eb4(cb8)  
        eb4 = torch.cat((eb4,cb1),1)
        cb9 = self.cb9(eb4)

        cb9 = torch.cat((cb9,cb0),1)
        cb10 = self.cb10(cb9)
        cb11= self.cb11(cb10)
        return cb11



class Block_Contracting(nn.Module):
    def __init__(self,padding,in_channels,out_channels,kernel_size=3):
        super(Block_Contracting,self).__init__()
        conv1 = nn.Conv3d(in_channels,out_channels,kernel_size,padding=padding)
        bn = nn.BatchNorm3d(out_channels)
        relu = nn.LeakyReLU()
        self.block = nn.Sequential(*[conv1,bn,relu])
    def forward(self,x):
        x = self.block(x)
        return x
        

class Block_Expansive(nn.Module):
    def __init__(self,padding,in_channels,out_channels,kernel_size=3,stride=(2,2,2)):
        super(Block_Expansive,self).__init__()
        conv1 = nn.ConvTranspose3d(in_channels,out_channels,kernel_size,padding=padding,stride=stride)
        bn = nn.BatchNorm3d(out_channels)
        relu = nn.LeakyReLU()
        self.block = nn.Sequential(*[conv1,bn,relu])
    def forward(self,x):
        x = self.block(x)
        return x       

