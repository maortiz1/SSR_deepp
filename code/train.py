
import torch.optim as optim
import torch
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import tqdm
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from torch.utils import data
import image_utils as utils
import glob
import os
import nibabel as nib 
class Trainer:
    def __init__(self, loader_train,loader_test,cuda,scale,model,lr):
        self.scale = scale
        self.data_loader_train = loader_train
        self.data_loader_test = loader_test
        self.model = model
        self.loss = nn.L1Loss()
        self.cuda = cuda
        self.optimizer =optim.Adam(model.parameters(),lr)
        self.error_last = 1e8
        self.ac_epoch= 0
        self.iteration = 0

    def train(self, max_epoch):
        for epoc in tqdm.trange(self.ac_epoch,max_epoch,desc='Train',ncols=80):
            self.ac_epoch = epoc
            self.train_epoch()
    

    def train_epoch(self):
        self.model.train()
        psnr_L=[]
        ssmi_L=[]
        psnr_c = []
        ssim_c = []
        for batch_idx,(data,target) in tqdm.tqdm(enumerate(self.data_loader_train),total=len(self.data_loader_train),desc='Train epoch =%d'%self.ac_epoch,ncols=80,leave=False):
            psnr_c = []
            ssim_c = []
            iteration = batch_idx + self.ac_epoch*len(self.data_loader_train)
            if self.iteration !=0 and (iteration -1 )!= self.iteration:
                continue
            self.iteration= iteration 
            if self.cuda:
                data, target = data.to('cuda'), target.to('cuda')
            self.optimizer.zero_grad()
            score = self.model(data)
            loss = self.loss(score,target)

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optimizer.step()
            p,s = self.metrics(target,score)
            psnr_c.append(p)
            ssim_c.append(s)
            print('Epoch: ',self.ac_epoch,'\t loss:',str(loss))


        
        
        psnr_L.append(np.mean(psnr_c))
        ssmi_L.append(np.mean(ssim_c))



    def metrics(self,true_img,pred_img):
        psnr_c = psnr(true_img,pred_img)
        ssim_c = ssim(true_img,pred_img)
        return psnr_c,ssim_c



class Preprocessing():
    def __init__(self,root_hr):
        self.root_hr = root_hr
        if os.path.isdir(root_hr):
            self.gt_hr= glob.glob(os.path.join(root_hr,'*.nii'))
        else:
            raise Exception('Root has to be a directory')
    def get_lr_ls(self,factor=3):
        lr =[]
        for i in self.root_hr:
            img = nib.load(i)
            data = img.get_fdata()
            lr_temp = utils.downsample(data, down_factor=factor)
            lr_nib = nib.nifti1.Nifti1Image(lr_temp ,np.eye(4))
            lr.append(lr_nib)
        self.lr = lr

    





class Dataset(data.Dataset):
    def __init__(self,data_hr,transform = None):
        self.data_hr =data_hr 
        self.transform = transform
    def __len__(self):
        return len(self.data_hr)
    def __getitem__(self,index):
        y = self.data_hr


        

