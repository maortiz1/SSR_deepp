
import torch.optim as optim
import torch
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import tqdm
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

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
            SR_pred = 
            p,s = self.metrics(target,score)
            psnr_c.append(p)
            ssim_c.append(s)
        
        pr
        psnr_L.append(np.mean(psnr_c))
        ssmi_L.append(np.mean(ssim_c))



    def metrics(self,true_img,pred_img):
        psnr_c = psnr(true_img,pred_img)
        ssim_c = ssim(true_img,pred_img)
        return psnr_c,ssim_c





    