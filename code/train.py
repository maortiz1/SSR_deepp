import torch.optim as optim
import torch
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import tqdm
from decimal import Decimal
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from torch.utils import data
import image_utils as utils
import glob
import os
import math
import nibabel as nib 
class Trainer:
    def __init__(self, loader_train,loader_test,cuda,scale,model,lr,out,device,epoch=0):
        self.scale = scale #scale_factor
        self.data_loader_train = loader_train #data loader for training dataset
        self.data_loader_test = loader_test #data loader for validation dataset
        self.model = model #model to be trained
        self.loss = nn.MSELoss() # loss for training 
        self.cuda = cuda #if cuda available is true for gpu usage
        self.psnr_L=[] #epoch training Peak to nosie ration
        self.ssmi_L=[] #epoch training structural similarity
    

        self.optimizer =optim.Adam(model.parameters(),lr) #optimizer for training

        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=50,gamma=0.1) #step scheduler for better learning convergence
        self.error_last = 1e8 # ideal last error
        self.ac_epoch= epoch # actual epoch intial value 0, if pretrained value passes as parameter
        self.iteration = 0 #iteration epoch 
        self.out_f = out #out folder for persistence values of training
        self.device = device # device for cuda if cuda available
        self.best_psnr = 0 # best psnr on validation set
        self.beat_ssmi = 0 # best ssmi on validation set
        self.best_model = [] # best model according to psnr validation results
        if not(os.path.isdir(self.out_f)):                  
           os.mkdir(self.out_f)
        self.mean_loss_epc=[] # mean losses per epochs
       
        

    def train(self, max_epoch):
        for epoc in tqdm.trange(self.ac_epoch,max_epoch,desc='Train',ncols=80):
            self.ac_epoch = epoc
            self.train_epoch()
    

    def train_epoch(self):
        self.model.train()
        lr=self.get_lr(self.optimizer)
        print('\n Learning Rate: {:.2e}'.format(Decimal(lr)))


        losses = []
        psnr_c = []
        ssim_c = []
        for batch_idx,(data,target) in tqdm.tqdm(enumerate(self.data_loader_train),total=len(self.data_loader_train),desc='Train epoch %d'%self.ac_epoch,ncols=80,leave=False):

            iteration = batch_idx + self.ac_epoch*len(self.data_loader_train)
            if self.iteration !=0 and (iteration -1 )!= self.iteration:
                continue
            self.iteration= iteration 
            if self.cuda:
                data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            score = self.model(data)
            loss = self.loss(score,target)
            losses.append(loss.item())
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optimizer.step()
            for k in range(0,score.shape[0]):   
              t = target[k,::,::,::,::] 

              s = score[k,::,::,::,::]   
                  
              p,s = self.metrics(t.squeeze(),s.squeeze())
              psnr_c.append(p)
              ssim_c.append(s)


        self.psnr_L.append(np.mean(psnr_c))
        self.ssmi_L.append(np.mean(ssim_c))
        self.mean_loss_epc.append(np.mean(losses))
        
        torch.save({'epoch':self.ac_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'psnr':self.psnr_L,
                    'ssim':self.ssmi_L,
                    'losses':losses,
                    'm_los':self.mean_loss_epc},os.path.join(self.out_f,'che_epoch_%d.pth.tar'%(self.ac_epoch)))
        print('\n Mean PSNR',str(np.mean(psnr_c)))
        print('\n Mean SSIM: ',str(np.mean(ssim_c)))
        print('\n Mean Loss',str(np.mean(losses)))
        self.lr_scheduler.step()       
        with torch.no_grad():
            self.model.eval()
            loss_ts=[]
            psnr_ts=[]
            ssim_ts=[]
            
            for  batch_idx,(data,target) in tqdm.tqdm(enumerate(self.data_loader_test),total=len(self.data_loader_test),desc='Test epoch %d'%self.ac_epoch,ncols=80,leave=False):

                if self.cuda:
                    data,target = data.to(self.device),target.to(self.device)
                score = self.model(data)   
                loss = self.loss(score,target)
                loss_ts.append(loss.item())
                for k in range(0,score.shape[0]):   
                  t = target[k,::,::,::,::] 
    
                  s = score[k,::,::,::,::]   
                      
                  p,s = self.metrics(t.squeeze(),s.squeeze())
                  psnr_ts.append(p)
                  ssim_ts.append(s)
            
            mean_psnr = np.mean(psnr_ts)
            mean_ssim = np.mean(ssim_ts)
            mean_loss = np.mean(loss_ts)
            if mean_psnr > self.best_psnr and mean_ssim > self.beat_ssmi:
                self.best_model = self.model
                torch.save({'epoch':self.ac_epoch,
                    'model_state_dict': self.best_model.state_dict(),
                    'psnr':self.psnr_L,
                    'ssmi':self.ssmi_L,
                    'mean_psnr_val':mean_psnr,
                    'mean_ssim_val':mean_ssim,
                    'losses':losses,
                    'm_los':self.mean_loss_epc},os.path.join(self.out_f,'best_model.pth.tar')) 
           
            print('\n Validation PSNR: ',str(mean_psnr))
            print('\n Validation SSIM: ',str(mean_ssim))
            print('\n Validation Loss: ',str(mean_loss))    
                

                
                


    def metrics(self,true_img,pred_img):
        
        psnr_c = psnr(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        ssim_c = ssim(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        return psnr_c,ssim_c
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


class Data_Preparation():
    def __init__(self,root_hr,factor=3,vox_size=(32,32),train_size=.7,n_samples=40):
        self.root_hr = root_hr
        self.vox_size = vox_size
        if os.path.isdir(root_hr):
            self.gt_hr= glob.glob(os.path.join(root_hr,'*.nii'))
        else:
            raise Exception('Root has to be a directory')
        self.generate_voxels(factor,vox_size,train_size)
       # self.generate_lr_ls_vx(factor,vox_size,train_size,n_samples)
        #self.cut_test(vox_size)

    def generate_voxels(self,factor=3,vox_size=(32,32),train_size=0.7):
        import nibabel as nib
        import image_utils as utils
        tr_size = round(len(self.gt_hr)*train_size)-1
        ts_size = len(self.gt_hr)-tr_size

        tr_samples = self.gt_hr[0:tr_size]
        ts_samples = self.gt_hr[tr_size:-1]
        lr_pcs = []
        hr_pcs = []
        n_pc = []
        indx = 0
        num_im=[]
        for i in  tr_samples:
            img = nib.load(i)
            data_img = img.get_fdata()
            
            lr_temp = utils.downsample(data_img,down_factor=factor)
            nib_f_lr = nib.nifti1.Nifti1Image(lr_temp ,np.eye(4))
            wh_norm_lr = utils.normalize_image_whitestripe(nib_f_lr,contrast='T1')
            lr_pc,n_x_lr,n_y_lr = utils.cropall(wh_norm_lr.get_fdata(),vox_size)


            wh_norm_hr = utils.normalize_image_whitestripe(img,contrast='T1')
            hr_pc,n_x_hr,n_y_hr = utils.cropall(wh_norm_hr.get_fdata(),vox_size)
            n_p_ls =[n_x_hr,n_y_hr]

            lr_pcs += lr_pc
            hr_pcs += hr_pc
            indx +=1
            num_im.append(indx)
            n_pc.append(n_p_ls)
        self.lr_pcs_tr = lr_pcs
        self.hr_pcs_tr = hr_pcs
        self.tr_cr_pcs = n_pc
        self.tr_num_img = num_im

        lr_pcs = []
        hr_pcs = []
        n_pc = []
        indx = 0
        num_im=[]
        for i in  ts_samples:
            img = nib.load(i)
            data_img = img.get_fdata()
            
            lr_temp = utils.downsample(data_img,down_factor=factor)
            nib_f_lr = nib.nifti1.Nifti1Image(lr_temp ,np.eye(4))
            wh_norm_lr = utils.normalize_image_whitestripe(nib_f_lr,contrast='T1')
            lr_pc,n_x_lr,n_y_lr = utils.cropall(wh_norm_lr.get_fdata(),vox_size)


            wh_norm_hr = utils.normalize_image_whitestripe(img,contrast='T1')
            hr_pc,n_x_hr,n_y_hr = utils.cropall(wh_norm_hr.get_fdata(),vox_size)
            n_p_ls =[n_x_hr,n_y_hr]

            lr_pcs += lr_pc
            hr_pcs += hr_pc
            indx +=1
            num_im.append(indx)
            n_pc.append(n_p_ls)
        self.lr_pcs_ts = lr_pcs
        self.hr_pcs_ts = hr_pcs
        self.ts_cr_pcs = n_pc
        self.ts_num_img = num_im



class Dataset(data.Dataset):
    """
    Class that builds the representation of the dataset inputs
    
    ...
    Attributes
    ----------
        data_hr: List
            Contains the high resolution data on voxxel form
        data_lr: List
            Contains the Low resolution data on voxel form
        transform: Function
            Transform Function to applied to input data
    
    Methods
    -------
        __getitem__(index)
            Returns item according to index performing if so the transformation function
        __len__
            Returns the len of the dataset
        """

    def __init__(self,vox_hr,vox_lr,transform=None):
        """
        Parameters
        ----------
            vox_hr: List
                The list containing voxels on high resolution
            vox_lr: List
                The List containing voxels on low resolution
            transform: Function,optional
                Function to transform data when asked for
        """
        self.data_hr = vox_hr
        self.data_lr = vox_lr
        self.transform = transform
    def __len__(self):
        """
        Denotes the total number of samples
        """
        return len(self.data_hr)
    def __getitem__(self,index):
        """
        Returns the item on the index with the corresponding transformation
        Parameters
        ----------
            index: int
                Position of the item
        """
        y = torch.from_numpy(np.expand_dims(self.data_hr[index],axis=0).astype(np.float32)).permute(0,3,1,2)
        x = torch.from_numpy(np.expand_dims(self.data_lr[index],axis=0).astype(np.float32)).permute(0,3,1,2)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x,y


root = os.path.join(os.getcwd(),'images')
dataprep = Data_Preparation(root)