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


import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class Trainer:
    def __init__(self, loader_train,loader_val,cuda,scale,model,lr,out,device,sch=True,st=50,epoch=0,pretrained = False,file=''):
        self.scale = scale #scale_factor
        self.data_loader_train = loader_train #data loader for training dataset
        self.data_loader_val = loader_val #data loader for validation dataset
        self.model = model #model to be trained
        self.loss = nn.MSELoss() # loss for training 
        self.cuda = cuda #if cuda available is true for gpu usage
        self.psnr_L=[] #epoch training Peak to nosie ration
        self.ssmi_L=[] #epoch training structural similarity
        self.file = file
        self.mean_loss_epc=[]
        self.ac_epoch= epoch # actual epoch intial value 0, if pretrained value passes as parameter
    

        self.optimizer =optim.Adam(model.parameters(),lr) #optimizer for training
        if pretrained:            
            self.optimizer.load_state_dict(self.file['optim_state_dict'])
            self.ac_epoch=file['epoch']
            self.psnr_L=file['psnr']
            self.ssmi_L = file['ssim']
            self.mean_loss_epc= file['m_los']

        if sch==True:
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=10) #step scheduler for better learning convergence

        self.sch  = True
        self.error_last = 1e8 # ideal last error
        self.iteration = 0 #iteration epoch 
        self.out_f = out #out folder for persistence values of training
        self.device = device # device for cuda if cuda available
        self.best_psnr = 0 # best psnr on validation set
        self.best_ssmi = 0 # best ssmi on validation set
        self.best_model = [] # best model according to psnr validation results
        if not(os.path.isdir(self.out_f)):                  
           os.mkdir(self.out_f)
         # mean losses per epochs
       
        

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
                    'optim_state_dict':self.optimizer.state_dict(),
                    'psnr':self.psnr_L,
                    'ssim':self.ssmi_L,
                    'losses':losses,
                    'm_los':self.mean_loss_epc},os.path.join(self.out_f,'che_epoch_%d.pth.tar'%(self.ac_epoch)))
        print('\n Mean PSNR',str(np.mean(psnr_c)))
        print('\n Mean SSIM: ',str(np.mean(ssim_c)))
        print('\n Mean Loss',str(np.mean(losses)))
        #self.lr_scheduler.step()       
        with torch.no_grad():
            self.model.eval()
            loss_ts=[]
            psnr_ts=[]
            ssim_ts=[]
            
            for  batch_idx,(data,target) in tqdm.tqdm(enumerate(self.data_loader_val),total=len(self.data_loader_val),desc='Test epoch %d'%self.ac_epoch,ncols=80,leave=False):

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
            if self.sch:
                self.lr_scheduler.step(mean_loss)  
                print('sch')
            if mean_psnr > self.best_psnr and mean_ssim > self.best_ssmi:
                self.best_model = self.model
                self.best_ssmi=mean_ssim
                self.best_psnr=mean_psnr
                torch.save({'epoch':self.ac_epoch,
                    'model_state_dict': self.best_model.state_dict(),
                    'optim_state_dict':self.optimizer.state_dict(),
                    'psnr':self.psnr_L,
                    'ssim':self.ssmi_L,
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
    def __init__(self,root_hr,crop=True,factor=[2,3,4,5],vox_size=(32,32),train_size=.6,n_samples=40,downfunction=utils.downsample,train=True,val=True,test=True):
        self.root_hr = root_hr
        self.vox_size = vox_size
        self.crop = crop
        self.mk_train = train
        self.mk_val = val
        self.mk_test=test
        if os.path.isdir(root_hr):
            self.gt_hr=sorted( glob.glob(os.path.join(root_hr,'*.nii')),key=numericalSort)
            print(self.gt_hr)
        else:
            raise Exception('Root has to be a directory')
        self.generate_voxels(factor,vox_size,train_size,downfunction=downfunction)


       # self.generate_lr_ls_vx(factor,vox_size,train_size,n_samples)
        #self.cut_test(vox_size)

    def vis_all(self):
        import matplotlib.pyplot as plt
        for num,i in enumerate(self.hr_pcs_tr):
            # img = nib.load(i)
            #header = img.header
            #print(header['pixdim'])
            # img = utils.normalize_image_whitestripe(img,contrast='T1')
            # img = utils.normalize(img.get_fdata())
            data_img = i
            y = data_img.shape[1]
            
            plt.title(num)
            image = []
            for y in range(0,y):
                if y == 0:
                    image=plt.imshow(data_img[::,y,::],cmap='gray',interpolation='nearest')
                    
                else:
                    image.set_data(data_img[::,y,::])

                plt.pause(.1)
                plt.draw()
            plt.waitforbuttonpress(0)
            

    def generate_voxels(self,factor,vox_size=(32,32),train_size=0.7,downfunction=utils.downsample):
        import nibabel as nib
        import image_utils as utils
        import matplotlib.pyplot as plt
        tr_size = round(len(self.gt_hr)*train_size)-1
        ts_size = len(self.gt_hr)-tr_size
        val_size = (ts_size)//2
        test_size=1-(ts_size)//2
        
        tr_samples = self.gt_hr[0:tr_size]
        val_samples = self.gt_hr[tr_size:tr_size+val_size]
        ts_samples = self.gt_hr[tr_size+val_size:-1]
        lr_pcs = []
        hr_pcs = []
        n_pc = []
        indx = 0
        num_im=[]
        print(downfunction)
        if self.mk_train:
            for k in factor:
                for i in  tr_samples:
                    if 'T1' in i:
                        img = nib.load(i)
                        data_img = img.get_fdata()
                        
                        lr_temp = downfunction(data_img,down_factor=k)
                        nib_f_lr = nib.nifti1.Nifti1Image(lr_temp ,np.eye(4))
                        wh_norm_lr = utils.normalize_image_whitestripe(nib_f_lr,contrast='T1')
                

                        wh_norm_hr = utils.normalize_image_whitestripe(img,contrast='T1')

                        wh_norm_hr = utils.normalize_image_whitestripe(img,contrast='T1')
                        if self.crop:
                            lr_pc,n_x_lr,n_y_lr = utils.cropall(wh_norm_lr,vox_size)
                            hr_pc,n_x_hr,n_y_hr = utils.cropall(wh_norm_hr,vox_size)
                            n_p_ls =[n_x_hr,n_y_hr]
                            n_pc.append(n_p_ls)
                            lr_pcs += lr_pc
                            hr_pcs += hr_pc
                        else:
                            lr_pcs.append(wh_norm_lr)
                            hr_pcs.append(wh_norm_hr)
                        img_T2 = nib.load(i.replace('T1','T2'))
                        data_img_T2 = img_T2.get_fdata()
                        
                        lr_temp_T2 = downfunction(data_img_T2,down_factor=k)
                        nib_f_lr_T2 = nib.nifti1.Nifti1Image(lr_temp_T2 ,np.eye(4))
                        wh_norm_lr_T2 = utils.normalize_image_whitestripe(nib_f_lr_T2,contrast='T2')           
                                    


                        indx +=1
                        num_im.append(indx)
            self.lr_pcs_tr = lr_pcs
            self.hr_pcs_tr = hr_pcs
            self.tr_cr_pcs = n_pc
            self.tr_num_img = num_im

        lr_pcs = []
        hr_pcs = []
        n_pc = []
        indx = 0
        num_im=[]
        if self.mk_val:
            for k in factor:
                for i in  val_samples:
                    img = nib.load(i)
                    data_img = img.get_fdata()
                    
                    lr_temp = downfunction(data_img,down_factor=k)
                    nib_f_lr = nib.nifti1.Nifti1Image(lr_temp ,np.eye(4))
                    wh_norm_lr = utils.normalize_image_whitestripe(nib_f_lr,contrast='T1')



                    wh_norm_hr = utils.normalize_image_whitestripe(img,contrast='T1')
                    if self.crop:
                        lr_pc,n_x_lr,n_y_lr = utils.cropall(wh_norm_lr,vox_size)
                        hr_pc,n_x_hr,n_y_hr = utils.cropall(wh_norm_hr,vox_size)
                        n_p_ls =[n_x_hr,n_y_hr]
                        n_pc.append(n_p_ls)
                        lr_pcs += lr_pc
                        hr_pcs += hr_pc
                    else:
                        lr_pcs.append(wh_norm_lr)
                        hr_pcs.append(wh_norm_hr)

                    indx +=1
                    num_im.append(indx)

            self.lr_pcs_val = lr_pcs
            self.hr_pcs_val = hr_pcs
            self.val_cr_pcs = n_pc
            self.val_num_img = num_im    

        lr_pcs = []
        hr_pcs = []
        n_pc = []
        indx = 0
        num_im=[]
        if self.mk_test:
            for k in factor:
                for i in  ts_samples:
                    img = nib.load(i)
                    data_img = img.get_fdata()
                    
                    lr_temp = downfunction(data_img,down_factor=k)
                    nib_f_lr = nib.nifti1.Nifti1Image(lr_temp ,np.eye(4))
                    wh_norm_lr = utils.normalize_image_whitestripe(nib_f_lr,contrast='T1')



                    wh_norm_hr = utils.normalize_image_whitestripe(img,contrast='T1')
                    if self.crop:
                        lr_pc,n_x_lr,n_y_lr = utils.cropall(wh_norm_lr,vox_size)
                        hr_pc,n_x_hr,n_y_hr = utils.cropall(wh_norm_hr,vox_size)
                        n_p_ls =[n_x_hr,n_y_hr]
                        n_pc.append(n_p_ls)
                        lr_pcs += lr_pc
                        hr_pcs += hr_pc
                    else:
                        lr_pcs.append(wh_norm_lr)
                        hr_pcs.append(wh_norm_hr)

                    indx +=1
                    num_im.append(indx)

            self.lr_pcs_ts = lr_pcs
            self.hr_pcs_ts = hr_pcs
            self.ts_cr_pcs = n_pc
            self.ts_num_img = num_im
    def reconstruct(self,scores,type='test'):
        all_join = []
        np_ts = self.ts_cr_pcs
        if type =='train':
            np_ts=self.tr_cr_pcs
        elif type== 'val':
            np_ts=self.val_cr_pcs
        i=0
        for indx, f in enumerate(np_ts):
            x_px = f[0]
            y_px = f[1]
            n_pz = y_px*x_px
            # print(f)
            # print('[%d,%d] length: %d'%(i,i+n_pz,len(scores)))
            a=[]
            l_pz = scores[i:i+n_pz]
            for j in range(0,x_px):
                y_pz = l_pz[j*y_px:j*y_px+y_px]

                cot = []
                for k in range(0,y_px):
                    ac = y_pz[k]
                    # plt.imshow(ac[::,::,100])
                    # plt.show()
                    # print(ac.shape)
                    if k == 0:
                        cot = ac
                    else:
                        cot=np.concatenate((cot,ac),axis=1)

                    # plt.imshow(cot[::,::,100])
                    # plt.show()
                if j == 0:
                    a = cot
                else:
                    a = np.concatenate((a,cot),axis=0)
            i=n_pz+i

            all_join.append(a)
        return all_join    








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




#import matplotlib.pyplot as plt
root = os.path.join(os.getcwd(),'images')
#import nibabel as nib
# img =nib.load(os.path.join(root,'T1_1.nii'))
# data = img.get_fdata()
# plt.imshow(data[::,50,::])
# plt.show()
dataprep = Data_Preparation(root)
#lr_tr = dataprep.hr_pcs_tr
#print(len(lr_tr))

# np_l = dataprep.tr_cr_pcs
# i = 0 
# all_join = []
# for indx, f in enumerate(np_l):
#     x_px = f[0]
#     y_px = f[1]
#     n_pz = y_px*x_px
#     print(f)
#     print('[%d,%d] length: %d'%(i,i+n_pz,len(lr_tr)))
#     a=[]
#     l_pz = lr_tr[i:i+n_pz]
#     for j in range(0,x_px):
#         y_pz = l_pz[j*y_px:j*y_px+y_px]
#         print(len(y_pz))
#         cot = []
#         for k in range(0,y_px):
#             ac = y_pz[k]
#             # plt.imshow(ac[::,::,100])
#             # plt.show()
#             # print(ac.shape)
#             if k == 0:
#                 cot = ac
#             else:
#                 cot=np.concatenate((cot,ac),axis=1)
#             print(cot.shape)
#             # plt.imshow(cot[::,::,100])
#             # plt.show()
#         if j == 0:
#             a = cot
#         else:
#             a = np.concatenate((a,cot),axis=0)
#     i=n_pz+i
#     all_join.append(a)
#     print(a.shape)
#     plt.imshow(a[::,50,::])
#     plt.draw()


# for img in all_join:
#     x = img.shape[1]
#     gr = plt.imshow(img[::,::,100])
#     plt.pause(.2)
#     plt.draw()
    
    # for k in range(0,x):
    #     if gr is None:
    #         gr = plt.imshow(img[::,k,::],cmap='gray')
    #     else: 
    #         gr.set_data(img[::,k,::])
    #     plt.pause(.2)
    #     plt.draw()


                



   # def reconstr_test(self,arr):
    #     import matplotlib.pyplot as plt
    #     size_arr = arr.shape
    #     num_img_arr = int(size_arr[0]/self.n_pieces_img)
    #     if size_arr[0]%self.n_pieces_img != 0:
    #         raise Exception('No enough pieces to form images, at least',str(self.n_pieces_img))
    #     ls_all_im = []
    #     for i in range(0,num_img_arr):
    #         ac_img = arr[i*self.n_pieces_img:i*self.n_pieces_img+self.n_pieces_img,::,::,::]
    #         a=[]
    #         for j in range(0,self.n_pieces_x):
    #             ac_piece= ac_img[j*self.n_pieces_y:j*self.n_pieces_y+self.n_pieces_y,::,::,::]#.squeeze(axis=1)
    #             x_cot = ac_piece[0,::,::,::]
    #             for k in range(1,self.n_pieces_y):
    #                 ac_s_piece = ac_piece[k,::,::,::]
    #                 x_cot = np.concatenate((x_cot,ac_s_piece),axis=1)
    #             if j == 0:
    #                 a = x_cot
    #             else:
    #                 a = np.concatenate((a,x_cot),axis=0)

    #         ls_all_im.append(a)
    #     return ls_all_im