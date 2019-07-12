
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
import math
import nibabel as nib 
class Trainer:
    def __init__(self, loader_train,loader_test,cuda,scale,model,lr,out,device):
        self.scale = scale
        self.data_loader_train = loader_train
        self.data_loader_test = loader_test
        self.model = model
        self.loss = nn.MSELoss()
        self.cuda = cuda
        self.optimizer =optim.Adam(model.parameters(),lr)
        self.error_last = 1e8
        self.ac_epoch= 0
        self.iteration = 0
        self.out_f = out
        self.device = device
        if not(os.path.isdir(self.out_f)):                  
           os.mkdir(self.out_f)
        self.mean_loss_epc=[]
       
        

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
        losses = []
        
        for batch_idx,(data,target) in tqdm.tqdm(enumerate(self.data_loader_train),total=len(self.data_loader_train),desc='Train epoch %d'%self.ac_epoch,ncols=80,leave=False):
            psnr_c = []
            ssim_c = []
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
            for k in range(0,self.data_loader_train.batch_size):   
              t = target[k,::,::,::,::] 

              s = score[k,::,::,::,::]   
                  
              p,s = self.metrics(t.squeeze(),s.squeeze())
              psnr_c.append(p)
              ssim_c.append(s)
            print('\n Epoch: ',self.ac_epoch,' \t loss:',str(loss.item()))


        
        
        psnr_L.append(np.mean(psnr_c))
        ssmi_L.append(np.mean(ssim_c))
        self.mean_loss_epc.append(np.mean(losses))
        
        torch.save({'epoch':self.ac_epoch,'model_state_dict': self.model.state_dict(),'model':self.model,'losses':losses,'m_los':self.mean_loss_epc},os.path.join(self.out_f,'che_epoch_%d.pth.tar'%(self.ac_epoch)))
        print('Mean PSNR',str(np.mean(psnr_c)))
        print('\n Mean Loss',str(np.mean(losses)))



    def metrics(self,true_img,pred_img):
        
        psnr_c = psnr(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        ssim_c = ssim(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        return psnr_c,ssim_c



class Data_Preparation():
    def __init__(self,root_hr,factor=3,vox_size=(32,32),train_size=.7,n_samples=40):
        self.root_hr = root_hr
        self.vox_size = vox_size
        if os.path.isdir(root_hr):
            self.gt_hr= glob.glob(os.path.join(root_hr,'*.nii'))
        else:
            raise Exception('Root has to be a directory')
        self.generate_lr_ls_vx(factor,vox_size,train_size,n_samples)
        self.cut_test(vox_size)

    def generate_lr_ls_vx(self,factor=3,vox_size=(32,32),train_size=.7,n_samples=40):
        tr_size = round(len(self.gt_hr)*train_size)-1
       # tst_size = len(self.gt_hr) - tr_size-1
        
        img = nib.load(self.gt_hr[0])
        data = img.get_fdata()
        z_size_hr = data.shape[2]
        lr_temp = utils.downsample(data, down_factor=factor)
        z_size_lr = lr_temp.shape[2]

      
        lr_vox = np.empty((n_samples*tr_size,vox_size[0],vox_size[1],z_size_lr))
        hr_vox = np.empty((n_samples*tr_size,vox_size[0],vox_size[1],z_size_hr))
        all_train_lr = []
        all_train_hr = []
        for i,img_r in enumerate(self.gt_hr[0:tr_size]):
            img = nib.load(img_r) # read data file
            print(img_r)
        
            data = img.get_fdata()#get data from file
            data_wh = utils.normalize_image_whitestripe(img)#normalize hr image to wh
            print(data.shape)

            lr_temp = utils.downsample(data, down_factor=factor)#downsampled hr image
            lr_nib = nib.nifti1.Nifti1Image(lr_temp ,np.eye(4))
            lr_norm_wh = utils.normalize_image_whitestripe(lr_nib)# normalize lr image

            voxels_lr,voxels_hr = utils.voxelize_image(lr_norm_wh.get_fdata(),data_wh.get_fdata(),vox_size,n_samples) # pairs of lr and hr voxels
            
            #all images lr, hr voxels
            lr_vox[i*n_samples:i*n_samples+n_samples,::,::,::] = voxels_lr
            hr_vox[i*n_samples:i*n_samples+n_samples,::,::,::] = voxels_hr
            
            #not voxelize data
            all_train_hr.append(data_wh.get_fdata())
            all_train_lr.append(lr_norm_wh.get_fdata())
        #all data 
        self.lr_train_vox = np.expand_dims(lr_vox,axis=1)
        self.lr_train_img = np.expand_dims(all_train_lr,axis=1)
        self.hr_train_vox = np.expand_dims(hr_vox,axis=1)
        self.hr_train_img = np.expand_dims(all_train_hr,axis=1)

        test_img_hr = []
        test_img_lr = []
        for i,img_r in enumerate(self.gt_hr[tr_size:-1]):
            img = nib.load(img_r) # read data file
            print(img_r)

        
            data = img.get_fdata()#get data from file
            print(data.shape)
            data_wh = utils.normalize_image_whitestripe(img)#normalize hr image to wh
            data_wh = np.expand_dims(data_wh.get_fdata(),axis=0)
            test_img_hr.append(data_wh)

            lr_temp = utils.downsample(data, down_factor=factor)#downsampled hr image
            lr_nib = nib.nifti1.Nifti1Image(lr_temp ,np.eye(4))
            lr_norm_wh = utils.normalize_image_whitestripe(lr_nib)# normalize lr image           
            lr_norm_wh = np.expand_dims(lr_norm_wh.get_fdata(),axis=0)
            test_img_lr.append(lr_norm_wh)


        self.test_hr_data = test_img_hr
        self.test_lr_data = test_img_lr
    
    def cut_test(self,size):
        x = size[0]
        y = size[1]

     
        hr = self.test_hr_data[0]
        lr = self.test_lr_data[0]
        shape_x = hr.shape[1]#x
        shape_y = hr.shape[2]#y
        
        n_pieces_x = int( math.floor(shape_x/x))
        n_pieces_y =int( math.floor(shape_y/y))
        self.n_pieces_x = n_pieces_x
        self.n_pieces_y = n_pieces_y
        self.n_pieces_img = n_pieces_x*n_pieces_y
        res_x = int((shape_x-(n_pieces_x*x))/2)
        res_y =  int((shape_y-(n_pieces_y*y))/2)
        
        str_x = res_x
        str_y = res_y
        ls_hr_pieces=[]
        ls_lr_pieces=[]
        arr_hr_pieces = np.empty((n_pieces_x*n_pieces_y*len(self.test_hr_data),hr.shape[0],x,y,hr.shape[3]))
        arr_lr_pieces = np.empty((n_pieces_x*n_pieces_y*len(self.test_lr_data),lr.shape[0],x,y,lr.shape[3]))
        for index, value in enumerate(zip(self.test_hr_data,self.test_lr_data)):        
            hr = value[0]
            lr = value[1]
            print(hr.shape)
            print(lr.shape)
            pc_hr_img = np.empty((n_pieces_x*n_pieces_y,hr.shape[0],x,y,hr.shape[3]))
            pc_lr_img = np.empty((n_pieces_x*n_pieces_y,lr.shape[0],x,y,lr.shape[3]))
            ind = 0
            for i in range(0,n_pieces_x):
                for j in range(0,n_pieces_y):
                    str_ind_x = str_x + i*x
                    str_ind_y = str_y + j*y 
                    pc_hr = hr[::,str_ind_x:str_ind_x+x,str_ind_y: str_ind_y+y,::]
                    pc_lr = lr[::,str_ind_x:str_ind_x+x,str_ind_y:str_ind_y+y ,::]
                    pc_hr_img[ind,::,::,::,::] = pc_hr
                    pc_lr_img[ind,::,::,:,::] = pc_lr
                    ind +=1

            arr_hr_pieces[index*n_pieces_x*n_pieces_y:index*n_pieces_x*n_pieces_y+n_pieces_x*n_pieces_y,::,::,::,::] = pc_hr_img
            arr_lr_pieces[index*n_pieces_x*n_pieces_y:index*n_pieces_x*n_pieces_y+n_pieces_x*n_pieces_y,::,::,::,::] = pc_lr_img

            ls_hr_pieces.append(pc_hr_img)
            ls_lr_pieces.append(pc_lr_img)

        self.ls_hr_pieces_test = ls_hr_pieces
        self.ls_lr_pieces_test = ls_lr_pieces
        self.arr_hr_pieces_test = arr_hr_pieces
        self.arr_lr_pieces_test = arr_lr_pieces
    def reconstr_test(self,arr):
        import matplotlib.pyplot as plt
        size_arr = arr.shape
        num_img_arr = int(size_arr[0]/self.n_pieces_img)
        if size_arr[0]%self.n_pieces_img != 0:
            raise Exception('No enough pieces to form images, at least',str(self.n_pieces_img))
        ls_all_im = []
        for i in range(0,num_img_arr):
            ac_img = arr[i*self.n_pieces_img:i*self.n_pieces_img+self.n_pieces_img,::,::,::]
            a=[]
            for j in range(0,self.n_pieces_x):
                ac_piece= ac_img[j*self.n_pieces_y:j*self.n_pieces_y+self.n_pieces_y,::,::,::]#.squeeze(axis=1)
                x_cot = ac_piece[0,::,::,::]
                for k in range(1,self.n_pieces_y):
                    ac_s_piece = ac_piece[k,::,::,::]
                    x_cot = np.concatenate((x_cot,ac_s_piece),axis=1)
                if j == 0:
                    a = x_cot
                else:
                    a = np.concatenate((a,x_cot),axis=0)

            ls_all_im.append(a)
        return ls_all_im
 

# import os
# root = os.path.join(os.getcwd(),'images')
# dataprep = Data_Preparation(root)
# import matplotlib.pyplot as plt

# hr= dataprep.arr_hr_pieces_test
# lr = dataprep.arr_lr_pieces_test
# print(hr.shape)
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(hr[20,::,::,20,::].squeeze(axis=0),cmap='gray')
# ax[1].imshow(lr[20,::,::,20,::].squeeze(axis=0),cmap='gray')

# plt.show()

# arr = dataprep.arr_lr_pieces_test
# recon = dataprep.reconstr_test(arr)
# plt.figure()
# plt.imshow(recon[0][::,::,50])
# plt.show()
# plt.figure()
# plt.imshow(recon[1][::,::,50])
# plt.show()

    
    




class Dataset(data.Dataset):
    def __init__(self,vox_hr,vox_lr,transform=None):
        self.data_hr = vox_hr
        self.data_lr = vox_lr
        self.transform = transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_hr)
    def __getitem__(self,index):
        y = torch.from_numpy(self.data_hr[index,::,::,::,::].astype(np.float32)).permute(0,3,1,2)
        x = torch.from_numpy(self.data_lr[index,::,::,::,::].astype(np.float32)).permute(0,3,1,2)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x,y




