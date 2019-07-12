
import os
import numpy as np
import torch
import model
import train
import image_utils
from torch.utils import data
import tqdm
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
class Test():
    def __init__(self, loader_test, file_R,cuda,device,vox_size,out_put,model):
        self.loader_test = loader_test
        self.root_M = file_R
        self.fileC = torch.load(self.root_M)
        model.load_state_dict(self.fileC['model_state_dict'])
        self.losses_epoc = self.fileC['m_los']
        if cuda:
            model.to(device)
          
        self.model = model
        self.device = device
        self.vox_size=vox_size
        self.output = out_put
     
    
        self.test_all()



    def test_all(self):

        import train
        all = np.empty((len(self.loader_test),self.output[0],self.output[1],self.output[2]))
        all_target = np.empty((len(self.loader_test),self.output[0],self.output[1],self.output[2]))
        all_data = np.empty((len(self.loader_test),self.output[0],self.output[1],85))
        psnrs=[]

        for batch_idx,(data,target) in tqdm.tqdm(enumerate(self.loader_test),total=len(self.loader_test)):
            if cuda:
                data, target = data.to(self.device), target.to(self.device)
         
            score = self.model(data)
            
            for k in range(0,self.loader_test.batch_size):   
                t = target[k,::,::,::,::] 
    
                s = score[k,::,::,::,::]   
                    
                p,s = self.metrics(t.squeeze(),s.squeeze())
                psnrs.append(p)           
           
           
           
            score_or = score.squeeze().permute(1,2,0)

            score_cpu = score_or.cpu().data.numpy()

            
            all[batch_idx,::,::,::]= score_cpu
            
            target_or = target.squeeze().permute(1,2,0)

            target_cpu = target_or.cpu().data.numpy()
            all_target[batch_idx,::,::,::]= target_cpu
            
                        
            data_or = data.squeeze().permute(1,2,0)

            data_cpu = data_or.cpu().data.numpy()
            all_data[batch_idx,::,::,::]= data_cpu



        self.all = all
        self.psnrs = psnrs
        self.all_target = all_target
        self.all_data = all_data
        
        print('Mean Psnr is: ',np.mean(psnrs))
    def vis_3(self):
       import matplotlib.pyplot as plt
       print('great')
       
    def metrics(self,true_img,pred_img):
        
        psnr_c = psnr(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        ssim_c = ssim(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        return psnr_c,ssim_c
        
    def test_all_models(self):
        print('not_yet')
    


            




        




gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.cuda.set_device(gpu)
device = 'cuda:1'
cuda = torch.cuda.is_available()

root = os.path.join(os.getcwd(),'..','images')
print(root)
bt_size = 1
shuffle = False
dataprep = train.Data_Preparation(root)
hr_test = dataprep.arr_hr_pieces_test
lr_test = dataprep.arr_lr_pieces_test
testDataset = train.Dataset(hr_test,lr_test,transform=image_utils.normalize)
testloader = data.DataLoader(testDataset,batch_size=bt_size,shuffle=shuffle)
#file_r='chkpt_r_52_bt_8_lr_0_001_res_0_1/che_epoch_24.pth.tar'
file_r='chkpt_r_52_bt_8_lr_0_001_res_0_1_V2/che_epoch_769.pth.tar'
output_size = (32,32,256)

n_resblock =52
output_sz =[ (*(testDataset[1][1]).squeeze().size())]

output_sz = (testDataset[1][1]).squeeze().size()
ResNet = model.ResNET(n_resblocks=n_resblock,scale=3,output_size=output_sz,res_scale=0.1)
test = Test(testloader,file_r,cuda,device,(32,32),output_size,ResNet)
top= test.all
hr = test.all_target
lr = test.all_data

import matplotlib.pyplot as plt


print(lr.shape)

fig,axes = plt.subplots(3,3)
axes[0,0].imshow(lr[50,::,15,::],cmap='gray')
axes[0,1].imshow(hr[50,::,15,::],cmap='gray')
axes[0,2].imshow(top[50,::,15,::],cmap='gray')
axes[1,0].imshow(lr[51,::,15,::],cmap='gray')
axes[1,1].imshow(hr[51,::,15,::],cmap='gray')
axes[1,2].imshow(top[51,::,15,::],cmap='gray')
axes[2,0].imshow(lr[52,::,15,::],cmap='gray')
axes[2,1].imshow(hr[52,::,15,::],cmap='gray')
axes[2,2].imshow(top[52,::,15,::],cmap='gray')


plt.show()




a = dataprep.reconstr_test(test.all)
plt.imshow(a[1][::,120,::],cmap='gray')
plt.show()


l = test.losses_epoc

plt.plot(l,'ro')
plt.show()




    