
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
import torch.nn as nn
class Test():
    """
    Class that allows testing a pretraned model
    """
    def __init__(self, loader_test,loader_train, file_R,cuda,device,vox_size,model):
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
  
        self.loss = nn.MSELoss()     
        self.loader_train = loader_train
        self.cuda = cuda


        self.test_all()



    def test_all(self):
        self.model.eval()
        loss_ts=[]
        psnr_ts=[]
        ssim_ts=[]
        self.scores = []
        self.targets=[]
        self.data=[]
            
        for batch_idx,(data,target) in tqdm.tqdm(enumerate(self.loader_test),total=len(self.loader_test),desc='Test epoch %d'%self.ac_epoch,ncols=80,leave=False):


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
                
            self.data.append(data.data.cpu().numpy())  
            self.scores.append(score.data.cpu().numpy())
            self.targets.append(target.data.cpu().numpy())  

        mean_psnr = np.mean(psnr_ts)
        mean_ssim = np.mean(ssim_ts)
        mean_loss = np.mean(loss_ts)
      
        
        print('\n Validation PSNR: ',str(mean_psnr))
        print('\n Validation SSIM: ',str(mean_ssim))
        print('\n Validation Loss: ',str(mean_loss)) 
        

    def vis_3(self):
        import matplotlib.pyplot as plt       

        ran_idx = np.random.randint(0,len(self.scores),3)
        fig, ax = plt.subplots(3,3)
        for ind,v in enumerate(ran_idx):
            ax[ind,0].imshow(self.data[v][::,16,::],cmap='gray')
            in_psnr = psnr(self.targets[v],self.data[v])
            ax[ind,0].title.set_text('Input Data: PSNR %.2f'%(in_psnr))
            ax[ind,0].axis('off')

            ax[ind,1].imshow(self.targets[v][::,16,::],cmap='gray')
            tt_psnr = psnr(self.targets[v],self.targets[v])
            ax[ind,1].title.set_text('Target Data: PSNR %.2f'%(tt_psnr))
            ax[ind,1].axis('off')


            ax[ind,2].imshow(self.scores[v][::,16,::],cmap='gray')
            st_psnr = psnr(self.targets[v],self.scores[v])
            ax[ind,2].title.set_text('Output Data: PSNR %.2f'%(st_psnr))
            ax[ind,2].axis('off')
        plt.show()


           


       
    def metrics(self,true_img,pred_img):
        
        psnr_c = psnr(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        ssim_c = ssim(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        return psnr_c,ssim_c
        
    def test_all_models(self):
        print('not_yet')
    


            




        




    