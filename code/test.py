
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
    def __init__(self,loader_test,loader_train,dataprep,file_R,cuda,device,model,crop=True):
        self.loader_test = loader_test
        self.loader_train=loader_train
        self.root_M = file_R
        self.dataprep=dataprep
        self.crop = crop
        self.fileC = torch.load(self.root_M,map_location='cpu')
        model.load_state_dict(self.fileC['model_state_dict'])
        self.losses_epoc = self.fileC['m_los']
        self.psnr = self.fileC['psnr']
        self.ssim = self.fileC['ssim']
        
        if cuda:
            model.to(device)
          
        self.model = model
        print(self.model)
        self.device = device
  
  
        self.loss = nn.MSELoss()     
        
        self.cuda = cuda


        self.test_all()
        if self.crop:
            self.reconstruct()
        


    def test_all(self):
        self.model.eval()

        loss_ts=[]
        psnr_ts=[]
        ssim_ts=[]
        self.scores = []
        self.targets=[]
        self.data=[]
            
        for batch_idx,(data,target) in tqdm.tqdm(enumerate(self.loader_test),total=len(self.loader_test),ncols=80,leave=False):


            if self.cuda:
                data,target = data.to(self.device),target.to(self.device)
            score = self.model(data)   
            loss = self.loss(score,target)
            loss_ts.append(loss.item())
            for k in range(0,score.shape[0]):  
                d = data[k,::,::,::,::] 
                t = target[k,::,::,::,::] 

                s = score[k,::,::,::,::]   
                    
                p,s = self.metrics(t.squeeze(),s.squeeze())
                psnr_ts.append(p)
                ssim_ts.append(s)
                
            d = data.squeeze().permute(1,2,0)
            d_cpu = d.cpu().data.numpy()
            t = target.squeeze().permute(1,2,0)
            t_cpu = t.cpu().data.numpy()
            s = score.squeeze().permute(1,2,0)
            s_cpu = s.cpu().data.numpy()
            
            
            self.data.append(d_cpu)
            
            self.scores.append(s_cpu)
            self.targets.append(t_cpu)  

        mean_psnr = np.mean(psnr_ts)
        mean_ssim = np.mean(ssim_ts)
        mean_loss = np.mean(loss_ts)
      
        
        print('\n Validation PSNR: ',str(mean_psnr))
        print('\n Validation SSIM: ',str(mean_ssim))
        print('\n Validation Loss: ',str(mean_loss)) 
        

    def vis_3(self,rep=3):
        from skimage.transform import resize 
        import matplotlib.pyplot as plt       
        
        for l in range(0,rep):
        
          ran_idx = np.random.randint(0,len(self.scores),3)
          print(ran_idx)
          fig, ax = plt.subplots(3,4)
          for ind,v in enumerate(ran_idx):
              ax[ind,0].imshow(self.data[v][::,25,::],cmap='gray')
  
              ax[ind,0].title.set_text('Input Data')
              ax[ind,0].axis('off')
  
              ax[ind,1].imshow(self.targets[v][::,25,::],cmap='gray')
              tt_psnr = psnr(self.targets[v],self.targets[v])
              ax[ind,1].title.set_text('Target Data: PSNR %.2f'%(tt_psnr))
              ax[ind,1].axis('off')
  
  
              ax[ind,2].imshow(self.scores[v][::,25,::],cmap='gray')
              st_psnr = psnr(self.targets[v],self.scores[v])
              ax[ind,2].title.set_text('Output Data: PSNR %.2f'%(st_psnr))
              ax[ind,2].axis('off')

              res = resize(self.data[v],output_shape=self.targets[v].shape,mode='symmetric',order=3)
              
              psnr2 = psnr(self.targets[v],res)
              ax[ind,3].imshow(res[::,25,::],cmap='gray')
              ax[ind,3].title.set_text('Interpolation: PSNR %.2f'%(psnr2))
              ax[ind,3].axis('off')



          plt.show()
    def plot_history_loss(self):
        import matplotlib.pyplot as plt

        X = [ (2,1,2,'Loss',self.losses_epoc), (2,2,1,'PSNR',self.psnr), (2,2,2,'SSIM',self.ssim)]
        
        for nrows, ncols, plot_number,title,data in X:
            ax=plt.subplot(nrows, ncols, plot_number)
            ax.plot(data)
            ax.title.set_text(title)
        plt.show()
    def vis(self):
        from skimage.transform import resize        
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,3)
        img_score = self.scores[65]
        img_data = self.data[65]
        img_target= self.targets[65]
        ax[2].imshow(img_score[::,15,::],cmap='gray')
        ax[1].imshow(img_data[::,15,::],cmap='gray')
        ax[0].imshow(img_target[::,15,::],cmap='gray')
        plt.show()
        ran_idx = np.random.randint(0,len(self.recons_org),4)
        for i,ra in enumerate(ran_idx):
          fig, ax = plt.subplots(2,2)
          psnr_d= psnr(self.recons_org[ra],self.recons_scores[ra])
          ax[0,0].imshow(self.recons_org[ra][::,50,::],cmap='gray')
          ax[0,0].axis('off')
          ax[0,0].title.set_text('Expected Output')
          ax[0,1].imshow(self.recons_scores[ra][::,50,::],cmap='gray')
          ax[0,1].axis('off')
          ax[0,1].title.set_text(('Output ResNet PSNR: %.2f dB')%(psnr_d))
          ax[1,0].imshow(self.recons_data[ra][::,50,::],cmap='gray')
          ax[1,0].axis('off')
          ax[1,0].title.set_text('Input Data')
          res = resize(self.recons_data[ra],output_shape=self.recons_org[ra].shape,mode='symmetric',order=3)
          psnr2 = psnr(self.recons_org[ra],res)
          ax[1,1].imshow(res[::,50,::],cmap='gray')
          ax[1,1].title.set_text('Interpolation: PSNR %.2f'%(psnr2))
          ax[1,1].axis('off')
          
          
          
        
          plt.show()
    def reconstruct(self):
        recons_test = self.dataprep.reconstruct(self.scores)
        recons_org = self.dataprep.reconstruct(self.targets)
        recons_input= self.dataprep.reconstruct(self.data)
        self.recons_scores = recons_test
        self.recons_org = recons_org
        self.recons_data = recons_input
        
    def test_1all(self):
        import matplotlib.pyplot as plt
        
      
            




           


       
    def metrics(self,true_img,pred_img):
        
        psnr_c = psnr(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        ssim_c = ssim(true_img.data.cpu().numpy(),pred_img.data.cpu().numpy())
        return psnr_c,ssim_c
        
    def test_all_models(self):
        print('not_yet')
    
# import  matplotlib.pyplot as plt
# X = [(2,2,3), (2,2,4) , (2,1,1)  ]
# for nrows, ncols, plot_number in X:
#     print(nrows)
#     print(ncols)
#     print(plot_number)
#     plt.subplot(nrows, ncols, plot_number)
# plt.show()








        




    