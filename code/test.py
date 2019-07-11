
import os
import numpy as np
import torch
import model
import train
import image_utils
from torch.utils import data
import tqdm

class Test():
    def __init__(self, loader_test, file_R,cuda,device,vox_size,out_put,model):
        self.loader_test = loader_test
        self.root_M = file_R
        self.fileC = torch.load(self.root_M)
        model.load_state_dict(self.fileC['model_state_dict'])  
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
        psnrs=[]

        for batch_idx,(data,target) in tqdm.tqdm(enumerate(self.loader_test),total=len(self.loader_test)):
            if cuda:
                data, target = data.to(self.device), target.to(self.device)
            
            score = self.model(data)
            t = target[0,::,::,::,::]
            s= score[0,::,::,::,::]
            p,s = train.metrics(t.squeeze(),s.squeeze())

            score_or = score.squeeze().permute(1,2,0)

            score_cpu = score_or.cpu().data.numpy()
            psnrs.append(psnrs)
            
            all[batch_idx,::,::,::]= score_cpu

        self.all = all
        self.psnrs = psnrs
        print('Mean Psnr is: ',np.mean(psnrs))
    def vis_3(self):
       import matplotlib.pyplot as plt
       print('great')
               

            




        




gpu = 2
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.cuda.set_device(gpu)
device = 'cuda:2'
cuda = torch.cuda.is_available()

root = os.path.join(os.getcwd(),'..','images')
bt_size = 1
shuffle = False
dataprep = train.Data_Preparation(root)
hr_test = dataprep.arr_hr_pieces_test
lr_test = dataprep.arr_lr_pieces_test
testDataset = train.Dataset(hr_test,lr_test,transform=image_utils.normalize)
testloader = data.DataLoader(testDataset,batch_size=bt_size,shuffle=shuffle)
file_r='chkpt_r_50_bt_8/che_epoch_237.pth.tar'
output_size = (32,32,256)

n_resblock =50
output_sz =[ (*(testDataset[1][1]).squeeze().size())]

output_sz = (testDataset[1][1]).squeeze().size()
ResNet = model.ResNET(n_resblocks=n_resblock,scale=3,output_size=output_sz,res_scale=0.1)
test = Test(testloader,file_r,cuda,device,(32,32),output_size,ResNet)
top= test.all[50,::,::,::]
import matplotlib.pyplot as plt
plt.figure
plt.imshow(top[::,15,::])
plt.show()
print(test.all.shape)

a = dataprep.reconstr_test(test.all)
plt.imshow(a[0][::,40,::])
plt.show()



    