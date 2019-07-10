
import os
import numpy as np
import torch
import model
import train
import image_utils
from torch.utils import data
import tqdm
class Test():
    def __init__(self, loader_test, file_R,cuda,device,vox_size,out_put):
        self.loader_test = loader_test
        self.root_M = file_R
        self.fileC = torch.load(self.root_M)
        self.model = self.fileC['model']
        self.device = device
        self.vox_size=vox_size
        self.output = out_put


    def test_all(self):
        all = np.empty((len(self.loader_test),self.output[0],self.output[1],self.output[2],self.output[3]))
        for batch_idx,(data,target) in tqdm.tqdm(enumerate(self.loader_test),total=len(self.loader_test)):
            if cuda:
                data, target = data.to(self.device), target.to(self.device)
            
            score = self.model(data)
            all[batch_idx*self.loader_test.batch_size:batch_idx*self.loader_test.batch_size+self.loader_test.batch_size,::,::,::,::]= score
            

            




        



if __name__ == '__test__':
     
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.cuda.set_device(gpu)
    device = 'cuda:0'
    cuda = torch.cuda.is_available()
    root = os.path.join(os.getcwd(),'images')
    bt_size = 2
    shuffle = False
    dataprep = train.Data_Preparation(root)
    hr_test = dataprep.arr_hr_pieces_test
    lr_test = dataprep.arr_lr_pieces_test
    testDataset = train.Dataset(hr_test,lr_test,transform=image_utils.normalize)
    testloader = data.DataLoader(testDataset,batch_size=bt_size,shuffle=shuffle)
    file_r=''
    output_size = (1,32,32,256)
    test = Test(testloader,file_r,cuda,device,(32,32),output_size)