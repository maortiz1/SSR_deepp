import image_utils
import train
import model
import torch
from torch.utils import data
import os
def main():
 
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    n_resblock = 26
    root = os.path.join(os.getcwd(),'..','images')
    dataprep = train.Data_Preparation(root)
    lr_train_vox = dataprep.lr_train_vox
    hr_train_vox = dataprep.hr_train_vox
    output_sz = hr_train_vox[1,::,::,::,::].squeeze()

    trainDataset = train.Dataset(hr_train_vox,lr_train_vox,transform=image_utils.normalize)
    output_sz = (trainDataset[1][1]).squeeze().size()
    bt_size = 5
    shuffle = True
    train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=shuffle)
    out_f = 'chkpt'
    
    print(output_sz)
    lr_test = dataprep.test_lr_data
    hr_test = dataprep.test_hr_data
    testDataset = train.Dataset(hr_test,lr_test,transform=image_utils.normalize)
    test_data_loader = data.DataLoader(testDataset,batch_size=bt_size,shuffle=shuffle)
    ResNet = model.ResNET(n_resblocks=n_resblock,scale=3,output_size=output_sz,res_scale=0.1)
    lr = 0.001
    if cuda:
      ResNet.to('cuda')
    trainer = train.Trainer(train_data_loader,test_data_loader,cuda,3,ResNet,lr,out_f)
    max_epoch = 1000
    trainer.train(max_epoch)

if __name__=='__main__':
    main()