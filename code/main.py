
import image_utils
import train
import model
import torch
from torch.utils import data
import os
import test


def main():
 
    gpu = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.cuda.set_device(gpu)
    device = 'cuda:2'
    cuda = torch.cuda.is_available()

    n_resblock = 52
    root = os.path.join(os.getcwd(),'..','images')
    dataprep = train.Data_Preparation(root)
    lr_train_vox = dataprep.lr_pcs_tr
    hr_train_vox = dataprep.hr_pcs_tr
    output_sz = hr_train_vox[0].squeeze()

    trainDataset = train.Dataset(hr_train_vox,lr_train_vox,transform=image_utils.normalize)
    output_sz = (256,32,32)

    bt_size = 9
    shuffle = True
    train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=shuffle)
    out_f = 'chkpt_r_52_bt_9_lr_0_001_res_0_1_sch_steplr'
    

    lr_test = dataprep.lr_pcs_ts
    hr_test = dataprep.hr_pcs_ts
    testDataset = train.Dataset(hr_test,lr_test,transform=image_utils.normalize)
    test_data_loader = data.DataLoader(testDataset,batch_size=bt_size,shuffle=False)
    ResNet = model.ResNET(n_resblocks=n_resblock,scale=3,output_size=output_sz,res_scale=0.1)
    
    lr = 0.001
    #pretrained
    if cuda:
      ResNet.to(device)

    
    
    
    trainer = train.Trainer(train_data_loader,test_data_loader,cuda,3,ResNet,lr,out_f,device)
    max_epoch = 1000
    trainer.train(max_epoch)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--test",help="runs modely only on validation if model is saved",dest='test',action='store_true')
    parser.add_argument("-tr","--train",help="runs only train",action='store_true',dest='train')
    parser.add_argument("-m","--model",default='ResNET',help='model to use')
    parser.add_argument("-p","--pretrained",help="if model is pretrained",dest='pretrained',action='store_true')
    parser.add_argument("-f","--file",default = "chkpt_r_52_bt_9_lr_0_001_res_0_1_sch_steplr/che_epoch_66.pth.tar",help="path where the pretrained model is for test or pretrained training")

    arguments = parser.parse_args()
    root = os.path.join(os.getcwd(),'..','images')
    dataprep = train.Data_Preparation(root)
    n_resblock = 52
    

    if arguments.test:
        file = arguments.file
        model = arguments.model


        bt_size = 9
        shuffle = True
        lr_train_vox = dataprep.lr_pcs_tr
        hr_train_vox = dataprep.hr_pcs_tr
        trainDataset = train.Dataset(hr_train_vox,lr_train_vox,transform=image_utils.normalize)
        train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=shuffle)


        lr_test = dataprep.lr_pcs_ts
        hr_test = dataprep.hr_pcs_ts
        test_dataset = train.Dataset(hr_test,lr_test,transform=image_utils.normalize)
        test_data_loader = data.DataLoader(test_dataset,batch_size=bt_size,shuffle=False)
        

        gpu = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        torch.cuda.set_device(gpu)
        device = 'cuda:2'
        cuda = torch.cuda.is_available()

        if arguments.model == 'ResNet':
          ResNet = model.ResNET(n_resblocks=n_resblock,scale=3,output_size=output_sz,res_scale=0.1)
        test = test.Test(test_data_loader,train_data_loader,file,cuda,device,)
        
    else:   

      main()