
import image_utils
import train
import model
import torch
from torch.utils import data
import os
import test


def main():
 
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.cuda.set_device(gpu)
    device = 'cuda:0'
    cuda = torch.cuda.is_available()

    n_resblock = 10
    root = os.path.join(os.getcwd(),'..','images')
    dataprep = train.Data_Preparation(root)
    lr_train_vox = dataprep.lr_pcs_tr
    hr_train_vox = dataprep.hr_pcs_tr
    output_sz = hr_train_vox[0].squeeze()

    trainDataset = train.Dataset(hr_train_vox,lr_train_vox,transform=image_utils.normalize)
    output_sz = (256,32,32)

    bt_size = 15
    shuffle = True
    train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=shuffle)
    out_f = 'chkpt_r_10_bt_15_lr_0_001_res_0_5_sch'
    

    lr_test = dataprep.lr_pcs_ts
    hr_test = dataprep.hr_pcs_ts
    testDataset = train.Dataset(hr_test,lr_test,transform=image_utils.normalize)
    test_data_loader = data.DataLoader(testDataset,batch_size=bt_size,shuffle=False)
    ResNet = model.ResNET(n_resblocks=n_resblock,output_size=output_sz,res_scale=0.5)
    
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
    parser.add_argument("-f","--file",default = "chkpt_r_52_bt_9_lr_0_001_res_0_1_sch/che_epoch_2.pth.tar",help="path where the pretrained model is for test or pretrained training")
    parser.add_argument("-o","--output_sz",default=(256,32,32),help="desire output size for training")
    parser.add_argument("-i","-images",default=os.path.join(os.getcwd(),'..','images'),help="folder that contains .nii files for training and validation (test) data")
    parser.add_argument("-nr","--n_resblock",default=52,help="Desire of number of resblocks")
    parser.add_argument("-cu","cuda",default="2",help="if cuda available number of cuda desire to be used")


    arguments = parser.parse_args()
    root = os.path.join(os.getcwd(),'..','images')
    print(arguments)
    n_resblock = 52
    
    

    if arguments.test:
        file = arguments.file

        dataprep = train.Data_Preparation(root)
        


        bt_size = 1
        shuffle = True
        lr_train_vox = dataprep.lr_pcs_tr
        hr_train_vox = dataprep.hr_pcs_tr
        trainDataset = train.Dataset(hr_train_vox,lr_train_vox)
        train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=shuffle)


        lr_test = dataprep.lr_pcs_ts
        hr_test = dataprep.hr_pcs_ts
        test_dataset = train.Dataset(hr_test,lr_test)
        test_data_loader = data.DataLoader(test_dataset,batch_size=bt_size,shuffle=False)
        

        gpu = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        torch.cuda.set_device(gpu)
        device = 'cuda:2'
        cuda = torch.cuda.is_available()



        if arguments.model == 'ResNET':
          ResNet = model.ResNET(n_resblocks=n_resblock,output_size=arguments.output_sz,res_scale=0.5)
          test = test.Test(test_data_loader,train_data_loader,dataprep,file,cuda,device,ResNet)
          
          test.vis_3()
          test.plot_history_loss()
          test.vis()
  
    elif arguments.train:
      if arguments.model == 'ResNET':
        n_resblock = arguments.n_resblock
        out_size = arguments.output_sz
        ResNet = model.ResNET(n_resblocks=n_resblock,output_size=out_size)

      main()