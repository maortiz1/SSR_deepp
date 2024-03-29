
import image_utils
import train
import model
import unet
import torchvision.transforms as t
import torch
from torch.utils import data
import os
import test
import numpy as np
import glob
import matplotlib.pyplot as plt
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
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--test",help="runs modely only on validation if model is saved",dest='test',action='store_true')
    parser.add_argument("-tr","--train",help="runs only train",action='store_true',dest='train')

    parser.add_argument("-m","--model",default='unet3d',help='model to use')
    parser.add_argument("-p","--pretrained",help="if model is pretrained",dest='pretrained',action='store_true')
    parser.add_argument("-f","--file",default = "ResNET_lr_0_0001_bt_5_rb_50/che_epoch_13.pth.tar",help="path where the pretrained model is for test or pretrained training")
    parser.add_argument("-o","--output_sz",default=(256,32,32),help="desire output size for training")
    parser.add_argument("-i","--images",default=os.path.join(os.getcwd(),'..','images'),help="folder that contains .nii files for training and validation (test) data")
    parser.add_argument("-nr","--n_resblock",default=50,help="Desire of number of resblocks")
    parser.add_argument("-cu","--cuda",default="2",help="if cuda available number of cuda desire to be used")
    parser.add_argument("-lr","--l_rate",default=0.0001,help="learning rate for training")
    parser.add_argument("-bt","--batch_size",default=3,help="Batch size for -bt ")
    parser.add_argument("-ft","--factor",default=3,help="Dowmsampling data for training")
    parser.add_argument("-pw","--pretWeights",action='store_true')
    parser.add_argument("-de","--demo",action='store_true')
    parser.add_argument("-con","--contt",action='store_true')
    parser.add_argument("-ex","--oneEx",action='store_true')
    
   # parser.add_argument("-af","--autof",action='store_true','')
   # parser.add_argument("-svf","--f_safe",help="folder to safe model")




    arguments = parser.parse_args()
    root = os.path.join(os.getcwd(),'..','images')
    print(arguments)
    n_resblock = 50
    
    

    if arguments.test:
        mode_tr=[]
        down_f=[]
        crop = True
        vox_size=[]
        factor=arguments.factor
        if arguments.model == 'ResNET':
          n_resblock = arguments.n_resblock
          out_size = arguments.output_sz
          mode_tr = model.ResNET(n_resblocks=n_resblock,output_size=out_size)
          down_f= image_utils.downsample
          vox_size = (32,32)
        elif arguments.model == 'ResNetIso':
          n_resblock=arguments.n_resblock
          mode_tr = model.ResNetIso(n_resblocks=n_resblock,res_scale=0.1)
          down_f = image_utils.downsample_isotropic
          vox_size = (32,32)
        elif arguments.model == 'unet3d':
          mode_tr = unet.Unet3D()
          print('in')
          down_f = image_utils.downsample_isotropic
          crop = True
          vox_size = (64,64)

        gpu = int(arguments.cuda)
        torch.cuda.set_device(gpu)
        device = 'cuda:%s'%(arguments.cuda)
        print(device)
        cuda = torch.cuda.is_available()

        dataprep = train.Data_Preparation(arguments.images,crop=crop,downfunction=down_f,factor=[int(factor)],vox_size=vox_size,train=False)
        #train dataset
        lr_train_vox = dataprep.lr_pcs_val
        hr_train_vox = dataprep.hr_pcs_val
        trainDataset = train.Dataset(hr_train_vox,lr_train_vox )       
        bt_size = int(arguments.batch_size)
        shuffle = True
        train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=False)
        #tEST
        lr_test = dataprep.lr_pcs_ts
        hr_test = dataprep.hr_pcs_ts
        val_Dataset = train.Dataset(hr_test,lr_test)
        val_data_loader = data.DataLoader(val_Dataset,batch_size=bt_size,shuffle=False)
        file = arguments.file
        
        
        test = test.Test(val_data_loader,train_data_loader,dataprep,file,cuda,device,mode_tr)
        
       # test.vis_3()
        #test.plot_history_loss()
        #test.vis()
        test.test_best()
  
    elif arguments.train:
      mode_tr=[]
      down_f=[]
      crop = True
      vox_size=[]
      factor = int(arguments.factor)
      if arguments.model == 'ResNET':
        n_resblock = arguments.n_resblock
        out_size = arguments.output_sz
        mode_tr = model.ResNET(n_resblocks=n_resblock,output_size=out_size)
        down_f= image_utils.downsample
        vox_size = (32,32)
        
      elif arguments.model == 'ResNetIso':
        n_resblock=arguments.n_resblock
        mode_tr = model.ResNetIso(n_resblocks=n_resblock,res_scale=0.1)
        down_f = image_utils.downsample_isotropic
        vox_size = (32,32)
      elif arguments.model == 'unet3d':
        mode_tr = unet.Unet3D()

        down_f = image_utils.downsample_isotropic
        crop = True
        vox_size = (64,64)
      if arguments.pretWeights:
        file = arguments.file
        mode_tr.load_state_dict(torch.load(file)['model_state_dict'])

      gpu = int(arguments.cuda)
      torch.cuda.set_device(gpu)
      device = 'cuda:%s'%(arguments.cuda)
      print(device)
      cuda = torch.cuda.is_available()

      dataprep = train.Data_Preparation(arguments.images,crop=crop,factor=[factor],downfunction=down_f,vox_size=vox_size,test=False)
      #train dataset
      lr_train_vox = dataprep.lr_pcs_tr
      hr_train_vox = dataprep.hr_pcs_tr
      trainDataset = train.Dataset(hr_train_vox,lr_train_vox )       
      bt_size = int(arguments.batch_size)
      shuffle = True
      train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=shuffle)
      #tEST
      lr_test = dataprep.lr_pcs_val
      hr_test = dataprep.hr_pcs_val
      val_Dataset = train.Dataset(hr_test,lr_test)
      val_data_loader = data.DataLoader(val_Dataset,batch_size=bt_size,shuffle=False)

      out_f= 'all_%s_lr_%s_bt_%d_rb_%d_ft_%d'%(arguments.model,str(arguments.l_rate).replace('.','_'),bt_size,n_resblock,factor)

      if cuda:
        mode_tr = mode_tr.to(device)
   


      trainer = train.Trainer(train_data_loader,val_data_loader,cuda,3,mode_tr,float(arguments.l_rate),out_f,device)
      max_epoch = 1000
      trainer.train(max_epoch)

    elif arguments.contt:
      mode_tr=[]
      down_f=[]
      file = []
      epoch_AC=[]
      optim_state_dic=[]
      factor = int(arguments.factor)
      if arguments.model == 'ResNET':
        n_resblock = arguments.n_resblock
        out_size = arguments.output_sz
        mode_tr = model.ResNET(n_resblocks=n_resblock,output_size=out_size)
        down_f= image_utils.downsample
        file = torch.load(arguments.file)
        mode_tr.load_state_dict(file['model_state_dict'])
        
        epoch_AC = file['epoch']

      elif arguments.model == 'ResNetIso':
        n_resblock=arguments.n_resblock
        mode_tr = model.ResNetIso(n_resblocks=n_resblock,res_scale=0.1)
        down_f = image_utils.downsample_isotropic
        file =  torch.load(arguments.file)
        mode_tr.load_state_dict(file['model_state_dict'])
        
        epoch_AC = file['epoch']

      gpu = int(arguments.cuda)
      torch.cuda.set_device(gpu)
      device = 'cuda:%s'%(arguments.cuda)
      print(device)
      cuda = torch.cuda.is_available()

      dataprep = train.Data_Preparation(arguments.images,crop=crop,factor=[factor],downfunction=down_f,vox_size=vox_size,test=False)
      #train dataset
      lr_train_vox = dataprep.lr_pcs_tr
      hr_train_vox = dataprep.hr_pcs_tr
      trainDataset = train.Dataset(hr_train_vox,lr_train_vox )       
      bt_size = 5
      shuffle = True
      train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=shuffle)
     
      lr_test = dataprep.lr_pcs_val
      hr_test = dataprep.hr_pcs_val
      testDataset = train.Dataset(hr_test,lr_test)
      test_data_loader = data.DataLoader(testDataset,batch_size=bt_size,shuffle=False)
      out_f= 'Continue_%s_lr_%s_bt_%d_rb_sc%d'%(arguments.model,str(arguments.l_rate).replace('.','_'),bt_size,n_resblock)




      if cuda:
        mode_tr = mode_tr.to(device)

      trainer = train.Trainer(train_data_loader,test_data_loader,cuda,3,mode_tr,float(arguments.l_rate),out_f,device,epoch=epoch_AC,pretrained=True,file=file)
      max_epoch = 1000
      trainer.train(max_epoch)

    elif arguments.demo:
      import nibabel as nib
      images = arguments.images
      file =arguments.file
      if arguments.model == 'ResNET':
        n_resblock = arguments.n_resblock
        out_size = arguments.output_sz
        mode_tr = model.ResNET(n_resblocks=n_resblock,output_size=out_size)
        down_f= image_utils.downsample
        vox_size = (32,32)
      elif arguments.model == 'ResNetIso':
        n_resblock=arguments.n_resblock
        mode_tr = model.ResNetIso(n_resblocks=n_resblock,res_scale=0.1)
        down_f = image_utils.downsample_isotropic
        vox_size = (32,32)
      elif arguments.model == 'unet3d':
        mode_tr = unet.Unet3D()

        down_f = image_utils.downsample_isotropic
        crop = True
        vox_size = (64,64)
      gpu = int(arguments.cuda)
      torch.cuda.set_device(gpu)
      device = 'cuda:%s'%(arguments.cuda)
      
      cuda = torch.cuda.is_available()
      file_m = torch.load(file,map_location='cpu')
      mode_tr.load_state_dict(file_m['model_state_dict'])
      if cuda:
        mode_tr=mode_tr.to(device)

      fds = glob.glob(os.path.join(images,'*.nii.gz'))
      import matplotlib.pyplot as plt
      for fa in fds:
        print(fa)
        fi= nib.load(fa)
        
        data = fi.get_fdata()
        data_in = image_utils.upsample_factor(data)
        data_in = abs(np.fft.ifftn(np.fft.ifftshift(np.fft.fftshift(np.fft.fftn(data_in)))))
        data_inwh = nib.nifti1.Nifti1Image(data_in,np.eye(4))
        
        data_in_wh = image_utils.normalize_image_whitestripe(data_inwh,contrast='T2')
        pcs,n_pz_x,n_pz_y = image_utils.cropall(data_in_wh,vox_size=(64,64))
        scr=[]
        for img in pcs:
          data_crop = np.expand_dims(img,axis=0)
          x = torch.from_numpy(np.expand_dims(data_crop,axis=0).astype(np.float32)).permute(0,1,4,2,3)
          if cuda:
            
            x= x.to(device)
          score = mode_tr(x)
          s = score.squeeze().permute(1,2,0)
          s_cpu = s.cpu().data.numpy()
          scr.append(s_cpu)
        recons = image_utils.reconstruct_npz(scr,[[n_pz_x,n_pz_y]])
       
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(data_in_wh[::,200,::],cmap='gray')
        axes[1].imshow(recons[0][::,200,::],cmap='gray')
        name = fa.split('/')[-1]
        name = name.split('.')[0]
        name = "test_"+name+'.nii.gz'
        file = os.path.join(os.getcwd(),name)
        print(len(recons))
        nib_file = nib.nifti1.Nifti1Image(recons[0],np.eye(4))
        nib.save(nib_file,file)
        # plt.show()
        recons = image_utils.reconstruct_npz(pcs,[[n_pz_x,n_pz_y]])
        nib_file = nib.nifti1.Nifti1Image(recons[0],np.eye(4))
        name = fa.split('/')[-1]
        name = name.split('.')[0]
        name = "test_down_"+name+'.nii.gz'
        file = os.path.join(os.getcwd(),name)
        nib.save(nib_file,file)

        data_in_wh = image_utils.normalize_image_whitestripe(fi,contrast='T2')
        pcs,n_pz_x,n_pz_y = image_utils.cropall(data_in_wh,vox_size=(64,64))
        recons = image_utils.reconstruct_npz(pcs,[[n_pz_x,n_pz_y]])
        nib_file = nib.nifti1.Nifti1Image(recons[0],np.eye(4))
        name = fa.split('/')[-1]
        name = name.split('.')[0]
        name = "test_ori"+name+'.nii.gz'
        file = os.path.join(os.getcwd(),name)
        nib.save(nib_file,file)
    elif arguments.pretrained:
      mode_tr=[]
      down_f=[]
      file = []
      epoch_AC=[]
      optim_state_dic=[]
      factor = int(arguments.factor)
      if arguments.model == 'ResNET':
        n_resblock = arguments.n_resblock
        out_size = arguments.output_sz
        mode_tr = model.ResNET(n_resblocks=n_resblock,output_size=out_size)
        down_f= image_utils.downsample
        file = torch.load(arguments.file,map_location='cpu')
        mode_tr.load_state_dict(file['model_state_dict'])
        del file
        
        epoch_AC = file['epoch']

      elif arguments.model == 'ResNetIso':
        n_resblock=arguments.n_resblock
        mode_tr = model.ResNetIso(n_resblocks=n_resblock,res_scale=0.1)
        down_f = image_utils.downsample_isotropic
        file =  torch.load(arguments.file,map_location='cpu')
        mode_tr.load_state_dict(file['model_state_dict'])
        del file
        
        epoch_AC = file['epoch']
      elif arguments.model == 'unet3d':
        mode_tr = unet.Unet3D()
        file =  torch.load(arguments.file,map_location='cpu')
        mode_tr.load_state_dict(file['model_state_dict'])
        del file
        down_f = image_utils.downsample_isotropic
        crop = True
        vox_size = (64,64)

      gpu = int(arguments.cuda)
      torch.cuda.set_device(gpu)
      device = 'cuda:%s'%(arguments.cuda)
      print(device)
      cuda = torch.cuda.is_available()

      dataprep = train.Data_Preparation(arguments.images,crop=crop,factor=[factor],downfunction=down_f,vox_size=vox_size,test=False)
      #train dataset
      lr_train_vox = dataprep.lr_pcs_tr
      hr_train_vox = dataprep.hr_pcs_tr
      trainDataset = train.Dataset(hr_train_vox,lr_train_vox )       
      bt_size = int(arguments.batch_size)
      shuffle = True
      train_data_loader = data.DataLoader(trainDataset,batch_size=bt_size,shuffle=shuffle)
     
      lr_test = dataprep.lr_pcs_val
      hr_test = dataprep.hr_pcs_val
      testDataset = train.Dataset(hr_test,lr_test)
      test_data_loader = data.DataLoader(testDataset,batch_size=bt_size,shuffle=shuffle)
      out_f= 'PretrainedW_%s_lr_%s_bt_%d_rb_fa%d'%(arguments.model,str(arguments.l_rate).replace('.','_'),bt_size,factor)
      
      




      if cuda:
        mode_tr=mode_tr.to(device)

      trainer = train.Trainer(train_data_loader,test_data_loader,cuda,3,mode_tr,float(arguments.l_rate),out_f,device)
      max_epoch = 1000
      trainer.train(max_epoch)        
    if arguments.oneEx:
      import nibabel as nib
      
      file =arguments.file
      if arguments.model == 'ResNET':
        n_resblock = arguments.n_resblock
        out_size = arguments.output_sz
        mode_tr = model.ResNET(n_resblocks=n_resblock,output_size=out_size)
        down_f= image_utils.downsample
        vox_size = (32,32)
      elif arguments.model == 'ResNetIso':
        n_resblock=arguments.n_resblock
        mode_tr = model.ResNetIso(n_resblocks=n_resblock,res_scale=0.1)
        down_f = image_utils.downsample_isotropic
        vox_size = (32,32)
      elif arguments.model == 'unet3d':
        mode_tr = unet.Unet3D()

        down_f = image_utils.downsample_isotropic
        crop = True
        vox_size = (64,64)
      gpu = int(arguments.cuda)
      torch.cuda.set_device(gpu)
      device = 'cuda:%s'%(arguments.cuda)
      
      cuda = torch.cuda.is_available()
      file_m = torch.load(file,map_location='cpu')
      mode_tr.load_state_dict(file_m['model_state_dict'])
      if cuda:
        mode_tr=mode_tr.to(device)


      image = arguments.images
      data_inwh= nib.load(image)  
      data_in = abs(np.fft.ifftn(np.fft.ifftshift(np.fft.fftshift(np.fft.fftn(data_inwh.get_fdata())))))
      # data_inwh = nib.nifti1.Nifti1Image(data_in,np.eye(4))
      data_in_wh = image_utils.normalize_image_whitestripe(data_inwh,contrast='T2')
      

      from skimage.transform import resize
      sz= data_in_wh.shape
      sz_out = (sz[0],sz[1],sz[2]*int(arguments.factor))
      res = resize(data_in_wh,sz_out,mode='symmetric',order=3)
      print(res.shape)
      
      pcs,n_pz_x,n_pz_y,n_pz_z = image_utils.cropall3(res,vox_size=(64,64,256))
      scr=[]
      for img in pcs:
        data_crop = np.expand_dims(img,axis=0)
        x = torch.from_numpy(np.expand_dims(data_crop,axis=0).astype(np.float32)).permute(0,1,4,2,3)
        if cuda:       
          x= x.to(device)
        score = mode_tr(x)
        s = score.squeeze().permute(1,2,0)
        s_cpu = s.cpu().data.numpy()
        plt.imshow(s_cpu[::,30,::],cmap='gray')       
        plt.show() 
        scr.append(s_cpu)
      recons = image_utils.reconstruct_npz2(scr,[[n_pz_x,n_pz_y,n_pz_z]])
      nib_file = nib.nifti1.Nifti1Image(recons[0],np.eye(4))
      name = image.split('/')[-1]
      name = name.split('.')[0]
      name = "oneEx_down_"+name+'.nii'
      file = os.path.join(os.getcwd(),name)
      nib.save(nib_file,file)

    










      
