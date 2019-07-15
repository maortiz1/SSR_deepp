
#Low resolution of image
def downsample(img,down_factor=3):
    import numpy as np
    import math
    
    fft_imag = np.fft.fftn(img)#n-fourier transform
    shift_fft_imag = np.fft.fftshift(fft_imag)#fft shifting(zero-frequency component to the center of the spectrum)
    z_size= shift_fft_imag.shape[2]#z_dimension size for cropping
    size = math.floor(z_size/(down_factor))#size of part according to down factor
    center = round(z_size/2)-1 #center for cutting
    if size%2==0:
        crop_ffs = shift_fft_imag[::,::,int(center-(size/2)):int(center+(size/2))]
    else:
        size = size+1        
        crop_ffs = shift_fft_imag[::,::,int(center-(size/2)+1):int(center+(size/2))]
    #crop_ffs = shift_fft_imag[::,::,center-size:center+size]#cropping kspace
    shift_ifft_crop_ffs = np.fft.ifftshift(crop_ffs)#inverse shiftt
    ifft_crop_ffs = np.fft.ifftn(shift_ifft_crop_ffs)#inverse fast fourier transform

    return abs(ifft_crop_ffs)#returns cropped version of the original image 


def normalize_image_whitestripe(img,contrast= 'T1'):
    from intensity_normalization.normalize import whitestripe

    mask = whitestripe.whitestripe(img,contrast)
    norm_image = whitestripe.whitestripe_norm(img,mask)
    return norm_image


#n number of sized voxels from inserted volume
def voxelize_image(img_lr,img_hr,vox_size,n_samples=40):
    import numpy as np
    import math
    size_x = img_lr.shape[0]
    size_y = img_lr.shape[1]
    # size_z = img.shape[2]

    voxels_lr = np.empty((n_samples,vox_size[0],vox_size[1],img_lr.shape[2]))
    voxels_hr = np.empty((n_samples,vox_size[0],vox_size[1],img_hr.shape[2]))
    if ((size_x/vox_size[0])<1) or ((size_y/vox_size[1]) <1):
        raise Exception('The size of the desired voxel is to big')
    
    for i in range(0,n_samples):
        r_x = int(np.floor(size_x-vox_size[0])* np.random.rand(1))
        r_y = int(np.floor(size_y-vox_size[1])* np.random.rand(1))
        # r_z = int(np.floor(img.shape[1]-vox_size[2])* np.random.rand(1))
        crop_lr = img_lr[r_x:r_x + vox_size[0],r_y:r_y+vox_size[1],::]
        voxels_lr[i,::,::,::] = crop_lr

        crop_hr = img_hr[r_x:r_x + vox_size[0],r_y:r_y+vox_size[1],::]
        voxels_hr[i,::,::,::] = crop_hr
    return voxels_lr,voxels_hr

def normalize(img):
    import numpy as np
    import torch
    maxA = img.max()
    minA = img.min()

    norm_img = (img - minA)/(maxA-minA)
    return norm_img



def cropall(img,vox_size):
    import numpy as np
    import math

    size_x = img.shape[0]
    size_y = img.shape[1]
    size_z = img.shape[2]
    v_x = vox_size[0]
    v_y = vox_size[1]

    n_pz_x = int(np.floor(size_x/v_x))
    n_pz_y = int(np.floor(size_y/v_x))
    res_x = size_x - n_pz_x*v_x
    res_y = size_y - n_pz_y*v_y

    if res_x%2 == 0:
        beg_x = int(res_x/2)
    else:
        beg_x = np.ceil(int(res_x/2))
    if res_y%2 == 0:
        beg_y = int(res_y/2)
    else:
        beg_y = np.ceil(int(res_y/2))

 #   pcs = np.empty((n_pz_x*n_pz_y,v_x,v_y,size_z))
    pcs = []
    ind = 0
    for i in range(0,n_pz_x):
        for j in range(0,n_pz_y):
            pz = img[beg_x*i:beg_x*i+v_x,beg_y*j:beg_y*j+v_y,::]
            #pcs[ind,::,::,::] = pz
            pcs.append(pz)
            ind +=1
    return pcs,n_pz_x,n_pz_y







import nibabel as nib
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# img = nib.load(os.path.join('images','T1_1.nii'))
# data = img.get_fdata()
# lr = downsample(data)
# lr_nib = nib.nifti1.Nifti1Image(lr ,np.eye(4))
# norm = normalize_image_whitestripe(lr_nib).get_fdata()

# plt.imshow(norm[::,50,::],cmap='gray')
# plt.show()

# #LR

# fig,ax = plt.subplots(1,2)

# a=ax[0].imshow(norm[::,50,::],cmap='gray')
# fig.colorbar(a,ax=ax[0])
# b= ax[1].imshow(lr[::,50,::],cmap='gray')
# fig.colorbar(b,ax=ax[1])
# plt.show()

# import intensity_normalization.plot.hist as hist

# lr_norm_down = downsample(img.get_fdata())
# lr_nib = nib.nifti1.Nifti1Image(lr ,np.eye(4)) 
# norm = normalize_image_whitestripe(lr_nib)
# fig,ax = plt.subplots(1,2)
# hist.hist(norm,ax=ax[0])
# hist.hist(img,ax=ax[1])
# plt.show()

# lr_norm_down=norm.get_fdata()
# vox = voxelize_image(lr_norm_down,(32,32,lr_norm_down.shape[2]),n_samples=40)
# print(vox)
# plt.imshow(vox[10,::,24,::],cmap='gray')
# plt.show()

# import nibabel as nib
# import os
# import matplotlib.pyplot as plt

import os
img = nib.load(os.path.join('images','T1_50.nii'))
img = img.get_fdata()
im_l,n_x,n_y = cropall(img,(32,32))
print(len(im_l))

# wh = normalize_image_whitestripe(img)
# data = wh.get_fdata()
# data_norm = normalize(data)

# fig,ax = plt.subplots(1,2)
# ax[0].imshow(data[::,50,::])
# ax[1].imshow(data_norm[::,50,::])
# plt.show()