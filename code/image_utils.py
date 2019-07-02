
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
        crop_ffs = shift_fft_imag[::,::,center-size/2:center+size/2]
    else:
        size = size+1        
        crop_ffs = shift_fft_imag[::,::,int(center-(size/2)+1):int(center+(size/2))]
    #crop_ffs = shift_fft_imag[::,::,center-size:center+size]#cropping kspace
    shift_ifft_crop_ffs = np.fft.ifftshift(crop_ffs)#inverse shiftt
    ifft_crop_ffs = np.fft.ifftn(shift_ifft_crop_ffs)#inverse fast fourier transform

    return abs(ifft_crop_ffs)#returns cropped version of the original image 


def normalize_image_whitestripe(img,contrast= 'T1'):
    from intensity_normalization.normalize import whitestripe
    from torchvision.transforms import normalize
    mask = whitestripe.whitestripe(img,contrast)
    norm_image = whitestripe.whitestripe_norm(img,mask)
    return norm_image



def voxelize_image(img,vox_size):
    size_x = img.shape[0]
    size_y = img.shape[1]
    size_z = img.shape[2]
    

import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
img = nib.load(os.path.join('images','T1_1.nii'))
data = img.get_fdata()
lr = downsample(data)
lr_nib = nib.nifti1.Nifti1Image(lr ,np.eye(4))
norm = normalize_image_whitestripe(lr_nib).get_fdata()

plt.imshow(norm[::,50,::],cmap='gray')
plt.show()

#LR

fig,ax = plt.subplots(1,2)

a=ax[0].imshow(norm[::,50,::],cmap='gray')
fig.colorbar(a,ax=ax[0])
b= ax[1].imshow(lr[::,50,::],cmap='gray')
fig.colorbar(b,ax=ax[1])
plt.show()

import intensity_normalization.plot.hist as hist
norm = normalize_image_whitestripe(img)
lr_norm_down = downsample(norm.get_fdata())
lr_nib = nib.nifti1.Nifti1Image(lr ,np.eye(4))
fig,ax = plt.subplots(1,2)
hist.hist(norm,ax=ax[0])
hist.hist(img,ax=ax[1])
plt.show()
