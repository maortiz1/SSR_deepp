import os
import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def downsample_z_axis(img, downsampling_factor=3):
    return block_reduce(img, block_size=(downsampling_factor,1,1), func=np.max)


def show_slices(slices):
    fig,axes = plt.subplots(1,len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    
    plt.show()


img = nib.load('T1_3.nii')


img2= img.get_fdata()

slices_0= img2[100,96:128,96:128]

downslampled = downsample_z_axis(img2,downsampling_factor=3)
print(downslampled.shape)

# try passing to fourier and filtering to get Downsampled images
from scipy import fftpack


img_loaded = nib.load('T1_3.nii')

img_data= img.get_fdata()#.transpose((2,1,0))
print(img_data.shape)

n_transform = np.fft.fftn(img_data)



nshift = np.fft.fftshift(n_transform,axes=0)
freq = np.fft.fftfreq(img_data.shape[0])
f=nshift
f[0:58,::,::]=0
f[117:-1,::,::]=0
f2=nshift[58:117,::,::]
f2_dows = np.fft.ifftn(f2).squeeze()
downsampled=np.fft.ifftn(f).squeeze()
print(downsampled.shape)
print(f2_dows.shape)
slice=100

indx= round(slice*f2.shape[0]/img_data.shape[1])
print(indx)
print(nshift.shape)
fig,ax=plt.subplots(1,3)
ax[0].imshow(abs(f2_dows[::,50,::]),cmap="gray")
ax[1].imshow(abs(downsampled[::,50,::]),cmap="gray")
ax[2].imshow(img_data[::,50,::],cmap="gray")
plt.show()




