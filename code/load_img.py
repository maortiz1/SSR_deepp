import os
import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import math

def downsample_z_axis(img, downsampling_factor=3):
    return block_reduce(img, block_size=(1,1,downsampling_factor), func=np.max)


def show_slices(slices):
    fig,axes = plt.subplots(1,len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    
    plt.show()

#function that receives 3D version image along z direction
#down_factor is the parts de space will be divided and only center will be kept
def downsample_onFSpace_3D(img,down_factor=3):
    import numpy as np
    
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
    print('Thickness of slice is now ',down_factor, 'times less that it originally was')
    return abs(ifft_crop_ffs)#returns cropped version of the origianl image 


def downsample_onFSpace_0pad_3D(img_nib,down_factor=3):
    import numpy as np
    import nibabel.processing as pr
    res = downsample_onFSpace_3D(img,down_factor)
    return res
    # fft_imag = np.fft.fftn(img)#n-fourier transform
    # shift_fft_imag = np.fft.fftshift(fft_imag)#fft shifting(zero-frequency component to the center of the spectrum)
    # z_size= shift_fft_imag.shape[2]#z_dimension size for cropping
    # size = math.floor(z_size/(down_factor))#size of part according to down factor
    # center = round(z_size/2)-1 #center for cutting
    # new_fft_zeros = np.zeros(shift_fft_imag.shape)
    # if size%2==0:
    #     new_fft_zeros[::,::,center-size/2:center+size/2] = shift_fft_imag[::,::,center-size/2:center+size/2]
    # else:
    #     size = size+1        
    #     new_fft_zeros[::,::,int(center-(size/2)+1):int(center+(size/2))] = shift_fft_imag[::,::,int(center-(size/2)+1):int(center+(size/2))]
    # #crop_ffs = shift_fft_imag[::,::,center-size:center+size]#cropping kspace
    # shift_ifft_crop_ffs = np.fft.ifftshift(new_fft_zeros)#inverse shiftt
    # ifft_crop_ffs = np.fft.ifftn(shift_ifft_crop_ffs)#inverse fast fourier transform
    # print('Thickness of slice is now ',down_factor, 'times less that it originally was')
    # return abs(ifft_crop_ffs)#returns cropped version of the origianl image 



img = nib.load(os.path.join('..','images','T1_60.nii'))
print(img.header['pixdim'])
data = img.get_fdata()

plt.imshow(data[100,::,::])
plt.title(str(data.shape))
print(data.shape)
plt.show()
fig,axes = plt.subplots(1,3)
axes[0].imshow(data[100,::,::])
axes[1].imshow(data[::,100,::])
axes[2].imshow(data[::,::,100])
plt.show()




img = nib.load(os.path.join('..','images','T1_68.nii'))
print(img.header['pixdim'])
data = img.get_fdata()



plt.imshow(data[100,::,::])
plt.title(str(data.shape))
print(data.shape)
plt.show()
fig,axes = plt.subplots(1,3)
axes[0].imshow(data[100,::,::])
axes[1].imshow(data[::,100,::])
axes[2].imshow(data[::,::,100])
plt.show()




img = nib.load(os.path.join('..','images','T1_96.nii'))
print(img.header['pixdim'])
data = img.get_fdata()



plt.imshow(data[100,::,::])
plt.title(str(data.shape))
print(data.shape)
plt.show()
fig,axes = plt.subplots(1,3)
axes[0].imshow(data[100,::,::])
axes[1].imshow(data[::,100,::])
axes[2].imshow(data[::,::,100])
plt.show()

