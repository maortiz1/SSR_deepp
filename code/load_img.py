import os
import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

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
    size = round(z_size/(2*down_factor))#size of part according to down factor
    center = round(z_size/2)-1 #center for cutting
    crop_ffs = shift_fft_imag[::,::,center-size:center+size]#cropping kspace
    shift_ifft_crop_ffs = np.fft.ifftshift(crop_ffs)#inverse shiftt
    ifft_crop_ffs = np.fft.ifftn(shift_ifft_crop_ffs)#inverse fast fourier transform
    print('Thickness of slice is now ',down_factor, 'times that it originally was')
    return abs(ifft_crop_ffs)#returns cropped version of the origianl image 
    



