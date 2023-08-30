#%%
#from iohub.ngff import open_ome_zarr
import os
import numpy as np
import tifffile as tiff
import numpy as np
import random
import zarr
from skimage import data
from skimage import filters

#%%
img_path = '/mnt/efs/shared_data/hack/data/20230811/raw/'
mask_path = '/mnt/efs/shared_data/hack/data/20230811/fixed_labels/'
store_path = '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr'

img_list = os.listdir(img_path)
gt_string = '_deconvolved_rho_0.0038_gamma_0.013_m2_Manual_Mask.tiff'

f = zarr.open(store_path, 'w')

#%%
n_files = len(img_list)
crop_size = 640
for file in range(n_files):
    # read files
    zyx = tiff.imread(img_path+img_list[file])
    gt_filename = img_list[file].replace('.tiff', gt_string)
    gt = tiff.imread(mask_path+gt_filename)    
    print(gt_filename)

    # add dimension to get TZYX
    z,y,x = zyx.shape
    tzyx = np.expand_dims(zyx, axis=0)
    # crop if needed
    if (y > crop_size and x > crop_size):
        tzyx = tzyx[:, 0:z, int(y/2 - crop_size//2) : int(y/2 + crop_size//2), 
                    int(x/2 - crop_size//2) : int(x/2 + crop_size//2)]
    
    # save to zarr 
    if file == 0:
        # f = zarr.open(store_path, 'w')
        f['raw'] = tzyx
        f['raw'].attrs['resolution'] = (0.25, 0.075, 0.075) # [um]
        f['ground_truth'] = gt
        f['ground_truth'].attrs['resolution'] = (0.25, 0.075, 0.075) # [um]
    else:
        f['raw'].append(tzyx, axis=0)
        f['ground_truth'].append(gt, axis=0)



# %%
