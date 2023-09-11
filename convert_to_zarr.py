#%%
import os
import numpy as np
import tifffile as tiff
import numpy as np
import random
import zarr
from skimage import data
from skimage import filters
from natsort import natsorted

#%%
img_path = '/mnt/efs/shared_data/hack/data/20230811/raw/'
mask_path = '/mnt/efs/shared_data/hack/data/20230811/fixed_labels/'
store_path = '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr'

img_list = natsorted(os.listdir(img_path))
gt_string = '_deconvolved_rho_0.0038_gamma_0.013_m2_Manual_Mask.tiff'
n_files = len(img_list)
#%%
f = zarr.group()
for i in range(n_files):
    f.create_group(f'fov{i}/raw')
    f.create_group(f'fov{i}/gt')
    f.create_group(f'fov{i}/fg_mask')
# %%

crop_size = 640

f = zarr.open(store_path, 'w')
for file in range(n_files):
    # read files
    zyx = tiff.imread(img_path+img_list[file])
    gt_filename = img_list[file].replace('.tiff', gt_string)
    gt = tiff.imread(mask_path+gt_filename)    
    print(gt_filename)

    z,y,x = zyx.shape

    # crop if needed
    if (y > crop_size and x > crop_size):
        zyx = zyx[0:z, int(y/2 - crop_size//2) : int(y/2 + crop_size//2), 
                    int(x/2 - crop_size//2) : int(x/2 + crop_size//2)]
    
    f[f'fov{file}/raw'] = zyx
    f[f'fov{file}/raw'].attrs['resolution'] = (0.25, 0.075, 0.075) # [um]
    f[f'fov{file}/gt'] = gt.astype('int64')
    f[f'fov{file}/gt'].attrs['resolution'] = (0.25, 0.075, 0.075) # [um]
    f[f'fov{file}/fg_mask'] = (gt>0).astype('uint8')
    f[f'fov{file}/fg_mask'].attrs['resolution'] = (0.25, 0.075, 0.075) # [um]
    os.system(f"chmod -R 777 {store_path}")
# %%

# %%
