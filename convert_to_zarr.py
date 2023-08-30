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
print(img_list)

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
        f = zarr.open(store_path, 'w')
        f['raw'] = tzyx
        f['raw'].attrs['resolution'] = (1, 1) # check
        f['ground_truth'] = gt
        f['ground_truth'].attrs['resolution'] = (1, 1) # check
    else:
        f['raw'].append(tzyx, axis=0)
        f['gt'].append(gt, axis=0)


    break


# #%%
# zyx = tiff.imread('./data/20230811/raw/' + imglist[1])
# z,y,x = zyx.shape
# tzyx = np.expand_dims(zyx, axis=0)
# print(tzyx.shape)
# if (y > 640 and x > 640):
#     tzyx = tzyx[:, 0:z, int(y/2 - 320) : int(y/2 + 320), int(x/2 - 320) : int(x/2 + 320)]
# gtname = imglist[1].replace('.tiff', '_deconvolved_rho_0.0038_gamma_0.013_m2_Manual_Mask.tiff')
# gt = tiff.imread('./data/20230811/fixed_labels/' + gtname)
# store_path = './data/20230811_raw.zarr'
# # with open_ome_zarr(
# #     store_path, layout="fov", mode="a", channel_names=["GFP"]
# # ) as dataset:
# #     dataset["img"] = zyx
# f = zarr.open('sample_data9.zarr', 'w')
# f['raw'] = tzyx
# f['raw'].attrs['resolution'] = (1, 1)
# f['ground_truth'] = gt
# f['ground_truth'].attrs['resolution'] = (1, 1)

# img2 = tiff.imread('./data/20230811/raw/' + imglist[2])
# img2 = np.expand_dims(img2, axis=0)
# img2 = img2[:, 0:z, int(y/2 - 320) : int(y/2 + 320), int(x/2 - 320) : int(x/2 + 320)]
# print(img2.shape)
# f['raw'].append(img2, axis=0)
# # %%
