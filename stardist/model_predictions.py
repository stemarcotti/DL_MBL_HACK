#%%
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import os

from glob import glob
from tqdm import tqdm
from tifffile import imread, imwrite
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

from natsort import natsorted
# import zarr

np.random.seed(42)
lbl_cmap = random_label_cmap()

#%%
# load test data
X1 = natsorted(glob('/mnt/efs/shared_data/hack/data/og_cellpose/denoised/*.TIF'))
X2 = natsorted(glob('/mnt/efs/shared_data/hack/data/20230811/denoised/*.tiff'))
X3 = natsorted(glob('/mnt/efs/shared_data/hack/data/20230504/denoised/*.tiff'))

Y1 = natsorted(glob('/mnt/efs/shared_data/hack/data/og_cellpose/fixed_labels/*Manual_Mask_median.tiff'))
Y2 = natsorted(glob('/mnt/efs/shared_data/hack/data/20230811/fixed_labels/*_Manual_Mask.tiff'))
Y3 = natsorted(glob('/mnt/efs/shared_data/hack/data/20230504/fixed_labels/*_Manual_Mask.tiff'))

# get test split
X1_test = X1[23:]
X2_test = X2[3:]
X3_test = X3[3]

Y1_test = Y1[23:]
Y2_test = Y2[3:]
Y3_test = Y3[3]

#%%
X_test = X1_test + X2_test + [X3_test]
Y_test = Y1_test + Y2_test + [Y3_test]

X_test = list(map(imread,X_test))
Y_test = list(map(imread,Y_test))

crop_size = 640
n_im_test = len(X_test)
X_test_crop = []

for i in range(n_im_test):
    img = X_test[i]
    z,y,x = img.shape

    # crop if needed
    if (y > crop_size and x > crop_size):
        img = img[0:z, int(y/2 - crop_size//2) : int(y/2 + crop_size//2), 
                    int(x/2 - crop_size//2) : int(x/2 + crop_size//2)]
        
    X_test_crop.append(img)
        
X_test = X_test_crop

axis_norm = (0,1,2)   # normalize channels independently

X_test = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_test)]
Y_test = [fill_label_holes(y) for y in tqdm(Y_test)]


#%%
# load model
model_name = 'stardist20230902'
model = StarDist3D(None, name=model_name, basedir='models')

#%%
# predict instance labels
Y_pred = [model.predict_instances(x, n_tiles=(1,2,2), show_tile_progress=False)[0]
              for x in tqdm(X_test)]
#%%
# save output
# !mkdir '/mnt/efs/shared_data/hack/stardist/stardist20230902_on_denoised/raw'
# !mkdir '/mnt/efs/shared_data/hack/stardist/stardist20230902_on_denoised/gt'
# !mkdir '/mnt/efs/shared_data/hack/stardist/stardist20230902_on_denoised/pred'
out_path = '/mnt/efs/shared_data/hack/stardist/stardist20230902_on_denoised'
os.system(f"chmod -R 777 {out_path}")
for i in range(len(X_test)):
    imwrite(os.path.join(out_path, f'raw/raw{i}.tiff'),X_test[i]) 
    imwrite(os.path.join(out_path, f'gt/gt{i}.tiff'),Y_test[i]) 
    imwrite(os.path.join(out_path, f'pred/pred{i}.tiff'),Y_pred[i]) 

#%%
# 

#%%
# 

#%%
# 