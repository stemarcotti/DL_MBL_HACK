#%%
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

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
#lbl_cmap = random_label_cmap()

#%%
# load test data
in_path = '/mnt/efs/shared_data/hack/data/20230410/pos006_timelapse'
img_list = natsorted(os.listdir(in_path))

axis_norm = (0,1,2)   # normalize channels independently
crop_size = 640
#%%
# load model
model_name = 'stardist20230902_denoised'
model = StarDist3D(None, name=model_name, basedir='models')

#%%
# save output
#! mkdir '/mnt/efs/shared_data/hack/stardist/stardist20230902_denoised/raw'
#! mkdir '/mnt/efs/shared_data/hack/stardist/stardist20230902_denoised/gt'
#! mkdir '/mnt/efs/shared_data/hack/stardist/stardist20230902_denoised/pred'
out_path = '/mnt/efs/shared_data/hack/data/20230410/pos006_timelapse_pred'
#os.system(f"chmod -R 777 {out_path}")
for imgname in img_list:
    print(imgname)
    img = imread(os.path.join(in_path, imgname))
    z,y,x = img.shape

    # crop if needed
    if (y > crop_size and x > crop_size):
        img = img[0:z, int(y/2 - crop_size//2) : int(y/2 + crop_size//2), 
                    int(x/2 - crop_size//2) : int(x/2 + crop_size//2)]
    img = normalize(img,1,99.8,axis=axis_norm)
    pred = model.predict_instances(img, n_tiles=(1,2,2), show_tile_progress=False)[0]
    imwrite(os.path.join(out_path, imgname),img)


#%%
# 

#%%
# 

#%%
# 