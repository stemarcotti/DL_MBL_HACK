#%%
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tifffile import imread, imwrite
import zipfile
from glob import glob
from natsort import natsorted
# %%
model_name = 'neuromast_3D'
model = N2V(config=None, name=model_name, basedir="noise2void/models")
# %%

!mkdir '/mnt/efs/shared_data/hack/data/20230811/denoised'
!mkdir '/mnt/efs/shared_data/hack/data/20230504/denoised'
!mkdir '/mnt/efs/shared_data/hack/data/og_cellpose/denoised'
#%%

#datagen = N2V_DataGenerator()
in_dirs = ['/mnt/efs/shared_data/hack/data/20230811',
            '/mnt/efs/shared_data/hack/data/20230504',
            '/mnt/efs/shared_data/hack/data/og_cellpose']
in_path = 'raw'
test_imgs_1 = natsorted(os.listdir(os.path.join(in_dirs[0], in_path)))
test_imgs_2 = natsorted(os.listdir(os.path.join(in_dirs[1], in_path)))
test_imgs_3 = natsorted(os.listdir(os.path.join(in_dirs[2], in_path)))

out_path = 'denoised'
for dir in in_dirs:
    test_imgs = natsorted(os.listdir(os.path.join(dir, in_path)))
    for imgname in test_imgs:
        img = imread(os.path.join(dir, in_path, imgname))
        predicted = model.predict(img, axes="ZYX", n_tiles=(1,2,2))
        imwrite(os.path.join(dir, out_path, imgname), predicted)
# test_imgs_1 = natsorted(glob('/mnt/efs/shared_data/hack/data/20230811/raw/*.tiff'))
# test_imgs_2 = natsorted(glob('/mnt/efs/shared_data/hack/data/20230504/raw/*.tiff'))
# test_imgs_3 = natsorted(glob('/mnt/efs/shared_data/hack/data/og_cellpose/raw/*.TIF'))

# %%
