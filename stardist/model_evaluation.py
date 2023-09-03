#%%
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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
# to run this you need to have an environment with stardist!
# the easiest way to do so is to create an environment from yml
# the yml can be found in the stardist repo (either clone or copy the yml locally)
# mamba env create -f stardist/extras/cuda_setup_conda/stardist_cuda_11.0.yml -n stardist
#%%
# load test data
in_path = '/mnt/efs/shared_data/hack/stardist/stardist20230902'

X = natsorted(glob(os.path.join(in_path, 'raw/*')))
Y = natsorted(glob(os.path.join(in_path, 'gt/*')))
Y_pred = natsorted(glob(os.path.join(in_path, 'pred/*')))

X = list(map(imread,X))
Y = list(map(imread,Y))
Y_pred = list(map(imread,Y_pred))

#%%
# compute stats
stats = matching_dataset(Y, Y_pred, thresh=0.7)
print(stats)

# %%
