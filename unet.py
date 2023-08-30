#%%
# import
import gunpowder as gp
from funlib.learn.torch.models import UNet
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import zarr
from skimage import data
from skimage import filters

#%%
folder_path = '/mnt/efs/shared_data/hack/data/20230811/'
zarr_file = folder_path+'20230811_raw.zarr'

#%%
# locate zarr containers
raw = gp.ArrayKey('RAW')
gt = gp.ArrayKey('GROUND_TRUTH')

source = gp.ZarrSource(
    zarr_file,
    {
      raw: 'raw',
      gt: 'ground_truth'
    },
    {
      raw: gp.ArraySpec(interpolatable=True, voxel_size=(1,1)),
      gt: gp.ArraySpec(interpolatable=False, voxel_size=(1,1))
    })
# %%
normalize = gp.Normalize(raw)
random_location = gp.RandomLocation()
simple_augment = gp.SimpleAugment()
intensity_augment = gp.IntensityAugment(
  raw,
  scale_min=0.8,
  scale_max=1.2,
  shift_min=-0.2,
  shift_max=0.2)
noise_augment = gp.NoiseAugment(raw)

# stack = gp.Stack(5)

request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0), (150, 150))
request[gt] = gp.Roi((0, 0), (150, 150))

pipeline = (
  source +
  normalize +
  random_location)

with gp.build(pipeline):
  batch = pipeline.request_batch(request)

# %%
fig,ax = plt.subplots(1,5)
for i in range(5):
  ax[i].imshow(batch[gt].data[i,32])
# %%
