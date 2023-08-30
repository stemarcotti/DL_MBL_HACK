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
raw = gp.ArrayKey('RAW')
gt = gp.ArrayKey('GROUND_TRUTH')

source = gp.ZarrSource(
    zarr_file,
    {
      raw: 'raw',
      gt: 'ground_truth'
    },
    {
      raw: gp.ArraySpec(interpolatable=True),
      gt: gp.ArraySpec(interpolatable=False)
    })
# %%


# %%
