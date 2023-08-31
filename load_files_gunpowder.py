import matplotlib.pyplot as plt
import numpy as np
import random
import zarr
from skimage import data
from skimage import filters
import gunpowder as gp
import os
import numpy as np
import tifffile as tiff
from funlib.learn.torch.models import UNet, ConvPass
import logging
import math
import torch
import multiprocessing
#%%
# This loads the zarr file
load_path = '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr'
f = zarr.open(load_path, 'r')
f['fov0/raw'].shape

#%%
# helper function to show images. 
# TODO: make this useful for different cases
def imshow(raw, slice_index=0, ground_truth=None, prediction=None):
    num_images = 1
    rows = 1

    cols = num_images

    # Update cols to 2 if ground_truth is provided
    if ground_truth is not None:
        cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8), sharex=True, sharey=True, squeeze=False)

    # Select and display the specified slice from the 'raw' data
    selected_slice = raw[slice_index, :, :]
    axes[0][0].imshow(selected_slice, cmap='gray')  # Assuming grayscale images

    # If ground_truth is provided, display it in the same row
    if ground_truth is not None:
        selected_slice_gt = ground_truth[slice_index, :, :]

        axes[0][1].imshow(selected_slice_gt, cmap='viridis',  vmin=0, vmax=20)  # Assuming grayscale ground truth

    plt.show()

#%% 
# Show the raw data and the ground truth

imshow(f['fov0/raw'], ground_truth = f['fov0/gt'], slice_index = 30)

#%%

# Make an array key for the raw data and the ground truth
raw = gp.ArrayKey('RAW')
gt = gp.ArrayKey('GROUND_TRUTH')

#%%
# This are the parameters for the augmentations we are using in the gunpowder pipeline

random_location = gp.RandomLocation()
simple_augment = gp.SimpleAugment()
stack = gp.Stack(5)

elastic_augment = gp.ElasticAugment(
  control_point_spacing=(20, 20),
  jitter_sigma=(1.0, 1.0),
  rotation_interval=(0, math.pi/2))

normalize = gp.Normalize(raw)
intensity_augment = gp.IntensityAugment(
  raw,
  scale_min=0.8,
  scale_max=1.2,
  shift_min=-0.2,
  shift_max=0.2)
noise_augment = gp.NoiseAugment(raw)

pad_raw = gp.Pad(raw, None)
pad_gt = gp.Pad(gt, 0)

#%%
# Gunpowder pipeline

source = tuple(gp.ZarrSource(
    load_path,
    {
      raw: f'fov{i}/raw',
      gt: f'fov{i}/gt'
    },
    {
      raw: gp.ArraySpec(interpolatable=True,
        voxel_size=(1,1)),
      gt: gp.ArraySpec(interpolatable=False,
        voxel_size=(1,1)),
    }) + normalize + pad_raw + pad_gt + random_location for i in [1])


# This is the pipeline we are using
pipeline = source
pipeline += gp.RandomProvider()
pipeline += simple_augment
pipeline += elastic_augment
pipeline += intensity_augment
pipeline += noise_augment
  # +stack
  # reject
  # )


# formulate a request for "raw"
request = gp.BatchRequest()
request[raw] = gp.Roi((0,0), (256, 256))
request[gt] = gp.Roi((0,0), (256, 256))

# build the pipeline...
with gp.build(pipeline):

    # ...and request a batch
    batch = pipeline.request_batch(request)

# show the content of the batch
print(f"batch returned: {batch}")

imshow(batch[raw].data, ground_truth=batch[gt].data, slice_index = 12)

#%%
# Test to check whether the shape of the output batch is correct
print(batch[raw].data.shape)
print(batch[gt].data.shape)
print("Well done you, you completed the gunpowder augmentations!")
