
#%%
from scipy.ndimage import maximum_filter1d, find_objects
import torch
import numpy as np
from tqdm import trange
from numba import njit, float32, int32, vectorize
import gunpowder as gp
import math
import zarr
import matplotlib.pyplot as plt
from cellpose import dynamics

import torch
from torch import optim, nn
TORCH_ENABLED = True 
torch_GPU = torch.device('cuda')
torch_CPU = torch.device('cpu')

#%%
# This loads the zarr file
load_path = '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr'
f = zarr.open(load_path, 'r')
f['fov0/raw'].shape

#%%

# add another dimension to the shape of mask
mask = f['fov0/gt'][20, 250:450,250:450]
maskexpand = np.expand_dims(mask, axis=0)
#%%

maskoutput = dynamics.masks_to_flows(maskexpand)

#%%
class maskstocellpose_flows(gp.BatchFilter):

  def __init__(self, in_array, out_array):
    self.in_array = in_array
    self.out_array = out_array

  def setup(self):

    # tell downstream nodes about the new array
    self.provides(
      self.out_array,
      self.spec[self.in_array].copy())

  def prepare(self, request):

    # to deliver inverted raw data, we need raw data in the same ROI
    deps = gp.BatchRequest()
    deps[self.in_array] = request[self.out_array].copy()

    return deps

  def process(self, batch, request):

    # take the masks and compute the flows
    flows = dynamics.masks_to_flows(batch[self.in_array].data)


    # create the array spec for the new array
    spec = batch[self.in_array].spec.copy()
    spec.roi = request[self.out_array].roi.copy()
    spec.dtype = np.float32 

    # create a new batch to hold the new array
    batch = gp.Batch()

    # create a new array
    flowarray = gp.Array(flows, spec)

    # store it in the batch
    batch[self.out_array] = flowarray

    # return the new batch
    return batch
  #%%

  # declare a new array key for inverted raw
flows_array = gp.ArrayKey('FLOW')
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
    }) + normalize + pad_raw + pad_gt  +random_location for i in [1])
#%%


## Got here - need to build in our pipeline there
pipeline = source
pipeline += gp.RandomProvider()
pipeline += simple_augment
pipeline += elastic_augment
pipeline += intensity_augment
pipeline += noise_augment
pipeline += maskstocellpose_flows(in_array=gt, out_array=flows_array)
  

request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0), (128, 128))
request[gt] = gp.Roi((0, 0), (128, 128))
request[flows_array] = gp.Roi((0, 0), (128, 128))

with gp.build(pipeline):
  batch = pipeline.request_batch(request)



print(batch[raw].data.shape)
print(batch[gt].data.shape)
print(batch[flows_array].data.shape)
# %%

plt.imshow(batch[flows_array].data[1,0])
# %%
