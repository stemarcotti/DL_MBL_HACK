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
import logging
#logging.basicConfig(level=logging.DEBUG)
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
      raw: 'fov0/raw',
      gt: 'fov0/gt'
    },
    {
      raw: gp.ArraySpec(interpolatable=True, voxel_size=(1,1,1)),
      gt: gp.ArraySpec(interpolatable=False, voxel_size=(1,1,1))
    })
# %%
#normalize = gp.Normalize(raw)
normalize_raw = gp.Normalize(raw)
normalize_labels = gp.Normalize(gt, factor=1.0/255.0)

random_location = gp.RandomLocation()
simple_augment = gp.SimpleAugment()
intensity_augment = gp.IntensityAugment(
  raw,
  scale_min=0.8,
  scale_max=1.2,
  shift_min=-0.2,
  shift_max=0.2)
noise_augment = gp.NoiseAugment(raw)

stack = gp.Stack(5)

request = gp.BatchRequest()
# request[raw] = gp.Roi((0, 0), (150, 150))
# request[gt] = gp.Roi((0, 0), (150, 150))

# pipeline = (
#   source +
#   normalize +
#   random_location +
#   stack)

# with gp.build(pipeline):
#   batch = pipeline.request_batch(request)

# %%
# fig,ax = plt.subplots(1,5)
# for i in range(5):
#   ax[i].imshow(batch[gt].data[i,32])
# %%
#%%
in_channels = 1
num_fmaps = 12
fmap_inc_factors = 2
downsample_factors = [[2,2],[2,2]]
kernel_sizes = [[(3,3), (3,3)],[(3,3), (3,3)], [(3,3), (3,3)]]
padding = 'same'

out_channels = 2
activation = torch.nn.Sigmoid()
#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.MSELoss()
#dtype = torch.LongTensor
final_kernel_size = 1

unet = UNet(in_channels=in_channels,
           num_fmaps=num_fmaps,
           fmap_inc_factor=fmap_inc_factors,
           kernel_size_down=kernel_sizes,
           kernel_size_up=kernel_sizes[:-1],
           downsample_factors=downsample_factors,
           padding=padding,
           fov=(1,1),
           voxel_size=(1,1)
          )

final_conv = torch.nn.Conv2d(
    in_channels=num_fmaps,
    out_channels=out_channels,
    kernel_size=final_kernel_size)

net = torch.nn.Sequential(unet,
                          final_conv,
                          activation,
                          torch.nn.Unflatten(1, torch.Size([out_channels,1])))
net.train(True)
optimizer = torch.optim.Adam(net.parameters())
# %%
# from torchsummary import summary
# summary(net)
# %%
# add affinities
affinities = gp.ArrayKey('AFFINITIES')
add_affinities = gp.AddAffinities([[0,0,1],[0,1,0]], gt, affinities, dtype=np.float32)
normalize_affs = gp.Normalize(affinities, 1.0, dtype=np.float32)
#add_affinities = gp.AddAffinities([[(0,1), (1,0)],[(0,1), (1,0)], [(0,1), (1,0)]], gt, affinities)

#%%
# create new array key for the network output
prediction = gp.ArrayKey('PREDICTION')
unsqueeze = gp.Unsqueeze([raw, gt], axis=0)
# create a train node using our model, loss, and optimizer
train = gp.torch.Train(
  net,
  loss_fn,
  optimizer,
  inputs = {
    'input': raw
  },
  loss_inputs = {
    0: prediction,
    1: affinities
  },
  outputs = {
    0: prediction
  })

stack = gp.Stack(5)
squeeze = gp.Squeeze([raw], axis=0)
pipeline = (
  source +
  normalize_raw +
  random_location +
  add_affinities + 
  #normalize_affs +
  #gp.Normalize(affinities)+
  #normalize_labels +
  stack +
  #unsqueeze +
  train
  )
request[gt] = gp.Roi((0,0,0), (1,128,128))
request[raw] = gp.Roi((0,0,0), (1,128,128))
request[prediction] = gp.Roi((0, 0, 0), ( 1, 128, 128))
request[affinities] = gp.Roi((0,0,0), (1, 128, 128))
with gp.build(pipeline):
   for i in range(1000):
     batch = pipeline.request_batch(request)
print(batch[affinities].data.shape)
print(batch[prediction].data.shape)
# %
# %%
fig,ax = plt.subplots(1,5)
for i in range(5):
  #ax[i].imshow(batch[affinities].data[i,1,0])
  ax[i].imshow(batch[prediction].data[i,1,0])
#plt.imshow(batch[prediction].data[0][0])
plt.show()
# %%
