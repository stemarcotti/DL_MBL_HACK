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
import datetime
from tqdm import tqdm
from validation_node import ValidationLoss

#logging.basicConfig(level=logging.DEBUG)
#%%
folder_path = '/mnt/efs/shared_data/hack/data/20230811/'
zarr_file = folder_path+'20230811_raw.zarr'
log_dir = '/mnt/efs/shared_data/hack/logs' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#%%
# locate zarr containers
raw = gp.ArrayKey('RAW')
gt = gp.ArrayKey('GROUND_TRUTH')
fg = gp.ArrayKey('FOREGROUND')

train_source = gp.ZarrSource(
    zarr_file,
    {
      raw: 'fov0/raw',
      gt: 'fov0/gt',
      fg: 'fov0/fg_mask'
    },
    {
      raw: gp.ArraySpec(interpolatable=True, voxel_size=(1,1,1)),
      gt: gp.ArraySpec(interpolatable=False, voxel_size=(1,1,1)),
      fg: gp.ArraySpec(interpolatable=False, voxel_size=(1,1,1))
    })
val_source = gp.ZarrSource(
    zarr_file,
    {
      raw: 'fov4/raw',
      gt: 'fov4/gt',
      fg: 'fov4/fg_mask'
    },
    {
      raw: gp.ArraySpec(interpolatable=True, voxel_size=(1,1,1)),
      gt: gp.ArraySpec(interpolatable=False, voxel_size=(1,1,1)),
      fg: gp.ArraySpec(interpolatable=False, voxel_size=(1,1,1))
    })
# %%
normalize_raw = gp.Normalize(raw)
random_location = gp.RandomLocation()

#%%
in_channels = 1
num_fmaps = 12
fmap_inc_factors = 2
downsample_factors = [[2,2],[2,2]]
kernel_sizes = [[(3,3), (3,3)],[(3,3), (3,3)], [(3,3), (3,3)]]
padding = 'same'

out_channels = 2
activation = torch.nn.Sigmoid()
loss_fn = torch.nn.MSELoss()
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
# add affinities
affinities = gp.ArrayKey('AFFINITIES')
add_affinities = gp.AddAffinities([[0,0,1],[0,1,0]], gt, affinities, dtype=np.float32)

#%%
# create new array key for the network output
prediction = gp.ArrayKey('PREDICTION')
roi = gp.Roi((0,0,0), (1,128,128))
request = gp.BatchRequest()
request[gt] = roi
request[raw] = roi
request[prediction] = roi
request[affinities] = roi
request[fg] = roi
#%%
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
  },
  log_dir = log_dir+'/train/',
  log_every = 10)

stack = gp.Stack(5)
# squeeze = gp.Squeeze([raw], axis=0)

snapshot = gp.Snapshot(
  {
    raw: 'raw',
    prediction: 'predictions',
    gt: 'gt',
    affinities: 'affinities'
  },
  output_filename = folder_path + 'iteration_{iteration}.zarr',
  every=1000
)

train_pipeline = (
  train_source +
  normalize_raw +
  random_location +
  gp.Reject(mask=fg, min_masked=0.1) +
  add_affinities + 
  stack +
  train
  )

#%%
val_roi = gp.Roi((0,0,0), (1,256,256))
val_request = gp.BatchRequest()
val_request[gt] = val_roi
val_request[raw] = val_roi
val_request[prediction] = val_roi
val_request[affinities] = val_roi

predict = gp.torch.Predict(
  model = net,
  inputs = {
    'input': raw
  },
  outputs = {
    0 : prediction,
  })

val_loss = ValidationLoss(loss = loss_fn,
               inputs = {0:prediction, 1:affinities},
               log_dir=log_dir+'/validation/',
               log_every=100)

val_pipeline = (
  val_source +
  gp.Normalize(raw) +
  gp.AddAffinities([[0,0,1],[0,1,0]], gt, affinities, dtype=np.float32) +
  gp.Stack(1) + 
  predict +
  gp.Scan(reference=request) +
  val_loss
)

#%%

with gp.build(train_pipeline), gp.build(val_pipeline):
   for i in tqdm(range(1000)):
     batch = train_pipeline.request_batch(request)

     if i % 100 == 0:
       print("validating")
       net.eval()
       batch_val = val_pipeline.request_batch(val_request)
       net.train(True)
print(batch[affinities].data.shape)
print(batch[prediction].data.shape)
# %
# %%
fig,ax = plt.subplots(1,5)
for i in range(5):
  #ax[i].imshow(batch[affinities].data[i,1,0])
  ax[i].imshow(batch[prediction].data[i,1,0])
plt.show()
# %%
