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
      raw: 'fov0/raw',
      gt: 'fov0/gt'
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

stack = gp.Stack(5)

request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0), (150, 150))
request[gt] = gp.Roi((0, 0), (150, 150))

pipeline = (
  source +
  normalize +
  random_location +
  stack)

with gp.build(pipeline):
  batch = pipeline.request_batch(request)

# %%
fig,ax = plt.subplots(1,5)
for i in range(5):
  ax[i].imshow(batch[gt].data[i,32])
# %%
def model_step(model, loss_fn, optimizer, feature, label, activation, prediction_type=None, train_step=True):
    
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()
    
    # forward - pass data through model to get logits
    logits = model(feature)
    
    if prediction_type == "three_class":
        label=torch.squeeze(label,1)
        
    # pass logits through final activation to get predictions
    predicted = activation(logits)

    # pass predictions through loss, compare to ground truth
    loss_value = loss_fn(input=predicted, target=label)
    
    # if training mode, backprop and optimizer step
    if train_step:
        loss_value.backward()
        optimizer.step()

    # return outputs and loss
    outputs = {
        'pred': predicted,
        'logits': logits,
    }
    
    return loss_value, outputs
#%%
in_channels = 1
num_fmaps = 12
fmap_inc_factors = 2
downsample_factors = [[2,2],[2,2]]
padding = 'same'

out_channels = 3
activation = torch.nn.Softmax(dim=1)
loss_fn = torch.nn.CrossEntropyLoss()
#dtype = torch.LongTensor
final_kernel_size = 1

unet = UNet(in_channels=1,
           num_fmaps=6,
           fmap_inc_factor=2,
           downsample_factors=downsample_factors,
           padding='same'
          )

final_conv = torch.nn.Conv2d(
    in_channels=num_fmaps,
    out_channels=out_channels,
    kernel_size=final_kernel_size)

net = torch.nn.Sequential(unet, final_conv, activation)
optimizer = torch.optim.Adam(net.parameters())
# %%
from torchsummary import summary
summary(net)
# %%
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
    1: gt
  },
  outputs = {
    0: prediction
  })

stack = gp.Stack(5)

pipeline = (
  source +
  normalize +
  random_location +
  train)

request[prediction] = gp.Roi((0,0), (150, 150))

with gp.build(pipeline):
   batch = pipeline.request_batch(request)
# %%
from funlib.learn.torch.models import UNet, ConvPass

# make sure we all see the same
torch.manual_seed(18)

unet = UNet(
  in_channels=3,
  num_fmaps=4,
  fmap_inc_factor=2,
  downsample_factors=[[2, 2], [2, 2]],
  kernel_size_down=[[[3, 3], [3, 3]]]*3,
  kernel_size_up=[[[3, 3], [3, 3]]]*2,
  padding='same')

model = torch.nn.Sequential(
  unet,
  ConvPass(4, 1, [(1, 1)], activation=None),
  torch.nn.Sigmoid())

loss = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters())
# %%
# create new array key for the network output
prediction = gp.ArrayKey('PREDICTION')

# create a train node using our model, loss, and optimizer
train = gp.torch.Train(
  model,
  loss,
  optimizer,
  inputs = {
    'input': raw
  },
  loss_inputs = {
    0: prediction,
    1: gt
  },
  outputs = {
    0: prediction
  })

pipeline = (
  source +
  normalize +
  random_location +
  simple_augment +
  intensity_augment +
  noise_augment +
  stack +
  train)

# add the prediction to the request
request[prediction] = gp.Roi((0, 0), (150, 150))

with gp.build(pipeline):
  batch = pipeline.request_batch(request)

#imshow(batch[raw].data, batch[gt].data, batch[prediction].data)
# %%
