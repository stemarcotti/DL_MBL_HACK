
#@title import packages

import gunpowder as gp
import h5py
import io
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import requests
import torch
import zarr

from funlib.learn.torch.models import UNet, ConvPass
# from lsd.gp import AddLocalShapeDescriptor
from lsd.train.gp import AddLocalShapeDescriptor
from gunpowder.torch import Train

# %matplotlib inline
logging.basicConfig(level=logging.DEBUG)
from gunpowder_augmentor import prepare_gunpowder_pipeline



##### GUNPOWDER PIPELINE #####

def prepare_gunpowder_pipeline(load_path, output_shape=(30, 128, 128), device='cuda', voxels=(250,75,75)): #stack_size=5
    output_shape = gp.Coordinate(output_shape)
    voxels = gp.Coordinate(voxels)
    # Define keys
    raw = gp.ArrayKey('RAW')
    gt = gp.ArrayKey('GROUND_TRUTH')
    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    gt_affs = gp.ArrayKey('GT_AFFS')

    # gt_lsds = tf.placeholder(tf.float32, shape=(10,) + output_shape)

    ########  Define paramters ########

    # How many stacks to request
    #stack = gp.Stack(stack_size)

    # random sampeling 
    random_location = gp.RandomLocation()

    # geometric augmentation
    simple_augment = gp.SimpleAugment(
        mirror_probs = [0,0,0],
        transpose_probs = [0,0,0])
    
    elastic_augment = gp.ElasticAugment(
        control_point_spacing=(30, 30, 30),
        jitter_sigma=(1.0, 1.0, 1.0),
        rotation_interval=(0, math.pi/2),
        spatial_dims = 3
    )

    # signal augmentations
    intensity_augment = gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.01,
        shift_max=0.01)

    noise_augment = gp.NoiseAugment(raw, mode='poisson')

    normalize_raw = gp.Normalize(raw)
    


    ##### Gunpowder pipeline #####

    # Load the data
    source = tuple(gp.ZarrSource(
    load_path,
    {
      raw: f'fov{i}/raw',
      gt: f'fov{i}/gt'
    },
    {
      raw: gp.ArraySpec(interpolatable=True,
        voxel_size=voxels),
      gt: gp.ArraySpec(interpolatable=False,
        voxel_size=voxels),
    })  + normalize_raw + random_location for i in [1]) 


    # Create the pipeline
    pipeline = source
    pipeline += gp.RandomProvider()
    pipeline += simple_augment
    # pipeline += elastic_augment
    # pipeline += stack
    pipeline += intensity_augment
    pipeline += noise_augment

    pipeline += AddLocalShapeDescriptor(
        gt,
        gt_lsds,
        lsds_mask=lsds_weights,
        sigma=1500,
        downsample=1)
    
    
    pipeline += gp.AddAffinities(
        affinity_neighborhood=[
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1]],
        labels=gt, #labels
        affinities=gt_affs,
        dtype=np.float32)
        

    # Define a batch request
    request = gp.BatchRequest()
    request[raw] = gp.Roi((0, 0, 0), output_shape*voxels)
    request[gt] = gp.Roi((0, 0, 0), output_shape*voxels)
    request[gt_lsds]= gp.Roi((0, 0, 0), output_shape*voxels)
    # request[lsds_weights]= gp.Roi((0, 0, 0), output_shape)
    request[gt_affs]= gp.Roi((0, 0, 0), output_shape*voxels)
    # request.add(gt_lsds, gp.Roi((0, 0, 0), output_shape))

    return pipeline, request
    
load_path = '/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr'
output_shape = (20, 265, 265)
stack_size = 6
device = 'cpu'

pipeline, request = prepare_gunpowder_pipeline(load_path, output_shape, device) #stack_size 


# Build the pipeline and request the batch
with gp.build(pipeline):
    batch = pipeline.request_batch(request)

# Move the batch data to the specified device
if device == 'cuda':
    batch[raw].data = torch.from_numpy(batch[raw].data).cuda()
    batch[gt].data = torch.from_numpy(batch[gt].data).cuda()

# Make some plots to check the data